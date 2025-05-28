# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch
from pathlib import Path
import signal
import sys
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter
import numpy as np

from PySide6.QtCore import (
    QAbstractListModel,
    QAbstractTableModel,
    QItemSelection,
    QModelIndex,
    QObject,
    # QSortFilterProxyModel,
    Qt,
)
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFileDialog,
    QLabel,
    QLineEdit,
    QListView,
    QHBoxLayout,
    QMainWindow,
    QProgressBar,
    QSplitter,
    QTableView,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from pffdtd.common.plot import plot_styles
from pffdtd.absorption.database import read_absorption_database_excel
from pffdtd.analysis.response import plot_musical_response
from pffdtd.analysis.rt60 import reverberation_time
from pffdtd.analysis.summary import plot_impulse_response_summary
from pffdtd.signals.octave import center_frequencies
from pffdtd.signals.wavfile import wavread


class MatplotLibCanvas(FigureCanvasQTAgg):
    fig: Figure
    axes: Axes | np.ndarray[Axes] | np.ndarray[np.ndarray[Axes]]

    def __init__(self, parent=None, width=5, height=4, dpi=100, subplot_args=None):
        subplot_args = subplot_args if subplot_args else {'nrows': 1, 'ncols': 1}
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.subplots(**subplot_args)
        super().__init__(self.fig)

    def clear_all_axes(self):
        if isinstance(self.axes, np.ndarray):
            for ax in self.axes.flat:
                ax.clear()
        else:
            self.axes.clear()


class MaterialTableModel(QAbstractTableModel):
    def __init__(self):
        super().__init__()
        self._df = read_absorption_database_excel('./sim_data/abstab_wf.xls')

    def data(self, index: QModelIndex, role: int) -> Any:
        if role == Qt.ItemDataRole.DisplayRole:
            value = self._df.iloc[index.row(), index.column()]
            return str(value)
        return None

    def rowCount(self, index):
        return self._df.shape[0]

    def columnCount(self, index):
        return self._df.shape[1]

    def headerData(self, section, orientation, role):
        # section is the index of the column/row.
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._df.columns[section])

            if orientation == Qt.Orientation.Vertical:
                return str(self._df.index[section])

        return None


class MaterialTable(QWidget):
    def __init__(self):
        super().__init__()

        # TABLE
        self.model = MaterialTableModel()
        # self.proxy_model = QSortFilterProxyModel()
        # self.proxy_model.setFilterKeyColumn(-1)  # Search all columns.
        # self.proxy_model.setSourceModel(self.model)
        # self.proxy_model.sort(0, Qt.SortOrder.AscendingOrder)

        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.selectionModel().selectionChanged.connect(self.updateSelection)

        self.search = QLineEdit()
        # self.search.textChanged.connect(self.proxy_model.setFilterFixedString)

        vbox = QVBoxLayout()
        vbox.addWidget(self.search)
        vbox.addWidget(self.table)

        tableWithSearch = QWidget()
        tableWithSearch.setLayout(vbox)

        # PLOT
        self.canvas = MatplotLibCanvas(self, width=5, height=4)

        # LAYOUT
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(tableWithSearch)
        splitter.addWidget(self.canvas)
        layout = QHBoxLayout()
        layout.addWidget(splitter)
        self.setLayout(layout)

    def updateSelection(self, selected: QItemSelection, deselected: QItemSelection):
        def get_id(idx):
            return self.model.headerData(idx.row(), Qt.Orientation.Vertical, Qt.ItemDataRole.DisplayRole)

        rows = self.table.selectionModel().selectedRows()
        ids = [get_id(r) for r in rows]

        ax = self.canvas.axes
        ax.clear()

        if len(ids) == 0:
            self.canvas.draw()
            return

        bands = [63, 125, 250, 500, 1000, 2000, 4000, 8000]
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel('Absorption [Sabs]')

        for idx in ids:
            coefficients = self.model._df.loc[int(idx)][bands]
            description = self.model._df.loc[int(idx)]['description']
            ax.semilogx(bands, coefficients, label=description[:150])
            ax.scatter(bands, coefficients, color='red')

        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        ax.xaxis.set_major_formatter(formatter)

        ax.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
        ax.legend(loc='upper left')
        self.canvas.draw()


class OpenFilesListModel(QAbstractListModel):
    _open_files: list[Path]

    def __init__(self, parent: QObject | None) -> None:
        super().__init__(parent=parent)
        self._open_files = []

    def data(self, index: QModelIndex, role: int) -> Any:
        if 0 <= index.row() < self.rowCount():
            if role == Qt.ItemDataRole.DisplayRole:
                path = self._open_files[index.row()]
                return path.stem

        return None

    def rowCount(self, index: QModelIndex = QModelIndex()) -> int:
        return len(self._open_files)

    def addFile(self, path: Path) -> None:
        self.beginInsertRows(QModelIndex(), self.rowCount(), self.rowCount())
        self._open_files.append(path)
        self.endInsertRows()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Status Bar
        self.status = QLabel()
        self.status.setText('Foo bar baz.')
        self.status.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        self.statusBar().addPermanentWidget(self.status, stretch=6)
        self.statusBar().addPermanentWidget(self.progress, stretch=4)

        # Create the "Import IR" action
        importImpulseAction = QAction('&Import IR...', self)
        importImpulseAction.setShortcut('Ctrl+Shift+I')
        importImpulseAction.setStatusTip('Open a impulse file')
        importImpulseAction.triggered.connect(self.onImportImpulseResponse)

        # Create the "Exit" action
        exitAction = QAction('&Exit...', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit the program')
        exitAction.triggered.connect(self.onExit)

        # Menu Bar
        menu = self.menuBar()
        fileMenu = menu.addMenu('&File')

        importMenu = fileMenu.addMenu('&Import')
        importMenu.addAction(importImpulseAction)

        fileMenu.addSeparator()
        fileMenu.addAction(exitAction)

        # Toolbar
        toolbar = QToolBar('My main toolbar')
        toolbar.addAction(importImpulseAction)
        toolbar.addSeparator()
        toolbar.addAction(exitAction)
        self.addToolBar(toolbar)

        # Open Files
        self.fileListModel = OpenFilesListModel(self)
        self.fileList = QListView()
        self.fileList.setModel(self.fileListModel)
        # self.fileList.selectionModel().selectionChanged.connect()

        # Tabs
        self.summary = MatplotLibCanvas(self, width=5, height=4, dpi=100, subplot_args={'nrows': 3, 'ncols': 2})
        self.musical = MatplotLibCanvas(self, width=5, height=4, dpi=100)
        self.edc = MatplotLibCanvas(self, width=5, height=4, dpi=100)
        self.materials = MaterialTable()

        self.tabs = QTabWidget(tabPosition=QTabWidget.TabPosition.North)
        self.tabs.addTab(self.summary, 'Summary')
        self.tabs.addTab(self.musical, 'Musical')
        self.tabs.addTab(self.edc, 'EDC')
        self.tabs.addTab(self.materials, 'Materials')

        # Main Layout
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.fileList)
        splitter.addWidget(self.tabs)

        self.setCentralWidget(splitter)
        self.setWindowTitle('PFFDTD')
        self.resize(1280, 720)

    def onExit(self):
        QApplication.quit()

    def onImportImpulseResponse(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            'Open WAV File',
            '',
            'WAV Files (*.wav);;All Files (*)',
            options=QFileDialog.Options() | QFileDialog.ReadOnly
        )
        if not path:
            return
        self.fileListModel.addFile(Path(path))

        fmax = 4000
        fs, buf = wavread(path)

        self.summary.clear_all_axes()
        plot_impulse_response_summary(buf, fs, ax=self.summary.axes, fmax=fmax)
        self.summary.draw()

        self.musical.clear_all_axes()
        plot_musical_response(path, fmax=fmax, ax=self.musical.axes, key_colors=False)
        self.musical.draw()

        self.edc.clear_all_axes()
        freqs = center_frequencies(3, 1000, 6, 5)
        freqs = freqs[(freqs >= 20) & (freqs <= fmax)]
        reverberation_time(buf, fs, freqs=freqs, plot=True, ax=self.edc.axes)
        self.edc.draw()


def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    plt.rcParams.update(plot_styles)

    app = QApplication(sys.argv)

    window = MainWindow()
    window.setStyleSheet('''
        QTabWidget::tab-bar {
            alignment: center;
        }
    ''')
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
