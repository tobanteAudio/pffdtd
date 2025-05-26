# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch
from pathlib import Path
import signal
import sys
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np

from PySide6.QtCore import (
    QAbstractListModel,
    QModelIndex,
    QObject,
    Qt,
)
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QListView,
    QHBoxLayout,
    QMainWindow,
    QProgressBar,
    QTabWidget,
    QToolBar,
    QWidget,
)

from pffdtd.common.plot import plot_styles
from pffdtd.analysis.response import plot_musical_response
from pffdtd.analysis.rt60 import reverberation_time
from pffdtd.analysis.summary import plot_impulse_response_summary
from pffdtd.signals.octave import center_frequencies
from pffdtd.signals.wavfile import wavread


class MatplotLibCanvas(FigureCanvas):
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


class OpenFilesListModel(QAbstractListModel):
    _open_files: list[Path]

    def __init__(self, parent: QObject | None) -> None:
        super().__init__(parent=parent)
        self._open_files = []

    def data(self, index, role) -> Any:
        if 0 <= index.row() < self.rowCount():
            if role == Qt.DisplayRole:
                path = self._open_files[index.row()]
                return path.stem

    def rowCount(self, index: QModelIndex = QModelIndex()) -> int:
        return len(self._open_files)

    def addFile(self, path: Path) -> None:
        self.beginInsertRows(QModelIndex(), self.rowCount(), self.rowCount())
        self._open_files.append(path)
        self.endInsertRows()


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        # Status Bar
        self.status = QLabel()
        self.status.setText('Foo bar baz.')

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        self.statusBar().addPermanentWidget(self.status)
        self.statusBar().addPermanentWidget(self.progress)

        # Toolbar
        toolbar = QToolBar('My main toolbar')
        self.addToolBar(toolbar)

        # Create the "Open" action
        openAction = QAction('&Open...', self)
        openAction.setShortcut('Ctrl+O')
        openAction.setStatusTip('Open a impulse file')
        openAction.triggered.connect(self.onOpenWavFile)

        # Create the "Exit" action
        exitAction = QAction('&Exit...', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit the program')
        exitAction.triggered.connect(self.onExit)

        # Menu Bar
        menu = self.menuBar()
        fileMenu = menu.addMenu('&File')
        fileMenu.addAction(openAction)
        fileMenu.addSeparator()
        fileMenu.addAction(exitAction)

        # Open Files
        self.fileListModel = OpenFilesListModel(self)
        self.fileList = QListView()
        self.fileList.setModel(self.fileListModel)

        # Tabs
        self.summary = MatplotLibCanvas(self, width=5, height=4, dpi=100, subplot_args={'nrows': 3, 'ncols': 2})
        self.musical = MatplotLibCanvas(self, width=5, height=4, dpi=100)
        self.edc = MatplotLibCanvas(self, width=5, height=4, dpi=100)

        self.tabs = QTabWidget(tabPosition=QTabWidget.TabPosition.North)
        self.tabs.addTab(self.summary, 'Summary')
        self.tabs.addTab(self.musical, 'Musical')
        self.tabs.addTab(self.edc, 'EDC')

        # Main Layout
        layout = QHBoxLayout()
        layout.addWidget(self.fileList, stretch=1)
        layout.addWidget(self.tabs, stretch=5)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.setWindowTitle('PFFDTD')
        self.resize(1280, 720)

    def onExit(self):
        QApplication.quit()

    def onOpenWavFile(self):
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
