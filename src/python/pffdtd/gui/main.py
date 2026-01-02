# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch
import json
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
from scipy.signal import sosfilt


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
    QSlider,
    QSplitter,
    QTableView,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from pffdtd.common.plot import plot_styles
from pffdtd.absorption.database import read_absorption_database_excel, ALL_BANDS
from pffdtd.analysis.response import plot_musical_response
from pffdtd.analysis.rt60 import reverberation_time
from pffdtd.analysis.summary import plot_impulse_response_summary
from pffdtd.signals.iir import linkwitz_riley_crossover
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
        self.canvas = MatplotLibCanvas(self, width=5, height=4, subplot_args={'nrows': 2, 'ncols': 1, 'sharex': True})

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

        if len(ids) == 0:
            self.canvas.draw()
            return

        bands = ALL_BANDS
        bands = [63, 125, 250, 500, 1000, 2000, 4000, 8000]

        ax = self.canvas.axes

        Sabs_plot: Axes = ax[0]
        dB_plot: Axes = ax[1]

        Sabs_plot.set_xscale('linear')
        dB_plot.set_xscale('linear')

        self.canvas.clear_all_axes()

        Sabs_plot.set_xlabel('Frequency [Hz]')
        Sabs_plot.set_ylabel('Absorption [Sabs]')
        Sabs_plot.set_xlim(50, 10000.0)
        Sabs_plot.set_ylim(0.0, 1.0)

        dB_plot.set_xlabel('Frequency [Hz]')
        dB_plot.set_ylabel('Reflection [dB]')
        dB_plot.set_xlim(50, 10000.0)
        dB_plot.set_ylim(-20.0, 0.0)

        for idx in ids:
            coefficients = self.model._df.loc[int(idx)][bands].to_numpy().astype(float)
            description = self.model._df.loc[int(idx)]['description']

            Sabs_plot.semilogx(bands, coefficients)
            Sabs_plot.scatter(bands, coefficients, color='red')

            dB = 10*np.log10(np.maximum(1-coefficients, 1e-6))
            dB_plot.semilogx(bands, dB, label=description[:150])
            dB_plot.scatter(bands, dB, color='red')

        formatter = ScalarFormatter()
        formatter.set_scientific(False)

        Sabs_plot.xaxis.set_major_formatter(formatter)
        Sabs_plot.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
        # Sabs_plot.legend(loc='lower left')

        dB_plot.xaxis.set_major_formatter(formatter)
        dB_plot.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
        dB_plot.legend(loc='lower left')

        # Sabs_plot.set_title()

        self.canvas.draw()


class XoverPreview(QWidget):
    def __init__(self):
        super().__init__()

        ir_low_path = 'sim_data/RedBullStudiosBerlin/gpu_sub/R001_out_native.wav'  # sys.argv[1]
        self.low = wavread(ir_low_path)

        ir_high_path = 'sim_data/RedBullStudiosBerlin/gpu/R001_out_native.wav'  # sys.argv[2]
        self.high = wavread(ir_high_path)

        model_path = 'models/private/RedBullStudiosBerlin/model.json'
        with open(model_path, 'r') as f:
            model = json.load(f)

        listener_pos = np.array(model['receivers'][0]['xyz'])
        top_pos = (np.array(model['sources'][0]['xyz'])+np.array(model['sources'][1]['xyz']))/2
        sub_pos = np.array(model['sources'][2]['xyz'])

        distance_top = np.linalg.norm(top_pos-listener_pos)
        distance_sub = np.linalg.norm(sub_pos-listener_pos)
        distance_delta = distance_sub-distance_top
        time_delta = 1e6/343.2*distance_delta

        print(f'Distance Top:   {distance_top*100:.1f} cm')
        print(f'Distance Sub:   {distance_sub*100:.1f} cm')
        print(f'Distance Delta: {distance_delta*100:.1f} cm')
        print(f'Distance Delta: {time_delta:.0f} us')

        self.frequency = QSlider()
        self.frequency.setRange(60, 200)
        self.frequency.setValue(80)
        self.frequency.setOrientation(Qt.Orientation.Horizontal)
        self.frequency.valueChanged.connect(self.updatePlot)

        self.delay = QSlider()
        self.delay.setRange(0, 10000)
        self.delay.setValue(int(time_delta))
        self.delay.setOrientation(Qt.Orientation.Horizontal)
        self.delay.valueChanged.connect(self.updatePlot)

        self.order = QLineEdit()

        vbox = QVBoxLayout()
        vbox.addWidget(self.frequency)
        vbox.addWidget(self.delay)
        vbox.addWidget(self.order)

        self.parameters = QWidget()
        self.parameters.setLayout(vbox)

        self.canvas = MatplotLibCanvas(self, width=5, height=4)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.parameters)
        splitter.addWidget(self.canvas)
        layout = QHBoxLayout()
        layout.addWidget(splitter)
        self.setLayout(layout)
        self.updatePlot()

    def updatePlot(self):
        self.canvas.clear_all_axes()

        fs_low, ir_low = self.low
        fs_high, ir_high = self.high
        assert fs_low == fs_high
        assert ir_low.shape == ir_high.shape

        delay_samples = max(int(np.ceil((self.delay.value()/1e6)*fs_low)), 0)
        ir_low = np.pad(ir_low.copy(), (0, delay_samples), 'constant', constant_values=0)
        ir_high = np.pad(ir_high.copy(), (delay_samples, 0), 'constant', constant_values=0)

        frequency = self.frequency.value()
        sos_low, sos_high = linkwitz_riley_crossover(frequency, fs_low, order=8)
        filt_low = sosfilt(sos_low, ir_low)
        filt_high = sosfilt(sos_high, ir_high)
        mix = filt_low+filt_high

        freqs = np.fft.rfftfreq(ir_low.shape[0], 1/fs_low)
        H_low = np.fft.rfft(filt_low)
        H_high = np.fft.rfft(filt_high)
        H_mix = np.fft.rfft(mix)

        N = 24
        fmax = 300
        freqs, H_low = rfft_to_N_per_octave(filt_low, fs_low, N, fmax)
        freqs, H_high = rfft_to_N_per_octave(filt_high, fs_high, N, fmax)
        freqs, H_mix = rfft_to_N_per_octave(mix, fs_low, N, fmax)

        # _, ax = plt.subplots(1, 1, constrained_layout=True)
        ax: Axes = self.canvas.axes
        ax.semilogx(freqs, 20*np.log10(np.maximum(np.abs(H_low), 1e-6)), linestyle='--', label='Low')
        ax.semilogx(freqs, 20*np.log10(np.maximum(np.abs(H_high), 1e-6)), linestyle='--', label='High')
        ax.semilogx(freqs, 20*np.log10(np.maximum(np.abs(H_mix), 1e-6)), label='Mix')
        ax.set_xlim(20, 300)
        ax.set_ylim(-40, 5)
        ax.set_ylabel('Amplitude [dB]')
        ax.set_title(f'Crossover @ {frequency:.2f} Hz - {self.delay.value()} us')
        ax.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
        ax.legend()

        self.canvas.draw()


def rfft_to_N_per_octave(x, fs, N, fmax):
    """
    Inputs:
      - x  : real-valued time signal (1D ndarray of length N)
      - fs : sampling rate in Hz
    Returns:
      - f_log : shape (n_bins,), the log-spaced freq array (96 bins/octave)
      - mag_log: shape (n_bins,), the interpolated magnitude |RFFT(x)| at f_log
    """
    # N = len(x)
    # 1) Compute RFFT and its linear frequency axis
    X = np.fft.rfft(x)
    f_lin = np.fft.rfftfreq(len(x), d=1.0/fs)    # [0, Δf, 2Δf, ..., fs/2], len = len(x)//2+1
    mag_lin = np.abs(X)

    # 2) Choose f_min and f_max
    f_min = fs / len(x)          # first nonzero bin
    f_max = min(fmax, fs / 2)          # Nyquist

    # 3) How many octaves from f_min to f_max?
    n_octaves = np.log2(f_max / f_min)     # = log2((fs/2)/(fs/N)) = log2(N/2)
    n_bins = int(np.floor(n_octaves * N)) + 1

    # 4) Build log-spaced frequencies: one bin per semitone
    k = np.arange(n_bins)                               # k = 0,1,...,n_bins-1
    f_log = f_min * (2.0 ** (k / N))                      # semitone steps

    # 5) Interpolate magnitude onto log axis
    mag_log = np.interp(f_log, f_lin, mag_lin)

    return f_log, mag_log


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
        # self.xover = XoverPreview()

        self.tabs = QTabWidget(tabPosition=QTabWidget.TabPosition.North)
        self.tabs.addTab(self.summary, 'Summary')
        self.tabs.addTab(self.musical, 'Musical')
        self.tabs.addTab(self.edc, 'EDC')
        self.tabs.addTab(self.materials, 'Materials')
        # self.tabs.addTab(self.xover, 'Xover')

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
