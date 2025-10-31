# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch
from pathlib import Path
import numpy as np

from pffdtd.absorption.admittance import convert_Sabs_to_Yn, write_freq_ind_mat_from_Yn
from pffdtd.sim3d.constants import SimConstants
from pffdtd.sim3d.model_builder import RoomModelBuilder
from pffdtd.sim3d.setup import Setup3D


class FlutterEcho(Setup3D):
    """
    Verify that the simulated decay time (RT60) agrees with theoretical
    predictions (Sabine and Eyring) across frequency bands, for a rectangular
    room with known boundary impedances.
    """

    model_file = 'model.json'
    mat_folder = '../../sim_data/FlutterEcho/materials'
    source_index = 1
    source_signal = 'impulse'
    diff_source = True
    materials = {
        'Ceiling': 'Sabs_9.h5',
        'Floor': 'Sabs_9.h5',
        'Walls': 'Sabs_05.h5',
    }
    duration = 1.0
    Tc = 20
    rh = 50
    fcc = False
    ppw = 10.5
    fmax = 3000.0
    save_folder = '../../sim_data/FlutterEcho/cpu'
    save_folder_gpu = '../../sim_data/FlutterEcho/gpu'
    compress = 0
    draw_vox = False
    draw_backend = 'polyscope'

    def generate_materials(self):
        self._print('Generate materials')

        folder = Path(self.mat_folder)
        write_freq_ind_mat_from_Yn(convert_Sabs_to_Yn(0.05), filename=folder / 'Sabs_05.h5')
        write_freq_ind_mat_from_Yn(convert_Sabs_to_Yn(0.9), filename=folder / 'Sabs_9.h5')

    def generate_model(self, constants: SimConstants):
        self._print('Generate model')
        L = 10.0
        W = 2.0
        H = 3.0

        room = RoomModelBuilder(L, W, H)
        room.with_colors({
            'Ceiling': [200, 200, 200],
            'Floor': [151, 134, 122],
            'Walls': [255, 255, 255],
        })

        room.add_source('S1', [0.5, 0.5, H/2])
        room.add_receiver('R1', [W-0.5, L-0.5, H/2])
        room.build(self.model_file)

    def _print(self, msg):
        print(f'--REVERBERATION_TIME: {msg}')
