# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch
from pathlib import Path

import numpy as np

from pffdtd.absorption.admittance import convert_Sabs_to_Yn, write_freq_ind_mat_from_Yn
from pffdtd.sim3d.model_builder import RoomModelBuilder
from pffdtd.sim3d.setup import Setup3D


class Localization(Setup3D):
    model_file = 'model.json'
    mat_folder = '../../sim_data/Localization/materials'
    source_index = 1
    source_signal = 'mls-11'
    diff_source = True
    materials = {
        'Ceiling': 'sabine_9512.h5',
        'Floor': 'sabine_9512.h5',
        'Walls': 'sabine_9512.h5',
    }
    duration = 0.35
    Tc = 20
    rh = 50
    fcc = False
    ppw = 10.5
    fmax = 6000.0
    save_folder = '../../sim_data/Localization/cpu'
    save_folder_gpu = '../../sim_data/Localization/gpu'
    compress = 0
    draw_vox = False
    draw_backend = 'polyscope'

    def generate_materials(self):
        self._print('Generate materials')

        folder = Path(self.mat_folder)
        filename = folder / 'sabine_9512.h5'
        write_freq_ind_mat_from_Yn(convert_Sabs_to_Yn(0.9512), filename=filename)

    def generate_model(self, constants):
        self._print('Generate model')
        L = 3.0
        W = 3.0
        H = 3.0

        source = [W/2, L-0.1, H/2]
        mics = [
            np.array([0, 0, 0]),
            np.array([1, 0, 0]),
            np.array([0.5, np.sqrt(3)/2, 0]),
            np.array([0.5, np.sqrt(3)/6, np.sqrt(6)/3]),
        ]

        room = RoomModelBuilder(L, W, H)
        room.with_colors({
            'Ceiling': [200, 200, 200],
            'Floor': [151, 134, 122],
            'Walls': [255, 255, 255],
        })

        room.add_source('S1', source)
        room.add_receiver('R1', list(mics[1-1]/2+[0.5, 0.5, 0.5]))
        room.add_receiver('R2', list(mics[2-1]/2+[0.5, 0.5, 0.5]))
        room.add_receiver('R3', list(mics[3-1]/2+[0.5, 0.5, 0.5]))
        room.add_receiver('R4', list(mics[4-1]/2+[0.5, 0.5, 0.5]))
        room.build(self.model_file)

    def _print(self, msg):
        print(f'--LOCALIZATION: {msg}')
