# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch
from pathlib import Path

from pffdtd.absorption.admittance import convert_Sabs_to_Yn, write_freq_ind_mat_from_Yn
from pffdtd.sim3d.model_builder import RoomModelBuilder
from pffdtd.sim3d.setup import Setup3D


class Modes(Setup3D):
    model_file = 'model.json'
    mat_folder = '../../sim_data/Modes/materials'
    source_index = 1
    source_signal = 'impulse'
    diff_source = True
    materials = {
        'Ceiling': 'sabine_03.h5',
        'Floor': 'sabine_03.h5',
        'Walls': 'sabine_03.h5',
    }
    duration = 7.0
    Tc = 20
    rh = 50
    fcc = False
    ppw = 10.5
    fmax = 1000.0
    save_folder = '../../sim_data/Modes/cpu'
    save_folder_gpu = '../../sim_data/Modes/gpu'
    compress = 0
    draw_vox = False
    draw_backend = 'polyscope'
    rot_az_el = (0, 0)

    def generate_materials(self):
        self._print('Generate materials')

        folder = Path(self.mat_folder)
        filename = folder / 'sabine_03.h5'
        write_freq_ind_mat_from_Yn(convert_Sabs_to_Yn(0.03), filename=filename)

    def generate_model(self, constants):
        self._print('Generate model')
        width = 3.65
        length = 6.0
        height = 3.12

        mul = 3.5 if self.fcc else 2.0
        offset = constants.h * mul

        room = RoomModelBuilder(length, width, height)
        room.with_colors({
            'Ceiling': [200, 200, 200],
            'Floor': [151, 134, 122],
            'Walls': [255, 255, 255],
        })
        room.add_source('S1', [offset, offset, offset])
        room.add_receiver('R1', [width-offset, length-offset, height-offset])
        room.build(self.model_file)

    def _print(self, msg):
        print(f'--MODES: {msg}')
