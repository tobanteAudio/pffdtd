# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch
from pathlib import Path

import numpy as np

from pffdtd.absorption.admittance import fit_to_Sabs_oct_11
from pffdtd.sim3d.model_builder import MeshModelBuilder
from pffdtd.sim3d.setup import Setup3D


class BassReflex(Setup3D):
    model_file = 'model.json'
    mat_folder = '../../sim_data/BassReflex/materials'
    materials = {
        'Wood': 'wood.h5',
    }
    source_index = 1
    source_signal = 'impulse'
    diff_source = True
    duration = 1.0
    Tc = 20
    rh = 50
    fcc = False
    ppw = 10.5
    fmax = 8000.0
    save_folder = '../../sim_data/BassReflex/cpu'
    save_folder_gpu = '../../sim_data/BassReflex/gpu'
    compress = 0
    draw_vox = True
    draw_backend = 'polyscope'
    bmin = [-0.05, -0.05, -0.05]
    bmax = [+0.65, +0.65, +0.80]

    def generate_materials(self):
        self._print('Generate materials')
        folder = Path(self.mat_folder)

        wood = np.array([0.10, 0.11, 0.13, 0.15, 0.11, 0.10, 0.07, 0.06, 0.07, 0.07, 0.07])
        fit_to_Sabs_oct_11(wood, filename=folder / 'wood.h5')

    def generate_model(self, constants):
        print('--BASS-REFLEX: Generate model')

        dir = Path('.')
        obj_dir = dir/'obj'

        m = MeshModelBuilder()
        m.add('Wood', obj_dir / 'box.obj', [25, 25, 25], reverse=True)
        m.add('_RIGID', obj_dir / 'reflex_port.obj', [125, 125, 125], reverse=True, sides=0)
        m.add_source('S1', [0.25, 0.25, 0.50])
        m.add_receiver('R1', [0.25, 0.00, 0.1875])
        m.write(self.model_file)

    def _print(self, msg):
        print(f'--BASS-REFLEX: {msg}')
