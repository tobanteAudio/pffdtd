# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch
from pathlib import Path

from pffdtd.sim3d.model_builder import MeshModelBuilder
from pffdtd.sim3d.setup import Setup3D


class BassReflex(Setup3D):
    model_file = 'model.json'
    mat_folder = '../../sim_data/BassReflex/materials'
    source_index = 1
    source_signal = 'impulse'
    diff_source = True
    duration = 10.0
    Tc = 20
    rh = 50
    fcc = False
    ppw = 10.5
    fmax = 4000.0
    save_folder = '../../sim_data/BassReflex/cpu'
    save_folder_gpu = '../../sim_data/BassReflex/gpu'
    compress = 0
    draw_vox = True
    draw_backend = 'polyscope'
    bmin = [-0.05, -0.05, -0.05]
    bmax = [+0.80, +0.75, +0.105]

    def generate_model(self, constants):
        print('--BASS-REFLEX: Generate model')

        dir = Path('.')
        obj_dir = dir/'obj'

        m = MeshModelBuilder()
        m.add('_RIGID', obj_dir / 'box-slot.obj', [125, 125, 125], reverse=True, sides=0)
        m.add_source('S1', [0.375, 0.05, 0.858/2])

        m.add_receiver('R1', [0.375, 0.00, 0.08/2])
        m.add_receiver('R2', [0.375/2, 0.00, 0.08/2])

        m.add_receiver('R3', [0.375, 0.20, 0.08/2])
        m.add_receiver('R4', [0.375/2, 0.20, 0.08/2])

        # m.add_receiver('R1', [0.375, 0.00, 0.1287])
        # m.add_receiver('R2', [0.375, 0.00, 0.7293])
        m.write(self.model_file)
