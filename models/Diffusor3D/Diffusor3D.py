# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch
from pathlib import Path

import numpy as np

from pffdtd.geometry.math import point_on_circle
from pffdtd.sim3d.model_builder import MeshModelBuilder
from pffdtd.sim3d.setup import Setup3D


class Diffusor3D(Setup3D):
    model_file = 'model.json'
    mat_folder = '../../sim_data/Diffusor3D/materials'
    source_index = 1
    source_signal = 'impulse'
    diff_source = True
    duration = 0.3
    Tc = 20
    rh = 50
    fcc = False
    ppw = 7.75
    fmax = 10000.0
    save_folder = '../../sim_data/Diffusor3D/cpu'
    save_folder_gpu = '../../sim_data/Diffusor3D/gpu'
    compress = 0
    draw_vox = True
    draw_backend = 'polyscope'
    bmin = [-1.1, +0.00, +0.0]
    bmax = [+1.1, +3.25, +0.40]

    def generate_model(self, constants):
        print('--DIFFUSOR-3D: Generate model')

        dir = Path('.')
        obj = dir/'obj'

        height = 400.0/1000.0

        def point_at_angle(angle):
            x, y = point_on_circle((0, 0), 1.0, np.deg2rad(angle))
            return [x, y, height/2]

        s1 = point_at_angle(90)
        s1[1] = 3.0

        m = MeshModelBuilder()
        m.add('_RIGID', obj / 'diffusor.obj', [25, 25, 25], reverse=True, sides=0)
        m.add_source('S1', s1)
        for i, angle in enumerate(range(30, 152, 2)):
            p = point_at_angle(angle)
            print(f'R{i}: {p}')
            m.add_receiver(f'R{i}', p)

        m.write(self.model_file)
