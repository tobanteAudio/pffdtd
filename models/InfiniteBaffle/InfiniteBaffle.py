# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch
import json

from pffdtd.sim3d.setup import Setup3D


class InfiniteBaffle(Setup3D):
    """Point source on infinite baffle in an anechoic chamber
    """
    fmax = 2500
    ppw = 10.5
    fcc = False
    model_file = 'model.json'
    mat_folder = '../../sim_data/InfiniteBaffle/materials'
    duration = 0.3
    source_index = 1
    source_signal = 'impulse'
    Tc = 20
    rh = 50
    save_folder = '../../sim_data/InfiniteBaffle/cpu'
    save_folder_gpu = '../../sim_data/InfiniteBaffle/gpu'
    draw_vox = True
    draw_backend = 'polyscope'
    compress = 0
    rot_az_el = [0, 0]
    bmax = [10.0, 2.0, 10.0]
    bmin = [0, 0, 0]

    def generate_model(self, constants):
        mul = 3.0 if self.fcc else 2.0
        offset = constants.h * mul

        width = self.bmax[0]
        length = self.bmax[1]
        height = self.bmax[2]

        model = {
            'mats_hash': {
                '_RIGID': {
                    'tris': [
                        [0, 2, 1],
                        [0, 3, 2],
                        [1, 5, 4],
                        [1, 2, 5]
                    ],
                    'pts': [
                        [0.0, length, 0.0],
                        [width/2, length, 0.0],
                        [width/2, length, height],
                        [0.0, length, height],
                        [width, length, 0.0],
                        [width, length, height]
                    ],
                    'color': [255, 255, 255],
                    'sides': [0, 0, 0, 0]
                }
            },
            'sources': [
                {'name': 'S1', 'xyz': [width/2, length-offset, height/2]},
            ],
            'receivers': [
                {'name': 'R1', 'xyz': [width/2, offset, height/2 - 0.75]},
                {'name': 'R2', 'xyz': [width/2, offset, height/2 - 0.25]},
                {'name': 'R3', 'xyz': [width/2, offset, height/2 + 0.00]},
                {'name': 'R4', 'xyz': [width/2, offset, height/2 + 0.25]},
                {'name': 'R5', 'xyz': [width/2, offset, height/2 + 0.75]},
            ]
        }

        with open(self.model_file, 'w') as file:
            json.dump(model, file)
            print('', file=file)
