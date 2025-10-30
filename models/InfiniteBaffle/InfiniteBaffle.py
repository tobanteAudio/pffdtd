# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch
import json

import numpy as np

from pffdtd.sim3d.setup import Setup3D

MULTI_SOURCE = False


class InfiniteBaffle(Setup3D):
    """Point source on infinite baffle in an anechoic chamber
    """
    fmax = 1000
    ppw = 10.5
    fcc = False
    model_file = 'model.json'
    mat_folder = '../../sim_data/InfiniteBaffle/materials'
    duration = 0.3
    source_index = [1, 2] if MULTI_SOURCE else 1
    source_signal = 'impulse'
    Tc = 20
    rh = 50
    save_folder = '../../sim_data/InfiniteBaffle/cpu'
    save_folder_gpu = '../../sim_data/InfiniteBaffle/gpu'
    draw_vox = False
    draw_backend = 'polyscope'
    compress = 0
    rot_az_el = (0, 0)
    bmax = [343.2/20, 2.0, 343.2/20]
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
            }
        }

        if MULTI_SOURCE:
            model['sources'] = [
                {'name': 'S1', 'xyz': [width/2, length-offset, (height/2)-0.5]},
                {'name': 'S2', 'xyz': [width/2, length-offset, (height/2)+1.5]},
            ]
            model['receivers'] = [
                {'name': 'R1', 'xyz': [width/2, offset, height/2 - 0.50]},
                {'name': 'R2', 'xyz': [width/2, offset, height/2 - 0.25]},
                {'name': 'R3', 'xyz': [width/2, offset, height/2 + 0.00]},
                {'name': 'R4', 'xyz': [width/2, offset, height/2 + 0.25]},
                {'name': 'R5', 'xyz': [width/2, offset, height/2 + 0.50]},
            ]
        else:
            model['sources'] = [{'name': 'S1', 'xyz': [width/2, length-offset, height/2]}]
            model['receivers'] = [{'name': 'R1', 'xyz': [width/2, offset, height/2]}]

        src = np.array(model['sources'][0]['xyz'])
        ref = np.linalg.norm(src - np.array(model['receivers'][0]['xyz']))
        for r in model['receivers']:
            distance = np.linalg.norm(src - np.array(r['xyz']))
            print(r['name'], 20*np.log10(ref/distance))

        with open(self.model_file, 'w') as file:
            json.dump(model, file)
            print('', file=file)
