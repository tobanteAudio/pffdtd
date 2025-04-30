# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch
from pathlib import Path

import numpy as np

from pffdtd.absorption.admittance import fit_to_Sabs_oct_11
from pffdtd.absorption.porous import porous_absorber
from pffdtd.dsp.octave import center_frequencies
from pffdtd.geometry.math import find_third_vertex
from pffdtd.sim3d.model_builder import RoomModelBuilder
from pffdtd.sim3d.setup import Setup3D


class Studio(Setup3D):
    model_file = 'model.json'
    mat_folder = '../../sim_data/Studio/materials'
    source_index = 1
    source_signal = 'impulse'
    diff_source = True
    materials = {
        'Absorber M': 'absorber_8000_100mm.h5',
        'Absorber L': 'absorber_8000_200mm.h5',
        'Ceiling': 'concrete_painted.h5',
        'Diffusor': 'wood.h5',
        'Floor': 'wood_on_concrete.h5',
        'Sofa': 'absorber_8000_100mm.h5',
        'Speaker_Cabinet': 'wood.h5',
        'Table': 'wood.h5',
        'Walls': 'concrete_painted.h5',
    }
    duration = 2.0
    Tc = 20
    rh = 50
    fcc = False
    ppw = 10.5
    fmax = 1000.0
    save_folder = '../../sim_data/Studio/cpu'
    save_folder_gpu = '../../sim_data/Studio/gpu'
    draw_vox = True
    draw_backend = 'polyscope'
    compress = 0
    rot_az_el = [0, 0]

    def generate_materials(self):
        self._print('Generate materials')
        folder = Path(self.mat_folder)
        iso_octaves = center_frequencies(1, 1000, 6, 5)

        # autopep8: off
        absorber_8000_100mm           = porous_absorber(0.1, 8000.0, frequency=iso_octaves, offset_zeros=True)
        absorber_8000_200mm           = porous_absorber(0.2, 8000.0, frequency=iso_octaves, offset_zeros=True)
        concrete_painted              = np.array([0.01, 0.01, 0.01, 0.05, 0.06, 0.07, 0.09, 0.08, 0.08, 0.08, 0.08])
        wood                          = np.array([0.10, 0.11, 0.13, 0.15, 0.11, 0.10, 0.07, 0.06, 0.07, 0.07, 0.07])
        wood_on_concrete              = np.array([0.01, 0.01, 0.01, 0.04, 0.04, 0.07, 0.06, 0.06, 0.07, 0.06, 0.06])

        fit_to_Sabs_oct_11(absorber_8000_100mm , filename=folder / 'absorber_8000_100mm.h5')
        fit_to_Sabs_oct_11(absorber_8000_200mm , filename=folder / 'absorber_8000_200mm.h5')
        fit_to_Sabs_oct_11(concrete_painted    , filename=folder / 'concrete_painted.h5'   )
        fit_to_Sabs_oct_11(wood                , filename=folder / 'wood.h5'               )
        fit_to_Sabs_oct_11(wood_on_concrete    , filename=folder / 'wood_on_concrete.h5'   )
        # autopep8: on

    def generate_model(self, constants):
        self._print('Generate model')
        S = 0.90
        L = 7.00*S
        W = 5.19*S
        H = 3.70*S

        src_height = 1.2
        src_backwall = 1
        src_distance = 2.2
        src_left = [W/2-src_distance/2, L-src_backwall, src_height]
        src_right = [W/2+src_distance/2, L-src_backwall, src_height]

        l1, l2 = find_third_vertex(src_left, src_right)
        listener = l1 if l1[1] < l2[1] else l2

        p1 = listener.copy()
        p1[2] = 0.9

        p2 = p1.copy()
        p2[2] = 1.5

        p3 = p1.copy()
        p3[2] = 2.0

        room = RoomModelBuilder(L, W, H)
        room.with_colors({
            'Absorber M': [111, 55, 10],
            'Absorber L': [111, 55, 10],
            'Ceiling': [200, 200, 200],
            'Diffusor': [140, 80, 35],
            'Floor': [151, 134, 122],
            'Table': [130, 75, 25],
            'Sofa': [25, 25, 25],
            'Speaker_Cabinet': [15, 15, 15],
            'Walls': [255, 255, 255],
        })

        room.add_box('Absorber M', [0.1, 2.5, 1.5], [0.2, L-2.5-0.75, 0.75])
        room.add_box('Absorber M', [0.1, 2.5, 1.5], [W-0.1-0.2, L-2.5-0.75, 0.75])

        room.add_box('Absorber L', [1.0, 0.2, 2.5], [src_left[0]-1.0/2, L-0.2-0.1, 0.25])
        room.add_box('Absorber L', [1.0, 0.2, 2.5], [src_right[0]-1.0/2, L-0.2-0.1, 0.25])

        room.add_box('Absorber L', [2.5, 2.0, 0.2], [W/2-2.5/2, L-2-0.5, H-0.3])
        room.add_box('Absorber L', [2.5, 2.0, 0.2], [W/2-2.5/2, L-2-2.5-0.5, H-0.3])

        room.add_diffusor_1d([2, 15*0.0254, 1.5], [W/2-1, 0.05, 0.7], 3*0.0254)
        room.add_box('Absorber L', [1.0, 0.2, 2.5], [0.1, 0.1, 0.25])
        room.add_box('Absorber L', [1.0, 0.2, 2.5], [W-1.0-0.1, 0.1, 0.25])

        room.add_box('Sofa', [2.52, 0.98, 0.48], [W/2-2.52/2, 0.4, 0.05])
        room.add_box('Table', [1.8, 0.8, 0.02], [W/2-1.8/2, listener[1]+0.4, 0.7])

        speaker_box = [0.435, 0.490, 0.650]
        speaker_mid = [0.217, -0.075, 0.520]
        room.add_cabinet_speaker('Speaker Left', src_left, speaker_box, speaker_mid)
        room.add_cabinet_speaker('Speaker Right', src_right, speaker_box, speaker_mid)
        # room.add_cabinet_speaker("Speaker Left", src_left)
        # room.add_cabinet_speaker("Speaker Right", src_right)
        # room.add_source("Speaker Left", src_left)
        # room.add_source("Speaker Right", src_right)
        room.add_receiver('Engineer', listener.tolist())
        room.add_receiver('P1', p1.tolist())
        room.add_receiver('P2', p2.tolist())
        room.add_receiver('P3', p3.tolist())
        room.build(self.model_file)

    def _print(self, msg):
        print(f'--STUDIO: {msg}')
