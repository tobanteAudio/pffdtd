# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch
from pathlib import Path

import numpy as np

from pffdtd.absorption.admittance import fit_to_Sabs_oct_11
from pffdtd.absorption.porous import porous_absorber
from pffdtd.filters.octave import center_frequencies
from pffdtd.sim3d.model_builder import MeshModelBuilder
from pffdtd.sim3d.setup import Setup3D


class LivingRoom(Setup3D):
    model_file = 'model.json'
    mat_folder = '../../sim_data/LivingRoom/materials'
    source_index = 1
    source_signal = 'impulse-highpass'
    diff_source = True
    materials = {
        'Book Shelf': 'wood.h5',
        'Ceiling': 'concrete_painted.h5',
        'Coffee Table': 'wood.h5',
        'Couch': 'absorber_8000_100mm.h5',
        'Desk': 'wood.h5',
        'Door': 'wood.h5',
        'Floor': 'wood.h5',
        'Kallax': 'wood.h5',
        'Monitors': 'wood.h5',
        'Speakers': 'wood.h5',
        'Speaker Stands': 'metal_iron.h5',
        'Table': 'wood.h5',
        'TV 42': 'wood.h5',
        'TV 55': 'wood.h5',
        'TV Table': 'wood.h5',
        'Walls': 'concrete_painted.h5',
        'Window': 'glas_thick.h5',
    }
    duration = 3.0
    Tc = 20
    rh = 50
    fcc = False
    ppw = 10.5
    fmax = 1600.0
    save_folder = '../../sim_data/LivingRoom/cpu'
    save_folder_gpu = '../../sim_data/LivingRoom/gpu'
    compress = 0
    draw_vox = True
    draw_backend = 'polyscope'

    def generate_materials(self):
        self._print('Generate materials')
        iso_octaves = center_frequencies(1, 1000, 6, 5)

        # autopep8: off
        absorber_8000_100mm = porous_absorber(0.1, 8000.0, frequency=iso_octaves, offset_zeros=True)
        concrete_painted    = np.array([0.01, 0.01, 0.01, 0.05, 0.06, 0.07, 0.09, 0.08, 0.08, 0.08, 0.08])
        glas_thick          = np.array([0.15, 0.30, 0.27, 0.18, 0.06, 0.04, 0.03, 0.02, 0.02, 0.02, 0.01])
        metal_iron          = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.03, 0.03, 0.03, 0.02])
        wood                = np.array([0.10, 0.11, 0.13, 0.15, 0.11, 0.10, 0.07, 0.06, 0.07, 0.07, 0.07])

        folder = Path(self.mat_folder)
        fit_to_Sabs_oct_11(absorber_8000_100mm , filename=folder / 'absorber_8000_100mm.h5')
        fit_to_Sabs_oct_11(concrete_painted    , filename=folder / 'concrete_painted.h5'   )
        fit_to_Sabs_oct_11(glas_thick          , filename=folder / 'glas_thick.h5'         )
        fit_to_Sabs_oct_11(metal_iron          , filename=folder / 'metal_iron.h5'         )
        fit_to_Sabs_oct_11(wood                , filename=folder / 'wood.h5'               )
        # autopep8: on

    def generate_model(self, constants):
        self._print('Generate model')

        dir = Path('.')
        obj = dir/'obj'

        s1 = [3.65-0.5, 6.0-0.3, 1.2]
        s2 = [3.65-0.5, 6.0-2.5, 1.2]

        r1 = [0.6, 6.0-1.3, 1.1]
        r2 = [0.6, 6.0-2.0, 1.1]

        m = MeshModelBuilder()
        m.add('Book Shelf', obj / 'book_shelf.obj', [200, 200, 200], reverse=True)
        m.add('Ceiling', obj / 'ceiling.obj', [150, 150, 150], reverse=True)
        m.add('Coffee Table', obj / 'coffee_table.obj', [10, 10, 10], reverse=True)
        m.add('Couch', obj / 'couch.obj', [29, 50, 112], reverse=True)
        m.add('Desk', obj / 'desk.obj', [103, 70, 55], reverse=True)
        m.add('Door', obj / 'door.obj', [103, 70, 55], reverse=True)
        m.add('Floor', obj / 'floor.obj', [133, 94, 66], reverse=True)
        m.add('Kallax', obj / 'kallax.obj', [200, 200, 200], reverse=True)
        m.add('Monitors', obj / 'monitors.obj', [15, 15, 15], reverse=True)
        m.add('Speakers', obj / 'speakers.obj', [25, 25, 25], reverse=True)
        m.add('Speaker Stands', obj / 'speaker_stands.obj', [5, 5, 5], reverse=True)
        m.add('Table', obj / 'table.obj', [200, 200, 200], reverse=True)
        m.add('TV 42', obj / 'tv_42.obj', [10, 10, 10], reverse=True)
        m.add('TV 55', obj / 'tv_55.obj', [10, 10, 10], reverse=True)
        m.add('TV Table', obj / 'tv_table.obj', [120, 120, 120], reverse=True)
        m.add('Walls', obj / 'walls.obj', [175, 175, 175], reverse=True)
        m.add('Window', obj / 'window.obj', [137, 207, 240], reverse=True)
        m.add_source('S1', s1)
        m.add_source('S2', s2)
        m.add_receiver('R1', r1)
        m.add_receiver('R2', r2)
        m.write(self.model_file)

    def _print(self, msg):
        print(f'--LIVING-ROOM: {msg}')
