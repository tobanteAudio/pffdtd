# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch
from pathlib import Path

import numpy as np

from pffdtd.absorption.admittance import fit_to_Sabs_oct_11
from pffdtd.absorption.porous import porous_absorber
from pffdtd.signals.octave import center_frequencies
from pffdtd.sim3d.model_builder import MeshModelBuilder
from pffdtd.sim3d.setup import Setup3D


class LivingRoom(Setup3D):
    model_file = 'model.json'
    mat_folder = '../../sim_data/LivingRoom/materials'
    source_index = 2
    source_signal = 'impulse-highpass-40'
    diff_source = True
    materials = {
        'Book Shelf': 'wood.h5',
        'Carpet': 'carpet.h5',
        'Ceiling': 'rabbitzdecke_330.h5',
        'Coffee Table Frame': 'wood.h5',
        'Coffee Table Top': 'sandstone_rough.h5',
        'Couch': 'absorber_8000_150mm.h5',
        'Desk Large Frame': 'metal_iron.h5',
        'Desk Large Top': 'wood.h5',
        'Desk Small': 'wood.h5',
        'Door': 'wood.h5',
        'Floor': 'parquet_on_counterfloor.h5',
        'Kallax': 'wood.h5',
        'Monitors': 'wood.h5',
        'Speakers': 'wood.h5',
        'Speaker Stands': 'metal_iron.h5',
        'TV 42': 'wood.h5',
        'TV 55': 'wood.h5',
        'TV Table': 'wood.h5',
        'Walls': 'vollziegel_mauerwerk_28.h5',
        'Window': 'glas_window_ordinary.h5',
    }
    duration = 2.0
    Tc = 20
    rh = 50
    fcc = True
    ppw = 10.5
    fmax = 3200.0
    save_folder = '../../sim_data/LivingRoom/cpu'
    save_folder_gpu = '../../sim_data/LivingRoom/gpu'
    compress = 0
    draw_vox = True
    draw_backend = 'polyscope'

    def generate_materials(self):
        self._print('Generate materials')

        # ISO octaves                       16     32    63    125   250   500   1000  2000  4000  8000  16000
        # autopep8: off
        carpet                  = np.array([0.01,  0.02, 0.05, 0.17, 0.18, 0.21, 0.50, 0.63, 0.83, 0.90, 0.92])
        concrete_painted_mod    = np.array([0.012, 0.02, 0.06, 0.14, 0.06, 0.07, 0.09, 0.08, 0.08, 0.08, 0.08])
        glas_window_ordinary    = np.array([0.20,  0.30, 0.40, 0.35, 0.25, 0.18, 0.12, 0.07, 0.04, 0.04, 0.02])
        metal_iron              = np.array([0.01,  0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.03, 0.03, 0.03, 0.02])
        rabbitzdecke_330        = np.array([0.02,  0.15, 0.24, 0.25, 0.20, 0.10, 0.05, 0.05, 0.06, 0.06, 0.05])
        parquet_on_counterfloor = np.array([0.10,  0.15, 0.20 ,0.20 ,0.15, 0.10, 0.10, 0.05, 0.10, 0.05, 0.05])
        sandstone_rough         = np.array([0.01,  0.01, 0.02, 0.02, 0.02, 0.03, 0.04, 0.05, 0.05, 0.06, 0.05])
        vollziegel_mauerwerk_28 = np.array([0.02,  0.02, 0.14, 0.16, 0.13, 0.15, 0.11, 0.13, 0.14, 0.10, 0.13])
        wood                    = np.array([0.10,  0.11, 0.13, 0.15, 0.11, 0.10, 0.07, 0.06, 0.07, 0.07, 0.07])
        # autopep8: on

        iso_octaves = center_frequencies(1, 1000, 6, 5)
        absorber_8000_150mm = porous_absorber(0.15, 8000.0, frequency=iso_octaves, offset_zeros=True, angle=45)

        folder = Path(self.mat_folder)
        fit_to_Sabs_oct_11(absorber_8000_150mm, filename=folder / 'absorber_8000_150mm.h5')
        fit_to_Sabs_oct_11(np.maximum(carpet, parquet_on_counterfloor), filename=folder / 'carpet.h5')
        fit_to_Sabs_oct_11(concrete_painted_mod, filename=folder / 'concrete_painted_mod.h5')
        fit_to_Sabs_oct_11(glas_window_ordinary, filename=folder / 'glas_window_ordinary.h5')
        fit_to_Sabs_oct_11(metal_iron, filename=folder / 'metal_iron.h5')
        fit_to_Sabs_oct_11(rabbitzdecke_330, filename=folder / 'rabbitzdecke_330.h5')
        fit_to_Sabs_oct_11(parquet_on_counterfloor, filename=folder / 'parquet_on_counterfloor.h5')
        fit_to_Sabs_oct_11(sandstone_rough, filename=folder / 'sandstone_rough.h5')
        fit_to_Sabs_oct_11(vollziegel_mauerwerk_28, filename=folder / 'vollziegel_mauerwerk_28.h5')
        fit_to_Sabs_oct_11(wood, filename=folder / 'wood.h5')

    def generate_model(self, constants):
        self._print('Generate model')

        dir = Path('.')
        obj = dir/'obj'

        s1 = [3.65-0.59, 6.0-0.3, 1.12]
        s2 = [3.65-0.59, 6.0-2.4, 1.12]

        r1 = s2.copy()
        r1[0] -= 0.7
        r2 = [0.6, 6.0-1.3, 1.1]
        r3 = [0.6, 6.0-2.0, 1.1]

        m = MeshModelBuilder()
        m.add('Book Shelf', obj / 'book_shelf.obj', [200, 200, 200], reverse=True)
        m.add('Carpet', obj / 'carpet.obj', [125, 31, 31], reverse=True)
        m.add('Ceiling', obj / 'ceiling.obj', [150, 150, 150], reverse=True)
        m.add('Coffee Table Frame', obj / 'coffee_table_frame.obj', [103, 70, 55], reverse=True)
        m.add('Coffee Table Top', obj / 'coffee_table_top.obj', [10, 10, 10], reverse=True)
        m.add('Couch', obj / 'couch.obj', [29, 50, 112], reverse=True)
        m.add('Desk Large Frame', obj / 'desk_large_frame.obj', [120, 120, 120], reverse=True)
        m.add('Desk Large Top', obj / 'desk_large_top.obj', [103, 70, 55], reverse=True)
        m.add('Desk Small', obj / 'desk_small.obj', [200, 200, 200], reverse=True)
        m.add('Door', obj / 'door.obj', [103, 70, 55], reverse=True)
        m.add('Floor', obj / 'floor.obj', [133, 94, 66], reverse=True)
        m.add('Kallax', obj / 'kallax.obj', [200, 200, 200], reverse=True)
        m.add('Monitors', obj / 'monitors.obj', [15, 15, 15], reverse=True)
        m.add('Speakers', obj / 'speakers.obj', [25, 25, 25], reverse=True)
        m.add('Speaker Stands', obj / 'speaker_stands.obj', [5, 5, 5], reverse=True)
        m.add('TV 42', obj / 'tv_42.obj', [10, 10, 10], reverse=True)
        m.add('TV 55', obj / 'tv_55.obj', [10, 10, 10], reverse=True)
        m.add('TV Table', obj / 'tv_table.obj', [120, 120, 120], reverse=True)
        m.add('Walls', obj / 'walls.obj', [175, 175, 175], reverse=True)
        m.add('Window', obj / 'window.obj', [137, 207, 240], reverse=True)
        m.add_source('S1', s1)
        m.add_source('S2', s2)
        m.add_receiver('R1', r1)
        # m.add_receiver('R2', r2)
        # m.add_receiver('R3', r3)
        m.write(self.model_file)

    def _print(self, msg):
        print(f'--LIVING-ROOM: {msg}')
