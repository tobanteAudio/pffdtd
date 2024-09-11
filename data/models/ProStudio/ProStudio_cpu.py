# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

from pffdtd.sim3d.setup import sim_setup_3d

sim_setup_3d(
    model_json_file='model.json',
    mat_folder='../../materials',
    source_num=1,
    insig_type='impulse',
    diff_source=True,
    mat_files_dict={
        'ATC Left': 'floor_wood.h5',
        'ATC Right': 'floor_wood.h5',
        'Ceiling': 'absorber_8000_200mm_gap_200mm.h5',
        'Console': 'door_iron.h5',
        'Couch': 'leather_arm_chair.h5',
        # 'Diffusor': 'floor_wood.h5',
        'Floor': 'floor_wood_on_concrete.h5',
        'Outboard': 'door_iron.h5',
        'Rack': 'floor_wood.h5',
        'Raised Floor': 'floor_wood.h5',
        'Walls Back': 'absorber_8000_200mm_gap_200mm.h5',
        'Walls Front': 'absorber_8000_200mm_gap_100mm.h5',
        'Walls Side': 'absorber_8000_50mm.h5',
        'Windows': 'glas_thick.h5',
    },
    duration=1.2,
    Tc=20,
    rh=50,
    fcc_flag=True,
    PPW=10.5,
    fmax=800.0,
    save_folder='../../sim_data/ProStudio/cpu',
    # save_folder_gpu='../../sim_data/ProStudio/gpu',
    draw_vox=True,
    draw_backend='polyscope',
    compress=0,
    rot_az_el=[0, 0],
)
