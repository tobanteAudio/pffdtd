# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

from pffdtd.sim3d.setup import sim_setup_3d

# will draw 'voxelization' (spheres are active boundary nodes, cubes rigid boundary nodes)
sim_setup_3d(
    model_json_file='model.json',
    mat_folder='../../materials',
    source_num=1,
    insig_type='dhann30',  # for viz
    diff_source=False,
    mat_files_dict={
        'Ceiling': 'concrete_painted.h5',
        'Floor': 'concrete_painted.h5',
        'Walls': 'concrete_painted.h5',
    },
    duration=0.1,
    Tc=20,
    rh=50,
    fcc_flag=False,
    PPW=10.5,
    fmax=500.0,
    save_folder='../../sim_data/Modes/viz',  # can run python from here
    compress=0,
    draw_vox=True,
    draw_backend='polyscope',
    rot_az_el=(0, 0)
)

# then run with python and 3D visualization:
#   pffdtd sim3d engine --sim_dir='../../sim_data/Modes/viz' --plot --draw_backend='polyscope' --json_model='model.json'
