# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2021 Brian Hamilton

from pathlib import Path

import numpy as np
import h5py


class SimMaterials:
    """Class to pack up materials from individual HDF5 files in ordering
    coresponding to room_geo materials.
    """

    def __init__(self, save_folder):
        save_folder = Path(save_folder)
        assert save_folder.exists()
        assert save_folder.is_dir()

        self.save_folder = save_folder

    def package(self, mat_files_dict, mat_list, read_folder, verbose=False):
        mat_list = mat_list[:]  # make copy of input
        if '_RIGID' in mat_list:
            mat_list.remove('_RIGID')
        mat_list.sort()
        mat_list2 = list(mat_files_dict.keys())
        mat_list2.sort()

        # mat dict coming in has to match list from room_geo
        assert mat_list == mat_list2

        save_folder = self.save_folder
        read_folder = Path(read_folder)
        DEF_list = []
        for mat in mat_list:
            with h5py.File(Path(read_folder / Path(mat_files_dict[mat])), 'r') as h5f:
                DEF_list.append(h5f['DEF'][()])

        Nmat = len(DEF_list)
        Mb = np.zeros((Nmat,), dtype=np.int8)  # number of circuit branches
        with h5py.File(Path(save_folder / Path('materials.h5')), 'w') as h5f:
            h5f.create_dataset('Nmat', data=np.int8(Nmat))
            for i in range(Nmat):
                mat = mat_list[i]
                DEF = DEF_list[i]
                assert DEF.ndim == 2
                assert DEF.shape[1] == 3

                if verbose:
                    self.print(f'{mat=} {DEF=}')
                else:
                    self.print(f'{mat=}')
                h5f.create_dataset(f'mat_{i:02d}_DEF', data=DEF)
                Mb[i] = DEF.shape[0]

            h5f.create_dataset('Mb', data=Mb)

    def print(self, fstring):
        print(f'--MATS: {fstring}')
