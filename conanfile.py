# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

from conan import ConanFile


class PFFDTD(ConanFile):
    settings = 'os', 'compiler', 'build_type', 'arch'
    generators = 'CMakeToolchain', 'CMakeDeps'

    def requirements(self):
        self.requires('cli11/2.7.2')
        self.requires('fmt/12.2.1')
        self.requires('hdf5/2.2.0')

    def config_options(self):
        pass
