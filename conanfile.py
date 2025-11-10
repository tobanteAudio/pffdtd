# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

from conan import ConanFile


class PFFDTD(ConanFile):
    settings = 'os', 'compiler', 'build_type', 'arch'
    generators = 'CMakeToolchain', 'CMakeDeps'

    def requirements(self):
        self.requires('cli11/2.6.0')
        self.requires('fmt/12.0.0')
        self.requires('hdf5/1.14.6')

    def config_options(self):
        pass
