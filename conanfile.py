# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

from conan import ConanFile


class PFFDTD(ConanFile):
    settings = 'os', 'compiler', 'build_type', 'arch'
    generators = 'CMakeToolchain', 'CMakeDeps'

    def requirements(self):
        self.requires('cli11/2.4.2')
        self.requires('fmt/11.1.3')

        if self.settings.os != 'Macos':
            self.requires('hdf5/1.14.5')

    def config_options(self):
        pass
