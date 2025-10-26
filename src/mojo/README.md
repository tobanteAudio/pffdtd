<!-- SPDX-License-Identifier: MIT -->
<!-- SPDX-FileCopyrightText: 2025 Tobias Hienzsch -->

# mojo

## Install

```sh
pip install --pre mojo --index-url https://dl.modular.com/public/nightly/python/simple/
make -C src/mojo clean all
```

## Run

```sh
SIM2D_DIR=../../sim_data/Diffusor/cpu make -C src/mojo sim2d
SIM3D_DIR=../../sim_data/ProStudio/cpu make -C src/mojo sim3d
```
