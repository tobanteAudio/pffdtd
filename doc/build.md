<!-- SPDX-License-Identifier: MIT -->
<!-- SPDX-FileCopyrightText: 2024 Tobias Hienzsch -->

# Build

- For builds without conan these CMake options can be removed:
  - `-D CMAKE_PROJECT_TOP_LEVEL_INCLUDES`
  - `-D CONAN_HOST_PROFILE`

```shell
cd /path/to/pffdtd

# Only required if building with conan package manager. Optional on Linux & macOS
git clone https://github.com/conan-io/cmake-conan.git -b develop2 external/cmake-conan
```

## Linux

```shell
# Always required
sudo dnf install libomp-devel ninja-build # Fedora
sudo apt install libomp-dev ninja-build   # Ubuntu

# Can be skipped if building with conan package manager
sudo dnf install hdf5-devel fmt-devel cli11-devel # Fedora
sudo apt install libhdf5-dev libfmt-dev libcli11-dev # Ubuntu
```

### GCC or Clang

```shell
cmake -S . -B build -G Ninja -D CMAKE_BUILD_TYPE=Release -D CMAKE_PROJECT_TOP_LEVEL_INCLUDES=external/cmake-conan/conan_provider.cmake
```

### AdaptiveCPP

Assumes that `AdaptiveCPP` was build with `clang++`:

```shell
cmake -S . -B build -G Ninja -D CMAKE_BUILD_TYPE=Release -D CMAKE_C_COMPILER=clang -D CMAKE_CXX_COMPILER=clang++ -D PFFDTD_ENABLE_SYCL_ACPP=ON -D AdaptiveCpp_DIR=/usr/local/lib/cmake/AdaptiveCpp -D ACPP_TARGETS="generic" -D CMAKE_PROJECT_TOP_LEVEL_INCLUDES=external/cmake-conan/conan_provider.cmake
```

### Intel oneAPI

- <https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#base-kit>
- Nvidia:
  - Drivers
  - <https://developer.codeplay.com/products/oneapi/nvidia/2024.2.1/guides/get-started-guide-nvidia>
  - <https://www.server-world.info/en/note?os=Ubuntu_24.04&p=nvidia&f=2>

```shell
source /opt/intel/oneapi/setvars.sh

SETVARS_ARGS="--force" cmake -S. -B build -G Ninja -D CMAKE_BUILD_TYPE=Release -D CMAKE_C_COMPILER=icx -D CMAKE_CXX_COMPILER=icpx -D PFFDTD_ENABLE_SYCL_ONEAPI=ON -D CMAKE_PROJECT_TOP_LEVEL_INCLUDES=external/cmake-conan/conan_provider.cmake -D CONAN_HOST_PROFILE=../cmake/profile/linux_sycl
cmake --build build

# Using hyper-threads is usally a slow down. Use the number of physical cores.
export DPCPP_CPU_PLACES=cores
export DPCPP_CPU_CU_AFFINITY=spread
export DPCPP_CPU_NUM_CUS=16
./run_2d.sh
```

### CUDA

```shell
cmake -S. -B build -G Ninja -D CMAKE_BUILD_TYPE=Release -D CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -D PFFDTD_ENABLE_CUDA=ON -D CMAKE_PROJECT_TOP_LEVEL_INCLUDES=external/cmake-conan/conan_provider.cmake
```

## Windows

```shell
winget install -e --id Kitware.CMake
winget install -e --id Ninja-build.Ninja
winget install -e --id JFrog.Conan
```

### MSVC

```shell
cmake -S . -B build -G Ninja -D CMAKE_BUILD_TYPE=Release -D CMAKE_PROJECT_TOP_LEVEL_INCLUDES=external/cmake-conan/conan_provider.cmake
```

### Clang

Currently not supported. Has issues building HDF5 from source via conan.

### Intel oneAPI

```shell
# Use "Intel oneAPI command prompt" application
# Or: Set environment variables
. 'C:\Program Files (x86)\Intel\oneAPI\setvars.bat'

cmake -S . -B build -G Ninja -D PFFDTD_ENABLE_SYCL_ONEAPI=ON -D CMAKE_BUILD_TYPE=Release -D CMAKE_C_COMPILER=icx -D CMAKE_CXX_COMPILER=icx -D CMAKE_PROJECT_TOP_LEVEL_INCLUDES=external/cmake-conan/conan_provider.cmake -D CONAN_HOST_PROFILE=../cmake/profile/windows_sycl
```
