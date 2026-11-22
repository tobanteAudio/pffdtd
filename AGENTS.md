<!-- SPDX-License-Identifier: MIT -->
<!-- SPDX-FileCopyrightText: 2026 Tobias Hienzsch -->

# Repository guidance

## Scope and safety

- Never read, modify, enumerate, or otherwise inspect `models/private/`. Treat it as out of scope even when searching the repository or running broad formatting commands.
- Ignore `src/mojo/` and `src/freecad/` unless a task explicitly asks for work there.
- Do not commit generated simulation data, build products, coverage output, caches, or virtual environments. In particular, avoid changes under `build/`, `external/`, `htmlcov/`, `sim_data/`, `.venv/`, and `__pycache__/`.
- Preserve unrelated user changes in the working tree.

## Architecture

- `src/python/pffdtd/` contains the Python package. It handles model preparation, geometry and voxelization, signal/material setup, reference engines, analysis, and post-processing.
- `src/cpp/` contains the performance FDTD engine and its CLI, built as `build/src/cpp/pffdtd-engine` (`.exe` on Windows). CPU support is the baseline; CUDA, Metal, and SYCL are optional CMake backends.
- Python setup code and the C++ engine communicate through HDF5 simulation files. Treat dataset names, shapes, ordering, precision, and scaling as an API: coordinated producer/consumer changes need tests on both sides.
- `src/js/` is a small Three.js viewer for JSON models produced by preprocessing. Keep its expected JSON schema aligned with the Python exporter/model builder. It has no local package build; `index.html` imports Three.js from a CDN.
- Python tests live in `src/python/test/`. Some integration tests invoke the compiled C++ engine through `PFFDTD_ENGINE_EXE`.

## Environment

Use the repository virtual environment for Python commands:

```shell
source .venv/bin/activate
```

The package supports Python 3.11 or newer. If dependencies need to be installed or refreshed, use:

```shell
python -m pip install --verbose -e '.[dev,test]'
```

Do not recreate `.venv` unless explicitly requested.

## Building the C++ engine

Follow `.github/workflows/build.yml`. The normal local CMake/Ninja build is:

```shell
source .venv/bin/activate
conan profile detect -f
[ -d external/cmake-conan ] || \
  git clone https://github.com/conan-io/cmake-conan.git -b develop2 external/cmake-conan
cmake -B build -G Ninja \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_PROJECT_TOP_LEVEL_INCLUDES=external/cmake-conan/conan_provider.cmake
cmake --build build
```

Reuse an existing `external/cmake-conan` checkout instead of cloning it again. The build also requires CMake, Ninja, HDF5, and OpenMP. On macOS, follow the workflow's Homebrew setup and pass its `OpenMP_ROOT` CMake option when OpenMP is not discovered automatically. Do not enable CUDA or SYCL merely to validate an unrelated change.

After a successful build, the engine's small built-in C++ check can be run with:

```shell
build/src/cpp/pffdtd-engine test
```

## Testing and validation

Run the smallest relevant tests while iterating. Python-only unit tests may be run directly, for example:

```shell
pytest src/python/test/test_geometry_math.py
```

After building the engine, run the full suite from the repository root with:

```shell
PFFDTD_ENGINE_EXE=build/src/cpp/pffdtd-engine \
  pytest --cov=pffdtd --cov-branch --cov-report=html
```

On Windows, point `PFFDTD_ENGINE_EXE` at `build/src/cpp/pffdtd-engine.exe`. Tests use strict markers/configuration, treat warnings as errors, and define a `slow` marker; do not weaken those settings to make a test pass. For a quick run that intentionally excludes expensive cases, use `pytest -m 'not slow'` and report that the slow tests were skipped.

Before handing off a broad change, run the relevant checks from `.pre-commit-config.yaml` against the files touched by the task. Do not use `pre-commit run --all-files`, because repository-wide tools must not traverse `models/private/`. The configured checks enforce REUSE/SPDX metadata, `autopep8` for Python, and `clang-format` for C-family, CUDA, Metal, and Objective-C files.

## Change guidelines

- Prefer focused changes and add or update tests for behavior changes and bug fixes.
- Keep numerical behavior explicit. Avoid silent dtype changes, reordered HDF5 data, altered coordinate conventions, or relaxed tolerances without a documented reason and parity tests.
- When changing preprocessing outputs, verify the affected Python consumer, C++ loader/engine, and JS viewer as applicable.
- Keep CPU execution working when changing an optional accelerator backend.
- Follow existing naming and style in nearby code. Add valid SPDX license and copyright headers to new files.
- Do not edit generated files to fix a source-level problem.
