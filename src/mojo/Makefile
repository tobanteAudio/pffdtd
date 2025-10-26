# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch

SIM2D_DIR ?= ../../sim_data/Diffusor/cpu
SIM3D_DIR ?= ../../sim_data/RedBullStudiosBerlin/cpu

export MOJO_ENABLE_STACK_TRACE_ON_ERROR = 1

.PHONY: all
all: cli gpuinfo vecadd matmul

bin:
	mkdir -p bin

.PHONY: cli
cli: bin
	mojo build -o bin/$@ src/$@.mojo
	bin/$@ -h
	bin/$@ foo.txt

.PHONY: gpuinfo
gpuinfo: bin
	mojo build -o bin/$@ src/$@.mojo
	bin/$@

.PHONY: vecadd
vecadd: bin
	mojo build -o bin/$@ src/$@.mojo
	bin/$@ 10

.PHONY: matmul
matmul: bin
	mojo build -o bin/$@ src/$@.mojo
	bin/$@

.PHONY: sim2d
sim2d: bin
	mojo build -o bin/$@ src/$@.mojo
	bin/$@ ${SIM2D_DIR}

.PHONY: sim3d
sim3d: bin
	mojo build -o bin/$@ src/$@.mojo
	bin/$@ ${SIM3D_DIR}

.PHONY: format
format: $(wildcard src/*.mojo)
	mojo format -l 120 $?

.PHONY: clean
clean:
	rm -rf bin
