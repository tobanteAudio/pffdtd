# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch

import FreeCAD
import Part

import numpy as np


def quadratic_residue_diffuser(prime, depth=None) -> np.ndarray:
    """
    Duplicate of pffdtd.duffusion.diffusor because FreeCAD
    doesn't have access to the pffdtd python package.
    """
    n = np.mod(np.arange(0, prime, 1)**2, prime)
    if depth:
        n = n / np.max(n)
        return n*depth
    return n


def main():
    doc = FreeCAD.ActiveDocument

    L = 400
    W = 30
    D = 100
    prime = 19
    small_prime = 3
    backing = 10
    fin = 5

    large_depths = quadratic_residue_diffuser(prime, D)
    large_depths = np.append(large_depths, large_depths[0])

    small_depths = quadratic_residue_diffuser(small_prime, W*0.75)
    small_depths = np.append(small_depths, small_depths[0])
    small_width = W/small_depths.shape[0]

    boxes = [Part.makeBox((W+fin)*large_depths.shape[0], L,
                          backing, FreeCAD.Vector(0, 0, -backing))]
    for i, well in enumerate(large_depths):
        x = (W+fin)*i
        if well > 0.0:
            boxes.append(Part.makeBox(W, L, well, FreeCAD.Vector(x, 0, 0)))

        for j, small_well in enumerate(small_depths):
            if small_well == 0.0:
                continue
            pos = FreeCAD.Vector(x+small_width*j, 0, well)
            sw = Part.makeBox(small_width, L, small_well, pos)
            boxes.append(sw)

        if i > 0:
            boxes.append(Part.makeBox(fin, L, D+W*0.75, FreeCAD.Vector(x-fin, 0, 0)))
    diffusor = boxes[0].fuse(boxes[1:])
    Part.show(diffusor)

    print(f"Total width = {(W+fin)*large_depths.shape[0]}")


if __name__ == '__main__':
    main()
