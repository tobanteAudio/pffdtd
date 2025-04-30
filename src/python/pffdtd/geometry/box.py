# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2021 Brian Hamilton
from typing import Any

import click
import numpy as np
import numpy.random as npr
from pffdtd.geometry.math import rotmatrix_ax_ang


def _default(a: Any, default: Any):
    # only used in box
    if a is None:
        a = default
    return a


class Box:
    """This is a class for a box. Used in a few places (vox_grid, tri_box).
    """

    def __init__(self, Lx=None, Ly=None, Lz=None, Rax=None, Rang=None, shift=None, centered=True):
        # defaults
        Lx = _default(Lx, 1.0)
        Ly = _default(Ly, 1.0)
        Lz = _default(Lz, 1.0)
        Rax = _default(Rax, np.array([1., 1., 1.]))
        Rang = _default(Rang, 0.)
        shift = _default(shift, np.array([0., 0., 0.]))

        self.centered = centered

        self.init(Lx, Ly, Lz, Rax, Rang, shift)

    def init(self, Lx, Ly, Lz, Rax, Rang, shift):

        verts = np.array([[0., 0., 0.], [0., 0., Lz], [0., Ly, 0.], [0., Ly, Lz], [
            Lx, 0., 0.], [Lx, 0., Lz], [Lx, Ly, 0.], [Lx, Ly, Lz]])
        if self.centered:
            verts -= 0.5*np.array([Lx, Ly, Lz])

        A = np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.],
                      [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

        if self.centered:
            b = 0.5*np.array([Lx, Ly, Lz, Lx, Ly, Lz])
        else:
            b = np.array([0, 0, 0, Lx, Ly, Lz])

        # rotate verts
        R = rotmatrix_ax_ang(Rax, Rang)
        verts = verts.dot(R.T) + shift

        # get AABB min/max
        bmin = np.amin(verts, 0)
        bmax = np.amax(verts, 0)

        # update A,b
        A = A.dot(R.T)
        b += shift.dot(A.T)

        # set member variables
        self.A = A
        self.b = b
        self.verts = verts
        self.bmin = bmin
        self.bmax = bmax

        self.edges = np.array([[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [
            2, 6], [4, 5], [4, 6], [3, 7], [5, 7], [6, 7]])
        self.tris = np.array([[0, 1, 3], [0, 3, 2], [1, 7, 3], [1, 5, 7], [0, 2, 6], [
            0, 6, 4], [4, 7, 5], [4, 6, 7], [2, 3, 7], [2, 7, 6], [0, 5, 1], [0, 4, 5]])
        self.quads = np.array([[0, 1, 3, 2], [0, 4, 5, 1], [4, 6, 7, 5], [
            1, 5, 7, 3], [2, 3, 7, 6], [0, 2, 6, 4]])

    def randomise(self):
        Lx = 10*npr.random()
        Ly = 10*npr.random()
        Lz = 10*npr.random()
        Rax = npr.rand(3)
        Rang = (-1.0 + 2.0*npr.random())*90
        shift = npr.randn(3)*2 - 1
        self.init(Lx, Ly, Lz, Rax, Rang, shift)

    # just draw (no show)
    def _draw(self, backend='mayavi', r=None, color=(0, 1, 0)):
        box = self

        if backend == 'mayavi':
            from mayavi import mlab
            for edge in box.edges:
                mlab.plot3d(box.verts[edge, 0], box.verts[edge, 1],
                            box.verts[edge, 2], color=color, tube_radius=r)
            mlab.draw()
            print('drawing box..')
        else:
            raise RuntimeError(f"invalid backend {backend}")

    # draw and show
    def draw(self, backend='mayavi', r=None, color=(0, 1, 0)):
        box = self
        if backend == 'mayavi':
            from mayavi import mlab
            from tvtk.api import tvtk
            fig = mlab.gcf()
            box._draw()
            box._draw_faces()
            # stacks as column vectors 2x3
            fake_verts = np.c_[box.bmin, box.bmax]
            mlab.plot3d(fake_verts[0], fake_verts[1],
                        fake_verts[2], transparent=True, opacity=0)
            mlab.axes(xlabel='x', ylabel='y', zlabel='z', color=(0., 0., 0.))
            fig.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()
            mlab.show()
        else:
            raise RuntimeError(f"invalid backend {backend}")

    def show(self, backend='mayavi'):
        if backend == 'mayavi':
            from mayavi import mlab
            mlab.show()
        elif backend == 'mplot3d':
            import matplotlib.pyplot as plt
            plt.show()

    # draw faces without show()
    def _draw_faces(self, backend='mayavi'):
        box = self

        if backend == 'mayavi':
            from mayavi import mlab
            x = box.verts[:, 0]
            y = box.verts[:, 1]
            z = box.verts[:, 2]
            mlab.triangular_mesh(
                x, y, z, box.tris, color=(0, 0, 0), opacity=0.6)
            mlab.draw()
            print('drawing box faces..')
        else:
            raise RuntimeError(f"invalid backend {backend}")


@click.command(name='box')
def main():
    # no shift, no rotation
    Lx = npr.random()
    Ly = npr.random()
    Lz = npr.random()
    print(f'Lx = {Lx:.2f}, Ly = {Ly:.2f}, Lz = {Lz:.2f}')

    box = Box(Lx, Ly, Lz)
    print(f'box 0 ... bmin = {box.bmin[0]:.2f},{box.bmin[1]:.2f},{box.bmin[2]:.2f}')
    print(f'box 0 ... bmax = {box.bmax[0]:.2f},{box.bmax[1]:.2f},{box.bmax[2]:.2f}')
    assert np.all(np.mean(box.verts, 0).dot(box.A.T) < box.b)
    assert np.all((box.verts*(1-1e-4)).dot(box.A.T) < box.b)
    assert np.any((box.verts*(1+1e-4)).dot(box.A.T) > box.b)

    # rotated
    Lx = npr.random()
    Ly = npr.random()
    Lz = npr.random()
    print(f'Lx = {Lx:.2f}, Ly = {Ly:.2f}, Lz = {Lz:.2f}')
    Rax = npr.rand(3)
    Rang = (-1.0 + 2.0*npr.random())*90
    print(f'Rax = {Rax[0]:.2f}, {Rax[1]:.2f}, {Rax[2]:.2f} Rang = {Rang:.2f} degrees')
    print()

    box = Box(Lx, Ly, Lz, Rax, Rang)
    print(f'Lx = {Lx:.2f}, Ly = {Ly:.2f}, Lz = {Lz:.2f}')
    print(f'Rax = {Rax[0]:.2f}, {Rax[1]:.2f}, {Rax[2]:.2f} Rang = {Rang:.2f} degrees')
    print(f'box 1 ... bmin = {box.bmin[0]:.2f}, {box.bmin[1]:.2f}, {box.bmin[2]:.2f}')
    print(f'box 1 ... bmax = {box.bmax[0]:.2f}, {box.bmax[1]:.2f}, {box.bmax[2]:.2f}')
    assert np.all(np.mean(box.verts, 0).dot(box.A.T) < box.b)
    assert np.all((box.verts*(1-1e-4)).dot(box.A.T) < box.b)
    assert np.any((box.verts*(1+1e-4)).dot(box.A.T) > box.b)
    print()

    # shifted and rotated
    Lx = npr.random()
    Ly = npr.random()
    Lz = npr.random()
    print(f'Lx = f{Lx:.2f}, Ly = f{Ly:.2f}, Lz = f{Lz:.2f}')
    Rax = npr.rand(3)
    Rang = (-1.0 + 2.0*npr.random())*90
    print(f'Rax = {Rax[0]:.2f},{Rax[1]:.2f},{Rax[2]:.2f}, {Rang=:.2f} degrees')
    shift = npr.rand(3)*100
    print(f'shift = {shift[0]:.2f},{shift[1]:.2f},{shift[2]:.2f}')

    box = Box(Lx, Ly, Lz, Rax, Rang, shift)
    print(f'box 2 ... bmin = {box.bmin[0]:.2f}, {box.bmin[1]:.2f}, {box.bmin[2]:.2f}')
    print(f'box 2 ... bmax = {box.bmax[0]:.2f}, {box.bmax[1]:.2f}, {box.bmax[2]:.2f}')
    assert np.all(np.mean(box.verts, 0).dot(box.A.T) < box.b)
    assert np.all((((box.verts-shift)*(1-1e-4))+shift).dot(box.A.T) < box.b)
    assert np.any((((box.verts-shift)*(1+1e-4))+shift).dot(box.A.T) > box.b)
    box.draw()
