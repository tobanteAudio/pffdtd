# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2021 Brian Hamilton
import json

import click
import numpy as np

from pffdtd.geometry.math import dotv, rotate_az_el_deg
from pffdtd.geometry.tris_precompute import tris_precompute


class RoomGeometry:
    """Class for room geometry, source/receiver positions, and materials (labels)

    - Reads JSON exported from Sketchup plugin
    - Also prunes triangles, prints some stats (surface areas, volume), rotates scene, and draws
    """

    def __init__(self, model_file=None, az_el=[0., 0.], area_eps=1e-6, bmin=None, bmax=None):
        # main dict for room data
        self.mats_dict = None
        # bmin and bmax may take custom bounds of scene
        if bmin is None:
            self.bmin = np.array([np.inf, np.inf, np.inf])
        else:
            self.bmin = bmin

        if bmax is None:
            self.bmax = -np.array([np.inf, np.inf, np.inf])
        else:
            self.bmax = bmax

        self.tris = None
        self.mat_side = None
        self.pts = None
        self.mat_ind = None
        self.mat_area = None
        self.mat_str = []
        self.vol = None
        self.colors = None
        if model_file is None:
            raise RuntimeError('model_file not specified')

        self.area_eps = area_eps

        # identity 3x3 matrix by default
        self.R, _, _ = rotate_az_el_deg(*az_el)

        if np.any(az_el != 0):
            self.print(f'az-el deg rotation: {az_el}')

        self.load_json(model_file)
        self.collapse_tris()
        self.calc_volume()

    def print(self, fstring):
        print(f'--ROOM_GEO: {fstring}')

    def load_json(self, json_filename):
        bmin = self.bmin
        bmax = self.bmax
        R = self.R

        with open(json_filename) as model_file:
            data = json.load(model_file)
        # print(data)

        # attach
        mats_dict = data['mats_hash']
        mat_str = list(mats_dict.keys())

        Nmat = len(mat_str)  # not including unmarked
        mat_str.sort()  # will process in alphabetical order
        if '_RIGID' in mat_str:
            # move to end (also corresponds to -1 index)
            mat_str.remove('_RIGID')
            mat_str.append('_RIGID')
            Nmat -= 1  # adjust Nmat

        # print(mat_str)

        colors = []
        # convert to np arrays
        for mat in mat_str:
            mats_dict[mat]['pts'] = np.array(mats_dict[mat]['pts'], dtype=np.float64) @ R  # also rotate pts here
            mats_dict[mat]['tris'] = np.array(mats_dict[mat]['tris'], dtype=np.int64)
            colors.append(mats_dict[mat]['color'])

        # calculate bmin/bmax
        for mat in mat_str:
            pts = mats_dict[mat]['pts']
            # tris = mats_dict[mat]['tris']
            bmin = np.min(np.r_[pts, bmin[None, :]], axis=0)
            bmax = np.max(np.r_[pts, bmax[None, :]], axis=0)

        assert len(data['sources']) > 0  # sources have to be defined in JSON
        assert len(data['receivers']) > 0  # receivers have to be defined in JSON
        Sxyz = np.atleast_2d(np.array([source['xyz'] for source in data['sources']], dtype=np.float64)) @ R
        assert np.all((Sxyz > bmin) & (Sxyz < bmax))

        Rxyz = np.atleast_2d(np.array([receiver['xyz'] for receiver in data['receivers']], dtype=np.float64)) @ R
        assert np.all((Rxyz > bmin) & (Rxyz < bmax))

        self.mats_dict = mats_dict
        self.mat_str = mat_str
        self.bmin = bmin
        self.bmax = bmax
        self.Nmat = Nmat
        self.colors = colors
        self.Sxyz = Sxyz
        self.Rxyz = Rxyz

    def collapse_tris(self):
        mats_dict = self.mats_dict
        mat_str = self.mat_str
        Nmat = self.Nmat
        # collapse tris and pts for easier computation
        pts = np.concatenate([mats_dict[mat]['pts'] for mat in mat_str], axis=0)

        # need to offset pt indices when concatenating tris
        tri_offsets = np.r_[0, np.cumsum([mats_dict[mat]['pts'].shape[0] for mat in mat_str])[:-1]]
        assert tri_offsets.size == len(mat_str)  # otherwise error, until PEP618 (Python 3.10)
        tris = np.concatenate([mats_dict[mat]['tris']+toff for mat, toff in zip(mat_str, tri_offsets)], axis=0)

        # can handle open scenes, but expects exported JSON to have at least four triangles
        assert tris.shape[0] >= 4  # if this is a problem just insert tiny fake triangles into scene, or modify code

        # use array of int8, -1 will be flag for unmarked (rigid)
        mat_ind = np.concatenate([np.ones(mats_dict[mat]['tris'].shape[0], dtype=np.int8)*ind for mat, ind in zip(mat_str, range(len(mat_str)))], axis=0)
        mat_ind[mat_ind == Nmat] = -1  # this should be anything on _RIGID (when len(mat_str)==Nmat+1)

        mat_side = np.concatenate([mats_dict[mat]['sides'] for mat in mat_str], axis=0)

        # all unmarked should have 0 for sidedness
        assert np.all(mat_side[mat_ind == -1] == 0)

        # print(f'{pts=}')
        # print(f'{tris=}')
        tris_pre = tris_precompute(tris=tris, pts=pts)

        self.pts = pts
        self.tris = tris
        self.mat_ind = mat_ind
        self.mat_side = mat_side
        self.tris_pre = tris_pre

        self.prune_by_area()  # delete small triangles for ray-tri or tri-box processing (not removing from original dict)
        self.calc_areas()

    def calc_areas(self):
        mat_ind = self.mat_ind
        mat_side = self.mat_side
        tris_pre = self.tris_pre
        Nmat = self.Nmat

        mat_area = np.empty((Nmat,), np.float64)
        for i in range(0, Nmat):  # ignores unmarked
            ii = np.nonzero(mat_ind == i)[0]
            sides = mat_side[ii]
            fac = np.zeros(sides.shape)
            fac[sides == 1] = 1.0  # back side only
            fac[sides == 2] = 1.0  # front side only
            fac[sides == 3] = 2.0  # both sides
            mat_area[i] = np.sum(tris_pre[ii]['area']*fac)
        self.mat_area = mat_area

    def prune_by_area(self):
        ii = np.nonzero(self.tris_pre['area'] < self.area_eps)[0]
        # self.print(f"deleting triangles: {ii}\n with areas {self.tris_pre[ii]['area']}")

        self.tris = np.delete(self.tris, ii, axis=0)
        self.mat_ind = np.delete(self.mat_ind, ii, axis=0)
        self.mat_side = np.delete(self.mat_side, ii, axis=0)
        self.tris_pre = np.delete(self.tris_pre, ii, axis=0)
        self.print(f'{ii.size} degenerate triangles deleted')

    def calc_volume(self):
        tris_pre = self.tris_pre

        # from discretized divergence theroem (volume somewhat misleading for open scenes)
        vol = np.sum(dotv(tris_pre['cent'], tris_pre['nor']))/6.0

        area = np.sum(tris_pre['area'])

        self.vol = vol
        self.area = area

    def draw(self, backend='mayavi', plot_normals=False, wireframe=False, enable_surface=False):
        mats_dict = self.mats_dict
        mat_str = self.mat_str
        Nmat = self.Nmat
        fig = None

        if wireframe:
            assert backend == 'mayavi'

        if backend == 'mayavi':
            # sources/receivers not drawn
            from mayavi import mlab
            from tvtk.api import tvtk

            fig = mlab.figure()
            # fig = mlab.gcf()

            for m in range(-1, Nmat):
                mat = mat_str[m]
                pts = mats_dict[mat]['pts']
                tris = mats_dict[mat]['tris']
                if m == -1:
                    color = (1, 1, 1)
                else:
                    color = tuple(np.array(mats_dict[mat]['color'])/255.0)

                # tris
                if not wireframe:
                    mlab.triangular_mesh(*(pts.T), tris, color=color, opacity=1.0)
                    mlab.triangular_mesh(*(pts.T), tris, color=(0, 0, 0), representation='wireframe', opacity=1.0)
                else:
                    mlab.triangular_mesh(*(pts.T), tris, color=color, representation='wireframe', opacity=1.0)

                # wireframe
                if plot_normals:
                    mtp = tris_precompute(tris=tris, pts=pts)  # precompute in mat-groups
                    mlab.quiver3d(*mtp['cent'].T, *mtp['nor'].T, color=color)

            mlab.orientation_axes()

            # mlab.axes binds to last object so draw line across bmin and bmax and attach axes
            fake_verts = np.c_[self.bmin, self.bmax]  # stacks as column vectors 2x3
            mlab.plot3d(*fake_verts, transparent=True, opacity=0)
            mlab.axes(xlabel='x', ylabel='y', zlabel='z', color=(0., 0., 0.))

            # fix z-up
            fig.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()

        elif backend == 'polyscope':
            import polyscope as ps
            # Initialize polyscope
            try:
                ps.init()  # gives error if run more than once (at least in ipython)
            except:
                self.print('polyscope already initialised?')
            # issue with saving settings on successive runs, need some close() function

            ps.set_SSAA_factor(4)
            ps.set_up_dir('z_up')

            sources = ps.register_point_cloud(
                name='Sources',
                points=self.Sxyz,
                enabled=True,
                point_render_mode='sphere',
                color=(0, 0, 1)
            )
            sources.set_radius(0.05, relative=False)

            receivers = ps.register_point_cloud(
                name='Receivers',
                points=self.Rxyz,
                enabled=True,
                point_render_mode='sphere',
                color=(0, 1, 0)
            )
            receivers.set_radius(0.05, relative=False)

            # Register a mesh
            for m in range(-1, Nmat):
                mat = mat_str[m]
                pts = mats_dict[mat]['pts']
                tris = mats_dict[mat]['tris']
                if m == -1:
                    color = (1, 1, 1)
                else:
                    color = tuple(np.array(mats_dict[mat]['color'])/255.0)

                mat_mesh = ps.register_surface_mesh(
                    name=mat,
                    vertices=pts,
                    faces=tris,
                    color=color,
                    edge_color=(0, 0, 0),
                    edge_width=1,
                    enabled=enable_surface,
                )

                if plot_normals:
                    mtp = tris_precompute(tris=tris, pts=pts)
                    mat_mesh.add_vector_quantity('Normals', mtp['nor'], enabled=True, defined_on='faces')

        else:
            raise RuntimeError(f"invalid backend {backend}")

        self.fig = fig

    def show(self, backend='mayavi'):
        if backend == 'mayavi':
            from mayavi import mlab
            mlab.show()
        elif backend == 'polyscope':
            import polyscope as ps
            # View the point cloud and mesh we just registered in the 3D UI
            ps.show()
        else:
            raise RuntimeError(f"invalid backend {backend}")

    def print_stats(self):
        rg = self
        self.print(f'npts =  {rg.pts.shape[0]}')
        self.print(f'ntris = {rg.tris.shape[0]}')
        self.print(f'bmin = {rg.bmin}')
        self.print(f'bmax = {rg.bmax}')
        self.print(f'Lxyz = {(rg.bmax-rg.bmin)}')
        self.print(f'bbvol = {np.prod(rg.bmax-rg.bmin):.3f}m³')
        self.print(f'room vol = {rg.vol}m³')
        self.print(f'room SA = {rg.area}m²')
        self.print(f'room Sxyz = {rg.Sxyz}')
        self.print(f'room Rxyz = {rg.Rxyz}')
        for i in range(rg.Nmat):
            self.print(f'mat {i}: {rg.mat_str[i]}, {rg.mat_area[i]:.3f}m²')


@click.command(name='room-geometry', help='Draw room geometry.')
@click.option('--model', type=click.Path(exists=True))
@click.option('--backend', default='polyscope', type=click.Choice(['mayavi', 'polyscope']))
@click.option('--nodraw', is_flag=True)
@click.option('--drawnormals', is_flag=True)
@click.option('--az_el', nargs=2, type=float, default=(0, 0))
def main(model, backend, nodraw, drawnormals, az_el):
    az_el = list(az_el)
    room = RoomGeometry(model, az_el=az_el)
    room.print_stats()
    if not nodraw:
        room.draw(backend, plot_normals=drawnormals, enable_surface=True)
        room.show(backend)
