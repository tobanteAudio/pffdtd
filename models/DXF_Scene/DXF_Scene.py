# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Tobias Hienzsch
import pathlib

import numpy as np
import ezdxf
from ezdxf import units
from ezdxf.addons import geo
from ezdxf.gfxattribs import GfxAttribs
from shapely import contains_xy, intersects_xy, prepare
from shapely.affinity import translate
from shapely.geometry import GeometryCollection, shape
from shapely.ops import unary_union

from pffdtd.diffusion.diffusor import quadratic_residue_diffusor
from pffdtd.geometry.math import to_ixy
from pffdtd.sim2d.setup import sim_setup_2d


def generate_dxf(path: pathlib.Path):
    doc = ezdxf.new(setup=True)
    doc.units = units.M
    doc.layers.add('DOMAIN')
    doc.layers.add('RIGID')
    doc.layers.add('SOURCES')
    doc.layers.add('RECEIVERS')

    width = 15
    length = 15
    msp = doc.modelspace()
    msp.add_lwpolyline(
        [(0, 0), (width, 0), (width, length), (0, length)],
        close=True,
        dxfattribs=GfxAttribs(layer='DOMAIN')
    )
    # msp.add_lwpolyline(
    #     [(0.5, 1.5), (1.5, 1.5), (1.5, 2.5), (0.5, 2.5)],
    #     close=True,
    #     dxfattribs=GfxAttribs(layer='RIGID')
    # )
    # msp.add_lwpolyline(
    #     [(3, 1.5), (4, 1.5), (4, 2.5), (3, 2.5)],
    #     close=True,
    #     dxfattribs=GfxAttribs(layer='RIGID')
    # )

    well_width = 0.07*2.0
    back_thickness = 0.025
    qrd = quadratic_residue_diffusor(97, depth=3.5)
    polyline = [(0, qrd[0]), (well_width, qrd[0])]
    for well in qrd[1:]:
        last_x = polyline[-1][0]
        polyline.append((last_x, well))
        polyline.append((last_x+well_width, well))

    polyline.append((polyline[-1][0], 0.0-back_thickness))
    polyline.append((0.0, 0.0-back_thickness))

    x_offset = width/2-max(x for x, _ in polyline)/2
    polyline = [(x+x_offset, y) for x, y in polyline]
    msp.add_lwpolyline(polyline, close=True, dxfattribs=GfxAttribs(layer='RIGID'))

    msp.add_point((width/2, length/2), dxfattribs=GfxAttribs(layer='SOURCES'))
    msp.add_point((1, 1), dxfattribs=GfxAttribs(layer='RECEIVERS'))

    doc.saveas(path)


def layer_geometry(modelspace, layer, tolerance):
    geometries = []

    for entity in modelspace:
        if entity.dxf.layer.upper() != layer or entity.dxftype() == 'POINT':
            continue

        geometry = shape(
            geo.proxy(entity, distance=tolerance).__geo_interface__
        )
        if geometry.geom_type not in {'Polygon', 'MultiPolygon'}:
            raise ValueError(
                f'{layer} entity {entity.dxftype()} is not a closed area'
            )
        geometries.append(geometry)

    if not geometries:
        return GeometryCollection()

    return unary_union(geometries)


def layer_points(modelspace, layer):
    return [
        (entity.dxf.location.x, entity.dxf.location.y)
        for entity in modelspace
        if entity.dxf.layer.upper() == layer
        and entity.dxftype() == 'POINT'
    ]


def load_dxf_scene(path, tolerance=1e-3):
    modelspace = ezdxf.readfile(path).modelspace()

    domain = layer_geometry(modelspace, 'DOMAIN', tolerance)
    rigid = layer_geometry(modelspace, 'RIGID', tolerance)

    if domain.is_empty:
        raise ValueError('DXF has no DOMAIN geometry')
    if not domain.is_valid:
        raise ValueError('DXF DOMAIN geometry is invalid')
    if not rigid.is_valid:
        raise ValueError('DXF RIGID geometry is invalid')

    min_x, min_y, max_x, max_y = domain.bounds
    room = (max_x-min_x, max_y-min_y)

    # sim_setup_2d expects the lower room corner at the origin.
    domain = translate(domain, xoff=-min_x, yoff=-min_y)
    rigid = translate(rigid, xoff=-min_x, yoff=-min_y)

    sources = [
        (x-min_x, y-min_y)
        for x, y in layer_points(modelspace, 'SOURCES')
    ]
    receivers = [
        (x-min_x, y-min_y)
        for x, y in layer_points(modelspace, 'RECEIVERS')
    ]

    if not sources:
        raise ValueError('DXF has no source POINT entities')
    if not receivers:
        raise ValueError('DXF has no receiver POINT entities')

    prepare(domain)
    prepare(rigid)

    def model(*, Lx, Ly, Nx, Ny, dx, X, Y, in_mask):
        del Lx, Ly, dx

        in_mask[:] = contains_xy(domain, X, Y)
        if not rigid.is_empty:
            in_mask[intersects_xy(rigid, X, Y)] = False

        def rasterize_points(points, kind):
            indices = []
            x_samples = X[:, 0]
            y_samples = Y[0, :]

            for x, y in points:
                ix = int(np.argmin(np.abs(x_samples-x)))
                iy = int(np.argmin(np.abs(y_samples-y)))

                if not in_mask[ix, iy]:
                    raise ValueError(
                        f'{kind} at {(x, y)} snaps outside the air domain'
                    )

                indices.append(to_ixy(ix, iy, Nx, Ny))

            return indices

        in_ixy = rasterize_points(sources, 'source')
        out_ixy = rasterize_points(receivers, 'receiver')
        return in_mask, in_ixy, out_ixy

    return room, model


def main():
    sim_dir = pathlib.Path('../../sim_data/DXF_Scene/cpu')
    sim_dir.mkdir(parents=True, exist_ok=True)

    dxf_path = sim_dir/'scene.dxf'
    generate_dxf(dxf_path)
    room, model = load_dxf_scene(dxf_path)

    sim_setup_2d(
        sim_dir=sim_dir,
        room=room,
        Tc=20,
        rh=50,
        fmax=5000,
        ppw=10.5,
        duration=0.05*10,
        refl_coeff=0.99,
        model_factory=model,
        apply_loss=True,
        diff=False,
        band_limited=True,
        image=True,
        verbose=True,
    )


if __name__ == '__main__':
    main()
