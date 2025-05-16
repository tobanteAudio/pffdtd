# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

from collections import defaultdict
import json
from typing import Any

import numpy as np

from pffdtd.geometry.math import make_box


def load_mesh(obj_file, reverse=False):
    with open(obj_file) as f:
        lines = [line.rstrip() for line in f]

    tris = []
    pts = []

    for line in lines:
        if not 'vn ' in line:
            if 'v ' in line:
                parts = line.split(' ')
                parts = parts[1:4]
                parts = [float(part)/1000.0 for part in parts]
                pts.append(parts)
            if 'f ' in line:
                parts = line.split(' ')
                parts = parts[1:4]
                parts = [int(part.split('/')[0])-1 for part in parts]
                tris.append(parts if not reverse else parts[::-1])

    return pts, tris


class MeshModelBuilder:
    def __init__(self) -> None:
        self.root: dict[str, Any] = {
            'mats_hash': {},
            'sources': [],
            'receivers': []
        }

    def add(self, name, obj_file, color, reverse=False, sides=1):
        assert name not in self.root
        pts, tris = load_mesh(obj_file, reverse=reverse)
        self.root['mats_hash'][name] = {
            'tris': tris,
            'pts': pts,
            'color': color,
            'sides': [sides]*len(tris)
        }

    def add_receiver(self, name, pos):
        self.root['receivers'].append({'name': name, 'xyz': pos})

    def add_source(self, name, pos):
        self.root['sources'].append({'name': name, 'xyz': pos})

    def write(self, model_file):
        src = np.array(self.root['sources'][0]['xyz'])
        ref = np.linalg.norm(src - np.array(self.root['receivers'][0]['xyz']))
        for r in self.root['receivers']:
            distance = np.linalg.norm(src - np.array(r['xyz']))
            print(r['name'], 20*np.log10(ref/distance))

        with open(model_file, 'w') as f:
            json.dump(self.root, f)
            print('', file=f)


class RoomModelBuilder:
    def __init__(self, length, width,  height):
        self.length = length
        self.width = width
        self.height = height
        self.sources = []
        self.cabinet_sources = []
        self.receivers = []
        self.boxes = defaultdict(list)
        self.colors = {}

    def get_color(self, material):
        if material in self.colors:
            return list(self.colors[material])
        return list(np.floor(np.random.rand(3)*255))

    def with_colors(self, colors):
        self.colors = colors
        return self

    def add_source(self, name, position):
        self.sources.append({
            'name': name,
            'xyz': position,
        })
        return self

    def add_cabinet_speaker(self, name, position, size=None, center=None):
        size = size if size else [0.25, 0.33, 0.39]
        if not center:
            center = [size[0]/2, -0.075, size[2]/2]

        self.add_source(name, position)
        self.cabinet_sources.append({
            'xyz': position,
            'size': size,
            'center': center,
        })

        return self

    def add_receiver(self, name, position):
        self.receivers.append({
            'name': name,
            'xyz': position,
        })
        return self

    def add_box(self, material, size, position, rotation=None):
        self.boxes[material].append({
            'size': size,
            'position': position,
            'rotation': rotation if rotation else [0, 0, 0],
        })
        return self

    def add_diffusor_1d(self, size, position, well_width):
        max_depth = size[1]
        depths = np.array([0.0, 0.25, 1.0, 0.5, 0.5, 1.0, 0.25]) * max_depth
        wells = int(size[0]/well_width)
        for i in range(wells):
            x = position[0] + well_width*i
            y = position[1]
            z = position[2]
            depth = depths[i % len(depths)]
            self.add_box('Diffusor', [well_width, depth, size[2]], [x, y, z])

    def build(self, file_path):
        L = self.length
        W = self.width
        H = self.height

        counter = 0
        for src in self.cabinet_sources:
            rot = [0, 0, 0]
            size = src['size']
            xyz = src['xyz']
            center = src['center']
            pos = list(np.array(xyz) - np.array(center))
            self.add_box('Speaker_Cabinet', size, pos, rot)

        model = {
            'mats_hash': {
                'Walls': {
                    'tris': [
                        [0, 1, 2],  # Back Wall
                        [0, 2, 3],

                        [6, 5, 4],  # Front Wall
                        [7, 6, 4],

                        [0, 3, 4],  # Left Wall
                        [7, 4, 3],

                        [5, 2, 1],  # Right Wall
                        [2, 5, 6],
                    ],
                    'pts': [
                        [0.0, 0.0, 0.0],  # Back Wall
                        [W, 0.0, 0.0],
                        [W, 0.0, H],
                        [0.0, 0.0, H],

                        [0.0, L, 0.0],  # Front Wall
                        [W, L, 0.0],
                        [W, L, H],
                        [0.0, L, H],
                    ],
                    'color': self.get_color('Walls'),
                    'sides': [1, 1, 1, 1, 1, 1, 1, 1]
                },
                'Ceiling': {
                    'tris': [[2, 1, 0], [2, 3, 1]],
                    'pts': [
                        [0, 0, H],
                        [0, L, H],
                        [W, 0, H],
                        [W, L, H]
                    ],
                    'color': self.get_color('Ceiling'),
                    'sides': [1, 1]
                },
                'Floor': {
                    'tris': [[0, 1, 2], [1, 3, 2]],
                    'pts': [
                        [0, 0, 0],
                        [0, L, 0],
                        [W, 0, 0],
                        [W, L, 0]
                    ],
                    'color': self.get_color('Floor'),
                    'sides': [1, 1]
                },
            },
            'sources': self.sources,
            'receivers': self.receivers,
        }

        for key, boxes in self.boxes.items():
            spec = {
                'tris': [],
                'pts': [],
                'color': self.get_color(key),
                'sides': []
            }

            counter = 0
            for box in boxes:
                size = box['size']
                pos = box['position']
                rot = box['rotation']
                ps, ts = make_box(size[0], size[1], size[2], pos, rot, counter)

                spec['tris'] += ts
                spec['pts'] += ps
                spec['sides'] += [2]*len(ts)

                counter += len(ps)

            model['mats_hash'][key] = spec

        with open(file_path, 'w') as file:
            json.dump(model, file)
            print('', file=file)
