# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch
import pathlib

import FreeCAD
import MeshPart


def main():
    doc = FreeCAD.ActiveDocument

    out_dir = pathlib.Path('/home/tobante/Developer/tobanteAudio/pffdtd/models/Atmos/obj')
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    model_name = 'Part'
    filename = f'{model_name.lower()}.obj'
    out_path = out_dir/filename

    obj = doc.getObject(model_name)
    mesh = MeshPart.meshFromShape(Shape=obj.Shape, LinearDeflection=0.1, Segments=True)
    mesh.write(Filename=str(out_path), Format='obj')


if __name__ == '__main__':
    main()
