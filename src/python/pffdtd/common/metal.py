# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Tobias Hienzsch

import ctypes
import sys


def metal_device_available():
    if sys.platform != 'darwin':
        return False

    try:
        metal = ctypes.CDLL('/System/Library/Frameworks/Metal.framework/Metal')
        create_device = metal.MTLCreateSystemDefaultDevice
        create_device.argtypes = []
        create_device.restype = ctypes.c_void_p
        return bool(create_device())
    except (OSError, AttributeError):
        return False
