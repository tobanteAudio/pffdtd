# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2021 Brian Hamilton

import numpy as np


def rotmatrix_ax_ang(Rax: np.ndarray, Rang: float) -> np.ndarray:
    # see https://en.wikipedia.org/wiki/Rotation_matrix
    assert isinstance(Rax, np.ndarray)
    assert Rax.shape == (3,)
    assert isinstance(Rang, float)

    Rax = Rax/np.linalg.norm(Rax)  # to be sure
    theta = np.deg2rad(Rang)
    ct = np.cos(theta)
    st = np.sin(theta)
    R = np.array([
        [ct + Rax[0]*Rax[0]*(1-ct), Rax[0]*Rax[1]*(1-ct) - Rax[2]*st, Rax[0]*Rax[2]*(1-ct) + Rax[1]*st],
        [Rax[1]*Rax[0]*(1-ct) + Rax[2]*st, ct + Rax[1]*Rax[1]*(1-ct), Rax[1]*Rax[2]*(1-ct) - Rax[0]*st],
        [Rax[2]*Rax[0]*(1-ct) - Rax[1]*st, Rax[2]*Rax[1]*(1-ct) + Rax[0]*st, ct + Rax[2]*Rax[2]*(1-ct)],
    ])

    assert np.linalg.norm(np.linalg.inv(R)-R.T) < 1e-8
    return R


def rotate_xyz_deg(thx_d, thy_d, thz_d):
    # R applies Rz then Ry then Rx (opposite to wikipedia)
    # rotations about x,y,z axes, right hand rule
    thx = np.deg2rad(thx_d)
    thy = np.deg2rad(thy_d)
    thz = np.deg2rad(thz_d)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(thx), -np.sin(thx)],
        [0, np.sin(thx), np.cos(thx)],
    ])
    Ry = np.array([
        [np.cos(thy), 0, np.sin(thy)],
        [0, 1, 0],
        [-np.sin(thy), 0, np.cos(thy)],
    ])
    Rz = np.array([
        [np.cos(thz), -np.sin(thz), 0],
        [np.sin(thz), np.cos(thz), 0],
        [0, 0, 1],
    ])

    R = Rx @ Ry @ Rz
    return R, Rx, Ry, Rz


def rotate_az_el_deg(az_d, el_d):
    # R applies Rel then Raz
    # Rel is rotation about negative y-axis, right hand rule
    # Raz is rotation z-axis, right hand rule
    # this uses matlab convention
    # note, this isn't the most general rotation
    _, _, Ry, Rz = rotate_xyz_deg(0, -el_d, az_d)
    Rel = Ry
    Raz = Rz
    R = Raz @ Rel
    return R, Raz, Rel


def dot2(v):
    return dotv(v, v)


def dotv(v1, v2):
    return np.sum(v1*v2, axis=-1)


def vecnorm(v1):
    return np.sqrt(dot2(v1))


def normalise(v1, eps=np.finfo(np.float64).eps):
    return (v1.T/(vecnorm(v1)+eps)).T


def ind2sub3d(ii, Nx, Ny, Nz):
    iz = ii % Nz
    iy = (ii - iz)//Nz % Ny
    ix = ((ii - iz)//Nz-iy)//Ny
    return ix, iy, iz


def rel_diff(x0, x1):
    # relative difference at machine epsilon level
    return (x0-x1)/(2.0**np.floor(np.log2(x0)))


def iceil(x):
    return np.int_(np.ceil(x))


def iround(x):
    return np.int_(np.round(x))


def to_ixy(x, y, Nx, Ny, order='row'):
    if order == 'row':
        return x*Ny+y
    return y*Nx+x


def point_on_circle(center, radius: float, angle: float):
    """
    Calculate the coordinates of a point on a circle arc.

    Parameters:
        center (tuple): (x, y) coordinates of the center of the circle.
        radius: Radius of the circle.
        angle: Angle in radians.

    Returns:
        tuple: (p_x, p_y) coordinates of the point on the circle arc.
    """
    x, y = center
    p_x = x + radius * np.cos(angle)
    p_y = y + radius * np.sin(angle)
    return (p_x, p_y)


def point_along_line(p1, p2, t):
    return [
        p1[0] + (p2[0]-p1[0]) * t,
        p1[1] + (p2[1]-p1[1]) * t,
        p1[2] + (p2[2]-p1[2]) * t,
    ]


def with_x_offset(p: list, offset):
    tmp = p.copy()
    tmp[0] += offset
    return tmp


def with_z(p: list, z):
    tmp = p.copy()
    tmp[2] = z
    return tmp


def find_third_vertex(A, B):
    A = np.array(A)
    B = np.array(B)

    # Calculate the vector from A to B
    AB = B - A

    # Calculate the distance AB
    d = np.linalg.norm(AB)

    # Normalize the vector AB
    AB_normalized = AB / d

    # Find two vectors perpendicular to AB and to each other
    if AB_normalized[0] != 0 or AB_normalized[1] != 0:
        perp_vector_1 = np.array([-AB_normalized[1], AB_normalized[0], 0])
    else:
        perp_vector_1 = np.array([0, AB_normalized[2], -AB_normalized[1]])

    perp_vector_1 = perp_vector_1 / np.linalg.norm(perp_vector_1)

    perp_vector_2 = np.cross(AB_normalized, perp_vector_1)
    perp_vector_2 = perp_vector_2 / np.linalg.norm(perp_vector_2)

    # Calculate the height of the equilateral triangle
    h = (np.sqrt(3) / 2) * d

    # Calculate the midpoint M
    M = (A + B) / 2

    # Calculate the two possible locations for the third vertex
    C1 = M + h * perp_vector_1
    C2 = M - h * perp_vector_1

    return C1, C2


def transform_point(point, scale, rotation, translation):
    """
    Transform a 3D point by scaling, rotating, and translating.

    Parameters:
        point: The original 3D point as a numpy array [x, y, z]
        scale: Scaling factors as a numpy array [sx, sy, sz]
        rotation: Rotation angles in degrees as a numpy array [rx, ry, rz]
        translation: Translation vector as a numpy array [tx, ty, tz]

    Returns:
        out: Transformed 3D point
    """
    # Scaling matrix
    S = np.diag(scale)

    # Rotation matrices for X, Y, Z axes
    rx, ry, rz = np.deg2rad(rotation)
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx), np.cos(rx)]])
    R_y = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]])
    R_z = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]])

    # Combined rotation matrix
    R = R_z @ R_y @ R_x

    # Scaling
    scaled_point = S @ point

    # Rotation
    rotated_point = R @ scaled_point

    return list(rotated_point + translation)


def make_box(W, L, H, translate, rotate=None, first_idx=0):
    if not rotate:
        rotate = [0, 0, 0]

    points = [
        # Back
        transform_point([0, 0, 0], [W, L, H], [0, 0, 0], translate),
        transform_point([1, 0, 0], [W, L, H], [0, 0, 0], translate),
        transform_point([1, 0, 1], [W, L, H], [0, 0, 0], translate),
        transform_point([0, 0, 1], [W, L, H], [0, 0, 0], translate),

        # Front
        transform_point([0, 1, 0], [W, L, H], [0, 0, 0], translate),
        transform_point([1, 1, 0], [W, L, H], [0, 0, 0], translate),
        transform_point([1, 1, 1], [W, L, H], [0, 0, 0], translate),
        transform_point([0, 1, 1], [W, L, H], [0, 0, 0], translate),
    ]

    triangles = [
        [first_idx+0, first_idx+1, first_idx+2],  # Back
        [first_idx+0, first_idx+2, first_idx+3],

        [first_idx+6, first_idx+5, first_idx+4],  # Front
        [first_idx+7, first_idx+6, first_idx+4],

        [first_idx+0, first_idx+3, first_idx+4],  # Left
        [first_idx+7, first_idx+4, first_idx+3],

        [first_idx+5, first_idx+2, first_idx+1],  # Right
        [first_idx+2, first_idx+5, first_idx+6],

        [first_idx+0, first_idx+5, first_idx+1],  # Bottom
        [first_idx+0, first_idx+4, first_idx+5],

        [first_idx+2, first_idx+6, first_idx+3],  # Top
        [first_idx+6, first_idx+7, first_idx+3],
    ]

    return points, triangles
