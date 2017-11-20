import numpy as np


def semi_major(P, M1, M2):
    return np.cbrt(P * P * G * (M1 + M2) / (4.0 * np.pi * np.pi))


def pol2cart(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def sph2cart(r, theta, phi, key=None):
    if key == 'grid':
        theta = np.atleast_2d(theta).T

    return (r * np.sin(theta) * np.cos(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(theta))


def rotation_matrix_2D(theta):
    """Rotation matrix"""
    cos = np.cos(theta)
    sin = np.sin(theta)
    return np.array([[cos, -sin],
                     [sin, cos]])

def rotation_matrix_3D(axis, theta):
    """
    Generates the 3x3 rotation matrix for the rotation of theta radians about
    the axis.

    NOTE: axis must be normalised!!!
    """
    axis = np.asarray(axis)

    # 'Rodriguez parameters'
    a = float(np.cos(theta / 2))
    b, c, d = -axis * np.sin(theta / 2)

    # Rotation matrix
    R =  np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                   [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                   [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])
    return R

