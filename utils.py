"""Utility helpers for stent FEA simulation."""

import numpy as np


def rotation_matrix_to_align(a, b):
    """Return 3x3 rotation matrix that rotates unit vector a onto unit vector b."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if abs(c + 1.0) < 1e-10:
        # 180-degree rotation — pick any perpendicular axis
        perp = np.array([1, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1, 0])
        v = np.cross(a, perp)
        v /= np.linalg.norm(v)
        return 2 * np.outer(v, v) - np.eye(3)
    s = np.linalg.norm(v)
    if s < 1e-12:
        return np.eye(3)
    kmat = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))


def cylindrical_to_cartesian(r, theta, z):
    return np.array([r * np.cos(theta), r * np.sin(theta), z])


def cartesian_to_cylindrical(x, y, z):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta, z


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def lerp(a, b, t):
    """Linear interpolation between a and b."""
    return a + t * (b - a)


def smooth_step(t):
    """Smooth step function (3t^2 - 2t^3) for easing transitions."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def bounding_box(points):
    """Return (min_xyz, max_xyz) of a point array (N,3)."""
    return points.min(axis=0), points.max(axis=0)


def centroid(points):
    return points.mean(axis=0)
