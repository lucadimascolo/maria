from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


def unjitted_offsets_to_phi_theta(dX, cphi, ctheta, xp=jnp):
    """
    A fast and well-conditioned method to convert from local dx/dy coordinates to phi/theta coordinates.
    Pass xp=np to compute it in numpy (e.g. in float64; JAX runs in float32).
    """

    dx, dy = dX[..., 0], dX[..., 1]

    r = xp.sqrt(dx**2 + dy**2)  # distance from the center
    p = xp.arctan2(-dx, -dy)  # 0 at the bottom, increases CCW to pi at the top

    # if we're looking at the north pole, we have (lon, lat) = (p, pi/2 - r)
    # a projection looking from the east
    proj_from_east = (xp.sin(r) * xp.cos(p) + 1j * xp.cos(r)) * xp.exp(
        1j * (ctheta - xp.pi / 2),
    )

    # the latitude is computed with arctan2 and not arcsin, which is ill-conditioned near the poles (in float32,
    # arcsin is off by up to 71 arcseconds within half a degree of the zenith)
    proj_y = xp.sin(r) * xp.sin(p)
    return xp.stack(
        [
            xp.arctan2(proj_y, xp.real(proj_from_east)) + cphi,
            xp.arctan2(xp.imag(proj_from_east), xp.hypot(proj_y, xp.real(proj_from_east))),
        ],
        axis=-1,
    )


offsets_to_phi_theta = jax.jit(unjitted_offsets_to_phi_theta, static_argnames=["cphi", "ctheta"])


@partial(jax.jit, static_argnames=["cphi", "ctheta"])
def phi_theta_to_offsets(pt, cphi, ctheta):
    """
    A fast and well-conditioned to convert from phi/theta coordinates to local dx/dy coordinates.
    """

    phi, theta = pt[..., 0], pt[..., 1]

    dphi = phi - cphi
    proj_from_east = (jnp.cos(dphi) * jnp.cos(theta) + 1j * jnp.sin(theta)) * jnp.exp(
        1j * (jnp.pi / 2 - ctheta),
    )
    dz = jnp.sin(dphi) * jnp.cos(theta) + 1j * jnp.real(proj_from_east)
    r = jnp.abs(dz)
    dz *= jnp.arcsin(r) / jnp.where(r > 0, r, 1.0)

    # negative, because we're looking at the observer
    return jnp.stack([-jnp.real(dz), -jnp.imag(dz)], axis=-1)


def unjitted_phi_theta_to_xyz(phi, theta, xp=jnp):
    """
    Project some angular coordinates phi (longitude) and theta (latitude) onto the 3d unit sphere.
    """
    cos_theta = xp.cos(theta)
    return xp.stack(
        [xp.cos(phi) * cos_theta, xp.sin(phi) * cos_theta, xp.sin(theta)],
        axis=-1,
    )


phi_theta_to_xyz = jax.jit(unjitted_phi_theta_to_xyz)


@jax.jit
def xyz_to_phi_theta(xyz):
    """
    Find the longitude and latitude of a 3-vector.
    """
    return jnp.arctan2(xyz[..., 1], xyz[..., 0]) % (2 * jnp.pi), jnp.arctan2(
        xyz[..., 2],
        jnp.hypot(xyz[..., 0], xyz[..., 1]),
    )


def get_center_phi_theta(phi, theta, keep_dims=()):
    """ """

    xyz = phi_theta_to_xyz(np.atleast_1d(phi), np.atleast_1d(theta))

    axes = list(range(xyz.ndim - 1))

    for dim in keep_dims:
        axes.pop(dim)

    center_xyz = xyz.mean(axis=tuple(axes))
    center_xyz /= jnp.sqrt(jnp.sum(jnp.square(center_xyz)))[..., None]

    return xyz_to_phi_theta(center_xyz)
