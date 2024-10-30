import jax.numpy as jnp


def signed_line_distance(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def dist_from_line(a, b, c):
    return jnp.abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def rmat_sc(s, c):
    return jnp.array([[c, -s], [s, c]])


def inv_rmat_sc(s, c):
    return jnp.array([[c, s], [-s, c]])
