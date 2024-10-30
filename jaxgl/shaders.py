from functools import partial

import jax
import jax.numpy as jnp

from jaxgl.maths import signed_line_distance


def fragment_shader_red(position, current_frag, unit_position, uniform):
    return jnp.array([255.0, 0.0, 0.0])


def fragment_shader_red_translucent(position, current_frag, unit_position, uniform):
    return jnp.array([128.0, 0.0, 0.0]) + 0.5 * current_frag


def fragment_shader_gradient(position, current_frag, unit_position, uniform):
    return jnp.array([unit_position[0], unit_position[1], 0.0]) * 255.0


def fragment_shader_colour(position, current_frag, unit_position, colour):
    return colour


def fragment_shader_circle(position, current_frag, unit_position, uniform):
    centre, radius, colour = uniform

    dist_sq = jnp.square(position - centre).sum()
    return jax.lax.select(dist_sq < jnp.square(radius), colour, current_frag)


def nearest_neighbour(texture, tex_coord):
    tex_coord = jnp.round(tex_coord).astype(jnp.int32)
    return texture[tex_coord[0], tex_coord[1]]


def linear_interp(texture, tex_coord):
    tex_coord_min = tex_coord.astype(jnp.int32)
    tex_coord_max = (1 + tex_coord).astype(jnp.int32)

    tex_frag_00 = texture[tex_coord_min[0], tex_coord_min[1]]
    tex_frag_01 = texture[tex_coord_min[0], tex_coord_max[1]]
    tex_frag_10 = texture[tex_coord_max[0], tex_coord_min[1]]
    tex_frag_11 = texture[tex_coord_max[0], tex_coord_max[1]]

    interp = tex_coord - tex_coord_min

    tex_frag_0m = interp[1] * tex_frag_01 + (1.0 - interp[1]) * tex_frag_00
    tex_frag_1m = interp[1] * tex_frag_11 + (1.0 - interp[1]) * tex_frag_10

    return interp[0] * tex_frag_1m + (1.0 - interp[0]) * tex_frag_0m


def make_fragment_shader_texture(tex_size, do_nearest_neighbour=False, alpha_channel=False):
    def fragment_shader_texture(position, current_frag, unit_position, texture):
        assert texture.shape[0] == tex_size[0]
        assert texture.shape[1] == tex_size[1]
        if alpha_channel:
            assert texture.shape[2] == 4
        else:
            assert texture.shape[2] == 3

        tex_coord = (
            jnp.array(
                [
                    tex_size[0] * unit_position[0],
                    tex_size[1] * unit_position[1],
                ]
            )
            - 0.5
        )

        if do_nearest_neighbour:
            tex_frag = nearest_neighbour(texture, tex_coord)
        else:
            tex_frag = linear_interp(texture, tex_coord)
        # Transparency wrt. current fragment
        if alpha_channel:
            tex_frag = (tex_frag[3] * tex_frag[:3]) + ((1.0 - tex_frag[3]) * current_frag)

        return tex_frag

    return fragment_shader_texture


def make_fragment_shader_convex_ngon(n):
    def fragment_shader_convex_ngon(position, current_frag, unit_position, uniform):
        vertices, colour = uniform
        assert vertices.shape == (n, 2)

        next_vertices = jnp.concatenate([vertices[1:], vertices[0][None, :]], axis=0)

        inside = True
        for i in range(n):
            side = signed_line_distance(position, vertices[i], next_vertices[i])
            inside &= side <= 0

        return jax.lax.select(inside, colour, current_frag)

    return fragment_shader_convex_ngon


fragment_shader_triangle = make_fragment_shader_convex_ngon(3)
fragment_shader_quad = make_fragment_shader_convex_ngon(4)


def make_fragment_shader_convex_ngon_with_edges(n, edge_thickness=2):
    def fragment_shader_convex_ngon(position, current_frag, unit_position, uniform):
        vertices, colour, edge_colour, mask = uniform
        assert vertices.shape == (n, 2)

        next_vertices = jnp.concatenate([vertices[1:], vertices[0][None, :]], axis=0)

        inside = True
        on_edge = False
        for i in range(n):
            side = signed_line_distance(position, vertices[i], next_vertices[i]) / jnp.linalg.norm(
                vertices[i] - next_vertices[i]
            )
            inside &= side <= 0
            on_edge |= (side > -edge_thickness) & (side <= 0)

        on_edge &= inside

        return jax.lax.select(inside & mask, jax.lax.select(on_edge, edge_colour, colour), current_frag)

    return fragment_shader_convex_ngon


fragment_shader_edged_triangle = make_fragment_shader_convex_ngon_with_edges(3)
fragment_shader_edged_quad = make_fragment_shader_convex_ngon_with_edges(4)


def make_fragment_shader_quad_textured(tex_size, do_nearest_neighbour=False):
    def fragment_shader_quad_textured(position, current_frag, unit_position, uniform):
        vertices, tex_coords, texture, mask = uniform

        assert vertices.shape == (4, 2)
        assert tex_coords.shape == (4, 2)

        # Barycentric 0, 1, 2
        b0_1 = (vertices[1, 1] - vertices[2, 1]) * (position[0] - vertices[2, 0]) + (
            vertices[2, 0] - vertices[1, 0]
        ) * (position[1] - vertices[2, 1])
        b0_1 /= (vertices[1, 1] - vertices[2, 1]) * (vertices[0, 0] - vertices[2, 0]) + (
            vertices[2, 0] - vertices[1, 0]
        ) * (vertices[0, 1] - vertices[2, 1])

        b0_2 = (vertices[2, 1] - vertices[0, 1]) * (position[0] - vertices[2, 0]) + (
            vertices[0, 0] - vertices[2, 0]
        ) * (position[1] - vertices[2, 1])
        b0_2 /= (vertices[1, 1] - vertices[2, 1]) * (vertices[0, 0] - vertices[2, 0]) + (
            vertices[2, 0] - vertices[1, 0]
        ) * (vertices[0, 1] - vertices[2, 1])

        b0_3 = 1 - b0_1 - b0_2

        # Barycentric 0, 2, 3
        b1_1 = (vertices[2, 1] - vertices[3, 1]) * (position[0] - vertices[3, 0]) + (
            vertices[3, 0] - vertices[2, 0]
        ) * (position[1] - vertices[3, 1])
        b1_1 /= (vertices[2, 1] - vertices[3, 1]) * (vertices[0, 0] - vertices[3, 0]) + (
            vertices[3, 0] - vertices[2, 0]
        ) * (vertices[0, 1] - vertices[3, 1])

        b1_2 = (vertices[3, 1] - vertices[0, 1]) * (position[0] - vertices[3, 0]) + (
            vertices[0, 0] - vertices[3, 0]
        ) * (position[1] - vertices[3, 1])
        b1_2 /= (vertices[2, 1] - vertices[3, 1]) * (vertices[0, 0] - vertices[3, 0]) + (
            vertices[3, 0] - vertices[2, 0]
        ) * (vertices[0, 1] - vertices[3, 1])

        b1_3 = 1 - b1_1 - b1_2

        t1 = tex_coords[0] * b0_1 + tex_coords[1] * b0_2 + tex_coords[2] * b0_3
        t2 = tex_coords[0] * b1_1 + tex_coords[2] * b1_2 + tex_coords[3] * b1_3

        t1 = t1 * jnp.array([tex_size[0], tex_size[1]])
        t2 = t2 * jnp.array([tex_size[0], tex_size[1]])

        if do_nearest_neighbour:
            tex_frag1 = nearest_neighbour(texture, t1)
            tex_frag2 = nearest_neighbour(texture, t2)
        else:
            tex_frag1 = linear_interp(texture, t1)
            tex_frag2 = linear_interp(texture, t2)

        inside1 = (b0_1 >= 0) & (b0_2 >= 0) & (b0_3 >= 0)
        inside2 = (b1_1 >= 0) & (b1_2 >= 0) & (b1_3 >= 0)

        return jax.lax.select((inside1 | inside2) & mask, jax.lax.select(inside1, tex_frag1, tex_frag2), current_frag)

    return fragment_shader_quad_textured
