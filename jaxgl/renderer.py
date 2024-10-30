from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnums=(0,))
def clear_screen(screen_size, colour):
    return jnp.ones((screen_size[0], screen_size[1], 3)) * colour[None, None, :]


def make_renderer(screen_size, shader, patch_size, batched=False):
    pixel_xs = jnp.arange(screen_size[0])
    pixel_ys = jnp.arange(screen_size[1])

    pixel_idxs = jnp.concatenate(
        [
            jnp.repeat(pixel_xs[:, None], screen_size[1], axis=1)[:, :, None],
            jnp.repeat(pixel_ys[None, :], screen_size[0], axis=0)[:, :, None],
        ],
        axis=2,
    )

    @jax.jit
    def render_and_apply_patches(pixels, positions, uniforms):
        def _render_patch(pixels, position_and_uniform):
            position, uniform = position_and_uniform

            idx_position = jnp.array([position[0], position[1], 0])
            idx_size = (patch_size[0], patch_size[1], 2)
            pixel_size = (patch_size[0], patch_size[1], 3)

            idxs_patch = jax.lax.dynamic_slice(pixel_idxs, idx_position, idx_size)
            pixels_patch = jax.lax.dynamic_slice(pixels, idx_position, pixel_size)

            unit_position_patch = jnp.concatenate(
                [
                    (
                        (idxs_patch[:, :, 0] - idxs_patch[:, :, 0].min())
                        / (idxs_patch[:, :, 0].max() - idxs_patch[:, :, 0].min())
                    )[:, :, None],
                    (
                        (idxs_patch[:, :, 1] - idxs_patch[:, :, 1].min())
                        / (idxs_patch[:, :, 1].max() - idxs_patch[:, :, 1].min())
                    )[:, :, None],
                ],
                axis=2,
            )

            pixels_patch = jax.vmap(jax.vmap(shader, in_axes=(0, 0, 0, None)), in_axes=(0, 0, 0, None))(
                idxs_patch, pixels_patch, unit_position_patch, uniform
            )

            pixels = jax.lax.dynamic_update_slice(pixels, pixels_patch, idx_position)

            return pixels, None

        if batched:
            pixels, _ = jax.lax.scan(_render_patch, pixels, (positions, uniforms))
        else:
            pixels, _ = _render_patch(pixels, (positions, uniforms))

        return pixels

    return render_and_apply_patches
