import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt

from jaxgl.renderer import clear_screen, make_renderer
from jaxgl.shaders import (
    fragment_shader_triangle,
)


def main():
    screen_size = (512, 512)

    clear_colour = jnp.array([0.0, 0.0, 0.0])
    pixels = clear_screen(screen_size, clear_colour)

    patch_size = (256, 256)

    # By setting batch=true we can now pass in multiple triangles at once
    triangle_batch_renderer = make_renderer(screen_size, fragment_shader_triangle, patch_size, batched=True)

    # Sample random positions
    N = 10
    rng = jax.random.PRNGKey(0)
    rng, _rng = jax.random.split(rng)
    triangle_positions = jax.random.uniform(_rng, shape=(N, 2)) * jnp.array([screen_size[0], screen_size[1]]) * 0.8
    triangle_positions = triangle_positions.astype(jnp.int32)

    # Use a fixed triangle and translate it by the random positions
    triangle_vertices = jnp.array([[0, 50], [0, 100], [100, 0]])
    triangle_vertices = jnp.repeat(triangle_vertices[None, ...], repeats=N, axis=0) + triangle_positions[:, None, :]

    # Sample random colours
    rng, _rng = jax.random.split(rng)
    triangle_colours = jax.random.uniform(_rng, shape=(N, 3)) * 255

    # Batched triangle data
    triangle_data = (
        triangle_vertices,
        triangle_colours,
    )

    # Render the triangles to the screen
    pixels = triangle_batch_renderer(pixels, triangle_positions, triangle_data)

    plt.imshow(pixels.astype(jnp.uint8))
    plt.show()


if __name__ == "__main__":
    main()
