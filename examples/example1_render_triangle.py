import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt

from jaxgl.renderer import clear_screen, make_renderer
from jaxgl.shaders import (
    fragment_shader_triangle,
)


def main():
    # 512x512 pixels
    screen_size = (512, 512)

    # Clear a fresh screen with a black background
    clear_colour = jnp.array([0.0, 0.0, 0.0])
    pixels = clear_screen(screen_size, clear_colour)

    # We render to a 256x256 'patch'
    patch_size = (256, 256)
    triangle_renderer = make_renderer(screen_size, fragment_shader_triangle, patch_size)

    # Patch position (top left corner)
    pos = jnp.array([128, 128])

    triangle_data = (
        # Vertices (note these must be anti-clockwise)
        jnp.array([[150, 200], [150, 300], [300, 150]]),
        # Colour
        jnp.array([255.0, 0.0, 0.0]),
    )

    # Render the triangle to the screen
    pixels = triangle_renderer(pixels, pos, triangle_data)

    # Display with matplotlib
    plt.imshow(pixels.astype(jnp.uint8))
    plt.show()


if __name__ == "__main__":
    main()
