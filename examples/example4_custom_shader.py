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

    # We make our own variation of the circle shader
    # We give both a central and edge colour and interpolate between these

    # Each fragment shader has access to
    # position: global position in screen space
    # current_frag: the current colour of the fragment (useful for transparency)
    # unit_position: the position inside the patch (scaled to between 0 and 1)
    # uniform: anything you want for your shader.  These are the same for every fragment.

    def my_shader(position, current_frag, unit_position, uniform):
        centre, radius, colour_centre, colour_outer = uniform

        dist = jnp.sqrt(jnp.square(position - centre).sum())
        colour_interp = dist / radius

        colour = colour_interp * colour_outer + (1 - colour_interp) * colour_centre

        return jax.lax.select(dist < radius, colour, current_frag)

    circle_renderer = make_renderer(screen_size, my_shader, patch_size)

    # Patch position (top left corner)
    pos = jnp.array([128, 128])

    # This is the uniform that is passed to the shader
    circle_data = (
        # Centre
        jnp.array([256.0, 256.0]),
        # Radius
        100.0,
        # Colour centre
        jnp.array([255.0, 0.0, 0.0]),
        # Colour outer
        jnp.array([0.0, 255.0, 0.0]),
    )

    # Render the triangle to the screen
    pixels = circle_renderer(pixels, pos, circle_data)

    # Display with matplotlib
    plt.imshow(pixels.astype(jnp.uint8))
    plt.show()


if __name__ == "__main__":
    main()
