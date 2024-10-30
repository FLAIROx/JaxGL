import os

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import imageio.v3 as iio

from jaxgl.renderer import clear_screen, make_renderer
from jaxgl.shaders import (
    fragment_shader_triangle,
    make_fragment_shader_texture,
)


def load_texture(filename, render_size):
    img = iio.imread(filename)
    jnp_img = jnp.array(img).astype(int)

    if jnp_img.shape[2] == 4:
        jnp_img = jnp_img.at[:, :, 3].set(jnp_img[:, :, 3] // 255)

    img = np.array(jnp_img, dtype=np.uint8)
    image = Image.fromarray(img)
    image = image.resize(render_size, resample=Image.NEAREST)
    jnp_img = jnp.array(image, dtype=jnp.int32)

    return jnp_img.transpose((1, 0, 2))


def main():
    # 512x512 pixels
    screen_size = (512, 512)

    # Clear a fresh screen with a black background
    clear_colour = jnp.array([255.0, 255.0, 255.0])
    pixels = clear_screen(screen_size, clear_colour)

    # Load texture
    img_size = (250, 145)
    texture = load_texture("jax_logo.png", img_size)

    # We statically create the fragment shader
    # Since our image has an alpha channel we specify this
    texture_shader = make_fragment_shader_texture(img_size, do_nearest_neighbour=False, alpha_channel=True)

    # We render the texture with a patch size equal to the image size
    # If we don't do this pixels will be resolved either by nearest neighbour or interpolation
    texture_renderer = make_renderer(screen_size, texture_shader, img_size)

    # Patch position (top left corner)
    pos = jnp.array([128, 128])

    # Render the triangle to the screen
    pixels = texture_renderer(pixels, pos, texture)

    # Display with matplotlib
    plt.imshow(pixels.astype(jnp.uint8).transpose(1, 0, 2))
    plt.show()


if __name__ == "__main__":
    main()
