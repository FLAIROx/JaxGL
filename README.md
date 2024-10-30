# JaxGL
<p align="center">
        <a href= "https://pypi.org/project/jaxgl/">
        <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue" /></a>
        <a href= "https://pypi.org/project/jaxgl/">
        <img src="https://img.shields.io/badge/pypi-1.0.0-green" /></a>
        <a href= "https://pepy.tech/project/jaxgl">
        <img src="https://static.pepy.tech/badge/jaxgl" /></a>
       <a href= "https://github.com/MichaelTMatthews/Craftax/blob/main/LICENSE">
        <img src="https://img.shields.io/badge/License-MIT-yellow" /></a>
       <a href= "https://github.com/psf/black">
        <img src="https://img.shields.io/badge/code%20style-black-000000.svg" /></a>
</p>

JaxGL is a simple and flexible graphics library written entirely in <a href="https://github.com/google/jax">JAX</a>.  JaxGL was created by [Michael Matthews](https://github.com/MichaelTMatthews) and [Michael Beukman](https://github.com/Michael-Beukman) for the [Kinetix](https://github.com/FLAIROx/Kinetix) project.

# Basic Usage
```python
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
```

This produces the following image:

<p align="center">
 <img width="20%" src="images/simple_render.png" />
</p>

# Custom Shaders
Arbitrary rendering effects can be achieved by writing your own shaders.
```python
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
```

<p align="center">
 <img width="20%" src="images/custom_shader.png" />
</p>

# In Kinetix
JaxGL is used for rendering in [Kinetix](https://github.com/FLAIROx/Kinetix). Shown below is an example robotics grasping task.
<p align="center">
 <img width="40%" src="images/kinetix.png" />
</p>

# Installation
To use JaxGL in your work you can install via PyPi:
```commandline
pip install jaxgl
```

If you want to extend JaxGL you can install as follows:
```commandline
git clone https://github.com/FLAIROx/JaxGL
cd JaxGL
pip install -e ".[dev]"
pre-commit install
```

# See Also
- [JAX Renderer](https://github.com/JoeyTeng/jaxrenderer) A more complete JAX renderer more suitable for 3D rendering.
- [Jax2D](https://github.com/MichaelTMatthews/Jax2D) 2D physics engine in JAX.
- [Kinetix](https://github.com/FLAIROx/Kinetix) physics-based reinforcement learning in JAX.