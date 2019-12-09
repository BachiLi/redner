redner pytorch module documentation
==========================================

For tutorials see https://github.com/BachiLi/redner/wiki

Camera
==========================================
.. autoclass:: pyredner.Camera
    :members:

Material
==========================================
.. autoclass:: pyredner.Material
    :members:

EnvironmentMap
==========================================
.. autoclass:: pyredner.EnvironmentMap
    :members:

Texture
==========================================
.. autoclass:: pyredner.Texture
    :members:

Object
==========================================
.. autoclass:: pyredner.Object
    :members:

Scene
==========================================
.. autoclass:: pyredner.Scene
    :members:

DeferredLight
==========================================
.. autoclass:: pyredner.DeferredLight
    :members:

AmbientLight
==========================================
.. autoclass:: pyredner.AmbientLight
    :members:

PointLight
==========================================
.. autoclass:: pyredner.PointLight
    :members:

DirectionalLight
==========================================
.. autoclass:: pyredner.DirectionalLight
    :members:

SpotLight
==========================================
.. autoclass:: pyredner.SpotLight
    :members:

Rendering
==========================================
.. autofunction:: pyredner.render_albedo

.. autofunction:: pyredner.render_deferred

.. autofunction:: pyredner.render_g_buffer

.. autofunction:: pyredner.render_pathtracing

automatic_camera_placement
==========================================
.. autofunction:: pyredner.automatic_camera_placement

device
==========================================
.. autofunction:: pyredner.set_device

.. autofunction:: pyredner.get_device

image writing/reading
==========================================
.. autofunction:: pyredner.imwrite

.. autofunction:: pyredner.imread

scene loading
==========================================
.. autofunction:: pyredner.load_mitsuba

.. autofunction:: pyredner.load_obj

.. autofunction:: pyredner.save_obj

gen_rotate_matrix
==========================================
.. autofunction:: pyredner.gen_rotate_matrix

generate_sphere
==========================================
.. autofunction:: pyredner.generate_sphere

generate_quad_light
==========================================
.. autofunction:: pyredner.generate_quad_light

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
