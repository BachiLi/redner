redner: Differentiable rendering without approximation.
=======================================================

redner is a differentiable renderer that can take the derivatives of rendering output with respect to arbitrary scene parameters, that is, you can backpropagate from the image to your 3D scene. One of the major usages of redner is inverse rendering (hence the name redner) through gradient descent. What sets redner apart are: 1) it computes correct rendering gradients stochastically without any approximation and 2) it has a physically-based mode -- which means it can simulate photons and produce realistic lighting phenomena, such as shadow and global illumination, and it handles the derivatives of these features correctly. You can also use redner in a fast deferred rendering mode for local shading: in this mode it still has correct gradient estimation and more elaborate material models compared to most differentiable renderers out there.

| For tutorials see https://github.com/BachiLi/redner/wiki
| For the theory of redner see https://people.csail.mit.edu/tzumao/diffrt/
  and Tzu-Mao Li's `PhD thesis <https://people.csail.mit.edu/tzumao/phdthesis/phdthesis.pdf>`_.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   pyredner
   pyredner_tensorflow

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
