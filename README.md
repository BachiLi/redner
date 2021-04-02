# redner-experimental: Unbiased differentiable rendering

This is an experimental branch of redner that supports two different techniques for differentiable rendering.

redner is a differentiable renderer that can take the derivatives of rendering outputs with respect to arbitrary scene parameters, that is, you can backpropagate from the image to your 3D scene. One of the major usages of redner is inverse rendering (hence the name redner) through gradient descent. What sets redner apart are: 1) it computes correct rendering gradients stochastically without any approximation by properly considering the discontinuities, and 2) it has a physically-based mode -- which means it can simulate photons and produce realistic lighting phenomena, such as shadow and global illumination, and it handles the derivatives of these features correctly. You can also use redner in a [fast deferred rendering mode](https://colab.research.google.com/github/BachiLi/redner/blob/master/tutorials/fast_local_shading.ipynb) for local shading: in this mode it still has correct gradient estimation and more elaborate material models compared to most differentiable renderers out there.

For more details on the rendering methods, what they can do, and the techniques they use for computing the derivatives, please
take a look at following papers:

## Edge-sampling
![](https://people.csail.mit.edu/tzumao/diffrt/teaser.jpg)
[Differentiable Monte Carlo Ray Tracing through Edge Sampling](https://people.csail.mit.edu/tzumao/diffrt/), Tzu-Mao Li, Miika Aittala, Fredo Durand, Jaakko Lehtinen.
See Tzu-Mao Li's [thesis](https://people.csail.mit.edu/tzumao/phdthesis/phdthesis.pdf) for even more details.

## Warped-area sampling (WAS)
![](https://www.saipraveenb.com/projects/was-2020/teaser.png)
[Unbiased Warped-area Sampling for Differentiable Rendering](https://www.saipraveenb.com/projects/was-2020/), Sai Praveen Bangaru, Tzu-Mao Li, Fredo Durand.


## Installation
This experimental branch must be compiled from source.
Clone this repository and run:

```
python setup.py install
```

It is generally advisable to use a new environment to avoid overwriting an existing version of redner.

## Documentation

A good starting point to learn how to use redner is to look at the [wiki](https://github.com/BachiLi/redner/wiki). The API documetation is [here](https://redner.readthedocs.io/en/latest/).
You can also take a look at the tests directories ([PyTorch](tests) and [TensorFlow](tests_tensorflow)) to have some ideas.

This branch differs slightly from the master, although the same API is supported for backwards compatibility.
To use the new renderer use the `serialize_scene_class(scene, integrator)` method instead of `serialize_scene(scene, *args)` to specify your choice of integrator (and its parameters).

The new tests `tests/test_single_triangle_was.py` and `tests/test_shadow_blocker_was.py` demonstrate this new feature.

## News

04/01/2021 - Now supports both differentiable rendering methods. Swap between `integrator=pyredner.integrators.EdgeSamplingIntegrator` `integrator=pyredner.integrators.WarpFieldIntegrator` to quickly try the different methods.

## Dependencies

redner depends on a few libraries/systems, which are all included in the repository:
- [Python 3.6 or above](https://www.python.org)
- [pybind11](https://github.com/pybind/pybind11)
- [PyTorch 1.0 or above](https://pytorch.org) (optional, required if TensorFlow is not installed)
- [Tensorflow 2.0](https://www.tensorflow.org/) (optional, required if PyTorch is not installed)
- [Embree](https://embree.github.io)
- [CUDA 10](https://developer.nvidia.com/cuda-downloads) (optional, need GPU at Kepler class or newer)
- [optix prime V6.5 or older](https://developer.nvidia.com/optix) (optional, required when compiled with CUDA)
- [Thrust](https://thrust.github.io)
- [miniz](https://github.com/richgel999/miniz)
- [xatlas](https://github.com/jpcy/xatlas)
- A few other python packages: numpy, scikit-image, and imageio

## Roadmap

The current WAS implementation is restricted to a smaller set of features compared to the master branch, but we intend to expand its capabilities to align with the master branch and eventually merge.

Current roadmap before merging:
- All reconstruction filters. WAS is currently restricted to using the Gaussian filter (the default for this branch).
- GPU support. (This implementation is currently CPU-only since it uses some STL classes)
- Tensorflow support. (This branch will crash with tensorflow since it does not contain the necessary hooks)
- Windows support. (Not tested)

## Citation
Please cite one or both of these papers, if you use this repository.

### Edge-sampling
```
@article{Li:2018:DMC,
    title = {Differentiable Monte Carlo Ray Tracing through Edge Sampling},
    author = {Li, Tzu-Mao and Aittala, Miika and Durand, Fr{\'e}do and Lehtinen, Jaakko},
    journal = {ACM Trans. Graph. (Proc. SIGGRAPH Asia)},
    volume = {37},
    number = {6},
    pages = {222:1--222:11},
    year = {2018}
}
```

### Warped-area sampling
```
@article{bangaru2020warpedsampling,
  title = {Unbiased Warped-Area Sampling for Differentiable Rendering},
  author = {Bangaru, Sai and Li, Tzu-Mao and Durand, Fr{\'e}do},
  journal = {ACM Trans. Graph.},
  volume = {39},
  number = {6}, 
  pages = {245:1--245:18},
  year = {2020},
  publisher = {ACM},
}
```

If you have any questions/comments/bug reports, feel free to open a github issue or e-mail to the authors Tzu-Mao Li (tzumao@mit.edu) and Sai Bangaru (sbangaru@mit.edu)
