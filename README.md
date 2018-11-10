# redner

redner is a differentiable Monte Carlo renderer that can differentiate the rendering output with respect to arbitrary scene parameters.
One of the major use of redner is for inverse rendering (hence the name redner).

For more detail of the renderer and the techniques it used for computing the derivatives, please
look at the paper:
"Differentiable Monte Carlo Ray Tracing through Edge Sampling", Tzu-Mao Li, Miika Aittala, Fredo Durand, Jaakko Lehtinen
[https://people.csail.mit.edu/tzumao/diffrt/]
Since the submission we have improved the renderer a bit. In particular we implemented a CUDA backend and accelerated
the continuous derivatives significantly by replacing automatic differentiation with hand derivatives.

redner is expected to use with [PyTorch](https://pytorch.org/), and can be used seamlessly with PyTorch operators.
While the documentation is work in progress, you can take a look at the [tests directory](tests) to have a basic sense.
redner inherits a subset of [Mitsuba](http://www.mitsuba-renderer.org) scene format,
see [tests/test_teapot_reflectance.py] and [tests/test_teapot_specular.py] for examples of loading a Mitsuba scene file.

redner depends on a few libraries/systems:
- [Python 3.6 or above](https://www.python.org) (required)
- [pybind11](https://github.com/pybind/pybind11) (required)
- [PyTorch 0.4.1 or 1.0](https://pytorch.org) (required)
- [OpenEXR](https://github.com/openexr/openexr) (required)
- [OpenEXR Python](https://github.com/jamesbowman/openexrpython) (required, just clone the repo and python setup.py install)
- [Embree](https://embree.github.io) (required)
- [Thrust](https://thrust.github.io) (required, included in a submodule)
- [miniz](https://github.com/richgel999/miniz) (already in this repository)
- [CUDA 10](https://developer.nvidia.com/cuda-downloads) (optional)
- [optix prime](https://developer.nvidia.com/optix) (optional, required when compiled with CUDA)
- [miniconda](https://conda.io/miniconda.html) (optional, but recommended)
- A few other python packages: numpy, scikit-image

I recommend using conda to setup the Python related dependencies, e.g.:
```
conda install -c conda-forge pybind11
conda install pytorch-nightly -c pytorch
(for some reason Mac OS users have to use conda-forge's Python, see [https://github.com/pybind/pybind11/issues/1579]).
conda install -c conda-forge python
```

redner uses [CMake](https://cmake.org) as its build system. You need CMake 3.12 or above to build redner.
The build procedure follows common CMake instructions.
Ideally,
```
mkdir build
cd build
cmake ..
make install -j 8
```
should build and install the project, but you may need to tell CMake where the dependencies are by defining
the following variables:
```
Python_INCLUDE_DIRS
Python_LIBRARIES
EMBREE_INCLUDE_DIRS
EMBREE_LIBRARY
OptiX_INCLUDE
CUDA_LIBRARIES
THRUST_INCLUDE_DIR
optix_prime_LIBRARY
CUDA_curand_LIBRARY
```
I suggest using ccmake or other interfaces of cmake to setup the variables.

redner is tested under MacOS with clang 6.0 and Ubuntu with gcc 7.0. Windows is not tested yet but should be
portable with moderate modification. In general any compiler with c++14 support should work.

The current development plan is to enhance the renderer. Following features will be added in the near future (not listed in any particular order):
- Environment map
- Non-black background
- Stratification of random number
- More BSDFs e.g. glass/GGX
- A properer secondary edge sampling strategy 
  (to make the renderer computation friendlier to GPU, we temporarily dropped the hierarchical edge sampling algorithm described in the paper, and instead used a resampling importance sample strategy.
   see [edge.cpp])
- Support for edge shared by more than two triangles
  (The code currently assumes every triangle edge is shared by at most two triangles.
   If your mesh doesn't satisfy this, you can preprocess it in other mesh processing software such as [MeshLab](http://www.meshlab.net))
- Source-to-source automatic differentiation
- Mipmapping
- Russian roulette
- Distribution effects: depth of field/motion blur
- Documentation

