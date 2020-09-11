cmake_minimum_required(VERSION 3.13)

project(redner VERSION 0.0.2 DESCRIPTION "Differentiable Ray Tracer")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

if(WIN32)
    find_package(Python 3.6 COMPONENTS Development REQUIRED)
else()
    find_package(Python 3.6 COMPONENTS Development REQUIRED)
endif()
add_subdirectory(pybind11)

option(REDNER_CUDA "Build redner with GPU code path?" OFF)


if(REDNER_CUDA)
    message(STATUS "Build with CUDA support")
    find_package(CUDA 10 REQUIRED)
    set(CMAKE_CUDA_STANDARD 11)
    if(NOT WIN32)
        # Hack: for some reason the line above doesn't work on some Linux systems.
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
    endif()
    # We don't set REQUIRED here as the main OptiX lib will not be found
    # (only OptiX Prime is required)
    find_package(OptiX)
else()
    message(STATUS "Build without CUDA support")
    find_package(Thrust REQUIRED)
endif()

find_package(Embree REQUIRED)

include_directories(${PYTHON_INCLUDE_PATH})
include_directories(${EMBREE_INCLUDE_PATH})
include_directories(pybind11/include)
if(REDNER_CUDA)
    include_directories(${OptiX_INCLUDE})
    link_directories(${CUDA_LIBRARIES})
else()
    include_directories(${THRUST_INCLUDE_DIR})
endif()

if(NOT MSVC)
  # These compile definitions are not meaningful for MSVC
  add_compile_options(-Wall -g -O3 -fvisibility=hidden -Wno-unknown-pragmas)
else()
  add_compile_options(/Wall /Zi)
  add_link_options(/DEBUG)
endif()

if(NOT REDNER_CUDA)
    add_compile_options("-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP")
endif()

set(SRCS src/aabb.h
         src/active_pixels.h
         src/area_light.h
         src/atomic.h
         src/automatic_uv_map.h
         src/bsdf_sample.h
         src/buffer.h
         src/camera.h
         src/camera_distortion.h
         src/channels.h
         src/cuda_utils.h
         src/edge.h
         src/edge_tree.h
         src/envmap.h
         src/frame.h
         src/intersection.h
         src/line_clip.h
         src/load_serialized.h
         src/ltc.inc
         src/material.h
         src/matrix.h
         src/miniz.h
         src/parallel.h
         src/path_contribution.h
         src/pathtracer.h
         src/pcg_sampler.h
         src/primary_contribution.h
         src/primary_intersection.h
         src/ptr.h
         src/ray.h
         src/rebuild_topology.h
         src/redner.h
         src/sampler.h
         src/scene.h
         src/shape.h
         src/sobol.inc
         src/sobol_sampler.h
         src/test_utils.h
         src/texture.h
         src/transform.h
         src/vector.h
         xatlas/xatlas.h
         src/aabb.cpp
         src/active_pixels.cpp
         src/atomic.cpp
         src/automatic_uv_map.cpp
         src/bsdf_sample.cpp
         src/camera.cpp
         src/camera_distortion.cpp
         src/channels.cpp
         src/edge.cpp
         src/edge_tree.cpp
         src/load_serialized.cpp
         src/material.cpp
         src/miniz.c
         src/parallel.cpp
         src/path_contribution.cpp
         src/pathtracer.cpp
         src/pcg_sampler.cpp
         src/primary_contribution.cpp
         src/primary_intersection.cpp
         src/rebuild_topology.cpp
         src/redner.cpp
         src/scene.cpp
         src/shape.cpp
         src/sobol_sampler.cpp
         xatlas/xatlas.cpp)

if(REDNER_CUDA)
    add_compile_definitions(COMPILE_WITH_CUDA)
    set_source_files_properties(
        src/aabb.cpp
        src/active_pixels.cpp
        src/bsdf_sample.cpp
        src/camera.cpp
        src/channels.cpp
        src/edge.cpp
        src/edge_tree.cpp
        src/material.cpp
        src/parallel.cpp
        src/path_contribution.cpp
        src/pathtracer.cpp
        src/pcg_sampler.cpp
        src/primary_contribution.cpp
        src/primary_intersection.cpp
        src/scene.cpp
        src/shape.cpp
        src/sobol_sampler.cpp
        PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)

    cuda_add_library(redner MODULE ${SRCS})
    target_link_libraries(redner ${optix_prime_LIBRARY})
else()
    add_library(redner MODULE ${SRCS})
endif()

if(APPLE)
    # The "-undefined dynamic_lookup" is a hack for systems with
    # multiple Python installed. If we link a particular Python version
    # here, and we import it with a different Python version later.
    # likely a segmentation fault.
    # The solution for Linux Mac OS machines, as mentioned in 
    # https://github.com/pybind/pybind11/blob/master/tools/pybind11Tools.cmake
    # is to not link against Python library at all and resolve the symbols
    # at compile time.
    set(DYNAMIC_LOOKUP "-undefined dynamic_lookup")
endif()

target_link_libraries(redner ${EMBREE_LIBRARY} ${DYNAMIC_LOOKUP})

if(WIN32)
    # See: https://pybind11.readthedocs.io/en/master/compiling.html#advanced-interface-library-target
    target_link_libraries(redner pybind11::module)
    set_target_properties(redner PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                            SUFFIX "${PYTHON_MODULE_EXTENSION}")
endif()

set_target_properties(redner PROPERTIES SKIP_BUILD_RPATH FALSE)
set_target_properties(redner PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
if(UNIX AND NOT APPLE)
    set_target_properties(redner PROPERTIES INSTALL_RPATH "$ORIGIN")
elseif(APPLE)
    set_target_properties(redner PROPERTIES INSTALL_RPATH "@loader_path")
endif()

set_property(TARGET redner PROPERTY CXX_STANDARD 11)
set_target_properties(redner PROPERTIES PREFIX "")
# Still enable assertion in release mode
string( REPLACE "/DNDEBUG" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
string( REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
string( REPLACE "/DNDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
string( REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
string( REPLACE "/DNDEBUG" "" CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
string( REPLACE "-DNDEBUG" "" CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
string( REPLACE "/DNDEBUG" "" CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
string( REPLACE "-DNDEBUG" "" CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")

if(MSVC)
    target_compile_options(redner PRIVATE /W3 /wd4244 /wd4146 /wd4244 /wd4305 /wd4334 /wd4996)
endif()

if(NOT WIN32)
    find_package(TensorFlow)
    if(TensorFlow_FOUND)
        add_subdirectory(pyredner_tensorflow/custom_ops)
    else()
        message(INFO " Building without TensorFlow support (not found)")
    endif()
endif()
