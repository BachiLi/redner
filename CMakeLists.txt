cmake_minimum_required(VERSION 3.13)

project(redner VERSION 0.0.2 DESCRIPTION "Differentiable Ray Tracer")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

if(WIN32)
    find_package(PythonLibs 3.6 COMPONENTS Development REQUIRED)
    add_subdirectory(pybind11)
else()
    find_package(Python 3.6 COMPONENTS Development REQUIRED)
endif()

option(REDNER_CUDA "Build redner with GPU code path?" OFF)

# TODO: CUDA does currently not work with Win32 as
#       pybind11_add_module creates the redner target but the
#       call to cuda_add_library tries to create it again.
if(WIN32 AND REDNER_CUDA)
  set(REDNER_CUDA OFF)
  message(WARNING "GPU build currently not supported for Win32.")
endif()

if(REDNER_CUDA)
    message(STATUS "Build with CUDA support")
    find_package(CUDA 10 REQUIRED)
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
if(REDNER_CUDA)
    include_directories(${OptiX_INCLUDE})
    link_directories(${CUDA_LIBRARIES})
else()
    include_directories(${THRUST_INCLUDE_DIR})
endif()

if(NOT MSVC)
  # These compile definitions are not meaningful for MSVC
  add_compile_options(-Wall -g -O3 -fvisibility=hidden)
else()
  add_compile_options(/Wall /Zi)
  add_link_options(/DEBUG)
endif()

if(NOT REDNER_CUDA)
    add_compile_options("-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP")
endif()

set(SRCS aabb.h
         active_pixels.h
         area_light.h
         atomic.h
         atomic_msvc.h
         automatic_uv_map.h
         bsdf_sample.h
         buffer.h
         camera.h
         channels.h
         cuda_utils.h
         edge.h
         edge_tree.h
         envmap.h
         frame.h
         intersection.h
         line_clip.h
         load_serialized.h
         material.h
         matrix.h
         miniz.h
         parallel.h
         path_contribution.h
         pathtracer.h
         pcg_sampler.h
         primary_contribution.h
         primary_intersection.h
         ptr.h
         py_utils.h
         ray.h
         rebuild_topology.h
         redner.h
         sampler.h
         scene.h
         shape.h
         sobol_sampler.h
         test_utils.h
         texture.h
         transform.h
         vector.h
         xatlas/xatlas.h
         aabb.cpp
         active_pixels.cpp
         automatic_uv_map.cpp
         bsdf_sample.cpp
         camera.cpp
         channels.cpp
         edge.cpp
         edge_tree.cpp
         envmap.cpp
         load_serialized.cpp
         material.cpp
         miniz.c
         parallel.cpp
         path_contribution.cpp
         pathtracer.cpp
         pcg_sampler.cpp
         primary_contribution.cpp
         primary_intersection.cpp
         rebuild_topology.cpp
         redner.cpp
         scene.cpp
         shape.cpp
         sobol_sampler.cpp
         xatlas/xatlas.cpp)

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
if (WIN32)
    pybind11_add_module(redner SHARED ${SRCS})
endif()

if(REDNER_CUDA)
    add_compile_definitions(COMPILE_WITH_CUDA)
    set_source_files_properties(
        aabb.cpp
        active_pixels.cpp
        bsdf_sample.cpp
        camera.cpp
        channels.cpp
        edge.cpp
        edge_tree.cpp
        envmap.cpp
        material.cpp
        parallel.cpp
        path_contribution.cpp
        pathtracer.cpp
        pcg_sampler.cpp
        primary_contribution.cpp
        primary_intersection.cpp
        scene.cpp
        shape.cpp
        sobol_sampler.cpp
        PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14")

    cuda_add_library(redner MODULE ${SRCS})
    target_link_libraries(redner
        ${EMBREE_LIBRARY}
        ${optix_prime_LIBRARY}
        ${DYNAMIC_LOOKUP})
else()
    # No CUDA
    if (NOT WIN32)
        add_library(redner MODULE ${SRCS})
    
        # The "-undefined dynamic_lookup" is a hack for systems with
        # multiple Python installed. If we link a particular Python version
        # here, and we import it with a different Python version later.
        # likely a segmentation fault.
        # The solution for Linux/Mac OS machines, as mentioned in 
        # https://github.com/pybind/pybind11/blob/master/tools/pybind11Tools.cmake
        # is to not link against Python library at all and resolve the symbols
        # at compile time.
        target_link_libraries(redner
            ${EMBREE_LIBRARY}
            ${DYNAMIC_LOOKUP})
    else()
        target_link_libraries(redner
            PRIVATE
            ${EMBREE_LIBRARY})
    endif()
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

find_package(TensorFlow)
if(TensorFlow_FOUND)
    add_subdirectory(pyredner_tensorflow/custom_ops)
else()
    message(INFO " Building without TensorFlow support (not found)")
endif()
