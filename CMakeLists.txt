cmake_minimum_required(VERSION 3.12)

project(redner VERSION 0.0.1 DESCRIPTION "Differentiable Path Tracer")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

if(WIN32)
    find_package(PythonLibs 3.6 COMPONENTS Development REQUIRED)
    add_subdirectory(pybind11)
else()
    find_package(Python 3.6 COMPONENTS Development REQUIRED)
endif()
find_package(CUDA 10)
if(CUDA_FOUND)
    find_package(OptiX REQUIRED)
else()
    find_package(Thrust REQUIRED)
endif()
find_package(Embree REQUIRED)
if(NOT EMBREE_FOUND)
    add_subdirectory(embree)
endif()

include_directories(${Python_INCLUDE_DIRS})
include_directories(${EMBREE_INCLUDE_PATH})
if(CUDA_FOUND)
    include_directories(${OptiX_INCLUDE})
    link_directories(${CUDA_LIBRARIES})
else()
    include_directories(${THRUST_INCLUDE_DIR})
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -Wall -g -O3 -fvisibility=hidden")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -g -O3 -fvisibility=hidden")
if(NOT CUDA_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP")
endif()

set(SRCS aabb.h
         active_pixels.h
         area_light.h
         atomic.h
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
         pathtracer.h
         ptr.h
         py_utils.h
         ray.h
         redner.h
         sampler.h
         scene.h
         shape.h
         test_utils.h
         texture.h
         transform.h
         vector.h
         aabb.cpp
         active_pixels.cpp
         area_light.cpp
         camera.cpp
         channels.cpp
         edge.cpp
         edge_tree.cpp
         envmap.cpp
         load_serialized.cpp
         material.cpp
         miniz.c
         parallel.cpp
         pathtracer.cpp
         redner.cpp
         sampler.cpp
         scene.cpp
         shape.cpp)

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
pybind11_add_module(redner SHARED  ${SRCS} )
endif()

if(CUDA_FOUND)
    add_compile_definitions(COMPILE_WITH_CUDA)
    set_source_files_properties(
        aabb.cpp
        active_pixels.cpp
        area_light.cpp
        camera.cpp
        channels.cpp
        edge.cpp
        edge_tree.cpp
        envmap.cpp
        material.cpp
        parallel.cpp
        pathtracer.cpp
        sampler.cpp
        scene.cpp
        shape.cpp
        PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14")

    cuda_add_library(redner MODULE ${SRCS})
    target_link_libraries(redner
        ${EMBREE_LIBRARY}
        ${optix_prime_LIBRARY}
        ${CUDA_curand_LIBRARY}
        ${DYNAMIC_LOOKUP})
else()
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
        ${EMBREE_LIBRARY} 
        )
endif()

endif()

set_target_properties(redner PROPERTIES PREFIX "")

execute_process(
    COMMAND "python" -c "if True:
        from distutils import sysconfig as sc
        print(sc.get_python_lib())"
    OUTPUT_VARIABLE PYTHON_SITE
    OUTPUT_STRIP_TRAILING_WHITESPACE)

install(
    TARGETS redner
    DESTINATION ${PYTHON_SITE})

# OpenEXR
find_package(OpenEXR REQUIRED)
if(NOT OPENEXR_FOUND)
    add_subdirectory(openexr)
endif()
# Install openexrpython
if(APPLE)
    set(ENV{MACOSX_DEPLOYMENT_TARGET} 10.9)
endif()
install(CODE "execute_process(COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/openexrpython/setup.py install
                              WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/openexrpython)")
install(CODE "execute_process(COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/setup.py install
                              WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})")
