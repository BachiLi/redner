cmake_minimum_required(VERSION 3.12)

project(redner VERSION 0.0.1 DESCRIPTION "Differentiable Path Tracer")

include(CheckLanguage)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

find_package(Python 3.6 COMPONENTS Development REQUIRED)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    find_package(CUDA REQUIRED)
    find_package(OptiX REQUIRED)
else()
    find_package(Thrust REQUIRED)
endif()
find_package(Embree REQUIRED)

include_directories(${Python_INCLUDE_DIRS})
include_directories(${EMBREE_INCLUDE_PATH})
if(CMAKE_CUDA_COMPILER)
    include_directories(${OptiX_INCLUDE})
    link_directories(${CUDA_LIBRARIES})
else()
    include_directories(${THRUST_INCLUDE_DIR})
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -Wall -g -O3 -fvisibility=hidden")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -g -O3 -fvisibility=hidden")
if(NOT CMAKE_CUDA_COMPILER)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP")
endif()

set(SRCS active_pixels.h
         buffer.h
         camera.h
         cuda_utils.h
         redner.h
         edge.h
         frame.h
         intersection.h
         light.h
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
         sampler.h
         scene.h
         shape.h
         test_utils.h
         transform.h
         vector.h
         active_pixels.cpp
         camera.cpp
         redner.cpp
         edge.cpp
         light.cpp
         load_serialized.cpp
         material.cpp
         miniz.c
         parallel.cpp
         pathtracer.cpp
         sampler.cpp
         scene.cpp
         shape.cpp)

if(CMAKE_CUDA_COMPILER)
    add_compile_definitions(COMPILE_WITH_CUDA)
    set_source_files_properties(
        active_pixels.cpp
        camera.cpp
        edge.cpp
        light.cpp
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
        ${Python_LIBRARIES}
        ${EMBREE_LIBRARY}
        ${optix_prime_LIBRARY}
        ${CUDA_curand_LIBRARY})
else()
    add_library(redner MODULE ${SRCS})
    target_link_libraries(redner
        ${Python_LIBRARIES}
        ${EMBREE_LIBRARY})
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
install(CODE "execute_process(COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/setup.py install
                              WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})")
