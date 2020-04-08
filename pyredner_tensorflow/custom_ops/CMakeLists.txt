cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(rednerTFCustomOp)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include_directories(SYSTEM ${TensorFlow_INCLUDE_DIR})

# Compile two versions of the library
add_library(redner_tf_data_ptr_cxx11_abi SHARED data_ptr.cc)
set_target_properties(redner_tf_data_ptr_cxx11_abi PROPERTIES COMPILE_FLAGS -D_GLIBCXX_USE_CXX11_ABI=1)
set_target_properties(redner_tf_data_ptr_cxx11_abi PROPERTIES LINK_FLAGS -D_GLIBCXX_USE_CXX11_ABI=1)
if(APPLE)
    # .so instead of .dylib
    set_target_properties(redner_tf_data_ptr_cxx11_abi PROPERTIES SUFFIX .so)
endif()
target_link_libraries(redner_tf_data_ptr_cxx11_abi ${TensorFlow_LIBRARY})

add_library(redner_tf_data_ptr_no_cxx11_abi SHARED data_ptr.cc)
set_target_properties(redner_tf_data_ptr_no_cxx11_abi PROPERTIES COMPILE_FLAGS -D_GLIBCXX_USE_CXX11_ABI=0)
set_target_properties(redner_tf_data_ptr_no_cxx11_abi PROPERTIES LINK_FLAGS -D_GLIBCXX_USE_CXX11_ABI=0)
if(APPLE)
    # .so instead of .dylib
    set_target_properties(redner_tf_data_ptr_no_cxx11_abi PROPERTIES SUFFIX .so)
endif()
target_link_libraries(redner_tf_data_ptr_no_cxx11_abi ${TensorFlow_LIBRARY})
