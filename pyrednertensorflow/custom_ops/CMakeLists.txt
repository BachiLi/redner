cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(rednerTFCustomOp)

# TODO: windows support
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -D_GLIBCXX_USE_CXX11_ABI=${TensorFlow_ABI}")
set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -fPIC --shared -D_GLIBCXX_USE_CXX11_ABI=${TensorFlow_ABI}")

include_directories(SYSTEM ${TensorFlow_INCLUDE_DIR})

add_library(redner_tf_data_ptr MODULE data_ptr.cc)
target_link_libraries(redner_tf_data_ptr
    ${TensorFlow_LIBRARY})

execute_process(
    COMMAND "python" -c "if True:
        from distutils import sysconfig as sc
        print(sc.get_python_lib())"
    OUTPUT_VARIABLE PYTHON_SITE
    OUTPUT_STRIP_TRAILING_WHITESPACE)
install(
    TARGETS redner_tf_data_ptr
    DESTINATION ${PYTHON_SITE})
