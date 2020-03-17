# https://github.com/PatWie/tensorflow-cmake/blob/master/cmake/modules/FindTensorFlow.cmake

execute_process(
    COMMAND python -c "exec(\"try:\\n  import tensorflow as tf; print(tf.__version__); print(tf.__cxx11_abi_flag__);print(tf.sysconfig.get_include()); print(tf.sysconfig.get_lib())\\nexcept ImportError:\\n  exit(1)\")"
    OUTPUT_VARIABLE TF_INFORMATION_STRING
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE retcode)

if("${retcode}" STREQUAL "0")
    string(REPLACE "\n" ";" TF_INFORMATION_LIST ${TF_INFORMATION_STRING})
    list(GET TF_INFORMATION_LIST 0 TF_DETECTED_VERSION)
    list(GET TF_INFORMATION_LIST 1 TF_DETECTED_ABI)
    list(GET TF_INFORMATION_LIST 2 TF_DETECTED_INCLUDE_DIR)
    list(GET TF_INFORMATION_LIST 3 TF_DETECTED_LIBRARY_DIR)
    if(WIN32)
        find_library(TF_DETECTED_LIBRARY NAMES _pywrap_tensorflow_internal PATHS 
            ${TF_DETECTED_LIBRARY_DIR}/python)        
    else()
        # For some reason my tensorflow doesn't have a .so file
        list(APPEND CMAKE_FIND_LIBRARY_SUFFIXES .so.1)
        list(APPEND CMAKE_FIND_LIBRARY_SUFFIXES .so.2)
        find_library(TF_DETECTED_LIBRARY NAMES tensorflow_framework PATHS 
            ${TF_DETECTED_LIBRARY_DIR})
    endif()
    set(TensorFlow_VERSION ${TF_DETECTED_VERSION})
    set(TensorFlow_ABI ${TF_DETECTED_ABI})
    set(TensorFlow_INCLUDE_DIR ${TF_DETECTED_INCLUDE_DIR})
    set(TensorFlow_LIBRARY ${TF_DETECTED_LIBRARY})
    if(TensorFlow_LIBRARY AND TensorFlow_INCLUDE_DIR)
        set(TensorFlow_FOUND TRUE)
    else()
        set(TensorFlow_FOUND FALSE)
    endif()
endif()
