find_path(EMBREE_INCLUDE_PATH embree3/rtcore.h
  ${CMAKE_SOURCE_DIR}/redner-dependencies/embree/include
  /usr/include
  /usr/local/include
  /opt/local/include)

find_library(EMBREE_LIBRARY NAMES embree3 PATHS 
  ${CMAKE_SOURCE_DIR}/redner-dependencies/embree/lib-linux
  ${CMAKE_SOURCE_DIR}/redner-dependencies/embree/lib-macos
  ${CMAKE_SOURCE_DIR}/redner-dependencies/embree/lib-win32
  /usr/lib 
  /usr/local/lib 
  /opt/local/lib)

if (EMBREE_INCLUDE_PATH AND EMBREE_LIBRARY)
  set(EMBREE_FOUND TRUE)
endif ()
