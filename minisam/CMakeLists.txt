set(MINISAM_SUBDIRS
  core
  linear
  nonlinear
  utils
)

list(APPEND MINISAM_SUBDIRS geometry)
list(APPEND MINISAM_SUBDIRS slam)

# add SuiteSparse module if selected
if(MINISAM_USE_CHOLMOD)
  list(APPEND MINISAM_SUBDIRS linear/cholmod)
endif()

if(MINISAM_USE_SPQR)
  list(APPEND MINISAM_SUBDIRS linear/spqr)
endif()

# add module
foreach(SUBDIR ${MINISAM_SUBDIRS})
  message(STATUS "Build module: ${SUBDIR}")

  # lib source
  file(GLOB MINISAM_SRC "${SUBDIR}/*.cpp" "${SUBDIR}/*.h")
  list(APPEND MINISAM_SRCS ${MINISAM_SRC})

  # install headers
  file(GLOB SUBFOLDER_HEADERS "${SUBDIR}/*.h")
  install(FILES ${SUBFOLDER_HEADERS} DESTINATION "include/${PROJECT_NAME}/${SUBDIR}")
endforeach()


# --------------------------------------
# lib

if(MINISAM_BUILD_SHARED_LIB)
  # shared
  message(STATUS "Library: build shared library")
  add_library(${PROJECT_NAME} SHARED ${MINISAM_SRCS})
  set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME ${PROJECT_NAME} CLEAN_DIRECT_OUTPUT 1)

  # link to dependency
  target_link_libraries(${PROJECT_NAME} ${MINISAM_ADDITIONAL_LIBRARIES})
  
  if(WIN32)
    set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
  endif()

  # install lib
  install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}-export LIBRARY DESTINATION lib 
      ARCHIVE DESTINATION lib RUNTIME DESTINATION bin)

else()
  # static
  message(STATUS "Library: build static library")
  add_library(${PROJECT_NAME} STATIC ${MINISAM_SRCS})
  set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME ${PROJECT_NAME} CLEAN_DIRECT_OUTPUT 1)

  # link to dependency
  target_link_libraries(${PROJECT_NAME} ${MINISAM_ADDITIONAL_LIBRARIES})

  if(WIN32) # Add 'lib' prefix to static library to avoid filename collision with shared library
    set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "lib")
  endif()

  # install lib
  install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}-export ARCHIVE DESTINATION lib)

endif()


# --------------------------------------
# 3rd party install

# sub-folders for source and headers
set(3RDPARTY_SUBDIRS
  eigen3
)

# add 3rd party folders
foreach(SUBDIR ${3RDPARTY_SUBDIRS})
  message(STATUS "Install 3rd-party module: ${SUBDIR}")
  install(DIRECTORY "3rdparty/${SUBDIR}" DESTINATION "include/${PROJECT_NAME}/3rdparty")
endforeach()


# --------------------------------------
# install lib

# export config file for CMake to search
set(${PROJECT_NAME}_INCLUDE_DIR "include")
set(CMAKE_CONFIG_DEST "lib/cmake/${PROJECT_NAME}")

configure_package_config_file( 
    "${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}-config.cmake.in" 
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}-config.cmake"
    INSTALL_DESTINATION ${CMAKE_CONFIG_DEST}
    PATH_VARS ${PROJECT_NAME}_INCLUDE_DIR)

# install config
install(EXPORT ${PROJECT_NAME}-export DESTINATION ${CMAKE_CONFIG_DEST} FILE "${PROJECT_NAME}-export.cmake" )
install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}-config.cmake" DESTINATION ${CMAKE_CONFIG_DEST})

add_subdirectory(3rdparty/pybind11 ${PROJECT_BINARY_DIR}/pybind11)
file(GLOB_RECURSE
  MINISAM_WRAPPER_SRCS
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} 
  minisam_wrapper/*.cpp
  sophus_wrapper/*.cpp
)
pybind11_add_module(_pybind11_minisam ${MINISAM_WRAPPER_SRCS})

# target_include_directories(_pybind11_minisam PRIVATE ${HEADER_ROOT})
target_link_libraries(_pybind11_minisam PRIVATE ${PROJECT_NAME})
target_compile_definitions(_pybind11_minisam
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})