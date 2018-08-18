project(libpngexternal)
cmake_minimum_required(VERSION 3.1)
include(ExternalProject)
set(CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${BASE_BUILD_DIR}")
list(APPEND CMAKE_ARGS "-DZLIB_ROOT=${BASE_BUILD_DIR}")
ExternalProject_Add(
	PNG
	PREFIX "${BASE_BUILD_DIR}/libpng/" 
	CMAKE_ARGS  "${CMAKE_ARGS}"
	GIT_REPOSITORY "https://github.com/glennrp/libpng.git" 
	GIT_TAG "v1.6.34"
)