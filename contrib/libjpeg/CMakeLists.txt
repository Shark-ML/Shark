project(libjpegexternal)
cmake_minimum_required(VERSION 3.1)
include(ExternalProject)
set(CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${BASE_BUILD_DIR}")
list(APPEND CMAKE_ARGS "-DBUILD_EXECUTABLES=OFF")
list(APPEND CMAKE_ARGS "-DBUILD_TESTS=OFF")
list(APPEND CMAKE_ARGS "-DBUILD_STATIC=ON")
message(STATUS "${CMAKE_ARGS}")
ExternalProject_Add(
	JPEG
	PREFIX "${BASE_BUILD_DIR}/libjpeg/" 
	CMAKE_ARGS  "${CMAKE_ARGS}"
	GIT_REPOSITORY "https://github.com/LuaDist/libjpeg.git" 
	GIT_TAG "master"
)