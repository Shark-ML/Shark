project(libZipexternal)
cmake_minimum_required(VERSION 3.1)
include(ExternalProject)
ExternalProject_Add(
	libZip
	PREFIX "${BASE_BUILD_DIR}/libzip/" 
	CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX='${BASE_BUILD_DIR}/'"
	GIT_REPOSITORY "https://github.com/nih-at/libzip.git" 
	GIT_TAG "rel-1-5-1"
)