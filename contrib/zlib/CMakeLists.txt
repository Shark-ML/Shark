project(zlibexternal)
cmake_minimum_required(VERSION 3.1)
include(ExternalProject)

ExternalProject_Add(
	ZLIB
	PREFIX "${BASE_BUILD_DIR}/zlib/" 
	CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX='${BASE_BUILD_DIR}/'"
	GIT_REPOSITORY "https://github.com/madler/zlib.git" 
	GIT_TAG "v1.2.11"
)