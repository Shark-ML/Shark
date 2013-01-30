#            Copyright (c) Dean Michael Berris 2010.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 2.6)
project(CPP-NETLIB)
find_package( Boost 1.41.0 )
set(CMAKE_VERBOSE_MAKEFILE true)
if (Boost_FOUND)
    set(Boost_USE_STATIC_LIBS ON)
    set(Boost_USE_MULTI_THREADED ON)
    include_directories(${Boost_INCLUDE_DIRS})
endif()
enable_testing()
add_subdirectory(libs/network/build)
add_subdirectory(libs/network/test)
add_subdirectory(libs/mime/test)
add_subdirectory(libs/network/example)

