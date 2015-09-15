# This file sets up include directories, link directories, and
# compiler settings for a project to use Shark.  It should not be
# included directly, but rather through the SHARK_USE_FILE setting
# obtained from SharkConfig.cmake.

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SHARK_REQUIRED_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SHARK_REQUIRED_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${SHARK_REQUIRED_EXE_LINKER_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${SHARK_REQUIRED_SHARED_LINKER_FLAGS}")
set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${SHARK_REQUIRED_MODULE_LINKER_FLAGS}")

# Add include directories needed to use Shark.
include_directories(BEFORE ${SHARK_INCLUDE_DIRS})

# Add link directories needed to use Shark.
link_directories(${SHARK_LIBRARY_DIRS})
