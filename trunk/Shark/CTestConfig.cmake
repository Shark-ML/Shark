
cmake_minimum_required(VERSION 2.8)

# directories
set(CTEST_SOURCE_DIRECTORY ./Test)
set(CTEST_BINARY_DIRECTORY ./Test/bin/)

# build
set(CTEST_BUILD_NAME "Linux")
set(CTEST_CMAKE_GENERATOR "cmake")
set(CTEST_BUILD_COMMAND "gcc")
set(CTEST_COVERAGE_COMMAND "/usr/bin/gcov")

set(CTEST_DROP_SITE_CDASH TRUE)
set(CTEST_PROJECT_NAME "Shark CTests")
set(CTEST_DROP_METHOD "cp")
set(CTEST_DROP_LOCATION "./Test/test_output")

# memory check
set(CTEST_MEMORYCHECK_COMMAND "/usr/bin/valgrind")
set(MEMORYCHECK_COMMAND_OPTIONS "--xml=yes --xml-file=test.xml")
set(ENV{COVFILE} "${CTEST_BINARY_DIRECTORY}/CMake.cov")

