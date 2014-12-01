
cmake_minimum_required(VERSION 3.8)

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
  set(CTEST_DROP_METHOD "xmlrpc")
  set(CTEST_DROP_LOCATION "./Test")

# memory check
set(CTEST_MEMORYCHECK_COMMAND "/usr/bin/valgrind")
set(MEMORYCHECK_COMMAND_OPTIONS "--xml=yes --xml-file=test.xml")
set(ENV{COVFILE} "${CTEST_BINARY_DIRECTORY}/CMake.cov")

ctest_empty_binary_directory("${CTEST_BINARY_DIRECTORY}")
ctest_start("Continous")

# ctest
set(CTEST_PROJECT_SUBPROJECTS SportTest SampleTest InterceptTest)
foreach(subproject ${CTEST_PROJECT_SUBPROJECTS})
    set_property(GLOBAL PROPERTY SubProject ${subproject})
    set_property(GLOBAL PROPERTY Label  ${subproject})

    ctest_configure()

    set(CTEST_BUILD_TARGET ${subproject})
    ctest_build(APPEND)
    ctest_test(INCLUDE_LABEL ${subproject})
    ctest_coverage()
    ctest_submit()
endforeach()

