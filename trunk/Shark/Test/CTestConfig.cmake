
cmake_minimum_required(VERSION 3.8)
 
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

