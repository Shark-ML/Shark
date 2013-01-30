include_directories(${CPP-NETLIB_SOURCE_DIR})
find_package ( Boost 1.41.0 COMPONENTS unit_test_framework )
set ( Boost_USE_STATIC_LIBS ON )
set ( Boost_USE_MULTITHREADED ON )

if ( Boost_FOUND )
    add_executable ( mime-roundtrip mime-roundtrip.cpp )
    target_link_libraries ( mime-roundtrip ${Boost_LIBRARIES} )
    add_test ( mime-roundtrip mime-roundtrip )
endif ()

