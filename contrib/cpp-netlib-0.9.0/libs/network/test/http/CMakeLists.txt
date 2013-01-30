
# Copyright 2010 Dean Michael Berris. 
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE_1_0.txt or copy at
# http://www.boost.org/LICENSE_1_0.txt)

include_directories(${CPP-NETLIB_SOURCE_DIR})
find_package( Boost 1.43.0 REQUIRED unit_test_framework system regex thread filesystem )
find_package( OpenSSL )

if (OPENSSL_FOUND)
    include_directories( ${OPENSSL_INCLUDE_DIR} )
    add_definitions(-DBOOST_NETWORK_ENABLE_HTTPS)
endif()

if (CMAKE_BUILD_TYPE MATCHES Debug)
    add_definitions(-DBOOST_NETWORK_DEBUG)
endif()

find_package( Threads )
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)

if (Boost_FOUND)
    set ( TESTS
        client_constructor_test
        client_get_test
        client_get_different_port_test
        client_get_timeout_test
        client_localhost_normal_test
        )
    if (OPENSSL_FOUND)
        set ( TESTS ${TESTS} client_localhost_ssl_test )
    endif (OPENSSL_FOUND)
    foreach ( test ${TESTS} )
        if (${CMAKE_CXX_COMPILER_ID} MATCHES GNU)
            set_source_files_properties(${test}.cpp
                PROPERTIES COMPILE_FLAGS "-Wall")
        endif()
        add_executable(cpp-netlib-http-${test} ${test}.cpp)
        add_dependencies(cpp-netlib-http-${test} cppnetlib-uri-parsers)
        target_link_libraries(cpp-netlib-http-${test} ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} cppnetlib-uri-parsers)
        if (OPENSSL_FOUND)
            target_link_libraries(cpp-netlib-http-${test} ${OPENSSL_LIBRARIES})
        endif()
        set_target_properties(cpp-netlib-http-${test}
            PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CPP-NETLIB_BINARY_DIR}/tests)
        add_test(cpp-netlib-http-${test} 
            ${CPP-NETLIB_BINARY_DIR}/tests/cpp-netlib-http-${test})
    endforeach (test)

    set ( SERVER_API_TESTS
        server_constructor_test
        )
    foreach ( test ${SERVER_API_TESTS} )
        if (${CMAKE_CXX_COMPILER_ID} MATCHES GNU)
            set_source_files_properties(${test}.cpp
                PROPERTIES COMPILE_FLAGS "-Wall")
        endif()
        add_executable(cpp-netlib-http-${test} ${test}.cpp)
        add_dependencies(cpp-netlib-http-${test} cppnetlib-server-parsers)
        target_link_libraries(cpp-netlib-http-${test} ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} cppnetlib-server-parsers)
        set_target_properties(cpp-netlib-http-${test}
            PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CPP-NETLIB_BINARY_DIR}/tests)
        add_test(cpp-netlib-http-${test}
            ${CPP-NETLIB_BINARY_DIR}/tests/cpp-netlib-http-${test})
    endforeach (test)

    set ( SERVER_TESTS
        server_hello_world
        server_async
        server_async_less_copy
        )
    set ( PORT 8000 )
    foreach ( test ${SERVER_TESTS} )
        if (${CMAKE_CXX_COMPILER_ID} MATCHES GNU)
            set_source_files_properties(${test}.cpp
                PROPERTIES COMPILE_FLAGS "-Wall")
        endif()
        add_executable(cpp-netlib-http-${test} ${test}.cpp)
        add_dependencies(cpp-netlib-http-${test} cppnetlib-server-parsers)
        target_link_libraries(cpp-netlib-http-${test} ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} cppnetlib-server-parsers)
        set_target_properties(cpp-netlib-http-${test}
            PROPERTIES RUNTIME_OUTPUT_DIRECTORY 
            ${CPP-NETLIB_BINARY_DIR}/tests)
        add_test(cpp-netlib-${test} 
            python 
            ${CPP-NETLIB_SOURCE_DIR}/libs/network/test/httplib_acceptance.py 
            ${CPP-NETLIB_BINARY_DIR}/tests/cpp-netlib-http-${test}
            ${PORT}
            ${CPP-NETLIB_BINARY_DIR}/tests/cpp-netlib-http-${test}.passed)
        math (EXPR PORT "${PORT} + 1")
    endforeach (test)

    set ( INLINED_TESTS
        client_include_inlined
        server_include_inlined
        )
    foreach ( test ${INLINED_TESTS} )
        if (${CMAKE_CXX_COMPILER_ID} MATCHES GNU)
            set_source_files_properties(${test}.cpp
                PROPERTIES COMPILE_FLAGS "-Wall")
        endif()
        add_executable(cpp-netlib-http-inlined-${test} ${test}.cpp)
        target_link_libraries(cpp-netlib-http-inlined-${test} ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
        if (OPENSSL_FOUND)
            target_link_libraries(cpp-netlib-http-inlined-${test} ${OPENSSL_LIBRARIES})
        endif()
        set_target_properties(cpp-netlib-http-inlined-${test}
            PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CPP-NETLIB_BINARY_DIR}/tests)
        add_test(cpp-netlib-http-inlined-${test}
            ${CPP-NETLIB_BINARY_DIR}/tests/cpp-netlib-http-inlined-${test})
    endforeach (test)

endif()
