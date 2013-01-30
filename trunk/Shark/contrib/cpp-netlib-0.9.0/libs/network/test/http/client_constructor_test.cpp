
// Copyright 2010 Dean Michael Berris. 
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_TEST_MODULE HTTP 1.0 Client Constructor Test
#include <boost/network/include/http/client.hpp>
#include <boost/test/unit_test.hpp>
#include "client_types.hpp"

namespace http = boost::network::http;

BOOST_AUTO_TEST_CASE_TEMPLATE(http_client_constructor_test, client, client_types) {
    client instance;
    boost::asio::io_service io_service;
    client instance2(io_service);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(http_cient_constructor_params_test, client, client_types) {
    client instance(
        http::_follow_redirects=true,
        http::_cache_resolved=true
        );
    boost::asio::io_service io_service;
    client instance2(
        http::_follow_redirects=true,
        http::_io_service=io_service,
        http::_cache_resolved=true
        );
    client instance3(
        http::_openssl_certificate="foo",
        http::_openssl_verify_path="bar"
        );
}

