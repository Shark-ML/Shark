
// Copyright 2010 Dean Michael Berris.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_TEST_MODULE HTTP Request Linearize Test
#include <boost/network/protocol/http/request.hpp>
#include <boost/network/protocol/http/algorithms/linearize.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/mpl/list.hpp>
#include <iostream>

namespace http = boost::network::http;
namespace tags = boost::network::http::tags;
namespace mpl  = boost::mpl;
namespace net  = boost::network;

typedef mpl::list<
    tags::http_default_8bit_tcp_resolve
    , tags::http_default_8bit_udp_resolve
    , tags::http_async_8bit_tcp_resolve
    , tags::http_async_8bit_udp_resolve
    > tag_types;

BOOST_AUTO_TEST_CASE_TEMPLATE(linearize_request, T, tag_types) {
    http::basic_request<T> request("http://www.boost.org");
    linearize(request, "GET", 1, 0, std::ostream_iterator<typename net::char_<T>::type>(std::cout));
    linearize(request, "GET", 1, 1, std::ostream_iterator<typename net::char_<T>::type>(std::cout));
}

