
// Copyright 2010 Dean Michael Berris. 
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_TEST_MODULE HTTP Server Construtor Tests

#include <boost/network/include/http/server.hpp>
#include <boost/test/unit_test.hpp>

namespace http = boost::network::http;
namespace util = boost::network::utils;

struct dummy_sync_handler;
struct dummy_async_handler;
typedef http::server<dummy_sync_handler> sync_server;
typedef http::async_server<dummy_async_handler> async_server;

struct dummy_sync_handler {
    void operator()(sync_server::request const & req,
                    sync_server::response & res) {
        // Really, this is just for testing purposes
    }

    void log(char const *) {
    }
};

struct dummy_async_handler {
    void operator()(async_server::request const & req,
                    async_server::connection_ptr conn) {
        // Really, this is just for testing purposes
    }
};

BOOST_AUTO_TEST_CASE(minimal_constructor) {
    dummy_sync_handler sync_handler;
    dummy_async_handler async_handler;
    util::thread_pool pool;

    BOOST_CHECK_NO_THROW(sync_server sync_instance("127.0.0.1", "80", sync_handler) );
    BOOST_CHECK_NO_THROW(async_server async_instance("127.0.0.1", "80", async_handler, pool) );
}

BOOST_AUTO_TEST_CASE(with_io_service_parameter) {
    dummy_sync_handler sync_handler;
    dummy_async_handler async_handler;
    util::thread_pool pool;
    boost::asio::io_service io_service;

    BOOST_CHECK_NO_THROW(sync_server sync_instance("127.0.0.1", "80", sync_handler, io_service));
    BOOST_CHECK_NO_THROW(async_server async_instance("127.0.0.1", "80", async_handler, pool, io_service));
}

BOOST_AUTO_TEST_CASE(with_socket_options_parameter) {
    dummy_sync_handler sync_handler;
    dummy_async_handler async_handler;
    util::thread_pool pool;

    BOOST_CHECK_NO_THROW(sync_server sync_instance("127.0.0.1", "80", sync_handler,
        http::_reuse_address=true,
        http::_report_aborted=true,
        http::_receive_buffer_size=4096,
        http::_send_buffer_size=4096,
        http::_receive_low_watermark=1024,
        http::_send_low_watermark=1024,
        http::_non_blocking_io=true,
        http::_linger=true,
        http::_linger_timeout=0
        ));
    BOOST_CHECK_NO_THROW(async_server async_instance("127.0.0.1", "80", async_handler, pool,
        http::_reuse_address=true,
        http::_report_aborted=true,
        http::_receive_buffer_size=4096,
        http::_send_buffer_size=4096,
        http::_receive_low_watermark=1024,
        http::_send_low_watermark=1024,
        http::_non_blocking_io=true,
        http::_linger=true,
        http::_linger_timeout=0
        ));

}
