// Copyright 2009 (c) Tarro, Inc.
// Copyright 2009 (c) Dean Michael Berris <mikhailberis@gmail.com>
// Copyright 2010 (c) Glyn Matthews
// Copyright 2003-2008 (c) Chris Kholhoff
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_NETWORK_HTTP_SERVER_HPP_
#define BOOST_NETWORK_HTTP_SERVER_HPP_

#include <boost/network/protocol/http/response.hpp>
#include <boost/network/protocol/http/request.hpp>
#include <boost/network/protocol/http/server/sync_server.hpp>
#include <boost/network/protocol/http/server/async_server.hpp>
#include <boost/network/protocol/http/server/parameters.hpp>

namespace boost { namespace network { namespace http {
    
    template <class Tag, class Handler, class Enable = void>
    struct server_base {
        typedef unsupported_tag<Tag> type;
    };
    
    template <class Tag, class Handler>
    struct server_base<Tag, Handler, typename enable_if<is_async<Tag> >::type> {
        typedef async_server_base<Tag, Handler> type;
    };
    
    template <class Tag, class Handler>
    struct server_base<Tag, Handler, typename enable_if<is_sync<Tag> >::type> {
        typedef sync_server_base<Tag, Handler> type;
    };

    template <class Tag, class Handler>
    struct basic_server : server_base<Tag, Handler>::type
    {};

    template <class Handler>
    struct server : server_base<tags::http_server, Handler>::type {
        typedef typename server_base<tags::http_server, Handler>::type
            server_base;

        BOOST_PARAMETER_CONSTRUCTOR(
            server, (server_base), tag,
            (required
                (address, (typename server_base::string_type const &))
                (port, (typename server_base::string_type const &))
                (in_out(handler), (Handler &)))
            (optional
                (in_out(io_service), (boost::asio::io_service &))
                (reuse_address, (bool))
                (report_aborted, (bool))
                (receive_buffer_size, (int))
                (send_buffer_size, (int))
                (receive_low_watermark, (int))
                (send_low_watermark, (int))
                (non_blocking_io, (int))
                (linger, (bool))
                (linger_timeout, (int)))
            )
    };

    template <class Handler>
    struct async_server : server_base<tags::http_async_server, Handler>::type
    {
        typedef typename server_base<tags::http_async_server, Handler>::type
            server_base;

        BOOST_PARAMETER_CONSTRUCTOR(
            async_server, (server_base), tag,
            (required
                (address, (typename server_base::string_type const &))
                (port, (typename server_base::string_type const &))
                (in_out(handler), (Handler&))
                (in_out(thread_pool), (utils::thread_pool&)))
            (optional
                (in_out(io_service), (boost::asio::io_service&))
                (reuse_address, (bool))
                (report_aborted, (bool))
                (receive_buffer_size, (int))
                (send_buffer_size, (int))
                (receive_low_watermark, (int))
                (send_low_watermark, (int))
                (non_blocking_io, (bool))
                (linger, (bool))
                (linger_timeout, (int)))
            )
    };

} // namespace http

} // namespace network

} // namespace boost

#endif // BOOST_NETWORK_HTTP_SERVER_HPP_

