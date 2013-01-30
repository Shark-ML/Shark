#ifndef BOOST_NETWORK_PROTOCOL_HTTP_SERVER_SOCKET_OPTIONS_BASE_HPP_20101210
#define BOOST_NETWORK_PROTOCOL_HTTP_SERVER_SOCKET_OPTIONS_BASE_HPP_20101210

// Copyright 2010 Dean Michael Berris. 
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include <boost/network/protocol/http/server/parameters.hpp>
#include <boost/optional.hpp>
#include <boost/utility/in_place_factory.hpp>

namespace boost { namespace network { namespace http {
    
    struct socket_options_base {
    protected:
        asio::socket_base::reuse_address acceptor_reuse_address;
        asio::socket_base::enable_connection_aborted acceptor_report_aborted;
        boost::optional<asio::socket_base::receive_buffer_size> receive_buffer_size;
        boost::optional<asio::socket_base::send_buffer_size> send_buffer_size;
        boost::optional<asio::socket_base::receive_low_watermark> receive_low_watermark;
        boost::optional<asio::socket_base::send_low_watermark> send_low_watermark;
        asio::socket_base::non_blocking_io non_blocking_io;
        asio::socket_base::linger linger;

        template <class ArgPack>
        socket_options_base(ArgPack const & args)
        : acceptor_reuse_address(args[_reuse_address|false])
        , acceptor_report_aborted(args[_report_aborted|false])
        , non_blocking_io(args[_non_blocking_io|true])
        , linger(args[_linger|true], args[_linger_timeout|0])
        {
            set_optional(receive_buffer_size, args, _receive_buffer_size);
            set_optional(send_buffer_size, args, _send_buffer_size);
            set_optional(receive_low_watermark, args, _receive_low_watermark);
            set_optional(send_low_watermark, args, _send_low_watermark);
        }

        void acceptor_options(boost::asio::ip::tcp::acceptor & acceptor) {
            acceptor.set_option(acceptor_reuse_address);
            acceptor.set_option(acceptor_report_aborted);
        }

        void socket_options(boost::asio::ip::tcp::socket & socket) {
            boost::system::error_code ignored;
            socket.io_control(non_blocking_io, ignored);
            socket.set_option(linger, ignored);
            if (receive_buffer_size) socket.set_option(*receive_buffer_size, ignored);
            if (receive_low_watermark) socket.set_option(*receive_low_watermark, ignored);
            if (send_buffer_size) socket.set_option(*send_buffer_size, ignored);
            if (send_low_watermark) socket.set_option(*send_low_watermark, ignored);
        }

    private:

        template <class Optional, class Args, class Keyword>
        typename boost::enable_if<
            mpl::not_<
                boost::is_same<
                    typename boost::parameter::value_type<Args, Keyword, void>::type
                    , void
                    >
                >
            , void
        >::type
        set_optional(Optional & option, Args const & args, Keyword const & keyword) {
            option = in_place<typename Optional::value_type>(args[keyword]);
        }

        template <class Optional, class Args, class Keyword>
        typename boost::enable_if<
            boost::is_same<
                typename boost::parameter::value_type<Args, Keyword, void>::type
                , void
                >
            , void
        >::type
        set_optional(Optional &, Args const &, Keyword const &) {
            // do nothing
        }

    };

} /* http */
    
} /* network */
    
} /* boost */

#endif /* BOOST_NETWORK_PROTOCOL_HTTP_SERVER_SOCKET_OPTIONS_BASE_HPP_20101210 */
