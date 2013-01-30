#ifndef BOOST_NETWORK_PROTOCOL_HTTP_SERVER_ASYNC_SERVER_HPP_20101025
#define BOOST_NETWORK_PROTOCOL_HTTP_SERVER_ASYNC_SERVER_HPP_20101025

// Copyright 2010 Dean Michael Berris. 
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include <boost/network/detail/debug.hpp>
#include <boost/network/protocol/http/server/async_connection.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/network/protocol/http/server/storage_base.hpp>
#include <boost/network/protocol/http/server/socket_options_base.hpp>
#include <boost/network/utils/thread_pool.hpp>

namespace boost { namespace network { namespace http {
    
    template <class Tag, class Handler>
    struct async_server_base : server_storage_base, socket_options_base {
        typedef basic_request<Tag> request;
        typedef basic_response<Tag> response;
        typedef typename string<Tag>::type string_type;
        typedef typename boost::network::http::response_header<Tag>::type response_header;
        typedef async_connection<Tag,Handler> connection;
        typedef shared_ptr<connection> connection_ptr;

        template <class ArgPack>
        async_server_base(ArgPack const & args)
        : server_storage_base(args,
            typename mpl::if_<
                is_same<
                    typename parameter::value_type<ArgPack, tag::io_service, void>::type,
                    void
                    >,
                server_storage_base::no_io_service,
                server_storage_base::has_io_service
                >::type())
        , socket_options_base(args)
        , handler(args[_handler])
        , address_(args[_address])
        , port_(args[_port])
        , thread_pool(args[_thread_pool])
        , acceptor(server_storage_base::service_)
        , stopping(false)
        , new_connection()
        , listening_mutex_()
        , listening(false)
        {}

        void run() {
            listen();
            service_.run();
        };

        void stop() {
            // stop accepting new requests and let all the existing
            // handlers finish.
            stopping = true;
            system::error_code ignored;
            acceptor.close(ignored);
        }

        void listen() {
            boost::unique_lock<boost::mutex> listening_lock(listening_mutex_);
            BOOST_NETWORK_MESSAGE("Listening on " << address_ << ':' << port_);
            if (!listening) start_listening();
            if (!listening) {
                BOOST_NETWORK_MESSAGE("Error listening on " << address_ << ':' << port_);
                boost::throw_exception(std::runtime_error("Error listening on provided port."));
            }
        }

    private:
        Handler & handler;
        string_type address_, port_;
        utils::thread_pool & thread_pool;
        asio::ip::tcp::acceptor acceptor;
        bool stopping;
        connection_ptr new_connection;
        boost::mutex listening_mutex_;
        bool listening;

        void handle_accept(boost::system::error_code const & ec) {
            if (!ec) {
                socket_options_base::socket_options(new_connection->socket());
                new_connection->start();
                if (!stopping) {
                    new_connection.reset(
                        new connection(
                            service_
                            , handler
                            , thread_pool
                            )
                        );
                    acceptor.async_accept(new_connection->socket(),
                        boost::bind(
                            &async_server_base<Tag,Handler>::handle_accept
                            , this
                            , boost::asio::placeholders::error
                            )
                        );
                }
            } else {
                BOOST_NETWORK_MESSAGE("Error accepting connection, reason: " << ec);
            }
        }
        
        void start_listening() {
            using boost::asio::ip::tcp;
            
            system::error_code error;
            tcp::resolver resolver(service_);
            tcp::resolver::query query(address_, port_);
            tcp::resolver::iterator endpoint_iterator = resolver.resolve(query, error);
            if (error) {
                BOOST_NETWORK_MESSAGE("Error resolving '" << address_ << ':' << port_);
                return;
            }
            tcp::endpoint endpoint = *endpoint_iterator;
            acceptor.open(endpoint.protocol(), error);
            if (error) {
                BOOST_NETWORK_MESSAGE("Error opening socket: " << address_ << ":" << port_);
                return;
            }
            acceptor.bind(endpoint, error);
            if (error) {
                BOOST_NETWORK_MESSAGE("Error binding socket: " << address_ << ":" << port_);
                return;
            }
            socket_options_base::acceptor_options(acceptor);
            acceptor.listen(asio::socket_base::max_connections, error);
            if (error) {
                BOOST_NETWORK_MESSAGE("Error listening on socket: '" << error << "' on " << address_ << ":" << port_);
                return;
            }
            new_connection.reset(new connection(service_, handler, thread_pool));
            acceptor.async_accept(new_connection->socket(),
                boost::bind(
                    &async_server_base<Tag,Handler>::handle_accept
                    , this
                    , boost::asio::placeholders::error));
            listening = true;
            BOOST_NETWORK_MESSAGE("Now listening on socket: '" << address_ << ":" << port_ << "'");
        }
    };
    
} /* http */
    
} /* network */
    
} /* boost */

#endif /* BOOST_NETWORK_PROTOCOL_HTTP_SERVER_ASYNC_SERVER_HPP_20101025 */
