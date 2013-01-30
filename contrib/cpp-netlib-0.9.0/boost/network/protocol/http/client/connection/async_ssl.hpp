#ifndef BOOST_NETWORK_PROTOCOL_HTTP_IMPL_HTTPS_ASYNC_CONNECTION_HPP_20100601
#define BOOST_NETWORK_PROTOCOL_HTTP_IMPL_HTTPS_ASYNC_CONNECTION_HPP_20100601

// Copyright 2010 (C) Dean Michael Berris
// Copyright 2010 (C) Sinefunc, Inc.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include <boost/asio/ssl.hpp>
#include <boost/asio/strand.hpp>
#include <boost/network/protocol/http/client/connection/async_protocol_handler.hpp>
#include <boost/assert.hpp>

namespace boost { namespace network { namespace http { namespace impl {

    template <class Tag, unsigned version_major, unsigned version_minor>
    struct async_connection_base;

    template <class Tag, unsigned version_major, unsigned version_minor>
    struct https_async_connection
        : async_connection_base<Tag,version_major,version_minor>,
        protected http_async_protocol_handler<Tag,version_major,version_minor>,
        boost::enable_shared_from_this<https_async_connection<Tag,version_major,version_minor> >
    {
            typedef async_connection_base<Tag,version_major,version_minor> base;
            typedef http_async_protocol_handler<Tag,version_major,version_minor> protocol_base;
            typedef typename base::resolver_type resolver_type;
            typedef typename base::resolver_base::resolver_iterator resolver_iterator;
            typedef typename base::resolver_base::resolver_iterator_pair resolver_iterator_pair;
            typedef typename base::response response;
            typedef typename base::string_type string_type;
            typedef typename base::request request;
            typedef typename base::resolver_base::resolve_function resolve_function;

            https_async_connection(
                resolver_type & resolver,
                resolve_function resolve, 
                bool follow_redirect,
                optional<string_type> const & certificate_filename = optional<string_type>(), 
                optional<string_type> const & verify_path = optional<string_type>()
                ) : 
                follow_redirect_(follow_redirect),
                resolver_(resolver),
                certificate_filename_(certificate_filename),
                verify_path_(verify_path),
                resolve_(resolve), 
                request_strand_(resolver.get_io_service())
            {}


            virtual response start(request const & request, string_type const & method, bool get_body) {
                response response_;
                this->init_response(response_, get_body);
                linearize(request, method, version_major, version_minor,
                    std::ostreambuf_iterator<typename char_<Tag>::type>(&command_streambuf));
                this->method = method;
                boost::uint16_t port_ = port(request);
                resolve_(resolver_, host(request), 
                    port_,
                    request_strand_.wrap(
                        boost::bind(
                        &https_async_connection<Tag,version_major,version_minor>::handle_resolved,
                        https_async_connection<Tag,version_major,version_minor>::shared_from_this(),
                        port_, get_body,
                        _1, _2)));
                return response_;
            }

    private:
        void handle_resolved(boost::uint16_t port, bool get_body, boost::system::error_code const & ec, resolver_iterator_pair endpoint_range) {
            resolver_iterator iter = boost::begin(endpoint_range);
            if (!ec && !boost::empty(endpoint_range)) {
                boost::asio::ip::tcp::endpoint endpoint(
                    iter->endpoint().address(),
                    port
                    );
                context_.reset(new boost::asio::ssl::context(
                    resolver_.get_io_service(),
                    boost::asio::ssl::context::sslv23_client
                    )
                );
                if (certificate_filename_ || verify_path_) {
                    context_->set_verify_mode(boost::asio::ssl::context::verify_peer);
                    if (certificate_filename_) context_->load_verify_file(*certificate_filename_);
                    if (verify_path_) context_->add_verify_path(*verify_path_);
                } else {
                    context_->set_verify_mode(boost::asio::ssl::context::verify_none);
                }
                socket_.reset(new boost::asio::ssl::stream<boost::asio::ip::tcp::socket>(
                    resolver_.get_io_service(),
                    *context_
                    )
                );
                socket_->lowest_layer().async_connect(
                    endpoint,
                    request_strand_.wrap(
                        boost::bind(
                            &https_async_connection<Tag,version_major,version_minor>::handle_connected,
                            https_async_connection<Tag,version_major,version_minor>::shared_from_this(),
                            port, get_body, std::make_pair(++iter, resolver_iterator()),
                            boost::asio::placeholders::error
                            )));
            } else {
                boost::system::system_error error(ec ? ec : boost::asio::error::host_not_found);
                this->version_promise.set_exception(boost::copy_exception(error));
                this->status_promise.set_exception(boost::copy_exception(error));
                this->status_message_promise.set_exception(boost::copy_exception(error));
                this->headers_promise.set_exception(boost::copy_exception(error));
                this->source_promise.set_exception(boost::copy_exception(error));
                this->destination_promise.set_exception(boost::copy_exception(error));
                this->body_promise.set_exception(boost::copy_exception(error));
            }
        }

        void handle_connected(boost::uint16_t port, bool get_body, resolver_iterator_pair endpoint_range, boost::system::error_code const & ec) {
            if (!ec) {
                socket_->async_handshake(boost::asio::ssl::stream_base::client,
                    request_strand_.wrap(
                        boost::bind(
                            &https_async_connection<Tag,version_major,version_minor>::handle_handshake,
                            https_async_connection<Tag,version_major,version_minor>::shared_from_this(),
                            port, get_body,
                            boost::asio::placeholders::error
                            )
                        )
                    );
            } else {
                if (!boost::empty(endpoint_range)) {
                    resolver_iterator iter = boost::begin(endpoint_range);
                    boost::asio::ip::tcp::endpoint endpoint(
                        iter->endpoint().address(),
                        port
                        );
                    socket_.reset(new boost::asio::ssl::stream<boost::asio::ip::tcp::socket>(
                        resolver_.get_io_service()
                        , *context_
                        )
                    );
                    socket_->lowest_layer().async_connect(
                        endpoint,
                        request_strand_.wrap(
                            boost::bind(
                                &https_async_connection<Tag,version_major,version_minor>::handle_connected,
                                https_async_connection<Tag,version_major,version_minor>::shared_from_this(),
                                port, get_body, std::make_pair(++iter, resolver_iterator()),
                                boost::asio::placeholders::error
                                )));
                } else {
                    boost::system::system_error error(
                        ec ? ec : boost::asio::error::host_not_found
                        );
                    this->version_promise.set_exception(boost::copy_exception(error));
                    this->status_promise.set_exception(boost::copy_exception(error));
                    this->status_message_promise.set_exception(boost::copy_exception(error));
                    this->headers_promise.set_exception(boost::copy_exception(error));
                    this->source_promise.set_exception(boost::copy_exception(error));
                    this->destination_promise.set_exception(boost::copy_exception(error));
                    this->body_promise.set_exception(boost::copy_exception(error));
                }
            }
        }

        void handle_handshake(boost::uint16_t port, bool get_body, boost::system::error_code const & ec) {
            if (!ec) {
                boost::asio::async_write(
                    *socket_
                    , command_streambuf
                    , request_strand_.wrap(
                        boost::bind(
                            &https_async_connection<Tag,version_major,version_minor>::handle_sent_request,
                            https_async_connection<Tag,version_major,version_minor>::shared_from_this(),
                            get_body,
                            boost::asio::placeholders::error,
                            boost::asio::placeholders::bytes_transferred
                            )));
            } else {
                    boost::system::system_error error(
                        ec ? ec : boost::asio::error::host_not_found
                        );
                    this->version_promise.set_exception(boost::copy_exception(error));
                    this->status_promise.set_exception(boost::copy_exception(error));
                    this->status_message_promise.set_exception(boost::copy_exception(error));
                    this->headers_promise.set_exception(boost::copy_exception(error));
                    this->source_promise.set_exception(boost::copy_exception(error));
                    this->destination_promise.set_exception(boost::copy_exception(error));
                    this->body_promise.set_exception(boost::copy_exception(error));
            }
        }

        enum state_t {
            version, status, status_message, headers, body
        };

        void handle_sent_request(bool get_body, boost::system::error_code const & ec, std::size_t bytes_transferred) {
            if (!ec) {
                boost::asio::async_read(
                    *socket_,
                    boost::asio::mutable_buffers_1(
                        this->part.c_array(), 
                        this->part.size()),
                    request_strand_.wrap(
                        boost::bind(
                            &https_async_connection<Tag,version_major,version_minor>::handle_received_data,
                            https_async_connection<Tag,version_major,version_minor>::shared_from_this(),
                            version, get_body, boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred)));
            } else {
                boost::system::system_error error(
                    ec ? ec : boost::asio::error::host_not_found
                    );
                this->version_promise.set_exception(boost::copy_exception(error));
                this->status_promise.set_exception(boost::copy_exception(error));
                this->status_message_promise.set_exception(boost::copy_exception(error));
                this->headers_promise.set_exception(boost::copy_exception(error));
                this->source_promise.set_exception(boost::copy_exception(error));
                this->destination_promise.set_exception(boost::copy_exception(error));
                this->body_promise.set_exception(boost::copy_exception(error));
            }
        }

        void handle_received_data(state_t state, bool get_body, boost::system::error_code const & ec, std::size_t bytes_transferred) {
            if (!ec || ec == boost::asio::error::eof) {
                logic::tribool parsed_ok;
                size_t remainder;
                switch(state) {
                    case version:
                        parsed_ok = 
                            this->parse_version(
                                *socket_,
                                request_strand_.wrap(
                                    boost::bind(
                                        &https_async_connection<Tag,version_major,version_minor>::handle_received_data,
                                        https_async_connection<Tag,version_major,version_minor>::shared_from_this(),
                                        status, get_body, boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred
                                        )
                                    )
                                );
                        if (parsed_ok != true) return;
                    case status:
                        parsed_ok =
                            this->parse_status(*socket_,
                                request_strand_.wrap(
                                    boost::bind(
                                        &https_async_connection<Tag,version_major,version_minor>::handle_received_data,
                                        https_async_connection<Tag,version_major,version_minor>::shared_from_this(),
                                        status_message, get_body, boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred
                                        )
                                    )
                                );
                        if (parsed_ok != true) return;
                    case status_message:
                        parsed_ok =
                            this->parse_status_message(*socket_,
                                request_strand_.wrap(
                                    boost::bind(
                                        &https_async_connection<Tag,version_major,version_minor>::handle_received_data,
                                        https_async_connection<Tag,version_major,version_minor>::shared_from_this(),
                                        headers, get_body, boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred
                                        )
                                    )
                                );
                        if (parsed_ok != true) return;
                    case headers:
                        fusion::tie(parsed_ok, remainder) =
                            this->parse_headers(*socket_,
                                request_strand_.wrap(
                                    boost::bind(
                                        &https_async_connection<Tag,version_major,version_minor>::handle_received_data,
                                        https_async_connection<Tag,version_major,version_minor>::shared_from_this(),
                                        body, get_body, boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred
                                        )
                                    )
                                );
                        if (parsed_ok != true) return;
                        if (!get_body) {
                            this->body_promise.set_value("");
                            return;
                        }
                        this->parse_body(
                            *socket_,
                            request_strand_.wrap(
                                boost::bind(
                                    &https_async_connection<Tag,version_major,version_minor>::handle_received_data,
                                    https_async_connection<Tag,version_major,version_minor>::shared_from_this(),
                                    body, get_body, boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred
                                    )
                                ),
                            remainder);
                        break;
                    case body:
                        if (!get_body) {
                            this->body_promise.set_value("");
                            return;
                        }
                        if (ec == boost::asio::error::eof) {
                            string_type body;
                            std::swap(body, this->partial_parsed);
                            body.append(
                                this->part.begin()
                                , bytes_transferred
                                );
                            this->body_promise.set_value(body);
                            // TODO set the destination value somewhere!
                            this->destination_promise.set_value("");
                            this->source_promise.set_value("");
                            this->part.assign('\0');
                            this->response_parser_.reset();
                        } else {
                            this->parse_body(
                                *socket_,
                                request_strand_.wrap(
                                    boost::bind(
                                        &https_async_connection<Tag,version_major,version_minor>::handle_received_data,
                                        https_async_connection<Tag,version_major,version_minor>::shared_from_this(),
                                        body, get_body, boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred
                                        )
                                    ),
                                bytes_transferred);
                        }
                        break;
                    default:
                        BOOST_ASSERT(false && "Bug, report this to the developers!");
                }
            } else {
                boost::system::system_error error(ec);
                    this->source_promise.set_exception(boost::copy_exception(error));
                    this->destination_promise.set_exception(boost::copy_exception(error));
                switch (state) {
                    case version:
                        this->version_promise.set_exception(boost::copy_exception(error));
                    case status:
                        this->status_promise.set_exception(boost::copy_exception(error));
                    case status_message:
                        this->status_message_promise.set_exception(boost::copy_exception(error));
                    case headers:
                        this->headers_promise.set_exception(boost::copy_exception(error));
                    case body:
                        if (get_body) this->body_promise.set_exception(boost::copy_exception(error));
                        break;
                    default:
                        BOOST_ASSERT(false && "Bug, report this to the developers!");
                }
            }
        }

        bool follow_redirect_;
        resolver_type & resolver_;
        optional<string_type> certificate_filename_, verify_path_;
        resolve_function resolve_;
        boost::shared_ptr<boost::asio::ssl::context> context_;
        boost::shared_ptr<boost::asio::ssl::stream<boost::asio::ip::tcp::socket> > socket_;
        boost::asio::io_service::strand request_strand_;
        boost::asio::streambuf command_streambuf;
        string_type method;
    };
    
} // namespace impl

} // namespace http

} // namespace network

} // namespace boost

#endif // BOOST_NETWORK_PROTOCOL_HTTP_IMPL_HTTPS_ASYNC_CONNECTION_HPP_20100601
