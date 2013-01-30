#ifndef BOOST_NETWORK_PROTOCOL_HTTP_IMPL_HTTP_ASYNC_PROTOCOL_HANDLER_HPP_20101015
#define BOOST_NETWORK_PROTOCOL_HTTP_IMPL_HTTP_ASYNC_PROTOCOL_HANDLER_HPP_20101015

// Copyright 2010 (C) Dean Michael Berris
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include <boost/network/protocol/http/algorithms/linearize.hpp>

namespace boost { namespace network { namespace http { namespace impl {

    template <class Tag, unsigned version_major, unsigned version_minor>
    struct http_async_protocol_handler {
    protected:

        typedef typename string<Tag>::type string_type;

        template <class ResponseType>
        void init_response(ResponseType & response_, bool get_body) {
            boost::shared_future<string_type> source_future(source_promise.get_future());
            source(response_, source_future);
            boost::shared_future<string_type> destination_future(destination_promise.get_future());
            destination(response_, destination_future);
            boost::shared_future<typename headers_container<Tag>::type> headers_future(headers_promise.get_future());
            headers(response_, headers_future);
            boost::shared_future<string_type> body_future(body_promise.get_future());
            body(response_, body_future);
            boost::shared_future<string_type> version_future(version_promise.get_future());
            version(response_, version_future);
            boost::shared_future<boost::uint16_t> status_future(status_promise.get_future());
            status(response_, status_future);
            boost::shared_future<string_type> status_message_future(status_message_promise.get_future());
            status_message(response_, status_message_future);
        }

        struct to_http_headers {
            typedef typename string<Tag>::type string_type;
            template <class U>
            string_type const operator() (U const & pair) const {
                typedef typename ostringstream<Tag>::type ostringstream_type;
                typedef constants<Tag> constants;
                ostringstream_type header_line;
                header_line << pair.first
                    << constants::colon()
                    << constants::space()
                    << pair.second
                    << constants::crlf();
                return header_line.str();
            }
        };

        template <class Socket, class Callback>
        logic::tribool parse_version(Socket & socket_, Callback callback) {
            logic::tribool parsed_ok;
            typename boost::iterator_range<typename buffer_type::const_iterator> result_range;
            fusion::tie(parsed_ok, result_range) = response_parser_.parse_until(
                response_parser_type::http_version_done,
                part);
            if (parsed_ok == true) {
                string_type version;
                std::swap(version, partial_parsed);
                version.append(boost::begin(result_range), boost::end(result_range));
                algorithm::trim(version);
                version_promise.set_value(version);
                part_begin = boost::end(result_range);
            } else if (parsed_ok == false) {
                std::runtime_error error("Invalid Version Part.");
                version_promise.set_exception(boost::copy_exception(error));
                status_promise.set_exception(boost::copy_exception(error));
                status_message_promise.set_exception(boost::copy_exception(error));
                headers_promise.set_exception(boost::copy_exception(error));
                source_promise.set_exception(boost::copy_exception(error));
                destination_promise.set_exception(boost::copy_exception(error));
                body_promise.set_exception(boost::copy_exception(error));
            } else {
                partial_parsed.append(
                    boost::begin(result_range),
                    boost::end(result_range)
                    );
                part_begin = part.begin();
                boost::asio::async_read(
                    socket_,
                    boost::asio::mutable_buffers_1(part.c_array(), part.size()),
                    callback
                    );
            }
            return parsed_ok;
        }

        template <class Socket, class Callback>
        logic::tribool parse_status(Socket & socket_, Callback callback) {
            logic::tribool parsed_ok;
            typename buffer_type::const_iterator part_end = part.end();
            typename boost::iterator_range<typename buffer_type::const_iterator> result_range,
                input_range = boost::make_iterator_range(part_begin, part_end);
            fusion::tie(parsed_ok, result_range) = response_parser_.parse_until(
                response_parser_type::http_status_done,
                input_range);
            if (parsed_ok == true) {
                string_type status;
                std::swap(status, partial_parsed);
                status.append(boost::begin(result_range), boost::end(result_range));
                trim(status);
                boost::uint16_t status_int = lexical_cast<boost::uint16_t>(status);
                status_promise.set_value(status_int);
                part_begin = boost::end(result_range);
            } else if (parsed_ok == false) {
                std::runtime_error error("Invalid status part.");
                status_promise.set_exception(boost::copy_exception(error));
                status_message_promise.set_exception(boost::copy_exception(error));
                headers_promise.set_exception(boost::copy_exception(error));
                source_promise.set_exception(boost::copy_exception(error));
                destination_promise.set_exception(boost::copy_exception(error));
                body_promise.set_exception(boost::copy_exception(error));
            } else {
                partial_parsed.append(
                    boost::begin(result_range), 
                    boost::end(result_range)
                    );
                part_begin = part.begin();
                boost::asio::async_read(socket_, 
                    boost::asio::mutable_buffers_1(part.c_array(), part.size()),
                    callback
                    );
            }
            return parsed_ok;
        }

        template <class Socket, class Callback>
        logic::tribool parse_status_message(Socket & socket_, Callback callback) {
            logic::tribool parsed_ok;
            typename buffer_type::const_iterator part_end = part.end();
            typename boost::iterator_range<typename buffer_type::const_iterator> result_range,
                input_range = boost::make_iterator_range(part_begin, part_end);
            fusion::tie(parsed_ok, result_range) = response_parser_.parse_until(
                response_parser_type::http_status_message_done,
                input_range);
            if (parsed_ok == true) {
                string_type status_message;
                std::swap(status_message, partial_parsed);
                status_message.append(boost::begin(result_range), boost::end(result_range));
                algorithm::trim(status_message);
                status_message_promise.set_value(status_message);
                part_begin = boost::end(result_range);
            } else if (parsed_ok == false) {
                std::runtime_error error("Invalid status message part.");
                status_message_promise.set_exception(boost::copy_exception(error));
                headers_promise.set_exception(boost::copy_exception(error));
                source_promise.set_exception(boost::copy_exception(error));
                destination_promise.set_exception(boost::copy_exception(error));
                body_promise.set_exception(boost::copy_exception(error));
            } else {
                partial_parsed.append(
                    boost::begin(result_range), 
                    boost::end(result_range));
                part_begin = part.begin();
                boost::asio::async_read(
                    socket_,
                    boost::asio::mutable_buffers_1(part.c_array(), part.size()),
                    callback
                    );
            }
            return parsed_ok;
        }

        void parse_headers_real(string_type & headers_part) {
            typename boost::iterator_range<typename string_type::const_iterator> 
                input_range = boost::make_iterator_range(headers_part)
                , result_range;
            logic::tribool parsed_ok;
            response_parser_type headers_parser(response_parser_type::http_header_line_done);
            typename headers_container<Tag>::type headers;
            std::pair<string_type,string_type> header_pair;
            while (!boost::empty(input_range)) {
                fusion::tie(parsed_ok, result_range) = headers_parser.parse_until(
                    response_parser_type::http_header_colon
                    , input_range);
                if (headers_parser.state() != response_parser_type::http_header_colon) break;
                header_pair.first = string_type(
                    boost::begin(result_range), 
                    boost::end(result_range));
                input_range.advance_begin(boost::distance(result_range));
                fusion::tie(parsed_ok, result_range) = headers_parser.parse_until(
                    response_parser_type::http_header_line_done
                    , input_range);
                header_pair.second = string_type(
                    boost::begin(result_range), 
                    boost::end(result_range));
                input_range.advance_begin(boost::distance(result_range));

                trim(header_pair.first);
                if (header_pair.first.size() > 1) {
                    header_pair.first.erase(
                        header_pair.first.size() - 1
                    );
                }
                trim(header_pair.second);
                headers.insert(header_pair);
            }
            headers_promise.set_value(headers);
        }

        template <class Socket, class Callback>
        fusion::tuple<logic::tribool, size_t> parse_headers(Socket & socket_, Callback callback) {
            logic::tribool parsed_ok;
            typename buffer_type::const_iterator part_end = part.end();
            typename boost::iterator_range<typename buffer_type::const_iterator> result_range,
                input_range = boost::make_iterator_range(part_begin, part_end);
            fusion::tie(parsed_ok, result_range) = response_parser_.parse_until(
                response_parser_type::http_headers_done,
                input_range);
            if (parsed_ok == true) {
                string_type headers_string;
                std::swap(headers_string, partial_parsed);
                headers_string.append(boost::begin(result_range), boost::end(result_range));
                part_begin = boost::end(result_range);
                this->parse_headers_real(headers_string);
            } else if (parsed_ok == false) {
                std::runtime_error error("Invalid header part.");
                headers_promise.set_exception(boost::copy_exception(error));
                body_promise.set_exception(boost::copy_exception(error));
                source_promise.set_exception(boost::copy_exception(error));
                destination_promise.set_exception(boost::copy_exception(error));
            } else {
                partial_parsed.append(boost::begin(result_range), boost::end(result_range));
                part_begin = part.begin();
                boost::asio::async_read(
                    socket_,
                    boost::asio::mutable_buffers_1(part.c_array(), part.size()),
                    callback
                    );
            }
            return fusion::make_tuple(
                parsed_ok, 
                std::distance(
                    boost::end(result_range)
                    , part_end
                    )
                );
        }

        template <class Socket, class Callback>
        void parse_body(Socket & socket_, Callback callback, size_t bytes) {
            partial_parsed.append(part_begin, bytes);
            part_begin = part.begin();
            boost::asio::async_read(
                socket_, 
                boost::asio::mutable_buffers_1(part.c_array(), part.size()),
                callback
                );
        }

        typedef response_parser<Tag> response_parser_type;
        typedef boost::array<typename char_<Tag>::type, 1024> buffer_type;

        response_parser_type response_parser_;
        boost::promise<string_type> version_promise;
        boost::promise<boost::uint16_t> status_promise;
        boost::promise<string_type> status_message_promise;
        boost::promise<typename headers_container<Tag>::type> headers_promise;
        boost::promise<string_type> source_promise;
        boost::promise<string_type> destination_promise;
        boost::promise<string_type> body_promise;
        buffer_type part;
        typename buffer_type::const_iterator part_begin;
        string_type partial_parsed;
    };

    
} /* impl */
    
} /* http */

} /* network */
    
} /* boost */

#endif /* BOOST_NETWORK_PROTOCOL_HTTP_IMPL_HTTP_ASYNC_PROTOCOL_HANDLER_HPP_20101015 */
