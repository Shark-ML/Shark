#ifndef BOOST_NETWORK_PROTOCOL_HTTP_ALGORITHMS_LINEARIZE_HPP_20101028
#define BOOST_NETWORK_PROTOCOL_HTTP_ALGORITHMS_LINEARIZE_HPP_20101028

// Copyright 2010 Dean Michael Berris.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include <boost/network/traits/string.hpp>
#include <boost/network/protocol/http/message/header/name.hpp>
#include <boost/network/protocol/http/message/header/value.hpp>
#include <boost/network/protocol/http/message/header_concept.hpp>
#include <boost/network/protocol/http/request_concept.hpp>
#include <boost/network/constants.hpp>
#include <boost/concept/requires.hpp>
#include <boost/optional.hpp>
#include <boost/range/algorithm/copy.hpp>

namespace boost { namespace network { namespace http {

    template <class Tag>
    struct linearize_header {
        typedef typename string<Tag>::type string_type;

        template <class Arguments>
        struct result;

        template <class This, class Arg>
        struct result<This(Arg)> {
            typedef string_type type;
        };

        template <class ValueType>
        BOOST_CONCEPT_REQUIRES(
            ((Header<ValueType>)),
            (string_type)
        ) operator()(ValueType & header) {
            typedef typename ostringstream<Tag>::type output_stream;
            typedef constants<Tag> consts;
            output_stream header_line;
            header_line << name(header) 
                << consts::colon() << consts::space() 
                << value(header) << consts::crlf();
            return header_line.str();
        }
    };

    template <class Request, class OutputIterator>
    BOOST_CONCEPT_REQUIRES(
        ((ClientRequest<Request>)),
        (OutputIterator)
    ) linearize(
        Request const & request, 
        typename Request::string_type const & method,
        unsigned version_major, 
        unsigned version_minor, 
        OutputIterator oi
        ) 
    {
        typedef typename Request::tag Tag;
        typedef constants<Tag> consts;
        typedef typename string<Tag>::type string_type;
        static string_type 
            http_slash = consts::http_slash()
            , accept   = consts::accept()
            , accept_mime = consts::default_accept_mime()
            , accept_encoding = consts::accept_encoding()
            , default_accept_encoding = consts::default_accept_encoding()
            , crlf = consts::crlf()
            , host = consts::host()
            , connection = consts::connection()
            , close = consts::close()
            ;
        boost::copy(method, oi);
        *oi = consts::space_char();
        if (request.path().empty() || request.path()[0] != consts::slash_char()) 
            *oi = consts::slash_char();
        boost::copy(request.path(), oi);
        if (!request.query().empty()) {
            *oi = consts::question_mark_char();
            boost::copy(request.query(), oi);
        }
        if (!request.anchor().empty()) {
            *oi = consts::hash_char();
            boost::copy(request.anchor(), oi);
        }
        *oi = consts::space_char();
        boost::copy(http_slash, oi);
        string_type version_major_str = boost::lexical_cast<string_type>(version_major),
                    version_minor_str = boost::lexical_cast<string_type>(version_minor);
        boost::copy(version_major_str, oi);
        *oi = consts::dot_char();
        boost::copy(version_minor_str, oi);
        boost::copy(crlf, oi);
        boost::copy(host, oi);
        *oi = consts::colon_char();
        *oi = consts::space_char();
        boost::copy(request.host(), oi);
        boost::optional<boost::uint16_t> port_ = port(request);
        if (port_) {
            string_type port_str = boost::lexical_cast<string_type>(*port_);
            *oi = consts::colon_char();
            boost::copy(port_str, oi);
        }
        boost::copy(crlf, oi);
        boost::copy(accept, oi);
        *oi = consts::colon_char();
        *oi = consts::space_char();
        boost::copy(accept_mime, oi);
        boost::copy(crlf, oi);
        if (version_major == 1u && version_minor == 1u) {
            boost::copy(accept_encoding, oi);
            *oi = consts::colon_char();
            *oi = consts::space_char();
            boost::copy(default_accept_encoding, oi);
            boost::copy(crlf, oi);
        }
        typedef typename headers_range<Request>::type headers_range;
        typedef typename range_iterator<headers_range>::type headers_iterator;
        headers_range request_headers = headers(request);
        headers_iterator iterator = boost::begin(request_headers),
                         end = boost::end(request_headers);
        for (; iterator != end; ++iterator) {
            string_type header_name = name(*iterator),
                        header_value = value(*iterator);
            boost::copy(header_name, oi);
            *oi = consts::colon_char();
            *oi = consts::space_char();
            boost::copy(header_value, oi);
            boost::copy(crlf, oi);
        }
        if (!connection_keepalive<Tag>::value) {
            boost::copy(connection, oi);
            *oi = consts::colon_char();
            *oi = consts::space_char();
            boost::copy(close, oi);
            boost::copy(crlf, oi);
        }
        boost::copy(crlf, oi);
        typename body_range<Request>::type body_data = body(request).range();
        return boost::copy(body_data, oi);
    }
    
} /* http */
    
} /* net */
    
} /* boost */

#endif /* BOOST_NETWORK_PROTOCOL_HTTP_ALGORITHMS_LINEARIZE_HPP_20101028 */

