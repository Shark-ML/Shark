#ifndef BOOST_NETWORK_URL_HTTP_URL_HPP_
#define BOOST_NETWORK_URL_HTTP_URL_HPP_

// Copyright 2009 Dean Michael Berris, Jeroen Habraken.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include <boost/cstdint.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/network/protocol/http/tags.hpp>
#include <boost/network/traits/string.hpp>
#include <boost/network/uri/basic_uri.hpp>
#include <boost/network/uri/http/detail/parse_specific.hpp>


namespace boost { namespace network { namespace uri {

template <>
class basic_uri<http::tags::http_default_8bit_tcp_resolve>
    : public uri_base<http::tags::http_default_8bit_tcp_resolve> {

public:
    basic_uri() : uri_base<http::tags::http_default_8bit_tcp_resolve>() {}
    basic_uri(uri_base<http::tags::http_default_8bit_tcp_resolve>::string_type const & uri) : uri_base<http::tags::http_default_8bit_tcp_resolve>(uri) {}

    boost::optional<boost::uint16_t> port() const {
        return parts_.port;
        return parts_.port ? *(parts_.port) :
            (boost::iequals(parts_.scheme, string_type("https")) ? 443 : 80);
    }

    string_type path() const {
        return (parts_.path == "") ? string_type("/") : parts_.path;
    }
};

} // namespace uri
} // namespace network
} // namespace boost

#endif

