#ifndef BOOST_NETWORK_URL_HTTP_DETAIL_PARSE_SPECIFIC_HPP_
#define BOOST_NETWORK_URL_HTTP_DETAIL_PARSE_SPECIFIC_HPP_

// Copyright 2009 Dean Michael Berris, Jeroen Habraken.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt of copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include <boost/network/protocol/http/tags.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/network/traits/string.hpp>
#include <boost/network/uri/detail/parse_uri.hpp>
#include <ciso646>

namespace boost { namespace network { namespace uri { 

    namespace detail {

        template <>
            inline bool parse_specific<http::tags::http_default_8bit_tcp_resolve>
            (uri_parts<http::tags::http_default_8bit_tcp_resolve> & parts)
            {
                if ((parts.scheme.size() < 4) || (parts.scheme.size() > 5))
                    return false;

                if (parts.scheme.size() == 4) {
                    if (not boost::iequals(parts.scheme, "http"))
                        return false;
                } else { // size is 5
                    if (not boost::iequals(parts.scheme, "https"))
                        return false;
                }

                if ((not parts.host) || parts.host->empty())
                    return false;

                return true;
            }

    } // namespace detail

} // namespace uri

} // namespace network

} // namespace boost

#endif

