#ifndef BOOST_NETWORK_URL_DETAIL_PARSE_URL_HPP_
#define BOOST_NETWORK_URL_DETAIL_PARSE_URL_HPP_

// Copyright 2009 Dean Michael Berris, Jeroen Habraken.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include <boost/network/uri/detail/uri_parts.hpp>
#include <boost/network/support/is_default_string.hpp>
#include <boost/range/iterator_range.hpp>
#ifdef BOOST_NETWORK_NO_LIB
#include <boost/network/uri/detail/impl/parse_uri.ipp>
#endif

namespace boost { namespace network { namespace uri { namespace detail {

template <class Tag>
inline bool parse_specific(uri_parts<Tag> & parts) {
    return true;
}

#ifndef BOOST_NETWORK_NO_LIB
extern bool parse_uri_impl(boost::iterator_range<std::string::const_iterator> & range, uri_parts_default_base & parts, tags::default_string);
extern bool parse_uri_impl(boost::iterator_range<std::wstring::const_iterator> & range, uri_parts_wide_base & parts, tags::default_wstring);
#endif

template <class Tag>
struct unsupported_tag;

template <class Range, class Tag>
inline bool parse_uri(Range & range, uri_parts<Tag> & parts) {
    typedef typename range_iterator<Range const>::type iterator;
    boost::iterator_range<iterator> local_range = boost::make_iterator_range(range);
    return parse_uri_impl(local_range, parts,
        typename mpl::if_<
            is_default_string<Tag>,
            tags::default_string,
            typename mpl::if_<
                is_default_wstring<Tag>,
                tags::default_wstring,
                unsupported_tag<Tag>
                >::type
        >::type());
}

} // namespace detail
} // namespace uri
} // namespace network
} // namespace boost

#endif

