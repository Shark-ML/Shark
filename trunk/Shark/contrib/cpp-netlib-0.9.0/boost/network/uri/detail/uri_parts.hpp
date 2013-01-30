#ifndef BOOST_NETWORK_URL_DETAIL_URL_PARTS_HPP_
#define BOOST_NETWORK_URL_DETAIL_URL_PARTS_HPP_

// Copyright 2009 Dean Michael Berris, Jeroen Habraken.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt of copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include <boost/cstdint.hpp>
#include <boost/fusion/tuple.hpp>
#include <boost/optional.hpp>

#include <boost/mpl/if.hpp>
#include <boost/network/traits/string.hpp>

namespace boost { namespace network { namespace uri {

namespace detail {
    
    struct uri_parts_default_base {
        typedef std::string string_type;
        string_type scheme;
        optional<string_type> user_info;
        optional<string_type> host;
        optional<boost::uint16_t> port;
        string_type path;
        optional<string_type> query;
        optional<string_type> fragment;
        
        uri_parts_default_base(uri_parts_default_base const & other)
        : scheme(other.scheme)
        , user_info(other.user_info)
        , host(other.host)
        , port(other.port)
        , path(other.path)
        , query(other.query)
        , fragment(other.fragment)
        {}
        
        uri_parts_default_base()
        {}
        
        uri_parts_default_base & operator=(uri_parts_default_base rhs)
        {
            rhs.swap(*this);
            return *this;
        }
        
        void swap(uri_parts_default_base & rhs)
        {
            std::swap(scheme, rhs.scheme);
            std::swap(user_info, rhs.user_info);
            std::swap(host, rhs.host);
            std::swap(port, rhs.port);
            std::swap(path, rhs.path);
            std::swap(query, rhs.query);
            std::swap(fragment, rhs.fragment);
        }
    };

    struct uri_parts_wide_base {
        typedef std::wstring string_type;
        string_type scheme;
        optional<string_type> user_info;
        optional<string_type> host;
        optional<boost::uint16_t> port;
        string_type path;
        optional<string_type> query;
        optional<string_type> fragment;
        
        uri_parts_wide_base(uri_parts_wide_base const & other)
        : scheme(other.scheme) 
        , user_info(other.user_info)
        , host(other.host)
        , port(other.port)
        , path(other.path)
        , query(other.query)
        , fragment(other.fragment)
        {}
        
        uri_parts_wide_base()
        {}
        
        uri_parts_wide_base & operator=(uri_parts_wide_base rhs)
        {
            rhs.swap(*this);
            return *this;
        }
        
        void swap(uri_parts_wide_base & rhs)
        {
            std::swap(this->scheme, rhs.scheme);
            std::swap(this->user_info, rhs.user_info);
            std::swap(this->host, rhs.host);
            std::swap(this->port, rhs.port);
            std::swap(this->path, rhs.path);
            std::swap(this->query, rhs.query);
            std::swap(this->fragment, rhs.fragment);
        }
    };

template <class Tag>
struct uri_parts :
    mpl::if_<
        is_default_string<Tag>
        , uri_parts_default_base
        , uri_parts_wide_base
    >::type
{
    typedef typename mpl::if_<
        is_default_string<Tag>
        , uri_parts_default_base
        , uri_parts_wide_base
        >::type base_type;
    uri_parts() : base_type() {}
    uri_parts(uri_parts const & other)
    : base_type(other)
    {}
    uri_parts & operator=(uri_parts rhs)
    {
        swap(*this, rhs);
        return *this;
    }
};

template <class Tag>
inline void swap(uri_parts<Tag> & l, uri_parts<Tag> & r) {
    using std::swap;

    swap(l.scheme, r.scheme);
    swap(l.user_info, r.user_info);
    swap(l.host, r.host);
    swap(l.port, r.port);
    swap(l.path, r.path);
    swap(l.query, r.query);
    swap(l.fragment, r.fragment);
}

template <class Tag>
inline
bool operator==(uri_parts<Tag> const & l, uri_parts<Tag> const & r) {
    return (l.scheme == r.scheme) &&
        (l.user_info == r.user_info) &&
        (l.host == r.host) &&
        (l.port == r.port) &&
        (l.path == r.path) &&
        (l.query == r.query) &&
        (l.fragment == r.fragment);
}

template <class Tag>
inline
bool operator!=(uri_parts<Tag> const & l, uri_parts<Tag> const & r) {
    return (l.scheme != r.scheme) &&
        (l.user_info != r.user_info) &&
        (l.host != r.host) &&
        (l.port != r.port) &&
        (l.path != r.path) &&
        (l.query != r.query) &&
        (l.fragment != r.fragment);
}

} // namespace detail
} // namespace uri
} // namespace network
} // namespace boost

#endif

