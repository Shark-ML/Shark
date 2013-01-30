#ifndef BOOST_NETWORK_URL_BASIC_URL_
#define BOOST_NETWORK_URL_BASIC_URL_

// Copyright 2009 Dean Michael Berris, Jeroen Habraken.
// Copyright 2010 Glyn Matthews.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)


#include <boost/network/uri/basic_uri_fwd.hpp>
#include <boost/network/uri/detail/parse_uri.hpp>
#include <boost/network/constants.hpp>
#include <boost/algorithm/string.hpp>

namespace boost { namespace network { namespace uri {

template <class Tag>
struct uri_base {
    typedef typename string<Tag>::type string_type;

    uri_base(string_type const & uri = string_type())
        :
        raw_(uri),
        parts_(),
        valid_(false)
    {
        valid_ = parse_uri(raw_, parts_);
    }

    uri_base(const uri_base & other)
        :
        raw_(other.raw_),
        parts_(other.parts_),
        valid_(other.valid_)
    { }

    uri_base & operator=(uri_base other) {
        other.swap(*this);
        return *this;
    }

    uri_base & operator=(string_type const & uri) {
        return *this = uri_base(uri);
    }

    void swap(uri_base & other) {
        using std::swap;

        swap(other.raw_, raw_);
        swap(other.parts_, parts_);
        swap(other.valid_, valid_);
    }

    string_type scheme() const {
        return parts_.scheme;
    }

    string_type user_info() const {
        return parts_.user_info ? *parts_.user_info : string_type();
    }

    string_type host() const {
        return parts_.host ? *parts_.host : string_type();
    }

    uint16_t port() const {
        return parts_.port ? *parts_.port : 0;
    }

    string_type path() const {
        return parts_.path;
    }

    string_type query() const {
        return parts_.query ? *parts_.query : string_type();
    }

    string_type fragment() const {
        return parts_.fragment ? *parts_.fragment : string_type();
    }

    string_type raw() const {
        return raw_;
    }

    string_type string() const {
        return raw_;
    }

    bool valid() const {
        return is_valid();
    }

    bool is_valid() const {
        return valid_;
    }

    bool operator == (uri_base const & other) const {
        return (raw_ == other.raw_) && (parts_ == other.parts_) && (valid_ == other.valid_);
    }

    bool operator != (uri_base const & other) const {
        return !(*this == other);
    }

protected:

    string_type raw_;
    detail::uri_parts<Tag> parts_;
    bool valid_;
};

template <class Tag>
class basic_uri : public uri_base<Tag> {

public:

    basic_uri() : uri_base<Tag>() {}
    basic_uri(typename uri_base<Tag>::string_type const & uri) : uri_base<Tag>(uri) {}
    basic_uri(basic_uri const & other) : uri_base<Tag>(other) {}

    basic_uri & operator= (basic_uri rhs) {
        rhs.swap(*this);
        return *this;
    }

    void swap(basic_uri & other) {
        uri_base<Tag>::swap(other);
    }

    using uri_base<Tag>::operator==;
    using uri_base<Tag>::operator!=;

};

template <class Tag>
inline
void swap(basic_uri<Tag> & left, basic_uri<Tag> & right) {
    right.swap(left);
}

template <class Tag>
inline
typename string<Tag>::type
scheme(basic_uri<Tag> const & uri) {
    return uri.scheme();
}

template <class Tag>
inline
typename string<Tag>::type
user_info(basic_uri<Tag> const & uri) {
    return uri.user_info();
}

template <class Tag>
inline
typename string<Tag>::type
host(basic_uri<Tag> const & uri) {
    return uri.host();
}

template <class Tag>
struct port_wrapper {
    basic_uri<Tag> const & uri;
    explicit port_wrapper(basic_uri<Tag> const & uri)
            : uri(uri)
    {}

    operator boost::optional<boost::uint16_t>() const {
        return uri.port();
    }

    operator boost::uint16_t() const {
        boost::optional<boost::uint16_t> const & port_ = uri.port();
        typedef typename string<Tag>::type string_type;
        typedef constants<Tag> consts;
        if (port_) return *port_;
        return boost::iequals(uri.scheme(), string_type(consts::https())) ? 443 : 80;
    }
};

template <class Tag>
inline
port_wrapper<Tag> const
port(basic_uri<Tag> const & uri) {
    return port_wrapper<Tag>(uri);
}

template <class Tag>
inline
typename string<Tag>::type
path(basic_uri<Tag> const & uri) {
    return uri.path();
}

template <class Tag>
inline
typename string<Tag>::type
query(basic_uri<Tag> const & uri) {
    return uri.query();
}

template <class Tag>
inline
typename string<Tag>::type
fragment(basic_uri<Tag> const & uri) {
    return uri.fragment();
}

template <class Tag>
inline
bool
valid(basic_uri<Tag> const & uri) {
    return uri.valid();
}

template <class Tag>
inline
bool
is_valid(basic_uri<Tag> const & uri) {
    return uri.is_valid();
}
} // namespace uri
} // namespace network
} // namespace boost

#ifdef BOOST_NETWORK_DEBUG
// Check that the URI concept is met by the basic_uri type.
#include <boost/network/uri/uri_concept.hpp>
#include <boost/network/tags.hpp>
namespace boost { namespace network { namespace uri {
BOOST_CONCEPT_ASSERT((URI<basic_uri<boost::network::tags::default_string> >));
BOOST_CONCEPT_ASSERT((URI<basic_uri<boost::network::tags::default_wstring> >));
} // namespace uri
} // namespace network
} // namespace boost
#endif // BOOST_NETWORK_DEBUG

#endif

