#ifndef BOOST_NETWORK_URL_URL_CONCEPT_HPP_
#define BOOST_NETWORK_URL_URL_CONCEPT_HPP_

// Copyright 2009 Dean Michael Berris, Jeroen Habraken.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include <boost/concept_check.hpp>

namespace boost { namespace network { namespace uri {

    template <class U>
        struct URI : DefaultConstructible<U>, EqualityComparable<U> {
            typedef typename U::string_type string_type;

            BOOST_CONCEPT_USAGE(URI)
            {
                U uri_(uri); // copy constructable

                U temp;
                swap(temp, uri_); // swappable

                string_type scheme_ = scheme(uri); // support functions
                string_type user_info_ = user_info(uri);
                string_type host_ = host(uri);
                uint16_t port_ = port(uri);
                port_ = 0;
                string_type path_ = path(uri);
                string_type query_ = query(uri);
                string_type fragment_ = fragment(uri);

                bool valid_ = is_valid(uri);
                valid_ = false;
            }

            private:
            U uri;
        };

} // namespace uri

} // namespace network

} // namespace boost

#endif

