#ifndef BOOST_NETWORK_VERSION_HPP_20091214
#define BOOST_NETWORK_VERSION_HPP_20091214

//          Copyright Dean Michael Berris 2009.
//          Copyright Glyn Matthews 2010.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <boost/config.hpp>

#define BOOST_NETLIB_VERSION_MAJOR 0
#define BOOST_NETLIB_VERSION_MINOR 9
#define BOOST_NETLIB_VERSION_INCREMENT 0

#ifndef BOOST_NETLIB_VERSION
# define BOOST_NETLIB_VERSION \
    BOOST_STRINGIZE(BOOST_NETLIB_VERSION_MAJOR) "." \
    BOOST_STRINGIZE(BOOST_NETLIB_VERSION_MINOR) "." \
    BOOST_STRINGIZE(BOOST_NETLIB_VERSION_INCREMENT)
#endif // BOOST_NETLIB_VERSION

#endif // BOOST_NETWORK_VERSION_HPP_20091214
