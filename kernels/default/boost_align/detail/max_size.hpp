/*
(c) 2014-2015 Glen Joseph Fernandes
<glenjofe -at- gmail.com>

Distributed under the Boost Software
License, Version 1.0.
http://boost.org/LICENSE_1_0.txt
*/
#ifndef BOOST_ALIGN_DETAIL_MAX_SIZE_HPP
#define BOOST_ALIGN_DETAIL_MAX_SIZE_HPP

#include <type_traits>
#include <cstddef>

namespace boost {
namespace alignment {
namespace detail {

template<std::size_t A, std::size_t B>
struct max_size
    : std::integral_constant<std::size_t, (A > B) ? A : B> { };

} /* .detail */
} /* .alignment */
} /* .boost */

#endif
