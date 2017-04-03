/*
(c) 2014 Glen Joseph Fernandes
<glenjofe -at- gmail.com>

Distributed under the Boost Software
License, Version 1.0.
http://boost.org/LICENSE_1_0.txt
*/
#ifndef BOOST_ALIGN_DETAIL_IS_ALIGNMENT_CONSTANT_HPP
#define BOOST_ALIGN_DETAIL_IS_ALIGNMENT_CONSTANT_HPP

#include <cstddef>
#include <type_traits>
namespace boost {
namespace alignment {
namespace detail {

template<std::size_t N>
struct is_alignment_constant
    : std::integral_constant<bool, (N > 0) && ((N & (N - 1)) == 0)> { };

} /* .detail */
} /* .alignment */
} /* .boost */

#endif
