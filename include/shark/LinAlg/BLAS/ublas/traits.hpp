//
//  Copyright (c) 2000-2002
//  Joerg Walter, Mathias Koch
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  GeNeSys mbH & Co. KG in producing this work.
//

#ifndef _BOOST_UBLAS_TRAITS_
#define _BOOST_UBLAS_TRAITS_

#include <iterator>
#include <complex>
#include <boost/config/no_tr1/cmath.hpp>

#include <shark/LinAlg/BLAS/ublas/detail/config.hpp>
#include <shark/LinAlg/BLAS/ublas/detail/iterator.hpp>
#include <shark/LinAlg/BLAS/ublas/detail/returntype_deduction.hpp>

#include <boost/type_traits.hpp>
#include <complex>
#include <boost/typeof/typeof.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_float.hpp>
#include <boost/type_traits/is_integral.hpp>
#include <boost/mpl/and.hpp>

namespace shark {
namespace blas {

// Use Joel de Guzman's return type deduction
// uBLAS assumes a common return type for all binary arithmetic operators
template<class X, class Y>
struct promote_traits {
	typedef type_deduction_detail::base_result_of<X, Y> base_type;
	static typename base_type::x_type x;
	static typename base_type::y_type y;
	static const std::size_t size = sizeof(
	        type_deduction_detail::test<
	        typename base_type::x_type
	        , typename base_type::y_type
	        >(x + y)     // Use x+y to stand of all the arithmetic actions
	        );

	static const std::size_t index = (size / sizeof(char)) - 1;
	typedef typename boost::mpl::at_c<
	typename base_type::types, index>::type id;
	typedef typename id::type promote_type;
};

// special case for bools. b1+b2 creates a boolean return type - which does not make sense
// for example when summing bools! therefore we use a signed int type
template<>
struct promote_traits<bool, bool> {
	typedef int promote_type;
};

template<typename R, typename I>
typename boost::enable_if<
boost::mpl::and_<
boost::is_float<R>,
      boost::is_integral<I>
      >,
std::complex<R> >::type inline operator+ (I in1, std::complex<R> const &in2) {
	return R(in1) + in2;
}

template<typename R, typename I>
typename boost::enable_if<
boost::mpl::and_<
boost::is_float<R>,
      boost::is_integral<I>
      >,
std::complex<R> >::type inline operator+ (std::complex<R> const &in1, I in2) {
	return in1 + R(in2);
}

template<typename R, typename I>
typename boost::enable_if<
boost::mpl::and_<
boost::is_float<R>,
      boost::is_integral<I>
      >,
std::complex<R> >::type inline operator- (I in1, std::complex<R> const &in2) {
	return R(in1) - in2;
}

template<typename R, typename I>
typename boost::enable_if<
boost::mpl::and_<
boost::is_float<R>,
      boost::is_integral<I>
      >,
std::complex<R> >::type inline operator- (std::complex<R> const &in1, I in2) {
	return in1 - R(in2);
}

template<typename R, typename I>
typename boost::enable_if<
boost::mpl::and_<
boost::is_float<R>,
      boost::is_integral<I>
      >,
std::complex<R> >::type inline operator* (I in1, std::complex<R> const &in2) {
	return R(in1) * in2;
}

template<typename R, typename I>
typename boost::enable_if<
boost::mpl::and_<
boost::is_float<R>,
      boost::is_integral<I>
      >,
std::complex<R> >::type inline operator* (std::complex<R> const &in1, I in2) {
	return in1 * R(in2);
}

template<typename R, typename I>
typename boost::enable_if<
boost::mpl::and_<
boost::is_float<R>,
      boost::is_integral<I>
      >,
std::complex<R> >::type inline operator/ (I in1, std::complex<R> const &in2) {
	return R(in1) / in2;
}

template<typename R, typename I>
typename boost::enable_if<
boost::mpl::and_<
boost::is_float<R>,
      boost::is_integral<I>
      >,
std::complex<R> >::type inline operator/ (std::complex<R> const &in1, I in2) {
	return in1 / R(in2);
}


template<class T>
struct real_traits{
	typedef T type;
};

template<class T>
struct real_traits<std::complex<T> >{
	typedef T type;
};

#ifdef BOOST_UBLAS_USE_INTERVAL
// Define scalar interval type traits
template<class T>
struct real_traits<boost::numeric::interval<T> >{
	typedef typename real_traits<T>::type type;
};
#endif


// Storage tags -- hierarchical definition of storage characteristics

struct unknown_storage_tag {};
struct sparse_proxy_tag: public unknown_storage_tag {};
struct sparse_tag: public sparse_proxy_tag {};
struct packed_proxy_tag: public sparse_proxy_tag {};
struct packed_tag: public packed_proxy_tag {};
struct dense_proxy_tag: public packed_proxy_tag {};
struct dense_tag: public dense_proxy_tag {};

template<class S1, class S2>
struct storage_restrict_traits {
	typedef S1 storage_category;
};

template<>
struct storage_restrict_traits<sparse_tag, dense_proxy_tag> {
	typedef sparse_proxy_tag storage_category;
};
template<>
struct storage_restrict_traits<sparse_tag, packed_proxy_tag> {
	typedef sparse_proxy_tag storage_category;
};
template<>
struct storage_restrict_traits<sparse_tag, sparse_proxy_tag> {
	typedef sparse_proxy_tag storage_category;
};

template<>
struct storage_restrict_traits<packed_tag, dense_proxy_tag> {
	typedef packed_proxy_tag storage_category;
};
template<>
struct storage_restrict_traits<packed_tag, packed_proxy_tag> {
	typedef packed_proxy_tag storage_category;
};
template<>
struct storage_restrict_traits<packed_tag, sparse_proxy_tag> {
	typedef sparse_proxy_tag storage_category;
};

template<>
struct storage_restrict_traits<packed_proxy_tag, sparse_proxy_tag> {
	typedef sparse_proxy_tag storage_category;
};

template<>
struct storage_restrict_traits<dense_tag, dense_proxy_tag> {
	typedef dense_proxy_tag storage_category;
};
template<>
struct storage_restrict_traits<dense_tag, packed_proxy_tag> {
	typedef packed_proxy_tag storage_category;
};
template<>
struct storage_restrict_traits<dense_tag, sparse_proxy_tag> {
	typedef sparse_proxy_tag storage_category;
};

template<>
struct storage_restrict_traits<dense_proxy_tag, packed_proxy_tag> {
	typedef packed_proxy_tag storage_category;
};
template<>
struct storage_restrict_traits<dense_proxy_tag, sparse_proxy_tag> {
	typedef sparse_proxy_tag storage_category;
};


// Iterator tags -- hierarchical definition of storage characteristics

struct sparse_bidirectional_iterator_tag : public std::bidirectional_iterator_tag {};
struct packed_random_access_iterator_tag : public std::random_access_iterator_tag {};
struct dense_random_access_iterator_tag : public packed_random_access_iterator_tag {};

// Thanks to Kresimir Fresl for convincing Comeau with iterator_base_traits ;-)
template<class IC>
struct iterator_base_traits {};

template<>
struct iterator_base_traits<std::forward_iterator_tag> {
	template<class I, class T>
	struct iterator_base {
		typedef forward_iterator_base<std::forward_iterator_tag, I, T> type;
	};
};

template<>
struct iterator_base_traits<std::bidirectional_iterator_tag> {
	template<class I, class T>
	struct iterator_base {
		typedef bidirectional_iterator_base<std::bidirectional_iterator_tag, I, T> type;
	};
};
template<>
struct iterator_base_traits<sparse_bidirectional_iterator_tag> {
	template<class I, class T>
	struct iterator_base {
		typedef bidirectional_iterator_base<sparse_bidirectional_iterator_tag, I, T> type;
	};
};

template<>
struct iterator_base_traits<std::random_access_iterator_tag> {
	template<class I, class T>
	struct iterator_base {
		typedef random_access_iterator_base<std::random_access_iterator_tag, I, T> type;
	};
};
template<>
struct iterator_base_traits<packed_random_access_iterator_tag> {
	template<class I, class T>
	struct iterator_base {
		typedef random_access_iterator_base<packed_random_access_iterator_tag, I, T> type;
	};
};
template<>
struct iterator_base_traits<dense_random_access_iterator_tag> {
	template<class I, class T>
	struct iterator_base {
		typedef random_access_iterator_base<dense_random_access_iterator_tag, I, T> type;
	};
};

template<class I1, class I2>
struct iterator_restrict_traits {
	typedef I1 iterator_category;
};

template<>
struct iterator_restrict_traits<packed_random_access_iterator_tag, sparse_bidirectional_iterator_tag> {
	typedef sparse_bidirectional_iterator_tag iterator_category;
};
template<>
struct iterator_restrict_traits<sparse_bidirectional_iterator_tag, packed_random_access_iterator_tag> {
	typedef sparse_bidirectional_iterator_tag iterator_category;
};

template<>
struct iterator_restrict_traits<dense_random_access_iterator_tag, sparse_bidirectional_iterator_tag> {
	typedef sparse_bidirectional_iterator_tag iterator_category;
};
template<>
struct iterator_restrict_traits<sparse_bidirectional_iterator_tag, dense_random_access_iterator_tag> {
	typedef sparse_bidirectional_iterator_tag iterator_category;
};

template<>
struct iterator_restrict_traits<dense_random_access_iterator_tag, packed_random_access_iterator_tag> {
	typedef packed_random_access_iterator_tag iterator_category;
};
template<>
struct iterator_restrict_traits<packed_random_access_iterator_tag, dense_random_access_iterator_tag> {
	typedef packed_random_access_iterator_tag iterator_category;
};

template<class I>

void increment(I &it, const I &it_end, typename I::difference_type compare, packed_random_access_iterator_tag) {
	it += (std::min)(compare, it_end - it);
}
template<class I>

void increment(I &it, const I &/* it_end */, typename I::difference_type /* compare */, sparse_bidirectional_iterator_tag) {
	++ it;
}
template<class I>

void increment(I &it, const I &it_end, typename I::difference_type compare) {
	increment(it, it_end, compare, typename I::iterator_category());
}

template<class I>

void increment(I &it, const I &it_end) {
	it = it_end;
}


}
}

#endif
