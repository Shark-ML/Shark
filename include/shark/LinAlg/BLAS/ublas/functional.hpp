//
//  Copyright (c) 2000-2009
//  Joerg Walter, Mathias Koch, Gunter Winkler
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  GeNeSys mbH & Co. KG in producing this work.
//

#ifndef _BOOST_UBLAS_FUNCTIONAL_
#define _BOOST_UBLAS_FUNCTIONAL_

#include <functional>
#include <boost/math/constants/constants.hpp>

#include <shark/LinAlg/BLAS/ublas/traits.hpp>
#include <shark/LinAlg/BLAS/ublas/detail/definitions.hpp>
#include <shark/Core/Exception.h>


namespace shark {
namespace blas {

namespace detail {
template<class T>
T maxExpInput() {
	return boost::math::constants::ln_two<T>()*std::numeric_limits<T>::max_exponent;
}
/// Minimum value for exp(x) allowed so that it is not 0.
template<class T>
T minExpInput() {
	return boost::math::constants::ln_two<T>()*std::numeric_limits<T>::min_exponent;
}
}

// Scalar functors
template<class T>
struct scalar_identity {
	typedef T argument_type;
	typedef argument_type result_type;
	static const bool zero_identity = true;

	result_type operator()(argument_type x)const {
		return x;
	}
};

template<class T>
struct scalar_negate {
	typedef T argument_type;
	typedef T result_type;
	static const bool zero_identity = true;

	result_type operator()(argument_type x)const {
		return -x;
	}
};

template<class T>
struct scalar_divide {
	typedef T argument_type;
	typedef argument_type result_type;
	static const bool zero_identity = true;

	scalar_divide(T divisor):m_divisor(divisor) {}
	result_type operator()(argument_type x)const {
		return x/m_divisor;
	}
private:
	T m_divisor;
};

template<class T>
struct scalar_multiply1 {
	typedef T argument_type;
	typedef argument_type result_type;
	static const bool zero_identity = true;

	scalar_multiply1(T factor):m_factor(factor) {}
	result_type operator()(argument_type x)const {
		return m_factor * x;
	}
private:
	T m_factor;
};

template<class T>
struct scalar_multiply2 {
	typedef T argument_type;
	typedef argument_type result_type;
	static const bool zero_identity = true;

	scalar_multiply2(T factor):m_factor(factor) {}
	result_type operator()(argument_type x)const {
		return x * m_factor;
	}
private:
	T m_factor;
};

template<class T>
struct scalar_conj {
	typedef T argument_type;
	typedef T result_type;
	static const bool zero_identity = true;

	result_type operator()(argument_type x)const {
		return x;
	}
};
template<class T>
struct scalar_conj<std::complex<T> > {
	typedef std::complex<T> argument_type;
	typedef T result_type;
	static const bool zero_identity = true;

	result_type operator()(argument_type x)const {
		return conj(x);
	}
};

template<class T>
struct scalar_real {
	typedef T argument_type;
	typedef typename real_traits<T>::real_type result_type;
	static const bool zero_identity = true;

	result_type operator()(argument_type x)const {
		return x;
	}
};
template<class T>
struct scalar_real<std::complex<T> > {
	typedef std::complex<T> argument_type;
	typedef typename real_traits<T>::real_type result_type;
	static const bool zero_identity = true;

	result_type operator()(argument_type x)const {
		return x.real();
	}
};

template<class T>
struct scalar_imag {
	typedef T argument_type;
	typedef typename real_traits<T>::real_type result_type;
	static const bool zero_identity = true;

	result_type operator()(argument_type x)const {
		return real_traits<T>::imag(x);
	}
};
template<class T>
struct scalar_imag<std::complex<T> > {
	typedef std::complex<T> argument_type;
	typedef typename real_traits<T>::real_type result_type;
	static const bool zero_identity = true;

	result_type operator()(argument_type x)const {
		return x.imag();
	}
};

template<class T>
struct scalar_abs {
	typedef T argument_type;
	typedef typename real_traits<T>::type result_type;
	static const bool zero_identity = true;

	result_type operator()(argument_type x)const {
		using std::abs;
		return abs(x);
	}
};

template<class T>
struct scalar_sqr{
	typedef T argument_type;
	typedef argument_type result_type;
	static const bool zero_identity = true;

	result_type operator()(argument_type x)const {
		return x*x;
	}
};

template<class T>
struct scalar_sqrt {
	typedef T argument_type;
	typedef argument_type result_type;
	static const bool zero_identity = true;

	result_type operator()(argument_type x)const {
		using std::sqrt;
		return sqrt(x);
	}
};

template<class T>
struct scalar_abs_sqr{
	typedef T argument_type;
	typedef T result_type;
	static const bool zero_identity = true;

	result_type operator()(argument_type x)const {
		using std::abs;
		result_type absolute_value = abs(x);
		return absolute_value*absolute_value;
	}
};

template<class T>
struct scalar_abs_sqr<std::complex<T> >{
	typedef std::complex<T> argument_type;
	typedef typename real_traits<T>::type result_type;
	static const bool zero_identity = true;

	result_type operator()(argument_type x)const{
		using std::abs;
		result_type abs_real = abs(x.real());
		result_type abs_imag = abs(x.imag());
		return abs_real*abs_real+abs_imag*abs_imag;
	}
};

template<class T>
struct scalar_exp {
	typedef T argument_type;
	typedef argument_type result_type;
	static const bool zero_identity = false;

	result_type operator()(argument_type x)const {
		using std::exp;
		return exp(x);
	}
};

template<class T>
struct scalar_log {
	typedef T argument_type;
	typedef argument_type result_type;
	static const bool zero_identity = false;

	result_type operator()(argument_type x)const {
		using std::log;
		return log(x);
	}
};

template<class T, class U>
struct scalar_pow{
	typedef T argument_type;
	typedef argument_type result_type;
	static const bool zero_identity = true;

	scalar_pow(U exponent):m_exponent(exponent) {}
	result_type operator()(argument_type x)const {
		using std::pow;
		return pow(x,m_exponent);
	}
private:
	U m_exponent;
};

template<class T>
struct scalar_tanh{
	typedef T argument_type;
	typedef argument_type result_type;
	static const bool zero_identity = true;

	result_type operator()(argument_type x)const {
		using std::tanh;
		return tanh(x);
	}
};

template<class T>
struct scalar_soft_plus {
	typedef T argument_type;
	typedef argument_type result_type;
	static const bool zero_identity = false;

	result_type operator()(argument_type x)const {
		using std::exp;
		using std::log;
		if (x < detail::minExpInput<argument_type>()) {
			return x;
		}
		if (x > detail::maxExpInput<argument_type>()) {
			return argument_type();
		}
		return log(argument_type(1.0)+exp(x));
	}
};

template<class T>
struct scalar_sigmoid {
	typedef T argument_type;
	typedef argument_type result_type;
	static const bool zero_identity = false;

	result_type operator()(argument_type x)const {
		using std::exp;
		if (x < detail::minExpInput<argument_type>()) {
			return 0;
		}
		if (x > detail::maxExpInput<argument_type>()) {
			return 1;
		}
		return 1.0/(1.0+exp(-x));
	}
};

template<class T>
struct scalar_less_than{
	typedef T argument_type;
	typedef bool result_type;
	static const bool zero_identity = false;

	scalar_less_than(T comparator):m_comparator(comparator) {}
	result_type operator()(argument_type x)const {
		return x < m_comparator;
	}
private:
	T m_comparator;
};

template<class T>
struct scalar_less_equal_than{
	typedef T argument_type;
	typedef bool result_type;
	static const bool zero_identity = false;

	scalar_less_equal_than(T comparator):m_comparator(comparator) {}
	result_type operator()(argument_type x)const {
		return x <= m_comparator;
	}
private:
	T m_comparator;
};

template<class T>
struct scalar_bigger_than{
	typedef T argument_type;
	typedef bool result_type;
	static const bool zero_identity = false;

	scalar_bigger_than(T comparator):m_comparator(comparator) {}
	result_type operator()(argument_type x)const {
		return x > m_comparator;
	}
private:
	T m_comparator;
};

template<class T>
struct scalar_bigger_equal_than{
	typedef T argument_type;
	typedef bool result_type;
	static const bool zero_identity = false;

	scalar_bigger_equal_than(T comparator):m_comparator(comparator) {}
	result_type operator()(argument_type x)const {
		return x > m_comparator;
	}
private:
	T m_comparator;
};

template<class T>
struct scalar_equal{
	typedef T argument_type;
	typedef bool result_type;
	static const bool zero_identity = false;

	scalar_equal(T comparator):m_comparator(comparator) {}
	result_type operator()(argument_type x)const {
		return x == m_comparator;
	}
private:
	T m_comparator;
};

template<class T>
struct scalar_not_equal{
	typedef T argument_type;
	typedef bool result_type;
	static const bool zero_identity = false;

	scalar_not_equal(T comparator):m_comparator(comparator) {}
	result_type operator()(argument_type x)const {
		return x != m_comparator;
	}
private:
	T m_comparator;
};


//////BINARY SCALAR OPRATIONS////////////////
template<class T1,class T2>
struct scalar_binary_plus {
	typedef T1 argument1_type;
	typedef T2 argument2_type;
	typedef typename promote_traits<T1, T2>::promote_type result_type;

	result_type operator()(argument1_type x, argument2_type y)const {
		return x+y;
	}
};
template<class T1,class T2>
struct scalar_binary_minus {
	typedef T1 argument1_type;
	typedef T2 argument2_type;
	typedef typename promote_traits<T1, T2>::promote_type result_type;

	result_type operator()(argument1_type x, argument2_type y)const {
		return x-y;
	}
};

template<class T1,class T2>
struct scalar_binary_multiply {
	typedef T1 argument1_type;
	typedef T2 argument2_type;
	typedef typename promote_traits<T1, T2>::promote_type result_type;

	result_type operator()(argument1_type x, argument2_type y)const {
		return x*y;
	}
};

template<class T1,class T2>
struct scalar_binary_divide {
	typedef T1 argument1_type;
	typedef T2 argument2_type;
	typedef typename promote_traits<T1, T2>::promote_type result_type;

	result_type operator()(argument1_type x, argument2_type y)const {
		return x/y;
	}
};
template<class T1,class T2>
struct scalar_binary_safe_divide {
	typedef T1 argument1_type;
	typedef T2 argument2_type;
	typedef typename promote_traits<T1, T2>::promote_type result_type;

	scalar_binary_safe_divide(result_type defaultValue):m_defaultValue(defaultValue) {}
	result_type operator()(argument1_type x, argument2_type y)const {
		return y == T2()? m_defaultValue : x/y;
	}
private:
	result_type m_defaultValue;
};

template<class T1,class T2>
struct scalar_binary_min{
	typedef T1 argument1_type;
	typedef T2 argument2_type;
	typedef typename promote_traits<T1, T2>::promote_type result_type;

	result_type operator()(argument1_type x, argument2_type y)const {
		using std::min;
		//convert to the bigger type to prevent std::min conversion errors.
		return min(result_type(x),result_type(y));
	}
};

template<class T1,class T2>
struct scalar_binary_max{
	typedef T1 argument1_type;
	typedef T2 argument2_type;
	typedef typename promote_traits<T1, T2>::promote_type result_type;

	result_type operator()(argument1_type x, argument2_type y)const {
		using std::max;
		//convert to the bigger type to prevent std::max conversion errors.
		return max(result_type(x),result_type(y));
	}
};


template<class T1, class T2>
struct scalar_binary_assign_functor {
	// ISSUE Remove reference to avoid reference to reference problems
	typedef T1 argument1_type;
	typedef T2 const& argument2_type;
};

struct assign_tag {};
struct computed_assign_tag {};

template<class T1, class T2>
struct scalar_assign:
	public scalar_binary_assign_functor<T1, T2> {
	typedef typename scalar_binary_assign_functor<T1, T2>::argument1_type argument1_type;
	typedef typename scalar_binary_assign_functor<T1, T2>::argument2_type argument2_type;
	static const bool computed = false ;

	static void apply(argument1_type t1, argument2_type t2) {
		t1 = t2;
	}

	template<class U1, class U2>
	struct rebind {
		typedef scalar_assign<U1, U2> other;
	};
};

template<class T1, class T2>
struct scalar_plus_assign:
	public scalar_binary_assign_functor<T1, T2> {
	typedef typename scalar_binary_assign_functor<T1, T2>::argument1_type argument1_type;
	typedef typename scalar_binary_assign_functor<T1, T2>::argument2_type argument2_type;
	static const bool computed = true ;

	static void apply(argument1_type t1, argument2_type t2) {
		t1 += t2;
	}

	template<class U1, class U2>
	struct rebind {
		typedef scalar_plus_assign<U1, U2> other;
	};
};

template<class T1, class T2>
struct scalar_minus_assign:
	public scalar_binary_assign_functor<T1, T2> {
	typedef typename scalar_binary_assign_functor<T1, T2>::argument1_type argument1_type;
	typedef typename scalar_binary_assign_functor<T1, T2>::argument2_type argument2_type;
	static const bool computed = true ;

	static void apply(argument1_type t1, argument2_type t2) {
		t1 -= t2;
	}

	template<class U1, class U2>
	struct rebind {
		typedef scalar_minus_assign<U1, U2> other;
	};
};

template<class T1, class T2>
struct scalar_multiplies_assign:
	public scalar_binary_assign_functor<T1, T2> {
	typedef typename scalar_binary_assign_functor<T1, T2>::argument1_type argument1_type;
	typedef typename scalar_binary_assign_functor<T1, T2>::argument2_type argument2_type;
	static const bool computed = true;

	static void apply(argument1_type t1, argument2_type t2) {
		t1 *= t2;
	}

	template<class U1, class U2>
	struct rebind {
		typedef scalar_multiplies_assign<U1, U2> other;
	};
};
template<class T1, class T2>
struct scalar_divides_assign:
	public scalar_binary_assign_functor<T1, T2> {
	typedef typename scalar_binary_assign_functor<T1, T2>::argument1_type argument1_type;
	typedef typename scalar_binary_assign_functor<T1, T2>::argument2_type argument2_type;
	static const bool computed = true;

	static void apply(argument1_type t1, argument2_type t2) {
		t1 /= t2;
	}

	template<class U1, class U2>
	struct rebind {
		typedef scalar_divides_assign<U1, U2> other;
	};
};

template<class T1, class T2>
struct scalar_swap{
	typedef typename boost::remove_reference<T1>::type& argument1_type;
	typedef typename boost::remove_reference<T2>::type& argument2_type;

	static 
	void apply(argument1_type t1, argument2_type t2) {
		using std::swap;
		swap(t1, t2);
	}

	template<class U1, class U2>
	struct rebind {
		typedef scalar_swap<U1, U2> other;
	};
};



///////////////////VECTOR REDUCTION FUNCTORS/////////////////////////

//Functor implementing reduction of the form f(v_n,f(v_{n-1},f(....f(v_0,seed))))
// we assume for sparse vectors that the following holds:
// f(0,0) = 0 and f(v,f(0,w))=f(f(v,w),0)
//second argument to the function is the default value(seed).
template<class F>
struct vector_fold{
	typedef F functor_type;
	typedef typename F::result_type result_type;
	
	vector_fold(functor_type const& f):m_functor(f){}
	vector_fold(){}
	
	template<class E>
	result_type operator()(
		vector_expression<E> const& v,
		result_type seed
	) {
		return apply(v(),seed, typename E::const_iterator::iterator_category());
	}
private:
	//Dense Case
	template<class E>
	result_type apply(
		E const& v,
		result_type seed,
		dense_random_access_iterator_tag
	) {
		std::size_t size = v.size();
		result_type result = seed;
		for(std::size_t i = 0; i != size; ++i){
			result = m_functor(result,v(i));
		}
		return result;
	}
	//Sparse Case
	template<class E>
	result_type apply(
		E const& v,
		result_type seed,
		sparse_bidirectional_iterator_tag
	) {
		typename E::const_iterator iter=v.begin();
		typename E::const_iterator end=v.end();
		
		result_type result = seed;
		std::size_t nnz = 0;
		for(;iter != end;++iter,++nnz){
			result = m_functor(result,*iter);
		}
		//apply final operator f(0,v)
		if(nnz != v.size())
			result = m_functor(result,*iter);
		return result;
	}
	functor_type m_functor;
};

// Functor implementing inner product of vectors v1 and v2 with return type T
template<class T>
struct vector_inner_prod{
	typedef T result_type;

	template<class E1, class E2>
	result_type operator()(
		vector_expression<E1> const&v1,
	        vector_expression<E2> const&v2
	) {
		SIZE_CHECK(v1().size()==v2().size());
		return apply(v1(),v2(),
			typename E1::const_iterator::iterator_category(),
			typename E2::const_iterator::iterator_category()
		);
	}
private:
	// Dense case
	template<class E1, class E2>
	static result_type apply(
		E1 const& v1, 
		E2 const& v2, 
		dense_random_access_iterator_tag,
		dense_random_access_iterator_tag
	) {
		std::size_t size = v1.size();
		result_type sum = result_type();
		for(std::size_t i = 0; i != size; ++i){
			sum += v1(i) * v2(i);
		}
		return sum;
	}
	// Sparse case
	template<class E1, class E2>
	static result_type apply(
		E1 const& v1, 
		E2 const& v2, 
		sparse_bidirectional_iterator_tag,
		sparse_bidirectional_iterator_tag
	) {
		typename E1::const_iterator iter1=v1.begin();
		typename E1::const_iterator end1=v1.end();
		typename E2::const_iterator iter2=v2.begin();
		typename E2::const_iterator end2=v2.end();
		result_type sum = result_type();
		//be aware of empty vectors!
		while(iter1 != end1 && iter2 != end2)
		{
			std::size_t index1=iter1.index();
			std::size_t index2=iter2.index();
			if(index1==index2){
				sum += *iter1 * *iter2;
				++iter1;
				++iter2;
			}
			else if(index1> index2){
				++iter2;
			}
			else {
				++iter1;
			}
		}
		return sum;
	}
	
	// Dense-Sparse case
	template<class E1, class E2>
	static result_type apply(
		E1 const& v1, 
		E2 const& v2, 
		dense_random_access_iterator_tag,
		sparse_bidirectional_iterator_tag
	) {
		typename E2::const_iterator iter2=v2.begin();
		typename E2::const_iterator end2=v2.end();
		result_type sum = result_type();
		for(;iter2 != end2;++iter2){
			sum += v1(iter2.index()) * *iter2;
		}
		return sum;
	}
	//Sparse-Dense case
	template<class E1, class E2>
	static result_type apply(
		E1 const& v1, 
		E2 const& v2, 
		sparse_bidirectional_iterator_tag t1,
		dense_random_access_iterator_tag t2
	) {
		//use commutativity!
		return apply(v2,v1,t2,t1);
	}
};

// Matrix functors

// Binary returning vector
template<class M1, class M2, class TV>
struct matrix_vector_binary_functor {
	typedef typename M1::size_type size_type;
	typedef typename M1::difference_type difference_type;
	typedef TV value_type;
	typedef TV result_type;
};

template<class M1, class M2, class TV>
struct matrix_vector_prod1:
	public matrix_vector_binary_functor<M1, M2, TV> {
	typedef typename matrix_vector_binary_functor<M1, M2, TV>::size_type size_type;
	typedef typename matrix_vector_binary_functor<M1, M2, TV>::difference_type difference_type;
	typedef typename matrix_vector_binary_functor<M1, M2, TV>::value_type value_type;
	typedef typename matrix_vector_binary_functor<M1, M2, TV>::result_type result_type;

	template<class E1, class E2>
	static result_type apply(
		const matrix_expression<E1> &e1,
	        const vector_expression<E2> &e2,
	        size_type i
	) {
		size_type size = BOOST_UBLAS_SAME(e1().size2(), e2().size());
		result_type t = result_type(0);
		for (size_type j = 0; j < size; ++ j)
			t += e1()(i, j) * e2()(j);
		return t;
	}
	// Dense case
	template<class I1, class I2>
	static 
	result_type apply(difference_type size, I1 it1, I2 it2) {
		result_type t = result_type(0);
		while (-- size >= 0)
			t += *it1 * *it2, ++ it1, ++ it2;
		return t;
	}
	// Packed case
	template<class I1, class I2>
	static 
	result_type apply(I1 it1, const I1 &it1_end, I2 it2, const I2 &it2_end) {
		result_type t = result_type(0);
		difference_type it1_size(it1_end - it1);
		difference_type it2_size(it2_end - it2);
		difference_type diff(0);
		if (it1_size > 0 && it2_size > 0)
			diff = it2.index() - it1.index2();
		if (diff != 0) {
			difference_type size = (std::min)(diff, it1_size);
			if (size > 0) {
				it1 += size;
				it1_size -= size;
				diff -= size;
			}
			size = (std::min)(- diff, it2_size);
			if (size > 0) {
				it2 += size;
				it2_size -= size;
				diff += size;
			}
		}
		difference_type size((std::min)(it1_size, it2_size));
		while (-- size >= 0)
			t += *it1 * *it2, ++ it1, ++ it2;
		return t;
	}
	// Sparse case
	template<class I1, class I2>
	static 
	result_type apply(I1 it1, const I1 &it1_end, I2 it2, const I2 &it2_end,
	        sparse_bidirectional_iterator_tag, sparse_bidirectional_iterator_tag) {
		result_type t = result_type(0);
		if (it1 != it1_end && it2 != it2_end) {
			size_type it1_index = it1.index2(), it2_index = it2.index();
			while (true) {
				difference_type compare = it1_index - it2_index;
				if (compare == 0) {
					t += *it1 * *it2, ++ it1, ++ it2;
					if (it1 != it1_end && it2 != it2_end) {
						it1_index = it1.index2();
						it2_index = it2.index();
					} else
						break;
				} else if (compare < 0) {
					increment(it1, it1_end, - compare);
					if (it1 != it1_end)
						it1_index = it1.index2();
					else
						break;
				} else if (compare > 0) {
					increment(it2, it2_end, compare);
					if (it2 != it2_end)
						it2_index = it2.index();
					else
						break;
				}
			}
		}
		return t;
	}
	// Sparse packed case
	template<class I1, class I2>
	static result_type apply(I1 it1, const I1 &it1_end, I2 it2, const I2 &/* it2_end */,
	        sparse_bidirectional_iterator_tag, packed_random_access_iterator_tag) {
		result_type t = result_type(0);
		while (it1 != it1_end) {
			t += *it1 * it2()(it1.index2());
			++ it1;
		}
		return t;
	}
	// Packed sparse case
	template<class I1, class I2>
	static result_type apply(I1 it1, const I1 &/* it1_end */, I2 it2, const I2 &it2_end,
	        packed_random_access_iterator_tag, sparse_bidirectional_iterator_tag) {
		result_type t = result_type(0);
		while (it2 != it2_end) {
			t += it1()(it1.index1(), it2.index()) * *it2;
			++ it2;
		}
		return t;
	}
	// Another dispatcher
	template<class I1, class I2>
	static result_type apply(I1 it1, const I1 &it1_end, I2 it2, const I2 &it2_end,
	        sparse_bidirectional_iterator_tag) {
		typedef typename I1::iterator_category iterator1_category;
		typedef typename I2::iterator_category iterator2_category;
		return apply(it1, it1_end, it2, it2_end, iterator1_category(), iterator2_category());
	}
};

template<class M1, class M2, class TV>
struct matrix_vector_prod2:
	public matrix_vector_binary_functor<M1, M2, TV> {
	typedef typename matrix_vector_binary_functor<M1, M2, TV>::size_type size_type;
	typedef typename matrix_vector_binary_functor<M1, M2, TV>::difference_type difference_type;
	typedef typename matrix_vector_binary_functor<M1, M2, TV>::value_type value_type;
	typedef typename matrix_vector_binary_functor<M1, M2, TV>::result_type result_type;

	template<class E1, class E2>
	static result_type apply(const vector_expression<E1> &e1,
	        const matrix_expression<E2> &e2,
	        size_type i) {
		size_type size = BOOST_UBLAS_SAME(e1().size(), e2().size1());
		result_type t = result_type(0);
		for (size_type j = 0; j < size; ++ j)
			t += e1()(j) * e2()(j, i);
		return t;
	}
	// Dense case
	template<class I1, class I2>
	static result_type apply(difference_type size, I1 it1, I2 it2) {
		result_type t = result_type(0);
		while (-- size >= 0)
			t += *it1 * *it2, ++ it1, ++ it2;
		return t;
	}
	// Packed case
	template<class I1, class I2>
	static result_type apply(I1 it1, const I1 &it1_end, I2 it2, const I2 &it2_end) {
		result_type t = result_type(0);
		difference_type it1_size(it1_end - it1);
		difference_type it2_size(it2_end - it2);
		difference_type diff(0);
		if (it1_size > 0 && it2_size > 0)
			diff = it2.index1() - it1.index();
		if (diff != 0) {
			difference_type size = (std::min)(diff, it1_size);
			if (size > 0) {
				it1 += size;
				it1_size -= size;
				diff -= size;
			}
			size = (std::min)(- diff, it2_size);
			if (size > 0) {
				it2 += size;
				it2_size -= size;
				diff += size;
			}
		}
		difference_type size((std::min)(it1_size, it2_size));
		while (-- size >= 0)
			t += *it1 * *it2, ++ it1, ++ it2;
		return t;
	}
	// Sparse case
	template<class I1, class I2>
	static result_type apply(I1 it1, const I1 &it1_end, I2 it2, const I2 &it2_end,
	        sparse_bidirectional_iterator_tag, sparse_bidirectional_iterator_tag) {
		result_type t = result_type(0);
		if (it1 != it1_end && it2 != it2_end) {
			size_type it1_index = it1.index(), it2_index = it2.index1();
			while (true) {
				difference_type compare = it1_index - it2_index;
				if (compare == 0) {
					t += *it1 * *it2, ++ it1, ++ it2;
					if (it1 != it1_end && it2 != it2_end) {
						it1_index = it1.index();
						it2_index = it2.index1();
					} else
						break;
				} else if (compare < 0) {
					increment(it1, it1_end, - compare);
					if (it1 != it1_end)
						it1_index = it1.index();
					else
						break;
				} else if (compare > 0) {
					increment(it2, it2_end, compare);
					if (it2 != it2_end)
						it2_index = it2.index1();
					else
						break;
				}
			}
		}
		return t;
	}
	// Packed sparse case
	template<class I1, class I2>
	static result_type apply(I1 it1, const I1 &/* it1_end */, I2 it2, const I2 &it2_end,
	        packed_random_access_iterator_tag, sparse_bidirectional_iterator_tag) {
		result_type t = result_type(0);
		while (it2 != it2_end) {
			t += it1()(it2.index1()) * *it2;
			++ it2;
		}
		return t;
	}
	// Sparse packed case
	template<class I1, class I2>
	static result_type apply(I1 it1, const I1 &it1_end, I2 it2, const I2 &/* it2_end */,
	        sparse_bidirectional_iterator_tag, packed_random_access_iterator_tag) {
		result_type t = result_type(0);
		while (it1 != it1_end) {
			t += *it1 * it2()(it1.index(), it2.index2());
			++ it1;
		}
		return t;
	}
	// Another dispatcher
	template<class I1, class I2>
	static result_type apply(I1 it1, const I1 &it1_end, I2 it2, const I2 &it2_end,
	        sparse_bidirectional_iterator_tag) {
		typedef typename I1::iterator_category iterator1_category;
		typedef typename I2::iterator_category iterator2_category;
		return apply(it1, it1_end, it2, it2_end, iterator1_category(), iterator2_category());
	}
};

// Binary returning matrix
template<class M1, class M2, class TV>
struct matrix_matrix_binary_functor {
	typedef typename M1::size_type size_type;
	typedef typename M1::difference_type difference_type;
	typedef TV value_type;
	typedef TV result_type;
};

template<class M1, class M2, class TV>
struct matrix_matrix_prod:
	public matrix_matrix_binary_functor<M1, M2, TV> {
	typedef typename matrix_matrix_binary_functor<M1, M2, TV>::size_type size_type;
	typedef typename matrix_matrix_binary_functor<M1, M2, TV>::difference_type difference_type;
	typedef typename matrix_matrix_binary_functor<M1, M2, TV>::value_type value_type;
	typedef typename matrix_matrix_binary_functor<M1, M2, TV>::result_type result_type;

	template<class E1, class E2>
	static result_type apply(const matrix_expression<E1> &e1,
	        const matrix_expression<E2> &e2,
	        size_type i, size_type j) {
		size_type size = BOOST_UBLAS_SAME(e1().size2(), e2().size1());
		result_type t = result_type(0);
		for (size_type k = 0; k < size; ++ k)
			t += e1()(i, k) * e2()(k, j);
		return t;
	}
	// Dense case
	template<class I1, class I2>
	static result_type apply(difference_type size, I1 it1, I2 it2) {
		result_type t = result_type(0);
		while (-- size >= 0)
			t += *it1 * *it2, ++ it1, ++ it2;
		return t;
	}
	// Packed case
	template<class I1, class I2>
	static result_type apply(I1 it1, const I1 &it1_end, I2 it2, const I2 &it2_end, packed_random_access_iterator_tag) {
		result_type t = result_type(0);
		difference_type it1_size(it1_end - it1);
		difference_type it2_size(it2_end - it2);
		difference_type diff(0);
		if (it1_size > 0 && it2_size > 0)
			diff = it2.index1() - it1.index2();
		if (diff != 0) {
			difference_type size = (std::min)(diff, it1_size);
			if (size > 0) {
				it1 += size;
				it1_size -= size;
				diff -= size;
			}
			size = (std::min)(- diff, it2_size);
			if (size > 0) {
				it2 += size;
				it2_size -= size;
				diff += size;
			}
		}
		difference_type size((std::min)(it1_size, it2_size));
		while (-- size >= 0)
			t += *it1 * *it2, ++ it1, ++ it2;
		return t;
	}
	// Sparse case
	template<class I1, class I2>
	static result_type apply(I1 it1, const I1 &it1_end, I2 it2, const I2 &it2_end, sparse_bidirectional_iterator_tag) {
		result_type t = result_type(0);
		if (it1 != it1_end && it2 != it2_end) {
			size_type it1_index = it1.index2(), it2_index = it2.index1();
			while (true) {
				difference_type compare = difference_type(it1_index - it2_index);
				if (compare == 0) {
					t += *it1 * *it2, ++ it1, ++ it2;
					if (it1 != it1_end && it2 != it2_end) {
						it1_index = it1.index2();
						it2_index = it2.index1();
					} else
						break;
				} else if (compare < 0) {
					increment(it1, it1_end, - compare);
					if (it1 != it1_end)
						it1_index = it1.index2();
					else
						break;
				} else if (compare > 0) {
					increment(it2, it2_end, compare);
					if (it2 != it2_end)
						it2_index = it2.index1();
					else
						break;
				}
			}
		}
		return t;
	}
};

// Unary returning scalar norm
template<class M>
struct matrix_scalar_real_unary_functor {
	typedef typename M::value_type value_type;
	typedef typename real_traits<value_type>::type type;
	typedef type result_type;
};

template<class M>
struct matrix_norm_1:
	public matrix_scalar_real_unary_functor<M> {
	typedef typename matrix_scalar_real_unary_functor<M>::value_type value_type;
	typedef typename matrix_scalar_real_unary_functor<M>::result_type result_type;

	template<class E>
	static 
	result_type apply(const matrix_expression<E> &e) {
		scalar_abs<value_type> abs;
		result_type t = result_type();
		
		typedef typename E::size_type matrix_size_type;
		matrix_size_type size2(e().size2());
		matrix_size_type size1(e().size1());
		for (matrix_size_type j = 0; j < size2; ++ j) {
			result_type u = result_type();
			for (matrix_size_type i = 0; i < size1; ++ i) {
				u += abs(e()(i, j));
			}
			if (u > t)
				t = u;
		}
		return t;
	}
};

template<class M>
struct matrix_norm_frobenius:
	public matrix_scalar_real_unary_functor<M> {
	typedef typename matrix_scalar_real_unary_functor<M>::value_type value_type;
	typedef typename matrix_scalar_real_unary_functor<M>::result_type result_type;

	template<class E>
	static result_type apply(const matrix_expression<E> &e) {
		scalar_abs_sqr<value_type> abs_sqr;
		result_type t = result_type();
		typedef typename E::size_type matrix_size_type;
		matrix_size_type size1(e().size1());
		matrix_size_type size2(e().size2());
		for (matrix_size_type i = 0; i < size1; ++ i) {
			for (matrix_size_type j = 0; j < size2; ++ j) {
				t +=  abs_sqr(e()(i, j));
			}
		}
		using std::sqrt;
		return sqrt(t);
	}
};

template<class M>
struct matrix_norm_inf:
	public matrix_scalar_real_unary_functor<M> {
	typedef typename matrix_scalar_real_unary_functor<M>::value_type value_type;
	typedef typename matrix_scalar_real_unary_functor<M>::result_type result_type;

	template<class E>
	static result_type apply(const matrix_expression<E> &e) {
		scalar_abs<value_type> abs;
		result_type t = result_type();
		typedef typename E::size_type matrix_size_type;
		matrix_size_type size1(e().size1());
		matrix_size_type size2(e().size2());
		for (matrix_size_type i = 0; i < size1; ++ i) {
			result_type u = result_type();
			for (matrix_size_type j = 0; j < size2; ++ j) {
				u += abs(e()(i, j));
			}
			if (u > t)
				t = u;
		}
		return t;
	}
};

// forward declaration
template <class Z, class D> struct basic_column_major;

// This functor defines storage layout and it's properties
// matrix (i,j) -> storage [i * size_i + j]
template <class Z, class D>
struct basic_row_major {
	typedef Z size_type;
	typedef D difference_type;
	typedef row_major_tag orientation_category;
	typedef basic_column_major<Z,D> transposed_layout;

	static size_type storage_size(size_type size_i, size_type size_j) {
		// Guard against size_type overflow
		BOOST_UBLAS_CHECK(size_j == 0 || size_i <= (std::numeric_limits<size_type>::max)() / size_j, bad_size());
		return size_i * size_j;
	}

	// Indexing conversion to storage element
	static size_type element(size_type i, size_type size_i, size_type j, size_type size_j) {
		BOOST_UBLAS_CHECK(i < size_i, bad_index());
		BOOST_UBLAS_CHECK(j < size_j, bad_index());
		detail::ignore_unused_variable_warning(size_i);
		// Guard against size_type overflow
		BOOST_UBLAS_CHECK(i <= ((std::numeric_limits<size_type>::max)() - j) / size_j, bad_index());
		return i * size_j + j;
	}
	static size_type address(size_type i, size_type size_i, size_type j, size_type size_j) {
		BOOST_UBLAS_CHECK(i <= size_i, bad_index());
		BOOST_UBLAS_CHECK(j <= size_j, bad_index());
		// Guard against size_type overflow - address may be size_j past end of storage
		BOOST_UBLAS_CHECK(size_j == 0 || i <= ((std::numeric_limits<size_type>::max)() - j) / size_j, bad_index());
		detail::ignore_unused_variable_warning(size_i);
		return i * size_j + j;
	}

	// Storage element to index conversion
	static difference_type distance_i(difference_type k, size_type /* size_i */, size_type size_j) {
		return size_j != 0 ? k / size_j : 0;
	}
	static difference_type distance_j(difference_type k, size_type /* size_i */, size_type /* size_j */) {
		return k;
	}
	static size_type index_i(difference_type k, size_type /* size_i */, size_type size_j) {
		return size_j != 0 ? k / size_j : 0;
	}
	static size_type index_j(difference_type k, size_type /* size_i */, size_type size_j) {
		return size_j != 0 ? k % size_j : 0;
	}
	static bool fast_i() {
		return false;
	}
	static bool fast_j() {
		return true;
	}

	// Iterating storage elements
	template<class I>
	static void increment_i(I &it, size_type /* size_i */, size_type size_j) {
		it += size_j;
	}
	template<class I>
	static
	
	void increment_i(I &it, difference_type n, size_type /* size_i */, size_type size_j) {
		it += n * size_j;
	}
	template<class I>
	static
	
	void decrement_i(I &it, size_type /* size_i */, size_type size_j) {
		it -= size_j;
	}
	template<class I>
	static
	
	void decrement_i(I &it, difference_type n, size_type /* size_i */, size_type size_j) {
		it -= n * size_j;
	}
	template<class I>
	static
	
	void increment_j(I &it, size_type /* size_i */, size_type /* size_j */) {
		++ it;
	}
	template<class I>
	static
	
	void increment_j(I &it, difference_type n, size_type /* size_i */, size_type /* size_j */) {
		it += n;
	}
	template<class I>
	static
	
	void decrement_j(I &it, size_type /* size_i */, size_type /* size_j */) {
		-- it;
	}
	template<class I>
	static
	
	void decrement_j(I &it, difference_type n, size_type /* size_i */, size_type /* size_j */) {
		it -= n;
	}

	// Triangular access
	static
	
	size_type triangular_size(size_type size_i, size_type size_j) {
		size_type size = (std::max)(size_i, size_j);
		// Guard against size_type overflow - siboost::mplified
		BOOST_UBLAS_CHECK(size == 0 || size / 2 < (std::numeric_limits<size_type>::max)() / size /* +1/2 */, bad_size());
		return ((size + 1) * size) / 2;
	}
	static
	
	size_type lower_element(size_type i, size_type size_i, size_type j, size_type size_j) {
		BOOST_UBLAS_CHECK(i < size_i, bad_index());
		BOOST_UBLAS_CHECK(j < size_j, bad_index());
		BOOST_UBLAS_CHECK(i >= j, bad_index());
		detail::ignore_unused_variable_warning(size_i);
		detail::ignore_unused_variable_warning(size_j);
		// FIXME size_type overflow
		// sigma_i (i + 1) = (i + 1) * i / 2
		// i = 0 1 2 3, sigma = 0 1 3 6
		return ((i + 1) * i) / 2 + j;
	}
	static
	
	size_type upper_element(size_type i, size_type size_i, size_type j, size_type size_j) {
		BOOST_UBLAS_CHECK(i < size_i, bad_index());
		BOOST_UBLAS_CHECK(j < size_j, bad_index());
		BOOST_UBLAS_CHECK(i <= j, bad_index());
		// FIXME size_type overflow
		// sigma_i (size - i) = size * i - i * (i - 1) / 2
		// i = 0 1 2 3, sigma = 0 4 7 9
		return (i * (2 * (std::max)(size_i, size_j) - i + 1)) / 2 + j - i;
	}

	// Major and minor indices
	static
	
	size_type index_M(size_type index1, size_type /* index2 */) {
		return index1;
	}
	static
	
	size_type index_m(size_type /* index1 */, size_type index2) {
		return index2;
	}
	static
	
	size_type size_M(size_type size_i, size_type /* size_j */) {
		return size_i;
	}
	static
	
	size_type size_m(size_type /* size_i */, size_type size_j) {
		return size_j;
	}
};

// This functor defines storage layout and it's properties
// matrix (i,j) -> storage [i + j * size_i]
template <class Z, class D>
struct basic_column_major {
	typedef Z size_type;
	typedef D difference_type;
	typedef column_major_tag orientation_category;
	typedef basic_row_major<Z,D> transposed_layout;

	static size_type storage_size(size_type size_i, size_type size_j) {
		// Guard against size_type overflow
		BOOST_UBLAS_CHECK(size_i == 0 || size_j <= (std::numeric_limits<size_type>::max)() / size_i, bad_size());
		return size_i * size_j;
	}

	// Indexing conversion to storage element
	static size_type element(size_type i, size_type size_i, size_type j, size_type size_j) {
		BOOST_UBLAS_CHECK(i < size_i, bad_index());
		BOOST_UBLAS_CHECK(j < size_j, bad_index());
		detail::ignore_unused_variable_warning(size_j);
		// Guard against size_type overflow
		BOOST_UBLAS_CHECK(j <= ((std::numeric_limits<size_type>::max)() - i) / size_i, bad_index());
		return i + j * size_i;
	}
	static size_type address(size_type i, size_type size_i, size_type j, size_type size_j) {
		BOOST_UBLAS_CHECK(i <= size_i, bad_index());
		BOOST_UBLAS_CHECK(j <= size_j, bad_index());
		detail::ignore_unused_variable_warning(size_j);
		// Guard against size_type overflow - address may be size_i past end of storage
		BOOST_UBLAS_CHECK(size_i == 0 || j <= ((std::numeric_limits<size_type>::max)() - i) / size_i, bad_index());
		return i + j * size_i;
	}

	// Storage element to index conversion
	static difference_type distance_i(difference_type k, size_type /* size_i */, size_type /* size_j */) {
		return k;
	}
	static difference_type distance_j(difference_type k, size_type size_i, size_type /* size_j */) {
		return size_i != 0 ? k / size_i : 0;
	}
	static size_type index_i(difference_type k, size_type size_i, size_type /* size_j */) {
		return size_i != 0 ? k % size_i : 0;
	}
	static size_type index_j(difference_type k, size_type size_i, size_type /* size_j */) {
		return size_i != 0 ? k / size_i : 0;
	}
	static bool fast_i() {
		return true;
	}
	static bool fast_j() {
		return false;
	}

	// Iterating
	template<class I>
	static void increment_i(I &it, size_type /* size_i */, size_type /* size_j */) {
		++ it;
	}
	template<class I>
	static void increment_i(I &it, difference_type n, size_type /* size_i */, size_type /* size_j */) {
		it += n;
	}
	template<class I>
	static void decrement_i(I &it, size_type /* size_i */, size_type /* size_j */) {
		-- it;
	}
	template<class I>
	static void decrement_i(I &it, difference_type n, size_type /* size_i */, size_type /* size_j */) {
		it -= n;
	}
	template<class I>
	static void increment_j(I &it, size_type size_i, size_type /* size_j */) {
		it += size_i;
	}
	template<class I>
	static void increment_j(I &it, difference_type n, size_type size_i, size_type /* size_j */) {
		it += n * size_i;
	}
	template<class I>
	static void decrement_j(I &it, size_type size_i, size_type /* size_j */) {
		it -= size_i;
	}
	template<class I>
	static void decrement_j(I &it, difference_type n, size_type size_i, size_type /* size_j */) {
		it -= n* size_i;
	}

	// Triangular access
	static size_type triangular_size(size_type size_i, size_type size_j) {
		size_type size = (std::max)(size_i, size_j);
		// Guard against size_type overflow - siboost::mplified
		BOOST_UBLAS_CHECK(size == 0 || size / 2 < (std::numeric_limits<size_type>::max)() / size /* +1/2 */, bad_size());
		return ((size + 1) * size) / 2;
	}
	static size_type lower_element(size_type i, size_type size_i, size_type j, size_type size_j) {
		BOOST_UBLAS_CHECK(i < size_i, bad_index());
		BOOST_UBLAS_CHECK(j < size_j, bad_index());
		BOOST_UBLAS_CHECK(i >= j, bad_index());
		// FIXME size_type overflow
		// sigma_j (size - j) = size * j - j * (j - 1) / 2
		// j = 0 1 2 3, sigma = 0 4 7 9
		return i - j + (j * (2 * (std::max)(size_i, size_j) - j + 1)) / 2;
	}
	static
	
	size_type upper_element(size_type i, size_type size_i, size_type j, size_type size_j) {
		BOOST_UBLAS_CHECK(i < size_i, bad_index());
		BOOST_UBLAS_CHECK(j < size_j, bad_index());
		BOOST_UBLAS_CHECK(i <= j, bad_index());
		// FIXME size_type overflow
		// sigma_j (j + 1) = (j + 1) * j / 2
		// j = 0 1 2 3, sigma = 0 1 3 6
		return i + ((j + 1) * j) / 2;
	}

	// Major and minor indices
	static size_type index_M(size_type /* index1 */, size_type index2) {
		return index2;
	}
	static size_type index_m(size_type index1, size_type /* index2 */) {
		return index1;
	}
	static
	
	size_type size_M(size_type /* size_i */, size_type size_j) {
		return size_j;
	}
	static size_type size_m(size_type size_i, size_type /* size_j */) {
		return size_i;
	}
};


template <class Z>
struct basic_full {
	typedef Z size_type;

	template<class L>
	static size_type packed_size(L, size_type size_i, size_type size_j) {
		return L::storage_size(size_i, size_j);
	}

	static bool zero(size_type /* i */, size_type /* j */) {
		return false;
	}
	static bool one(size_type /* i */, size_type /* j */) {
		return false;
	}
	static bool other(size_type /* i */, size_type /* j */) {
		return true;
	}
	// FIXME: this should not be used at all
	static size_type restrict1(size_type i, size_type /* j */) {
		return i;
	}
	static size_type restrict2(size_type /* i */, size_type j) {
		return j;
	}
	static size_type mutable_restrict1(size_type i, size_type /* j */) {
		return i;
	}
	static size_type mutable_restrict2(size_type /* i */, size_type j) {
		return j;
	}
};

}
}

#endif
