#ifndef SHARK_LINALG_BLAS_UBLAS_FUNCTIONAL_HPP
#define SHARK_LINALG_BLAS_UBLAS_FUNCTIONAL_HPP

#include <boost/math/constants/constants.hpp>
#include <boost/type_traits/remove_reference.hpp> 

#include "traits.hpp"
#include <shark/Core/Exception.h>
#include <shark/Core/Math.h>

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
	typedef typename real_traits<T>::type result_type;
	static const bool zero_identity = true;

	result_type operator()(argument_type x)const {
		return x;
	}
};
template<class T>
struct scalar_real<std::complex<T> > {
	typedef std::complex<T> argument_type;
	typedef typename real_traits<T>::type result_type;
	static const bool zero_identity = true;

	result_type operator()(argument_type x)const {
		return x.real();
	}
};

template<class T>
struct scalar_imag {
	typedef T argument_type;
	typedef typename real_traits<T>::type result_type;
	static const bool zero_identity = true;

	result_type operator()(argument_type x)const {
		return real_traits<T>::imag(x);
	}
};
template<class T>
struct scalar_imag<std::complex<T> > {
	typedef std::complex<T> argument_type;
	typedef typename real_traits<T>::type result_type;
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
	typedef typename real_traits<T>::type result_type;
	static const bool zero_identity = true;

	result_type operator()(argument_type x)const {
		result_type absolute_value = std::abs(x);
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
		return shark::softPlus(x);
	}
};

template<class T>
struct scalar_sigmoid {
	typedef T argument_type;
	typedef argument_type result_type;
	static const bool zero_identity = false;

	result_type operator()(argument_type x)const {
		return shark::sigmoid(x);
	}
};

template<class T>
struct scalar_less_than{
	typedef T argument_type;
	typedef int result_type;
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
	typedef int result_type;
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
	typedef int result_type;
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
	typedef int result_type;
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
	typedef int result_type;
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
	typedef int result_type;
	static const bool zero_identity = false;

	scalar_not_equal(T comparator):m_comparator(comparator) {}
	result_type operator()(argument_type x)const {
		return x != m_comparator;
	}
private:
	T m_comparator;
};

template<class T>
struct scalar_min {
	typedef T argument_type;
	typedef argument_type result_type;
	static const bool zero_identity = true;

	scalar_min(T argument):m_argument(argument) {}
	result_type operator()(argument_type x)const {
		using std::min;
		return min(x,m_argument);
	}
private:
	T m_argument;
};
template<class T>
struct scalar_max {
	typedef T argument_type;
	typedef argument_type result_type;
	static const bool zero_identity = true;

	scalar_max(T argument):m_argument(argument) {}
	result_type operator()(argument_type x)const {
		using std::max;
		return max(x,m_argument);
	}
private:
	T m_argument;
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
struct scalar_plus_assign{
	typedef T1 argument1_type;
	typedef T2 argument2_type;
	 void operator()(argument1_type t1, argument2_type t2) {
		t1 += static_cast<typename boost::remove_reference<T1>::type const>(t2);
	}
};

template<class T1, class T2>
struct scalar_minus_assign{
	typedef T1 argument1_type;
	typedef T2 argument2_type;

	void operator()(argument1_type t1, argument2_type t2) {
		t1 -= t2;
	}
};

template<class T1, class T2>
struct scalar_multiply_assign{
	typedef T1 argument1_type;
	typedef T2 argument2_type;

	void operator()(argument1_type t1, argument2_type t2) {
		t1 *= t2;
	}
};
template<class T1, class T2>
struct scalar_divide_assign{
	typedef T1 argument1_type;
	typedef T2 argument2_type;
	
	void operator()(argument1_type t1, argument2_type t2) {
		t1 /= t2;
	}
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
// Unary returning scalar norm
template<class M>
struct matrix_norm_1 {
	typedef typename M::value_type value_type;
	typedef typename  real_traits<value_type>::type result_type;

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
struct matrix_norm_inf{
	typedef typename M::value_type value_type;
	typedef typename  real_traits<value_type>::type result_type;

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

struct range {
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;
	typedef size_type value_type;
	typedef value_type const_reference;
	typedef const_reference reference;
	typedef const value_type *const_pointer;
	typedef value_type *pointer;

	// Construction and destruction

	range():m_start(0), m_size(0) {}

	range(size_type start, size_type stop):m_start(start), m_size(stop - start) {
		RANGE_CHECK(start <= stop);
	}

	size_type start() const {
		return m_start;
	}
	size_type size() const {
		return m_size;
	}

	// Random Access Container
	size_type max_size() const {
		return m_size;
	}


	bool empty() const {
		return m_size == 0;
	}
	    
	// Element access
	const_reference operator()(size_type i) const {
		RANGE_CHECK(i < m_size);
		return m_start + i;
	}

	// Comparison
	bool operator ==(range const& r) const {
		return m_start == r.m_start && m_size == r.m_size;
	}
	bool operator !=(range const& r) const {
		return !(*this == r);
	}

private:
	size_type m_start;
	size_type m_size;
};

}}

#endif
