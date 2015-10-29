#ifndef SHARK_LINALG_BLAS_VECTOR_EXPRESSION_HPP
#define SHARK_LINALG_BLAS_VECTOR_EXPRESSION_HPP

#include <boost/type_traits/is_convertible.hpp> 
#include <boost/utility/enable_if.hpp> 
#include "vector_proxy.hpp"
#include "kernels/dot.hpp"

namespace shark {
namespace blas {
	
///\brief Implements multiplications of a vector by a scalar
template<class E>
class vector_scalar_multiply:
	public vector_expression<vector_scalar_multiply <E> > {
	typedef vector_scalar_multiply<E> self_type;
public:
	typedef typename E::const_closure_type expression_closure_type;
	typedef typename E::size_type size_type;
	typedef typename E::difference_type difference_type;
	typedef typename E::value_type value_type;
	typedef typename E::scalar_type scalar_type;
	typedef value_type const_reference;
	typedef value_type reference;
	typedef value_type const * const_pointer;
	typedef value_type const*  pointer;

	typedef typename E::index_type index_type;
	typedef typename E::const_index_pointer const_index_pointer;
	typedef typename index_pointer<E>::type index_pointer;

	typedef self_type const const_closure_type;
	typedef self_type closure_type;
	typedef unknown_storage_tag storage_category;
	typedef typename E::evaluation_category evaluation_category;

	// Construction and destruction
	// May be used as mutable expression.
	vector_scalar_multiply(vector_expression<E> const &e, scalar_type scalar):
		m_expression(e()), m_scalar(scalar) {}

	// Accessors
	size_type size() const {
		return m_expression.size();
	}

	// Expression accessors
	expression_closure_type const &expression() const {
		return m_expression;
	}

public:
	// Element access
	const_reference operator()(index_type i) const {
		return m_scalar * m_expression(i);
	}

	const_reference operator[](index_type i) const {
		return m_scalar * m_expression(i);
	}
	
	
	//computation kernels
	template<class VecX>
	void assign_to(vector_expression<VecX>& x, scalar_type alpha = scalar_type(1) )const{
		m_expression.assign_to(x,alpha*m_scalar);
	}
	template<class VecX>
	void plus_assign_to(vector_expression<VecX>& x, scalar_type alpha = scalar_type(1) )const{
		m_expression.plus_assign_to(x,alpha*m_scalar);
	}
	
	template<class VecX>
	void minus_assign_to(vector_expression<VecX>& x, scalar_type alpha = scalar_type(1) )const{
		m_expression.minus_assign_to(x,alpha*m_scalar);
	}

	
	//iterators
	typedef transform_iterator<typename E::const_iterator,scalar_multiply1<value_type, scalar_type> > const_iterator;
	typedef const_iterator iterator;
	
	const_iterator begin() const {
		return const_iterator(m_expression.begin(),scalar_multiply1<value_type, scalar_type>(m_scalar));
	}
	const_iterator end() const {
		return const_iterator(m_expression.end(),scalar_multiply1<value_type, scalar_type>(m_scalar));
	}
private:
	expression_closure_type m_expression;
	scalar_type m_scalar;
};


template<class T, class E>
typename boost::enable_if<
	boost::is_convertible<T, typename E::scalar_type >,
        vector_scalar_multiply<E>
>::type
operator* (vector_expression<E> const& e, T scalar){
	typedef typename E::scalar_type scalar_type;
	return vector_scalar_multiply<E>(e, scalar_type(scalar));
}
template<class T, class E>
typename boost::enable_if<
	boost::is_convertible<T, typename E::scalar_type >,
        vector_scalar_multiply<E>
>::type
operator* (T scalar, vector_expression<E> const& e){
	typedef typename E::scalar_type scalar_type;
	return vector_scalar_multiply<E>(e, scalar_type(scalar));//explicit cast prevents warning, alternative would be to template vector_scalar_multiply on T as well
}

template<class E>
vector_scalar_multiply<E> operator-(vector_expression<E> const& e){
	typedef typename E::scalar_type scalar_type;
	return vector_scalar_multiply<E>(e, scalar_type(-1));//explicit cast prevents warning, alternative would be to template vector_scalar_multiply on T as well
}
	
/// \brief Vector expression representing a constant valued vector.
template<class T>
class scalar_vector:public vector_expression<scalar_vector<T> > {

	typedef scalar_vector<T> self_type;
public:
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;
	typedef T value_type;
	typedef T scalar_type;
	typedef const T& const_reference;
	typedef T& reference;
	typedef T* pointer;
	typedef const T *const_pointer;

	typedef std::size_t index_type;
	typedef index_type const* const_index_pointer;
	typedef index_type index_pointer;

	typedef const vector_reference<const self_type> const_closure_type;
	typedef vector_reference<self_type> closure_type;
	typedef unknown_storage_tag storage_category;
	typedef elementwise_tag evaluation_category;

	// Construction and destruction
	scalar_vector()
	:m_size(0), m_value() {}
	explicit scalar_vector(size_type size, value_type value)
	:m_size(size), m_value(value) {}
	scalar_vector(const scalar_vector& v)
	:m_size(v.m_size), m_value(v.m_value) {}

	// Accessors
	size_type size() const {
		return m_size;
	}

	// Resizing
	void resize(size_type size, bool /*preserve*/ = true) {
		m_size = size;
	}

	// Element access
	const_reference operator()(index_type /*i*/) const {
		return m_value;
	}

	const_reference operator [](index_type /*i*/) const {
		return m_value;
	}

public:
	typedef constant_iterator<T> iterator;
	typedef constant_iterator<T> const_iterator;

	const_iterator begin() const {
		return const_iterator(0,m_value);
	}
	const_iterator end() const {
		return const_iterator(m_size,m_value);
	}

private:
	size_type m_size;
	value_type m_value;
};

///\brief Creates a vector having a constant value.
///
///@param scalar the value which is repeated
///@param elements the size of the resulting vector
template<class T>
typename boost::enable_if<boost::is_arithmetic<T>, scalar_vector<T> >::type
repeat(T scalar, std::size_t elements){
	return scalar_vector<T>(elements,scalar);
}


///\brief Class implementing vector transformation expressions.
///
///transforms a vector Expression e of type E using a Function f of type F as an elementwise transformation f(e(i))
///This transformation needs f to be constant, meaning that applying f(x), f(y), f(z) yields the same results independent of the
///order of application. Also F must provide a type F::result_type indicating the result type of the functor.
template<class E, class F>
class vector_unary:
	public vector_expression<vector_unary<E, F> > {
	typedef vector_unary<E, F> self_type;
public:
	typedef F functor_type;
	typedef typename E::const_closure_type expression_closure_type;
	typedef typename E::size_type size_type;
	typedef typename E::difference_type difference_type;
	typedef typename F::result_type value_type;
	typedef value_type scalar_type;
	typedef value_type const_reference;
	typedef value_type reference;
	typedef value_type const * const_pointer;
	typedef value_type const*  pointer;

	typedef typename E::index_type index_type;
	typedef typename E::const_index_pointer const_index_pointer;
	typedef typename index_pointer<E>::type index_pointer;

	typedef self_type const const_closure_type;
	typedef self_type closure_type;
	typedef unknown_storage_tag storage_category;
	typedef typename E::evaluation_category evaluation_category;

	// Construction and destruction
	// May be used as mutable expression.
	vector_unary(vector_expression<E> const &e, F const &functor):
		m_expression(e()), m_functor(functor) {}

	// Accessors
	size_type size() const {
		return m_expression.size();
	}

	// Expression accessors
	expression_closure_type const &expression() const {
		return m_expression;
	}
	
	//computation kernels
	template<class VecX>
	void assign_to(vector_expression<VecX>& x, scalar_type alpha = scalar_type(1) )const{
		//compute this by first assigning the result of the argument and then applying
		//the function to every element
		assign(x,m_expression);
		typename VecX::iterator end=x().end();
		for(typename VecX::iterator pos =x().begin(); pos != end; ++pos){
			*pos= alpha * m_functor(*pos);
		}
	}
	template<class VecX>
	void plus_assign_to(vector_expression<VecX>& x, scalar_type alpha = scalar_type(1) )const{
		//First assign result of this expression to a temporary and then perform plus_assignment to x
		typename vector_temporary<self_type>::type temporary(size());
		assign_to(temporary,alpha);
		plus_assign_to(x,temporary);
	}
	
	template<class VecX>
	void minus_assign_to(vector_expression<VecX>& x, scalar_type alpha = scalar_type(1) )const{
		//First assign result of this expression to a temporary and then perform minus_assignment to x
		typename vector_temporary<self_type>::type temporary(size());
		assign_to(temporary,alpha);
		minus_assign_to(x,temporary);
	}

public:
	// Element access
	const_reference operator()(index_type i) const {
		return m_functor(m_expression(i));
	}

	const_reference operator[](index_type i) const {
		return m_functor(m_expression[i]);
	}

	typedef transform_iterator<typename E::const_iterator,functor_type> const_iterator;
	typedef const_iterator iterator;

	// Element lookup
	const_iterator begin() const {
		return const_iterator(m_expression.begin(),m_functor);
	}
	const_iterator end() const {
		return const_iterator(m_expression.end(),m_functor);
	}
private:
	expression_closure_type m_expression;
	F m_functor;
};



#define SHARK_UNARY_VECTOR_TRANSFORMATION(name, F)\
template<class E>\
vector_unary<E,F<typename E::value_type> >\
name(vector_expression<E> const& e){\
	typedef F<typename E::value_type> functor_type;\
	return vector_unary<E, functor_type>(e, functor_type());\
}
SHARK_UNARY_VECTOR_TRANSFORMATION(abs, scalar_abs)
SHARK_UNARY_VECTOR_TRANSFORMATION(log, scalar_log)
SHARK_UNARY_VECTOR_TRANSFORMATION(exp, scalar_exp)
SHARK_UNARY_VECTOR_TRANSFORMATION(cos, scalar_cos)
SHARK_UNARY_VECTOR_TRANSFORMATION(sin, scalar_sin)
SHARK_UNARY_VECTOR_TRANSFORMATION(tanh,scalar_tanh)
SHARK_UNARY_VECTOR_TRANSFORMATION(atanh,scalar_atanh)
SHARK_UNARY_VECTOR_TRANSFORMATION(sqr, scalar_sqr)
SHARK_UNARY_VECTOR_TRANSFORMATION(abs_sqr, scalar_abs_sqr)
SHARK_UNARY_VECTOR_TRANSFORMATION(sqrt, scalar_sqrt)
SHARK_UNARY_VECTOR_TRANSFORMATION(sigmoid, scalar_sigmoid)
SHARK_UNARY_VECTOR_TRANSFORMATION(softPlus, scalar_soft_plus)
SHARK_UNARY_VECTOR_TRANSFORMATION(elem_inv, scalar_inverse)
#undef SHARK_UNARY_VECTOR_TRANSFORMATION


//operations of the form op(v,t)[i] = op(v[i],t)
#define SHARK_VECTOR_SCALAR_TRANSFORMATION(name, F)\
template<class T, class E> \
typename boost::enable_if< \
	boost::is_convertible<T, typename E::value_type >,\
        vector_unary<E,F<typename E::value_type,T> > \
>::type \
name (vector_expression<E> const& e, T scalar){ \
	typedef F<typename E::value_type,T> functor_type; \
	return vector_unary<E, functor_type>(e, functor_type(scalar)); \
}
SHARK_VECTOR_SCALAR_TRANSFORMATION(operator/, scalar_divide)
SHARK_VECTOR_SCALAR_TRANSFORMATION(operator<, scalar_less_than)
SHARK_VECTOR_SCALAR_TRANSFORMATION(operator<=, scalar_less_equal_than)
SHARK_VECTOR_SCALAR_TRANSFORMATION(operator>, scalar_bigger_than)
SHARK_VECTOR_SCALAR_TRANSFORMATION(operator>=, scalar_bigger_equal_than)
SHARK_VECTOR_SCALAR_TRANSFORMATION(operator==, scalar_equal)
SHARK_VECTOR_SCALAR_TRANSFORMATION(operator!=, scalar_not_equal)
SHARK_VECTOR_SCALAR_TRANSFORMATION(min, scalar_min)
SHARK_VECTOR_SCALAR_TRANSFORMATION(max, scalar_max)
SHARK_VECTOR_SCALAR_TRANSFORMATION(pow, scalar_pow)
#undef SHARK_VECTOR_SCALAR_TRANSFORMATION

// operations of the form op(t,v)[i] = op(t,v[i])
#define SHARK_VECTOR_SCALAR_TRANSFORMATION_2(name, F)\
template<class T, class E> \
typename boost::enable_if< \
	boost::is_convertible<T, typename E::value_type >,\
        vector_unary<E,F<typename E::value_type,T> > \
>::type \
name (T scalar, vector_expression<E> const& e){ \
	typedef F<typename E::value_type,T> functor_type; \
	return vector_unary<E, functor_type>(e, functor_type(scalar)); \
}
SHARK_VECTOR_SCALAR_TRANSFORMATION_2(min, scalar_min)
SHARK_VECTOR_SCALAR_TRANSFORMATION_2(max, scalar_max)
#undef SHARK_VECTOR_SCALAR_TRANSFORMATION_2

template<class E1, class E2>
class vector_addition: public vector_expression<vector_addition<E1,E2> > {
private:
	typedef scalar_binary_plus<
		typename E1::value_type,
		typename E2::value_type
	> functor_type;
public:
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;
	typedef typename functor_type::result_type value_type;
	typedef value_type scalar_type;
	typedef value_type const_reference;
	typedef value_type reference;
	typedef value_type const * const_pointer;
	typedef value_type const*  pointer;

	typedef typename E1::index_type index_type;
	typedef typename E1::const_index_pointer const_index_pointer;
	typedef typename index_pointer<E1>::type index_pointer;

	typedef typename E1::const_closure_type expression_closure1_type;
	typedef typename E2::const_closure_type expression_closure2_type;
	
	typedef vector_addition<E1,E2> const_closure_type;
	typedef const_closure_type closure_type;
	typedef unknown_storage_tag storage_category;
	typedef typename evaluation_restrict_traits<E1,E2>::type evaluation_category;

	// Construction and destruction
	explicit vector_addition (
		expression_closure1_type e1, 
		expression_closure2_type e2
	):m_expression1(e1),m_expression2(e2){
		SIZE_CHECK(e1.size() == e2.size());
	}

	// Accessors
	size_type size() const {
		return m_expression1.size();
	}

	// Expression accessors
	expression_closure1_type const& expression1() const {
		return m_expression1;
	}
	expression_closure2_type const& expression2() const {
		return m_expression2;
	}

	// Element access
	const_reference operator() (index_type i) const {
		SIZE_CHECK(i < size());
		return m_expression1(i) + m_expression2(i);
	}

	const_reference operator[] (index_type i) const {
		SIZE_CHECK(i < size());
		return m_expression1(i) + m_expression2(i);
	}
	
	//computation kernels
	template<class VecX>
	void assign_to(vector_expression<VecX>& x, scalar_type alpha = scalar_type(1) )const{
		assign(x,alpha*m_expression1);
		plus_assign(x,alpha*m_expression2);
	}
	template<class VecX>
	void plus_assign_to(vector_expression<VecX>& x, scalar_type alpha = scalar_type(1) )const{
		plus_assign(x,alpha*m_expression1);
		plus_assign(x,alpha*m_expression2);
	}
	
	template<class VecX>
	void minus_assign_to(vector_expression<VecX>& x, scalar_type alpha = scalar_type(1) )const{
		minus_assign(x,alpha*m_expression1);
		minus_assign(x,alpha*m_expression2);
	}

	// Iterator types
	typedef binary_transform_iterator<
		typename E1::const_iterator,
		typename E2::const_iterator,
		functor_type
	> const_iterator;
	typedef const_iterator iterator;

	const_iterator begin () const {
		return const_iterator(functor_type(),
			m_expression1.begin(),m_expression1.end(),
			m_expression2.begin(),m_expression2.end()
		);
	}
	const_iterator end() const {
		return const_iterator(functor_type(),
			m_expression1.end(),m_expression1.end(),
			m_expression2.end(),m_expression2.end()
		);
	}

private:
	expression_closure1_type m_expression1;
	expression_closure2_type m_expression2;
};

///\brief Adds two vectors
template<class E1, class E2>
vector_addition<E1, E2 > operator+ (
	vector_expression<E1> const& e1,
	vector_expression<E2> const& e2
){
	return vector_addition<E1, E2>(e1(),e2());
}
///\brief Subtracts two vectors
template<class E1, class E2>
vector_addition<E1, vector_scalar_multiply<E2> > operator- (
	vector_expression<E1> const& e1,
	vector_expression<E2> const& e2
){
	return vector_addition<E1, vector_scalar_multiply<E2> >(e1(),-e2());
}

///\brief Adds a vector plus a scalr which is interpreted as a constant vector
template<class E, class T>
typename boost::enable_if<
	boost::is_convertible<T, typename E::value_type>, 
	vector_addition<E, scalar_vector<T> >
>::type operator+ (
	vector_expression<E> const& e,
	T t
){
	return e + scalar_vector<T>(e().size(),t);
}

///\brief Adds a vector plus a scalar which is interpreted as a constant vector
template<class T, class E>
typename boost::enable_if<
	boost::is_convertible<T, typename E::value_type>,
	vector_addition<E, scalar_vector<T> >
>::type operator+ (
	T t,
	vector_expression<E> const& e
){
	return e + scalar_vector<T>(e().size(),t);
}

///\brief Subtracts a scalar which is interpreted as a constant vector from a vector.
template<class E, class T>
typename boost::enable_if<
	boost::is_convertible<T, typename E::value_type> ,
	vector_addition<E, vector_scalar_multiply<scalar_vector<T> > >
>::type operator- (
	vector_expression<E> const& e,
	T t
){
	return e - scalar_vector<T>(e().size(),t);
}

///\brief Subtracts a vector from a scalar which is interpreted as a constant vector
template<class E, class T>
typename boost::enable_if<
	boost::is_convertible<T, typename E::value_type>,
	vector_addition<scalar_vector<T>, vector_scalar_multiply<E> >
>::type operator- (
	T t,
	vector_expression<E> const& e
){
	return scalar_vector<T>(e().size(),t) - e;
}


template<class E1, class E2, class F>
class vector_binary:
	public vector_expression<vector_binary<E1,E2, F> > {
	typedef vector_binary<E1,E2, F> self_type;
	typedef E1 const expression1_type;
	typedef E2 const expression2_type;
public:
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;
	typedef typename F::result_type value_type;
	typedef value_type scalar_type;
	typedef value_type const_reference;
	typedef value_type reference;
	typedef value_type const * const_pointer;
	typedef value_type const*  pointer;

	typedef typename E1::index_type index_type;
	typedef typename E1::const_index_pointer const_index_pointer;
	typedef typename index_pointer<E1>::type index_pointer;

	typedef F functor_type;
	typedef typename E1::const_closure_type expression_closure1_type;
	typedef typename E2::const_closure_type expression_closure2_type;
	
	typedef self_type const const_closure_type;
	typedef self_type closure_type;
	typedef unknown_storage_tag storage_category;
	typedef typename evaluation_restrict_traits<E1,E2>::type evaluation_category;

	// Construction and destruction
	explicit vector_binary (
		expression_closure1_type e1, 
		expression_closure2_type e2,
		F functor
	):m_expression1(e1),m_expression2(e2), m_functor(functor) {
		SIZE_CHECK(e1.size() == e2.size());
	}

	// Accessors
	size_type size() const {
		return m_expression1.size ();
	}

	// Expression accessors
	expression_closure1_type const& expression1() const {
		return m_expression1;
	}
	expression_closure2_type const& expression2() const {
		return m_expression2;
	}

	// Element access
	const_reference operator() (index_type i) const {
		SIZE_CHECK(i < size());
		return m_functor(m_expression1(i),m_expression2(i));
	}

	const_reference operator[] (index_type i) const {
		SIZE_CHECK(i < size());
		return m_functor(m_expression1(i),m_expression2(i));
	}

	// Iterator types
	
	// Iterator enhances the iterator of the referenced expressions
	// with the unary functor.
	typedef binary_transform_iterator<
		typename E1::const_iterator,
		typename E2::const_iterator,
		functor_type
	> const_iterator;
	typedef const_iterator iterator;

	const_iterator begin () const {
		return const_iterator (m_functor,
			m_expression1.begin(),m_expression1.end(),
			m_expression2.begin(),m_expression2.end()
		);
	}
	const_iterator end() const {
		return const_iterator (m_functor,
			m_expression1.end(),m_expression1.end(),
			m_expression2.end(),m_expression2.end()
		);
	}

private:
	expression_closure1_type m_expression1;
	expression_closure2_type m_expression2;
	F m_functor;
};

#define SHARK_BINARY_VECTOR_EXPRESSION(name, F)\
template<class E1, class E2>\
vector_binary<E1, E2, F<typename E1::value_type, typename E2::value_type> >\
name(vector_expression<E1> const& e1, vector_expression<E2> const& e2){\
	typedef F<typename E1::value_type, typename E2::value_type> functor_type;\
	return vector_binary<E1, E2, functor_type>(e1(),e2(), functor_type());\
}
SHARK_BINARY_VECTOR_EXPRESSION(operator*, scalar_binary_multiply)
SHARK_BINARY_VECTOR_EXPRESSION(element_prod, scalar_binary_multiply)
SHARK_BINARY_VECTOR_EXPRESSION(operator/, scalar_binary_divide)
SHARK_BINARY_VECTOR_EXPRESSION(element_div, scalar_binary_divide)
SHARK_BINARY_VECTOR_EXPRESSION(min, scalar_binary_min)
SHARK_BINARY_VECTOR_EXPRESSION(max, scalar_binary_max)
#undef SHARK_BINARY_VECTOR_EXPRESSION

template<class E1, class E2>
vector_binary<E1, E2, 
	scalar_binary_safe_divide<typename E1::value_type, typename E2::value_type> 
>
safe_div(
	vector_expression<E1> const& e1, 
	vector_expression<E2> const& e2, 
	typename promote_traits<
		typename E1::value_type, 
		typename E2::value_type
	>::promote_type defaultValue
){
	typedef scalar_binary_safe_divide<typename E1::value_type, typename E2::value_type> functor_type;
	return vector_binary<E1, E2, functor_type>(e1(),e2(), functor_type(defaultValue));
}

/////VECTOR REDUCTIONS

/// \brief sum v = sum_i v_i
template<class E>
typename E::value_type
sum(const vector_expression<E> &e) {
	typedef typename E::value_type value_type;
	vector_fold<scalar_binary_plus<value_type, value_type> > kernel;
	return kernel(e,value_type());
}

/// \brief max v = max_i v_i
template<class E>
typename E::value_type
max(const vector_expression<E> &e) {
	typedef typename E::value_type value_type;
	vector_fold<scalar_binary_max<value_type, value_type> > kernel;
	return kernel(e,e()(0));
}

/// \brief min v = min_i v_i
template<class E>
typename E::value_type
min(const vector_expression<E> &e) {
	typedef typename E::value_type value_type;
	vector_fold<scalar_binary_min<value_type, value_type> > kernel;
	return kernel(e,e()(0));
}

/// \brief arg_max v = arg max_i v_i
template<class E>
std::size_t arg_max(const vector_expression<E> &e) {
	SIZE_CHECK(e().size() > 0);
	return std::max_element(e().begin(),e().end()).index();
}

/// \brief arg_min v = arg min_i v_i
template<class E>
std::size_t arg_min(const vector_expression<E> &e) {
	SIZE_CHECK(e().size() > 0);
	return arg_max(-e);
}

/// \brief soft_max v = ln(sum(exp(v)))
///
/// Be aware that this is NOT the same function as used in machine learning: exp(v)/sum(exp(v))
///
/// The function is computed in an numerically stable way to prevent that too high values of v_i produce inf or nan.
/// The name of the function comes from the fact that it behaves like a continuous version of max in the respect that soft_max v <= v.size()*max(v)
/// max is reached in the limit as the gap between the biggest value and the rest grows to infinity.
template<class E>
typename E::value_type
soft_max(const vector_expression<E> &e) {
	typename E::value_type maximum = max(e);
	return std::log(sum(exp(e - maximum))) + maximum;
}


////implement all the norms based on sum!

/// \brief norm_1 v = sum_i |v_i|
template<class E>
typename real_traits<typename E::value_type >::type
norm_1(const vector_expression<E> &e) {
	return sum(abs(eval_block(e)));
}

/// \brief norm_2 v = sum_i |v_i|^2
template<class E>
typename real_traits<typename E::value_type >::type
norm_sqr(const vector_expression<E> &e) {
	return sum(abs_sqr(eval_block(e)));
}

/// \brief norm_2 v = sqrt (sum_i |v_i|^2 )
template<class E>
typename real_traits<typename E::value_type >::type
norm_2(const vector_expression<E> &e) {
	using std::sqrt;
	return sqrt(norm_sqr(e));
}

/// \brief norm_inf v = max_i |v_i|
template<class E>
typename real_traits<typename E::value_type >::type
norm_inf(vector_expression<E> const &e){
	return max(abs(eval_block(e)));
}

/// \brief index_norm_inf v = arg max_i |v_i|
template<class E>
std::size_t index_norm_inf(vector_expression<E> const &e){
	return arg_max(abs(eval_block(e)));
}

// inner_prod (v1, v2) = sum_i v1_i * v2_i
template<class E1, class E2>
typename promote_traits<
	typename E1::value_type,
	typename E2::value_type
>::promote_type
inner_prod(
	vector_expression<E1> const& e1,
	vector_expression<E2> const& e2
) {
	typedef typename promote_traits<
		typename E1::value_type,
		typename E2::value_type
	>::promote_type value_type;
	value_type result = value_type();
	kernels::dot(eval_block(e1),eval_block(e2),result);
	return result;
}

}

}

#endif
