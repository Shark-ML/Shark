#ifndef SHARK_LINALG_BLAS_MATRIX_EXPRESSION_HPP
#define SHARK_LINALG_BLAS_MATRIX_EXPRESSION_HPP

#include <boost/type_traits/is_convertible.hpp> 
#include "matrix_proxy.hpp"
#include "operation.hpp"

namespace shark {
namespace blas {

template<class E1, class E2, class F>
class vector_matrix_binary:public matrix_expression<vector_matrix_binary<E1, E2, F> > {
	typedef vector_matrix_binary<E1, E2, F> self_type;
		
	class Binder1{
	public:
		typedef typename F::argument2_type argument_type;
		typedef typename F::result_type result_type;
		
		Binder1(F functor, typename F::argument1_type argument)
		: m_functor(functor), m_argument(argument){}
			
		result_type operator()(argument_type x)const {
			return m_functor(m_argument,x);
		}
	private:
		F m_functor;
		typename F::argument1_type m_argument;
	};
	
	class Binder2{
	public:
		typedef typename F::argument1_type argument_type;
		typedef typename F::result_type result_type;
		
		Binder2(F functor, typename F::argument2_type argument)
		: m_functor(functor), m_argument(argument){}
			
		result_type operator()(argument_type x)const {
			return m_functor(x, m_argument);
		}
	private:
		F m_functor;
		typename F::argument2_type m_argument;
	};
public:

	typedef typename E1::const_closure_type expression1_closure_type;
	typedef typename E2::const_closure_type expression2_closure_type;

	typedef F functor_type;
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;
	typedef typename F::result_type value_type;
	typedef value_type scalar_type;
	typedef value_type const_reference;
	typedef const_reference reference;
	typedef value_type const* const_pointer;
	typedef const_pointer pointer;

	typedef typename E1::index_type index_type;
	typedef typename E1::const_index_pointer const_index_pointer;
	typedef typename index_pointer<E1>::type index_pointer;

	typedef const self_type const_closure_type;
	typedef const_closure_type closure_type;
	typedef unknown_orientation orientation;
	typedef unknown_storage_tag storage_category;

	// Construction and destruction
	
	vector_matrix_binary(expression1_closure_type e1, expression2_closure_type e2, functor_type functor)
	:m_expression1(e1), m_expression2(e2), m_functor(functor) {}

	// Accessors
	size_type size1() const {
		return m_expression1.size();
	}
	size_type size2() const {
		return m_expression2.size();
	}

	// Expression accessors
	const expression1_closure_type &expression1() const {
		return m_expression1;
	}
	const expression2_closure_type &expression2() const {
		return m_expression2;
	}
	// Element access
	const_reference operator()(index_type i, index_type j) const {
		return m_functor(m_expression1(i), m_expression2(j));
	}

	// Closure comparison
	bool same_closure(const vector_matrix_binary &vmb) const {
		return expression1().same_closure(vmb.expression1()) &&
		       expression2().same_closure(vmb.expression2());
	}

	typedef transform_iterator<typename E2::const_iterator,Binder1> const_row_iterator;
	typedef transform_iterator<typename E1::const_iterator,Binder2> const_column_iterator;
	typedef const_row_iterator row_iterator;
	typedef const_column_iterator column_iterator;
	
	const_row_iterator row_begin(index_type i) const {
		return const_row_iterator(m_expression2.begin(),
			Binder1(m_functor,m_expression1(i))
		);
	}
	const_row_iterator row_end(index_type i) const {
		return const_row_iterator(m_expression2.end(),
			Binder1(m_functor,m_expression1(i))
		);
	}

	const_column_iterator column_begin(index_type i) const {
		return const_column_iterator(m_expression1.begin(),
			Binder2(m_functor,m_expression2(i))
		);
	}
	const_column_iterator column_end(index_type i) const {
		return const_column_iterator(m_expression1.end(),
			Binder2(m_functor,m_expression2(i))
		);
	}
private:
	expression1_closure_type m_expression1;
	expression2_closure_type m_expression2;
	functor_type m_functor;
};

template<class E1, class E2, class F>
struct vector_matrix_binary_traits {
	typedef vector_matrix_binary<E1, E2, F> expression_type;
	typedef expression_type result_type;
};

// (outer_prod (v1, v2)) [i] [j] = v1 [i] * v2 [j]
template<class E1, class E2>
vector_matrix_binary<E1, E2, scalar_binary_multiply<typename E1::value_type, typename E2::value_type> >
outer_prod(
	vector_expression<E1> const& e1,
        vector_expression<E2> const& e2
) {
	typedef scalar_binary_multiply<typename E1::value_type, typename E2::value_type> Multiplier;
	return vector_matrix_binary<E1, E2, Multiplier>(e1(), e2(),Multiplier());
}

// (outer_plus (v1, v2)) [i] [j] = v1 [i] + v2 [j]
template<class E1, class E2>
vector_matrix_binary<E1, E2, scalar_binary_multiply<typename E1::value_type, typename E2::value_type> >
outer_plus(
	vector_expression<E1> const& e1,
        vector_expression<E2> const& e2
) {
	typedef scalar_binary_plus<typename E1::value_type, typename E2::value_type> Multiplier;
	return vector_matrix_binary<E1, E2, Multiplier>(e1(), e2(),Multiplier());
}

template<class V>
class vector_repeater:public blas::matrix_expression<vector_repeater<V> > {
private:
	typedef V expression_type;
	typedef vector_repeater<V> self_type;
	typedef typename V::const_iterator const_subiterator_type;
public:
	typedef typename V::const_closure_type expression_closure_type;
	typedef typename V::size_type size_type;
	typedef typename V::difference_type difference_type;
	typedef typename V::value_type value_type;
	typedef value_type scalar_type;
	typedef value_type const_reference;
	typedef const_reference reference;
	typedef value_type const* pointer;
	typedef value_type const* const_pointer;

	typedef typename V::index_type index_type;
	typedef typename V::const_index_pointer const_index_pointer;
	typedef typename index_pointer<V>::type index_pointer;

	typedef self_type const const_closure_type;
	typedef const_closure_type closure_type;
	typedef blas::row_major orientation;
	typedef blas::unknown_storage_tag storage_category;

	// Construction and destruction
	explicit vector_repeater (expression_type const& e, std::size_t rows):
	m_vector(e), m_rows(rows) {}

	// Accessors
	size_type size1() const {
		return m_rows;
	}
	size_type size2() const {
		return m_vector.size();
	}

	// Expression accessors
	const expression_closure_type &expression () const {
		return m_vector;
	}

	// Element access
	const_reference operator() (index_type i, index_type j) const {
		return m_vector(j);
	}

	// Closure comparison
	bool same_closure (const vector_repeater &other) const {
		return (*this).expression ().same_closure (other.expression ());
	}

	// Iterator types
	typedef typename V::const_iterator const_row_iterator;
	typedef const_row_iterator row_iterator;
	typedef blas::constant_iterator<value_type>  const_column_iterator;
	typedef const_column_iterator column_iterator;

	// Element lookup
	
	const_row_iterator row_begin(std::size_t i) const {
		RANGE_CHECK( i < size1());
		return m_vector.begin();
	}
	const_row_iterator row_end(std::size_t i) const {
		RANGE_CHECK( i < size1());
		return m_vector.end();
	}
	
	const_column_iterator column_begin(std::size_t j) const {
		RANGE_CHECK( j < size2());
		return const_column_iterator(0,m_vector(j));
	}
	const_column_iterator column_end(std::size_t j) const {
		RANGE_CHECK( j < size2());
		return const_column_iterator(size1(),m_vector(j));
	}
private:
	expression_closure_type m_vector;
	std::size_t m_rows;
};

///\brief class which allows for matrix transformations
///
///transforms a matrix expression e of type E using a Functiof f of type F as an elementwise transformation f(e(i,j))
///This transformation needs to leave f constant, meaning that applying f(x), f(y), f(z) yields the same results independent of the
///order of application. Also F must provide a type F::result_type indicating the result type of the functor.
///F must further provide a boolean flag F::zero_identity which indicates that f(0) = 0. This is needed for correct usage with sparse
///arguments - if f(0) != 0 this expression will be dense!
///todo: desification is not implemented
template<class E, class F>
class matrix_unary:public blas::matrix_expression<matrix_unary<E, F> > {
private:
	typedef matrix_unary<E, F> self_type;
	typedef E expression_type;
	typedef typename expression_type::const_row_iterator const_subrow_iterator_type;
	typedef typename expression_type::const_column_iterator const_subcolumn_iterator_type;

public:
	typedef typename expression_type::const_closure_type expression_closure_type;

	typedef F functor_type;
	typedef typename functor_type::result_type value_type;
	typedef value_type scalar_type;
	typedef value_type const_reference;
	typedef const_reference reference;
	typedef value_type const *const_pointer;
	typedef value_type *pointer;
	typedef typename expression_type::size_type size_type;
	typedef typename expression_type::difference_type difference_type;

	typedef typename E::index_type index_type;
	typedef typename E::const_index_pointer const_index_pointer;
	typedef typename index_pointer<E>::type index_pointer;

	typedef self_type const const_closure_type;
	typedef const_closure_type closure_type;
	typedef typename E::orientation orientation;
	typedef blas::unknown_storage_tag storage_category;

	// Construction and destruction
	matrix_unary(blas::matrix_expression<E> const &e, F const &functor):
		m_expression(e()), m_functor(functor) {}

	// Accessors
	size_type size1() const {
		return m_expression.size1();
	}
	size_type size2() const {
		return m_expression.size2();
	}

	// Element access
	const_reference operator()(index_type i, index_type j) const {
		return m_functor(m_expression(i, j));
	}

	// Closure comparison
	bool same_closure(matrix_unary const &other) const {
		return m_expression.same_closure(other.m_expression);
	}

	// Iterator types
	typedef transform_iterator<typename E::const_row_iterator, F> const_row_iterator;
	typedef transform_iterator<typename E::const_column_iterator, F> const_column_iterator;
	typedef const_row_iterator row_iterator;
	typedef const_column_iterator column_iterator;
	
	const_row_iterator row_begin(index_type i) const {
		return const_row_iterator(m_expression.row_begin(i),m_functor);
	}
	const_row_iterator row_end(index_type i) const {
		return const_row_iterator(m_expression.row_end(i),m_functor);
	}

	const_column_iterator column_begin(index_type i) const {
		return const_row_iterator(m_expression.column_begin(i),m_functor);
	}
	const_column_iterator column_end(index_type i) const {
		return const_row_iterator(m_expression.column_end(i),m_functor);
	}

private:
	expression_closure_type m_expression;
	functor_type m_functor;
};

#define SHARK_UNARY_MATRIX_TRANSFORMATION(name, F)\
template<class E>\
matrix_unary<E,F<typename E::value_type> >\
name(matrix_expression<E> const& e){\
	typedef F<typename E::value_type> functor_type;\
	return matrix_unary<E, functor_type>(e, functor_type());\
}
SHARK_UNARY_MATRIX_TRANSFORMATION(operator-, scalar_negate)
SHARK_UNARY_MATRIX_TRANSFORMATION(conj, scalar_conj)
SHARK_UNARY_MATRIX_TRANSFORMATION(real, scalar_real)
SHARK_UNARY_MATRIX_TRANSFORMATION(imag, scalar_imag)
SHARK_UNARY_MATRIX_TRANSFORMATION(abs, scalar_abs)
SHARK_UNARY_MATRIX_TRANSFORMATION(log, scalar_log)
SHARK_UNARY_MATRIX_TRANSFORMATION(exp, scalar_exp)
SHARK_UNARY_MATRIX_TRANSFORMATION(sin, scalar_sin)
SHARK_UNARY_MATRIX_TRANSFORMATION(cos, scalar_cos)
SHARK_UNARY_MATRIX_TRANSFORMATION(tanh,scalar_tanh)
SHARK_UNARY_MATRIX_TRANSFORMATION(sqr, scalar_sqr)
SHARK_UNARY_MATRIX_TRANSFORMATION(abs_sqr, scalar_abs_sqr)
SHARK_UNARY_MATRIX_TRANSFORMATION(sqrt, scalar_sqrt)
SHARK_UNARY_MATRIX_TRANSFORMATION(sigmoid, scalar_sigmoid)
SHARK_UNARY_MATRIX_TRANSFORMATION(softPlus, scalar_soft_plus)
SHARK_UNARY_MATRIX_TRANSFORMATION(elem_inv, scalar_inverse)
#undef SHARK_UNARY_MATRIX_TRANSFORMATION

#define SHARK_MATRIX_SCALAR_TRANSFORMATION(name, F)\
template<class E, class T> \
typename boost::enable_if< \
	boost::is_convertible<T, typename E::value_type >,\
        matrix_unary<E,F<typename E::value_type,T> > \
>::type \
name (matrix_expression<E> const& e, T scalar){ \
	typedef F<typename E::value_type, T> functor_type; \
	return matrix_unary<E, functor_type>(e, functor_type(scalar)); \
}
SHARK_MATRIX_SCALAR_TRANSFORMATION(operator+, scalar_add)
SHARK_MATRIX_SCALAR_TRANSFORMATION(operator-, scalar_subtract2)
SHARK_MATRIX_SCALAR_TRANSFORMATION(operator*, scalar_multiply2)
SHARK_MATRIX_SCALAR_TRANSFORMATION(operator/, scalar_divide)
SHARK_MATRIX_SCALAR_TRANSFORMATION(operator<, scalar_less_than)
SHARK_MATRIX_SCALAR_TRANSFORMATION(operator<=, scalar_less_equal_than)
SHARK_MATRIX_SCALAR_TRANSFORMATION(operator>, scalar_bigger_than)
SHARK_MATRIX_SCALAR_TRANSFORMATION(operator>=, scalar_bigger_equal_than)
SHARK_MATRIX_SCALAR_TRANSFORMATION(operator==, scalar_equal)
SHARK_MATRIX_SCALAR_TRANSFORMATION(operator!=, scalar_not_equal)
SHARK_MATRIX_SCALAR_TRANSFORMATION(min, scalar_min)
SHARK_MATRIX_SCALAR_TRANSFORMATION(max, scalar_max)
SHARK_MATRIX_SCALAR_TRANSFORMATION(pow, scalar_pow)
#undef SHARK_MATRIX_SCALAR_TRANSFORMATION

// operations of the form op(t,v)[i,j] = op(t,v[i,j])
#define SHARK_MATRIX_SCALAR_TRANSFORMATION_2(name, F)\
template<class T, class E> \
typename boost::enable_if< \
	boost::is_convertible<T, typename E::value_type >,\
        matrix_unary<E,F<typename E::value_type,T> > \
>::type \
name (T scalar, matrix_expression<E> const& e){ \
	typedef F<typename E::value_type, T> functor_type; \
	return matrix_unary<E, functor_type>(e, functor_type(scalar)); \
}
SHARK_MATRIX_SCALAR_TRANSFORMATION_2(operator+, scalar_add)
SHARK_MATRIX_SCALAR_TRANSFORMATION_2(operator-, scalar_subtract1)
SHARK_MATRIX_SCALAR_TRANSFORMATION_2(operator*, scalar_multiply1)
SHARK_MATRIX_SCALAR_TRANSFORMATION_2(min, scalar_min)
SHARK_MATRIX_SCALAR_TRANSFORMATION_2(max, scalar_max)
#undef SHARK_MATRIX_SCALAR_TRANSFORMATION_2

template<class E1, class E2, class F>
class matrix_binary:
	public blas::matrix_expression<matrix_binary<E1, E2, F> > {
private:
	typedef matrix_binary<E1, E2, F> self_type;
public:
	typedef E1 expression1_type;
	typedef E2 expression2_type;
	typedef typename E1::const_closure_type expression1_closure_type;
	typedef typename E2::const_closure_type expression2_closure_type;

	typedef typename E1::size_type size_type;
	typedef typename E1::difference_type difference_type;
	typedef typename F::result_type value_type;
	typedef value_type scalar_type;
	typedef value_type const_reference;
	typedef const_reference reference;
	typedef value_type const* const_pointer;
	typedef const_pointer pointer;

	typedef typename E1::index_type index_type;
	typedef typename E1::const_index_pointer const_index_pointer;
	typedef typename index_pointer<E1>::type index_pointer;

	typedef const matrix_binary<E1, E2, F> const_closure_type;
	typedef const_closure_type closure_type;
	typedef typename E1::orientation orientation;
	typedef blas::unknown_storage_tag storage_category;

	typedef F functor_type;

        // Construction and destruction

        matrix_binary (
		matrix_expression<E1> const&e1,  matrix_expression<E2> const& e2, functor_type functor 
	): m_expression1 (e1()), m_expression2 (e2()),m_functor(functor) {}

        // Accessors
        size_type size1 () const {
		return m_expression1.size1 ();
        }
        size_type size2 () const {
		return m_expression1.size2 ();
        }

        const_reference operator () (index_type i, index_type j) const {
		return m_functor( m_expression1 (i, j), m_expression2(i,j));
        }

        // Closure comparison
        bool same_closure (matrix_binary const&mbs2) const {
		return m_expression1.same_closure (mbs2.m_expression1) ||
		m_expression2.same_closure (mbs2.m_expression2);
        }

	// Iterator types
private:
	typedef typename E1::const_row_iterator const_row_iterator1_type;
	typedef typename E1::const_column_iterator const_row_column_iterator_type;
	typedef typename E2::const_row_iterator const_column_iterator1_type;
	typedef typename E2::const_column_iterator const_column_iterator2_type;

public:
	typedef binary_transform_iterator<
		typename E1::const_row_iterator,typename E2::const_row_iterator,functor_type
	> const_row_iterator;
	typedef binary_transform_iterator<
		typename E1::const_column_iterator,typename E2::const_column_iterator,functor_type
	> const_column_iterator;
	typedef const_row_iterator row_iterator;
	typedef const_column_iterator column_iterator;

	const_row_iterator row_begin(std::size_t i) const {
		return const_row_iterator (m_functor,
			m_expression1.row_begin(i),m_expression1.row_end(i),
			m_expression2.row_begin(i),m_expression2.row_end(i)
		);
	}
	const_row_iterator row_end(std::size_t i) const {
		return const_row_iterator (m_functor,
			m_expression1.row_end(i),m_expression1.row_end(i),
			m_expression2.row_end(i),m_expression2.row_end(i)
		);
	}

	const_column_iterator column_begin(std::size_t j) const {
		return const_column_iterator (m_functor,
			m_expression1.column_begin(j),m_expression1.column_end(j),
			m_expression2.column_begin(j),m_expression2.column_end(j)
		);
	}
	const_column_iterator column_end(std::size_t j) const {
		return const_column_iterator (m_functor,
			m_expression1.column_begin(j),m_expression1.column_end(j),
			m_expression2.column_begin(j),m_expression2.column_end(j)
		);
	}

private:
	expression1_closure_type m_expression1;
        expression2_closure_type m_expression2;
	functor_type m_functor;
};

#define SHARK_BINARY_MATRIX_EXPRESSION(name, F)\
template<class E1, class E2>\
matrix_binary<E1, E2, F<typename E1::value_type, typename E2::value_type> >\
name(matrix_expression<E1> const& e1, matrix_expression<E2> const& e2){\
	typedef F<typename E1::value_type, typename E2::value_type> functor_type;\
	return matrix_binary<E1, E2, functor_type>(e1,e2, functor_type());\
}
SHARK_BINARY_MATRIX_EXPRESSION(operator+, scalar_binary_plus)
SHARK_BINARY_MATRIX_EXPRESSION(operator-, scalar_binary_minus)
SHARK_BINARY_MATRIX_EXPRESSION(operator*, scalar_binary_multiply)
SHARK_BINARY_MATRIX_EXPRESSION(element_prod, scalar_binary_multiply)
SHARK_BINARY_MATRIX_EXPRESSION(operator/, scalar_binary_divide)
SHARK_BINARY_MATRIX_EXPRESSION(element_div, scalar_binary_divide)
#undef SHARK_BINARY_MATRIX_EXPRESSION

template<class E1, class E2>
matrix_binary<E1, E2, 
	scalar_binary_safe_divide<typename E1::value_type, typename E2::value_type> 
>
safe_div(
	matrix_expression<E1> const& e1, 
	matrix_expression<E2> const& e2, 
	typename promote_traits<
		typename E1::value_type, 
		typename E2::value_type
	>::promote_type defaultValue
){
	typedef scalar_binary_safe_divide<typename E1::value_type, typename E2::value_type> functor_type;
	return matrix_binary<E1, E2, functor_type>(e1,e2, functor_type(defaultValue));
}

template<class E1, class E2, class F>
class matrix_vector_binary1:
	public vector_expression<matrix_vector_binary1<E1, E2, F> > {

public:
	typedef E1 expression1_type;
	typedef E2 expression2_type;
	typedef F functor_type;
	typedef typename E1::const_closure_type expression1_closure_type;
	typedef typename E2::const_closure_type expression2_closure_type;
private:
	typedef matrix_vector_binary1<E1, E2, F> self_type;
public:
	typedef typename promote_traits<
		typename E1::size_type, typename E2::size_type
	>::promote_type size_type;
	typedef typename promote_traits<
		typename E1::difference_type, typename E2::difference_type
	>::promote_type difference_type;
	typedef typename F::result_type value_type;
	typedef value_type scalar_type;
	typedef value_type const_reference;
	typedef const_reference reference;
	typedef value_type const* const_pointer;
	typedef const_pointer pointer;

	typedef typename E1::index_type index_type;
	typedef typename E1::const_index_pointer const_index_pointer;
	typedef typename index_pointer<E1>::type index_pointer;

	typedef const self_type const_closure_type;
	typedef const_closure_type closure_type;
	typedef unknown_storage_tag storage_category;

	// Construction and destruction
	matrix_vector_binary1(const expression1_type &e1, const expression2_type &e2):
		m_expression1(e1), m_expression2(e2) {}

	// Accessors
	size_type size() const {
		return m_expression1.size1();
	}

	// Expression accessors
	const expression1_closure_type &expression1() const {
		return m_expression1;
	}
	const expression2_closure_type &expression2() const {
		return m_expression2;
	}

	// Element access
	const_reference operator()(index_type i) const {
		return functor_type::apply(m_expression1, m_expression2, i);
	}

	// Closure comparison
	bool same_closure(const matrix_vector_binary1 &mvb1) const {
		return (*this).expression1().same_closure(mvb1.expression1()) &&
		       (*this).expression2().same_closure(mvb1.expression2());
	}

	// Iterator types
private:
	typedef typename E1::const_row_iterator const_subrow_iterator_type;
public:
	typedef indexed_iterator<const_closure_type> const_iterator;
	typedef const_iterator iterator;

	const_iterator begin() const {
		return const_iterator(*this,0);
	}
	const_iterator end() const {
		return const_iterator(*this,size());
	}
private:
	expression1_closure_type m_expression1;
	expression2_closure_type m_expression2;
};

template<class E1, class E2>
struct matrix_vector_binary_traits {
private:
	template<class T>
	struct matrix_vector_prod{
		typedef T value_type;
		typedef T result_type;

		template<class U,class V>
		static result_type apply(
			const matrix_expression<U> &e1,
			const vector_expression<V> &e2,
			std::size_t i
		) {
			return inner_prod(row(e1,i),e2);
		}
	};
public:
	typedef typename E1::value_type T1;
	typedef typename E2::value_type T2;
	typedef unknown_storage_tag storage_category;
	typedef row_major orientation;
	typedef typename promote_traits<T1, T2>::promote_type promote_type;
	typedef matrix_vector_binary1<E1, E2, matrix_vector_prod<promote_type> > expression_type;
	typedef expression_type result_type;
};

template<class M, class V>
typename matrix_vector_binary_traits<M,V>::result_type
prod(matrix_expression<M> const& matrix,vector_expression<V> const& vector){
	typedef typename matrix_vector_binary_traits<M,V>::expression_type expression;
	return expression(matrix(),vector());
}

template<class V, class M>
typename matrix_vector_binary_traits<matrix_transpose<M const>,V>::result_type
prod(vector_expression<V> const& vector,matrix_expression<M> const& matrix){
	typedef typename matrix_vector_binary_traits<matrix_transpose<M const>,V>::expression_type expression;
	return expression(trans(matrix),vector());
}

//matrix-matrix prod

//FIXME: return type is not optimally chosen. We need to take both arguments into account
//             especially take into account that the type is correct.
//FIXME: better make this a block expression so that we know the result type and don't need
// a temporary
template<class Result, class E1, class E2>
Result prod(matrix_expression<E1> const& e1,matrix_expression<E2> const& e2) {
	Result result(e1().size1(),e2().size2(),typename Result::value_type());
	axpy_prod(e1,e2,result,false);
	return result;
}

template<class E1, class E2>
typename matrix_temporary<E1>::type
prod(matrix_expression<E1> const& e1,matrix_expression<E2> const& e2) {
	typename matrix_temporary<E1>::type result(e1().size1(),e2().size2(),typename E1::value_type());
	axpy_prod(e1,e2,result,false);
	return result;
}

namespace detail{
	
template<class MatA,class VecB>
void sum_rows_impl(MatA const& matA, VecB& vecB, column_major){
	for(std::size_t i = 0; i != matA.size2(); ++i){ 
		vecB(i)+=sum(column(matA,i));
	}
}

template<class MatA,class VecB>
void sum_rows_impl(MatA const& matA, VecB& vecB, row_major){
	for(std::size_t i = 0; i != matA.size1(); ++i){ 
		noalias(vecB) += row(matA,i);
	}
}

//dispatcher for triangular matrix
template<class MatA,class VecB,class Orientation,class Triangular>
void sum_rows_impl(MatA const& matA, VecB& vecB, packed<Orientation,Triangular>){
	sum_rows_impl(matA,vecB,Orientation());
}

template<class MatA>
typename MatA::value_type sum_impl(MatA const& matA, column_major){
	typename MatA::value_type totalSum = 0;
	for(std::size_t j = 0; j != matA.size2(); ++j){
		totalSum += sum(column(matA,j));
	}
	return totalSum;
}

template<class MatA>
typename MatA::value_type sum_impl(MatA const& matA, row_major){
	typename MatA::value_type totalSum = 0;
	for(std::size_t i = 0; i != matA.size1(); ++i){
		totalSum += sum(row(matA,i));
	}
	return totalSum;
}

//dispatcher for triangular matrix
template<class MatA,class Orientation,class Triangular>
typename MatA::value_type sum_impl(MatA const& matA, packed<Orientation,Triangular>){
	return sum_impl(matA,Orientation());
}

template<class MatA>
typename MatA::value_type max_impl(MatA const& matA, column_major){
	typename MatA::value_type maximum = 0;
	for(std::size_t j = 0; j != matA.size2(); ++j){
		maximum= std::max(maximum, max(column(matA,j)));
	}
	return maximum;
}

template<class MatA>
typename MatA::value_type max_impl(MatA const& matA, row_major){
	typename MatA::value_type maximum = 0;
	for(std::size_t i = 0; i != matA.size1(); ++i){
		maximum= std::max(maximum, max(row(matA,i)));
	}
	return maximum;
}

//dispatcher for triangular matrix
template<class MatA,class Orientation,class Triangular>
typename MatA::value_type max_impl(MatA const& matA, packed<Orientation,Triangular>){
	return std::max(max_impl(matA,Orientation()),0.0);
}

template<class MatA>
typename MatA::value_type min_impl(MatA const& matA, column_major){
	typename MatA::value_type minimum = 0;
	for(std::size_t j = 0; j != matA.size2(); ++j){
		minimum= std::min(minimum, min(column(matA,j)));
	}
	return minimum;
}

template<class MatA>
typename MatA::value_type min_impl(MatA const& matA, row_major){
	typename MatA::value_type minimum = 0;
	for(std::size_t i = 0; i != matA.size1(); ++i){
		minimum= std::min(minimum, min(row(matA,i)));
	}
	return minimum;
}

//dispatcher for triangular matrix
template<class MatA,class Orientation,class Triangular>
typename MatA::value_type min_impl(MatA const& matA, packed<Orientation,Triangular>){
	return std::max(min_impl(matA,Orientation()),0.0);
}

}//end detail

//dispatcher
template<class MatA>
typename vector_temporary_type<
	typename MatA::value_type,
	dense_random_access_iterator_tag
>::type
sum_rows(matrix_expression<MatA> const& A){
	typename vector_temporary_type<
		typename MatA::value_type,
		dense_random_access_iterator_tag
	>::type result(A().size2(),0.0);
	detail::sum_rows_impl(A(),result,typename MatA::orientation());
	return result;
}

template<class MatA>
typename vector_temporary_type<
	typename MatA::value_type,
	dense_random_access_iterator_tag
>::type
sum_columns(matrix_expression<MatA> const& A){
	//implemented using sum_rows_impl by transposing A
	typename vector_temporary_type<
		typename MatA::value_type,
		dense_random_access_iterator_tag
	>::type result(A().size1(),0.0);
	detail::sum_rows_impl(trans(A),result,typename MatA::orientation::transposed_orientation());
	return result;
}


template<class MatA>
typename MatA::value_type sum(matrix_expression<MatA> const& A){
	return detail::sum_impl(A(),typename MatA::orientation());
}

template<class MatA>
typename MatA::value_type max(matrix_expression<MatA> const& A){
	return detail::max_impl(A(),typename MatA::orientation());
}

template<class MatA>
typename MatA::value_type min(matrix_expression<MatA> const& A){
	return detail::min_impl(A(),typename MatA::orientation());
}

/// \brief Returns the frobenius inner-product between matrices exprssions 1 and e2.
///
///The frobenius inner product is defined as \f$ <A,B>_F=\sum_{ij} A_ij*B_{ij} \f$. It induces the
/// Frobenius norm \f$ ||A||_F = \sqrt{<A,A>_F} \f$
template<class E1, class E2>
typename promote_traits <typename E1::value_type,typename E2::value_type>::promote_type
frobenius_prod(
	matrix_expression<E1> const& e1,
	matrix_expression<E2> const& e2
) {
	return sum(e1*e2);
}


template<class E>
typename matrix_norm_1<E>::result_type
norm_1(const matrix_expression<E> &e) {
	return matrix_norm_1<E>::apply(e());
}

template<class E>
typename real_traits<typename E::value_type>::type
norm_frobenius(const matrix_expression<E> &e) {
	using std::sqrt;
	return sqrt(sum(abs_sqr(e)));
}

template<class E>
typename matrix_norm_inf<E>::result_type
norm_inf(const matrix_expression<E> &e) {
	return matrix_norm_inf<E>::apply(e());
}

/*!
 *  \brief Evaluates the sum of the values at the diagonal of
 *         matrix "v".
 *
 *  Example:
 *  \f[
 *      \left(
 *      \begin{array}{*{4}{c}}
 *          {\bf 1} & 5       & 9        & 13\\
 *          2       & {\bf 6} & 10       & 14\\
 *          3       & 7       & {\bf 11} & 15\\
 *          4       & 8       & 12       & {\bf 16}\\
 *      \end{array}
 *      \right)
 *      \longrightarrow 1 + 6 + 11 + 16 = 34
 *  \f]
 *
 *      \param  m square matrix
 *      \return the sum of the values at the diagonal of \em m
 */
template < class MatrixT >
typename MatrixT::value_type trace(matrix_expression<MatrixT> const& m)
{
	SIZE_CHECK(m().size1() == m().size2());

	typename MatrixT::value_type t(m()(0, 0));
	for (unsigned i = 1; i < m().size1(); ++i)
		t += m()(i, i);
	return t;
}


///\brief Creates a matrix from a vector by repeating the vector in every row of the matrix
///
///example: vector = (1,2,3)
///repeat(vector,3) results in
///(1,2,3)
///(1,2,3)
///(1,2,3)
///@param vector the vector which is to be repeated as the rows of the resulting matrix
///@param rows the number of rows of the matrix
template<class Vector>
vector_repeater<Vector> repeat(vector_expression<Vector> const& vector, std::size_t rows){
	return vector_repeater<Vector>(vector(),rows);
}

/** \brief An identity matrix with values of type \c T
 *
 * Elements or cordinates \f$(i,i)\f$ are equal to 1 (one) and all others to 0 (zero).
 */
template<class T>
class identity_matrix: public diagonal_matrix<scalar_vector<T> > {
	typedef diagonal_matrix<scalar_vector<T> > base_type;
public:
	identity_matrix(){}
	identity_matrix(std::size_t size):base_type(scalar_vector<T>(size,T(1))){}
};

/** \brief A matrix with all values of type \c T equal to the same value
 *
 * Changing one value has the effect of changing all the values. Assigning it to a normal matrix will copy
 * the same value everywhere in this matrix. All accesses are constant time, due to the trivial value.
 *
 * \tparam T the type of object stored in the matrix (like double, float, complex, etc...)
 * \tparam ALLOC an allocator for storing the unique value. By default, a standar allocator is used.
 */
template<class T>
class scalar_matrix:
	public matrix_container<scalar_matrix<T> > {

	typedef scalar_matrix<T> self_type;
public:
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;
	typedef T value_type;
	typedef const T& const_reference;
	typedef T& reference;
	typedef value_type const* const_pointer;
	typedef value_type scalar_type;
	typedef const_pointer pointer;

	typedef std::size_t index_type;
	typedef index_type const* const_index_pointer;
	typedef index_type index_pointer;

	typedef const matrix_reference<const self_type> const_closure_type;
	typedef matrix_reference<self_type> closure_type;
	typedef dense_tag storage_category;
	typedef unknown_orientation orientation;

	// Construction and destruction
	scalar_matrix():
		m_size1(0), m_size2(0), m_value() {}
	scalar_matrix(size_type size1, size_type size2, const value_type& value = value_type(1)):
		m_size1(size1), m_size2(size2), m_value(value) {}
	scalar_matrix(const scalar_matrix& m):
		m_size1(m.m_size1), m_size2(m.m_size2), m_value(m.m_value) {}

	// Accessors
	size_type size1() const {
		return m_size1;
	}
	size_type size2() const {
		return m_size2;
	}

	// Element access
	const_reference operator()(size_type /*i*/, size_type /*j*/) const {
		return m_value;
	}
	
	//Iterators
	typedef constant_iterator<value_type> const_row_iterator;
	typedef constant_iterator<value_type> const_column_iterator;
	typedef const_row_iterator row_iterator;
	typedef const_column_iterator column_iterator;

	const_row_iterator row_begin(std::size_t i) const {
		return const_row_iterator(0, m_value);
	}
	const_row_iterator row_end(std::size_t i) const {
		return const_row_iterator(size2(), m_value);
	}
	
	const_row_iterator column_begin(std::size_t j) const {
		return const_row_iterator(0, m_value);
	}
	const_row_iterator column_end(std::size_t j) const {
		return const_row_iterator(size1(), m_value);
	}
private:
	size_type m_size1;
	size_type m_size2;
	value_type m_value;
};

///brief repeats a single element to form a matrix  of size rows x columns
///
///@param scalar the value which is repeated
///@param rows the number of rows of the resulting vector
///@param columns the number of columns of the resulting vector
template<class T>
typename boost::enable_if<boost::is_arithmetic<T>, scalar_matrix<T> >::type
repeat(T scalar, std::size_t rows, std::size_t columns){
	return scalar_matrix<T>(rows, columns, scalar);
}


}
}

#endif
