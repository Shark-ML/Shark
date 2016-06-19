/*!
 * \brief       Classes used for matrix expressions.
 * 
 * \author      O. Krause
 * \date        2016
 *
 *
 * \par Copyright 1995-2015 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://image.diku.dk/shark/>
 * 
 * Shark is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Shark is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with Shark.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
 #ifndef SHARK_LINALG_BLAS_MATRIX_EXPRESSION_CLASSES_HPP
#define SHARK_LINALG_BLAS_MATRIX_EXPRESSION_CLASSES_HPP

#include "../kernels/gemv.hpp"
#include "../kernels/tpmv.hpp"
#include "../kernels/trmv.hpp"
#include "../kernels/gemm.hpp"
#include "../kernels/trmm.hpp"
#include "../vector_expression.hpp"
#include "../matrix_proxy.hpp"
#include <type_traits>

namespace shark {
namespace blas {

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

	typedef self_type const_closure_type;
	typedef const_closure_type closure_type;
	typedef blas::row_major orientation;
	typedef blas::unknown_storage_tag storage_category;
	typedef typename V::evaluation_category evaluation_category;

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

/// \brief A matrix with all values of type \c T equal to the same value
///
/// \tparam T the type of object stored in the matrix (like double, float, complex, etc...)
template<class T>
class scalar_matrix:
	public matrix_container<scalar_matrix<T> > {

	typedef scalar_matrix<T> self_type;
public:
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;
	typedef T value_type;
	typedef const T& const_reference;
	typedef const_reference reference;
	typedef value_type const* const_pointer;
	typedef value_type scalar_type;
	typedef const_pointer pointer;

	typedef std::size_t index_type;
	typedef index_type const* const_index_pointer;
	typedef index_type index_pointer;

	typedef self_type const_closure_type;
	typedef self_type closure_type;
	typedef dense_tag storage_category;
	typedef unknown_orientation orientation;
	typedef elementwise_tag evaluation_category;

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

///\brief class which allows for matrix transformations
///
///transforms a matrix expression e of type E using a Function f of type F as an elementwise transformation f(e(i,j))
///This transformation needs to leave f constant, meaning that applying f(x), f(y), f(z) yields the same results independent of the
///order of application.
///F must provide a boolean flag F::zero_identity which indicates that f(0) = 0. This is needed for correct usage with sparse
///arguments - if f(0) != 0 this expression will be dense!
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
	typedef typename std::result_of<F(typename E::value_type)>::type value_type;
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

	typedef self_type const_closure_type;
	typedef const_closure_type closure_type;
	typedef typename E::orientation orientation;
	typedef blas::unknown_storage_tag storage_category;
	typedef typename E::evaluation_category evaluation_category;

	// Construction and destruction
	explicit matrix_unary(blas::matrix_expression<E> const &e, F const &functor):
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

template<class E1, class E2, class F>
class matrix_binary:
	public blas::matrix_expression<matrix_binary<E1, E2, F> > {
private:
	typedef matrix_binary<E1, E2, F> self_type;
public:
	typedef E1 lhs_type;
	typedef E2 rhs_type;
	typedef typename E1::const_closure_type lhs_closure_type;
	typedef typename E2::const_closure_type rhs_closure_type;

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

	typedef self_type const_closure_type;
	typedef const_closure_type closure_type;
	typedef typename E1::orientation orientation;
	typedef blas::unknown_storage_tag storage_category;
	typedef typename evaluation_restrict_traits<E1,E2>::type evaluation_category;

	typedef F functor_type;

        // Construction and destruction

        explicit matrix_binary (
		matrix_expression<E1> const&e1,  matrix_expression<E2> const& e2, functor_type functor 
	): m_lhs (e1()), m_rhs (e2()),m_functor(functor) {}

        // Accessors
        size_type size1 () const {
		return m_lhs.size1 ();
        }
        size_type size2 () const {
		return m_lhs.size2 ();
        }

        const_reference operator () (index_type i, index_type j) const {
		return m_functor( m_lhs (i, j), m_rhs(i,j));
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
			m_lhs.row_begin(i),m_lhs.row_end(i),
			m_rhs.row_begin(i),m_rhs.row_end(i)
		);
	}
	const_row_iterator row_end(std::size_t i) const {
		return const_row_iterator (m_functor,
			m_lhs.row_end(i),m_lhs.row_end(i),
			m_rhs.row_end(i),m_rhs.row_end(i)
		);
	}

	const_column_iterator column_begin(std::size_t j) const {
		return const_column_iterator (m_functor,
			m_lhs.column_begin(j),m_lhs.column_end(j),
			m_rhs.column_begin(j),m_rhs.column_end(j)
		);
	}
	const_column_iterator column_end(std::size_t j) const {
		return const_column_iterator (m_functor,
			m_lhs.column_begin(j),m_lhs.column_end(j),
			m_rhs.column_begin(j),m_rhs.column_end(j)
		);
	}

private:
	lhs_closure_type m_lhs;
        rhs_closure_type m_rhs;
	functor_type m_functor;
};

template<class E1, class E2>
class matrix_addition: public blas::matrix_expression<matrix_addition<E1, E2> > {
private:
	typedef scalar_binary_plus<
		typename E1::value_type,
		typename E2::value_type
	> functor_type;
public:
	typedef typename E1::const_closure_type lhs_closure_type;
	typedef typename E2::const_closure_type rhs_closure_type;

	typedef typename E1::size_type size_type;
	typedef typename E1::difference_type difference_type;
	typedef typename functor_type::result_type value_type;
	typedef value_type scalar_type;
	typedef value_type const_reference;
	typedef const_reference reference;
	typedef value_type const* const_pointer;
	typedef const_pointer pointer;

	typedef typename E1::index_type index_type;
	typedef typename E1::const_index_pointer const_index_pointer;
	typedef typename index_pointer<E1>::type index_pointer;

	typedef matrix_addition const_closure_type;
	typedef const_closure_type closure_type;
	typedef typename E1::orientation orientation;
	typedef blas::unknown_storage_tag storage_category;
	typedef typename evaluation_restrict_traits<E1,E2>::type evaluation_category;

        // Construction
        explicit matrix_addition(
		lhs_closure_type const& e1,
		rhs_closure_type const& e2
	): m_lhs (e1), m_rhs (e2){}

        // Accessors
        size_type size1 () const {
		return m_lhs.size1 ();
        }
        size_type size2 () const {
		return m_lhs.size2 ();
        }
	
	lhs_closure_type const& lhs()const{
		return m_lhs;
	}
	
	rhs_closure_type const& rhs()const{
		return m_rhs;
	}

        const_reference operator () (index_type i, index_type j) const {
		return m_lhs(i, j) + m_rhs(i,j);
        }
	
	//computation kernels
	template<class MatX>
	void assign_to(matrix_expression<MatX>& X, scalar_type alpha = scalar_type(1) )const{
		assign(X,alpha * m_lhs);
		plus_assign(X,alpha * m_rhs);
	}
	template<class MatX>
	void plus_assign_to(matrix_expression<MatX>& X, scalar_type alpha = scalar_type(1) )const{
		plus_assign(X,alpha * m_lhs);
		plus_assign(X,alpha * m_rhs);
	}
	
	template<class MatX>
	void minus_assign_to(matrix_expression<MatX>& X, scalar_type alpha = scalar_type(1) )const{
		minus_assign(X,alpha * m_lhs);
		minus_assign(X,alpha * m_rhs);
	}

	// Iterator types
private:
	typedef typename E1::const_row_iterator const_row_iterator1_type;
	typedef typename E1::const_column_iterator const_row_column_iterator_type;
	typedef typename E2::const_row_iterator const_column_iterator1_type;
	typedef typename E2::const_column_iterator const_column_iterator2_type;

public:
	typedef binary_transform_iterator<
		typename E1::const_row_iterator,
		typename E2::const_row_iterator,
		functor_type
	> const_row_iterator;
	typedef binary_transform_iterator<
		typename E1::const_column_iterator,
		typename E2::const_column_iterator,
		functor_type
	> const_column_iterator;
	typedef const_row_iterator row_iterator;
	typedef const_column_iterator column_iterator;

	const_row_iterator row_begin(std::size_t i) const {
		return const_row_iterator (functor_type(),
			m_lhs.row_begin(i),m_lhs.row_end(i),
			m_rhs.row_begin(i),m_rhs.row_end(i)
		);
	}
	const_row_iterator row_end(std::size_t i) const {
		return const_row_iterator (functor_type(),
			m_lhs.row_end(i),m_lhs.row_end(i),
			m_rhs.row_end(i),m_rhs.row_end(i)
		);
	}

	const_column_iterator column_begin(std::size_t j) const {
		return const_column_iterator (functor_type(),
			m_lhs.column_begin(j),m_lhs.column_end(j),
			m_rhs.column_begin(j),m_rhs.column_end(j)
		);
	}
	const_column_iterator column_end(std::size_t j) const {
		return const_column_iterator (functor_type(),
			m_lhs.column_end(j),m_lhs.column_end(j),
			m_rhs.column_end(j),m_rhs.column_end(j)
		);
	}

private:
	lhs_closure_type m_lhs;
        rhs_closure_type m_rhs;
	functor_type m_functor;
};

template<class E>
class matrix_scalar_multiply:public blas::matrix_expression<matrix_scalar_multiply<E> > {
private:
	typedef typename E::const_row_iterator const_subrow_iterator_type;
	typedef typename E::const_column_iterator const_subcolumn_iterator_type;
	typedef scalar_multiply1<typename E::value_type, typename E::scalar_type> functor_type;
public:
	typedef typename E::const_closure_type expression_closure_type;

	typedef typename functor_type::result_type value_type;
	typedef typename E::scalar_type scalar_type;
	typedef value_type const_reference;
	typedef const_reference reference;
	typedef value_type const *const_pointer;
	typedef value_type *pointer;
	typedef typename E::size_type size_type;
	typedef typename E::difference_type difference_type;

	typedef typename E::index_type index_type;
	typedef typename E::const_index_pointer const_index_pointer;
	typedef typename index_pointer<E>::type index_pointer;

	typedef matrix_scalar_multiply const_closure_type;
	typedef const_closure_type closure_type;
	typedef typename E::orientation orientation;
	typedef blas::unknown_storage_tag storage_category;
	typedef typename E::evaluation_category evaluation_category;

	// Construction and destruction
	explicit matrix_scalar_multiply(blas::matrix_expression<E> const &e, scalar_type scalar):
		m_expression(e()), m_scalar(scalar){}

	// Accessors
	size_type size1() const {
		return m_expression.size1();
	}
	size_type size2() const {
		return m_expression.size2();
	}

	// Element access
	const_reference operator()(index_type i, index_type j) const {
		return m_scalar * m_expression(i, j);
	}
	
	//computation kernels
	template<class MatX>
	void assign_to(matrix_expression<MatX>& X, scalar_type alpha = scalar_type(1) )const{
		m_expression.assign_to(X,alpha*m_scalar);
	}
	template<class MatX>
	void plus_assign_to(matrix_expression<MatX>& X, scalar_type alpha = scalar_type(1) )const{
		m_expression.plus_assign_to(X,alpha*m_scalar);
	}
	
	template<class MatX>
	void minus_assign_to(matrix_expression<MatX>& X, scalar_type alpha = scalar_type(1) )const{
		m_expression.minus_assign_to(X,alpha*m_scalar);
	}

	// Iterator types
	typedef transform_iterator<typename E::const_row_iterator, functor_type> const_row_iterator;
	typedef transform_iterator<typename E::const_column_iterator, functor_type> const_column_iterator;
	typedef const_row_iterator row_iterator;
	typedef const_column_iterator column_iterator;
	
	const_row_iterator row_begin(index_type i) const {
		return const_row_iterator(m_expression.row_begin(i),functor_type(m_scalar));
	}
	const_row_iterator row_end(index_type i) const {
		return const_row_iterator(m_expression.row_end(i),functor_type(m_scalar));
	}

	const_column_iterator column_begin(index_type i) const {
		return const_row_iterator(m_expression.column_begin(i),functor_type(m_scalar));
	}
	const_column_iterator column_end(index_type i) const {
		return const_row_iterator(m_expression.column_end(i),functor_type(m_scalar));
	}

private:
	expression_closure_type m_expression;
	scalar_type m_scalar;
};

template<class E1, class E2>
class outer_product:public matrix_expression<outer_product<E1, E2> > {
	typedef scalar_multiply1<typename E1::value_type, typename E2::value_type> functor_type;
public:
	typedef typename E1::const_closure_type lhs_closure_type;
	typedef typename E2::const_closure_type rhs_closure_type;

	
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;
	typedef typename functor_type::result_type value_type;
	typedef value_type scalar_type;
	typedef value_type const_reference;
	typedef const_reference reference;
	typedef value_type const* const_pointer;
	typedef const_pointer pointer;

	typedef typename E1::index_type index_type;
	typedef typename E1::const_index_pointer const_index_pointer;
	typedef typename index_pointer<E1>::type index_pointer;

	typedef outer_product const_closure_type;
	typedef outer_product closure_type;
	typedef unknown_orientation orientation;
	typedef unknown_storage_tag storage_category;
	typedef typename evaluation_restrict_traits<E1,E2>::type evaluation_category;

	// Construction and destruction
	
	explicit outer_product(lhs_closure_type const& e1, rhs_closure_type const& e2)
	:m_lhs(e1), m_rhs(e2){}

	// Accessors
	size_type size1() const {
		return m_lhs.size();
	}
	size_type size2() const {
		return m_rhs.size();
	}

	// Expression accessors
	lhs_closure_type const& lhs() const {
		return m_lhs;
	}
	rhs_closure_type const& rhs() const {
		return m_rhs;
	}
	// Element access
	const_reference operator()(index_type i, index_type j) const {
		return m_lhs(i) * m_rhs(j);
	}

	typedef transform_iterator<typename E2::const_iterator,functor_type> const_row_iterator;
	typedef transform_iterator<typename E1::const_iterator,functor_type> const_column_iterator;
	typedef const_row_iterator row_iterator;
	typedef const_column_iterator column_iterator;
	
	const_row_iterator row_begin(index_type i) const {
		return const_row_iterator(m_rhs.begin(),
			functor_type(m_lhs(i))
		);
	}
	const_row_iterator row_end(index_type i) const {
		return const_row_iterator(m_rhs.end(),
			functor_type(m_lhs(i))
		);
	}

	const_column_iterator column_begin(index_type i) const {
		return const_column_iterator(m_lhs.begin(),
			functor_type(m_rhs(i))
		);
	}
	const_column_iterator column_end(index_type i) const {
		return const_column_iterator(m_lhs.end(),
			functor_type(m_rhs(i))
		);
	}
private:
	lhs_closure_type m_lhs;
	rhs_closure_type m_rhs;
};

template<class MatA, class VecV>
class matrix_vector_prod:
	public vector_expression<matrix_vector_prod<MatA, VecV> > {
public:
	typedef typename MatA::const_closure_type matrix_closure_type;
	typedef typename VecV::const_closure_type vector_closure_type;
public:
	typedef decltype(
		typename MatA::scalar_type() * typename VecV::scalar_type()
	) scalar_type;
	typedef scalar_type value_type;
	typedef typename MatA::size_type size_type;
	typedef typename MatA::difference_type difference_type;
	typedef typename MatA::index_type index_type;
	typedef value_type const& const_reference;
	typedef const_reference reference;
	typedef value_type const* const_pointer;
	typedef const_pointer pointer;
	typedef typename MatA::index_pointer index_pointer;
	typedef typename MatA::const_index_pointer const_index_pointer;

	typedef matrix_vector_prod<MatA, VecV> const_closure_type;
	typedef const_closure_type closure_type;
	typedef unknown_storage_tag storage_category;
	typedef blockwise_tag evaluation_category;


	//FIXME: This workaround is required to be able to generate
	// temporary vectors
	typedef typename MatA::const_row_iterator const_iterator;
	typedef const_iterator iterator;

	// Construction and destruction
	explicit matrix_vector_prod(
		matrix_closure_type const& matrix,
		vector_closure_type  const& vector
	):m_matrix(matrix), m_vector(vector) {}

	size_type size() const {
		return m_matrix.size1();
	}
	
	matrix_closure_type const& matrix() const {
		return m_matrix;
	}
	vector_closure_type const& vector() const {
		return m_vector;
	}
	
	//dispatcher to computation kernels
	template<class VecX>
	void assign_to(vector_expression<VecX>& x, scalar_type alpha = scalar_type(1) )const{
		assign_to(x, alpha, typename MatA::orientation(), typename MatA::storage_category());
	}
	template<class VecX>
	void plus_assign_to(vector_expression<VecX>& x, scalar_type alpha = scalar_type(1) )const{
		plus_assign_to(x, alpha, typename MatA::orientation(), typename MatA::storage_category());
	}
	
	template<class VecX>
	void minus_assign_to(vector_expression<VecX>& x, scalar_type alpha = scalar_type(1) )const{
		plus_assign_to(x,-alpha, typename MatA::orientation(), typename MatA::storage_category());
	}
	
private:
	//gemv
	template<class VecX, class Category>
	void assign_to(vector_expression<VecX>& x, scalar_type alpha, linear_structure, Category c)const{
		x().clear();
		plus_assign_to(x,alpha, linear_structure(), c);
	}
	template<class VecX, class Category>
	void plus_assign_to(vector_expression<VecX>& x, scalar_type alpha, linear_structure, Category )const{
		kernels::gemv(eval_block(m_matrix), eval_block(m_vector), x, alpha);
	}
	//tpmv
	template<class VecX>
	void assign_to(vector_expression<VecX>& x, scalar_type alpha, triangular_structure, packed_tag )const{
		noalias(x) = eval_block(alpha * m_vector);
		kernels::tpmv(eval_block(m_matrix), x);
	}
	template<class VecX>
	void plus_assign_to(vector_expression<VecX>& x, scalar_type alpha, triangular_structure, packed_tag )const{
		//computation of tpmv is in-place so we need a temporary for plus-assign.
		typename vector_temporary<VecX>::type temp( eval_block(alpha * m_vector));
		kernels::tpmv(eval_block(m_matrix), temp);
		noalias(x) += temp;
	}
	//trmv
	template<class VecX>
	void assign_to(vector_expression<VecX>& x, scalar_type alpha, triangular_structure, dense_tag )const{
		noalias(x) = eval_block(alpha * m_vector);
		kernels::trmv<MatA::orientation::is_upper, MatA::orientation::is_unit>(m_matrix.expression(), x);
	}
	template<class VecX>
	void plus_assign_to(vector_expression<VecX>& x, scalar_type alpha, triangular_structure, dense_tag )const{
		//computation of tpmv is in-place so we need a temporary for plus-assign.
		typename vector_temporary<VecX>::type temp( eval_block(alpha * m_vector));
		kernels::trmv<MatA::orientation::is_upper, MatA::orientation::is_unit>(m_matrix.expression(), temp);
		noalias(x) += temp;
	}
	matrix_closure_type m_matrix;
	vector_closure_type m_vector;
};

namespace detail{
template<class M, class TriangularType>
class dense_triangular_proxy: public matrix_expression<dense_triangular_proxy<M, TriangularType> > {
	typedef dense_triangular_proxy<M, TriangularType> self_type;
public:
	static_assert(std::is_same<typename M::storage_category, dense_tag>::value, "Can only create triangular proxies of dense matrices");

	typedef typename M::size_type size_type;
	typedef typename M::difference_type difference_type;
	typedef typename M::value_type value_type;
	typedef typename M::scalar_type scalar_type;
	typedef typename M::const_reference const_reference;
	typedef typename reference<M>::type reference;
	typedef typename M::const_pointer const_pointer;
	typedef typename pointer<M>::type pointer;

	typedef typename M::index_type index_type;
	typedef typename M::const_index_pointer const_index_pointer;
	typedef typename index_pointer<M>::type index_pointer;

	typedef typename closure<M>::type matrix_closure_type;
	typedef dense_triangular_proxy<typename const_expression<M>::type,TriangularType> const_closure_type;
	typedef dense_triangular_proxy<M,TriangularType> closure_type;

	typedef dense_tag storage_category;
	typedef elementwise_tag evaluation_category;
	typedef triangular<typename M::orientation,TriangularType> orientation;

	// Construction and destruction
	explicit dense_triangular_proxy(matrix_closure_type const& m)
	: m_expression(m) {
		SIZE_CHECK(m.size1() == m.size2());
	}

	// Expression accessors
	matrix_closure_type const& expression() const{
		return m_expression;
	}
	matrix_closure_type expression(){
		return m_expression;
	}
	
	 size_type size1 () const {
		return expression().size1 ();
        }
        size_type size2 () const {
		return expression().size2 ();
        }
	
	typedef typename row_iterator<M>::type row_iterator;
	typedef row_iterator const_row_iterator;
	typedef typename column_iterator<M>::type column_iterator;
	typedef column_iterator const_column_iterator;
private:
	matrix_closure_type m_expression;
};
}

//matrix-matrix prod
template<class MatA, class MatB>
class matrix_matrix_prod: public matrix_expression<matrix_matrix_prod<MatA, MatB> > {
public:
	typedef typename MatA::const_closure_type matrix_closure_typeA;
	typedef typename MatB::const_closure_type matrix_closure_typeB;
public:
	typedef decltype(
		typename MatA::scalar_type() * typename MatB::scalar_type()
	) scalar_type;
	typedef scalar_type value_type;
	typedef typename MatA::size_type size_type;
	typedef typename MatA::difference_type difference_type;
	typedef typename MatA::index_type index_type;
	typedef value_type const& const_reference;
	typedef const_reference reference;
	typedef value_type const* const_pointer;
	typedef const_pointer pointer;

	typedef typename MatA::index_pointer index_pointer;
	typedef typename MatA::const_index_pointer const_index_pointer;

	typedef matrix_matrix_prod<MatA, MatB> const_closure_type;
	typedef const_closure_type closure_type;
	typedef unknown_storage_tag storage_category;
	typedef blockwise_tag evaluation_category;
	typedef unknown_orientation orientation;


	//FIXME: This workaround is required to be able to generate
	// temporary matrices
	typedef typename MatA::const_row_iterator const_row_iterator;
	typedef typename MatA::const_column_iterator const_column_iterator;
	typedef const_row_iterator row_iterator;
	typedef const_column_iterator column_iterator;

	// Construction and destruction
	explicit matrix_matrix_prod(
		matrix_closure_typeA const& lhs,
		matrix_closure_typeB const& rhs
	):m_lhs(lhs), m_rhs(rhs) {}

	size_type size1() const {
		return m_lhs.size1();
	}
	size_type size2() const {
		return m_rhs.size2();
	}
	
	matrix_closure_typeA const& lhs() const {
		return m_lhs;
	}
	matrix_closure_typeB const& rhs() const {
		return m_rhs;
	}
	
	//dispatcher to computation kernels
	template<class MatX>
	void assign_to(matrix_expression<MatX>& X, scalar_type alpha = scalar_type(1) )const{
		assign_to(X, alpha, typename MatA::orientation(), typename MatA::storage_category());
	}
	template<class MatX>
	void plus_assign_to(matrix_expression<MatX>& X, scalar_type alpha = scalar_type(1) )const{
		plus_assign_to(X, alpha, typename MatA::orientation(), typename MatA::storage_category());
	}
	
	template<class MatX>
	void minus_assign_to(matrix_expression<MatX>& X, scalar_type alpha = scalar_type(1) )const{
		plus_assign_to(X, -alpha, typename MatA::orientation(), typename MatA::storage_category());
	}
	
private:
	//gemm
	template<class MatX, class Category>
	void assign_to(matrix_expression<MatX>& X, scalar_type alpha, linear_structure, Category c )const{
		X().clear();
		plus_assign_to(X,alpha, linear_structure(), c);
	}
	template<class MatX, class Category>
	void plus_assign_to(matrix_expression<MatX>& X, scalar_type alpha, linear_structure, Category )const{
		kernels::gemm(eval_block(m_lhs), eval_block(m_rhs), X, alpha);
	}
	//trmv
	template<class MatX>
	void assign_to(matrix_expression<MatX>& X, scalar_type alpha, triangular_structure, dense_tag)const{
		noalias(X) = eval_block(alpha * m_rhs);
		kernels::trmm<MatA::orientation::is_upper, MatA::orientation::is_unit>(m_lhs.expression(), X);
	}
	template<class MatX>
	void plus_assign_to(matrix_expression<MatX>& X, scalar_type alpha, triangular_structure, dense_tag )const{
		//computation of tpmv is in-place so we need a temporary for plus-assign.
		typename matrix_temporary<MatX>::type temp( eval_block(alpha * m_rhs));
		kernels::trmm<MatA::orientation::is_upper, MatA::orientation::is_unit>(m_lhs.expression(), temp);
		noalias(X) += temp;
	}
private:
	matrix_closure_typeA m_lhs;
	matrix_closure_typeB m_rhs;
};

namespace detail{
//  a traits class which optimizes and matrix-vector expressions to be more efficient. This is useful in case a user
// gives a complex expression as an argument to a function, e.g. a general CG implementation.
	
//while not crucial for speed, it makes most expressions much easier to handle if they get simplified
//e.g. we do not have to look two steps deep in to the matrix-vector product if the matrix is transposed (e.g. transpose of matrix-sum)
template<class M>
struct matrix_transpose_optimizer{
	typedef matrix_transpose<M> type;
	
	static type create(typename closure<M>::type const& m){
		return type(m);
	}
};

//(M1^T)^T = M1
template<class M>
struct matrix_transpose_optimizer<matrix_transpose<M> >{
	typedef typename closure<M>::type type;
	
	static type create(matrix_transpose<M> const& m){
		return m.expression();
	}
};

//(M1+M2)^T=M1^T+M2^T
template<class M1, class M2>
struct matrix_transpose_optimizer<matrix_addition<M1,M2> >{
	typedef matrix_transpose_optimizer<typename const_expression<M1>::type > left_opt;
	typedef matrix_transpose_optimizer<typename const_expression<M2>::type > right_opt;
	typedef matrix_addition<typename left_opt::type,typename right_opt::type > type;
	
	static type create(matrix_addition<M1,M2> const& m){
		return type(left_opt::create(m.lhs()),right_opt::create(m.rhs()));
	}
};

//(v1v2^T)^T = v2 v1^T
template<class V1, class V2>
struct matrix_transpose_optimizer<outer_product<V1,V2> >{
	typedef outer_product<V2,V1> type;
	
	static type create(outer_product<V1,V2> const& m){
		return type(m.rhs(),m.lhs());
	}
};

//(M1M2)^T = M2^T M1^T
template<class M1, class M2>
struct matrix_transpose_optimizer<matrix_matrix_prod<M1,M2> >{
	typedef matrix_transpose_optimizer<typename const_expression<M2>::type> left_opt;
	typedef matrix_transpose_optimizer<typename const_expression<M1>::type> right_opt;
	typedef matrix_matrix_prod<typename left_opt::type,typename right_opt::type > type;
	
	static type create(matrix_matrix_prod<M1,M2> const& m){
		return type(left_opt::create(m.rhs()),right_opt::create(m.lhs()));
	}
};
	
//matrix-vector multiplications
template<class M, class V>
struct matrix_vector_prod_optimizer{
	typedef matrix_vector_prod<M,V> type;
	
	static type create(typename M::const_closure_type const& m, typename V::const_closure_type const& v){
		return type(m,v);
	}
};

//the helper guards against the case that applying the simplifications of matrix_transpose
//can be the identity -> guard against infinite loops
template<class M1, class M1Simplified, class V>
struct matrix_vector_prod_transpose_helper{
private:
	typedef matrix_vector_prod_optimizer<M1Simplified, V> inner_opt;
public:
	typedef typename inner_opt::type type;
	static type create(typename M1Simplified::const_closure_type const& m, typename V::const_closure_type const& v){
		return inner_opt::create(m,v);
	}
};
template<class M, class V>
struct matrix_vector_prod_transpose_helper<M,M,V>{
	typedef matrix_vector_prod<M,V> type;
	
	static type create(typename M::const_closure_type const& m, typename V::const_closure_type const& v){
		return type(m,v);
	}
};

//simplify expressions with transposed matrix arguments.(used for product of types xA=>A^Tx)
template<class M, class V>
struct matrix_vector_prod_optimizer<matrix_transpose<M>,V>{
private:
	typedef typename matrix_transpose<M>::const_closure_type closure;
	typedef matrix_transpose_optimizer<typename const_expression<M>::type > transpose_opt;//simplify the matrix transpose statement
	typedef matrix_vector_prod_transpose_helper<
		closure, 
		typename transpose_opt::type,V
	> inner_opt;//call recursively on the simplified type and guard against identity transformations
public:
	typedef typename inner_opt::type type;
	static type create(closure const& m, typename V::const_closure_type const& v){
		return inner_opt::create(transpose_opt::create(m.expression()),v);
	}
};

//(M1*M2)*V=M1*(M2*V)
template<class M1,class M2, class V>
struct matrix_vector_prod_optimizer<matrix_matrix_prod<M1,M2>,V>{
private:
	typedef matrix_vector_prod_optimizer<M2,V> inner_opt;
	typedef matrix_vector_prod_optimizer<M1, typename inner_opt::type> outer_opt;
public:
	typedef typename outer_opt::type type;
	
	static type create(matrix_matrix_prod<M1,M2> const& m, typename V::const_closure_type const& v){
		auto inner_result = inner_opt::create(m.rhs(),v);
		return outer_opt::create(m.lhs(),inner_result);
	}
};

//(M1+M2)*V=M1*V+M2*V
template<class M1,class M2, class V>
struct matrix_vector_prod_optimizer<matrix_addition<M1,M2>,V>{
private:
	typedef matrix_vector_prod_optimizer<M1,V> left_opt;
	typedef matrix_vector_prod_optimizer<M2,V> right_opt;
public:
	typedef vector_addition<typename left_opt::type ,typename right_opt::type> type;
	
	static type create(matrix_addition<M1,M2> const& m, typename V::const_closure_type const& v){
		auto lhs = left_opt::create(m.lhs(),v);
		auto rhs = right_opt::create(m.rhs(),v);
		return type(lhs,rhs);
	}
};

//(v1*v2^T)*v3= v1*(v2^T*v3)
template<class V1,class V2, class V3>
struct matrix_vector_prod_optimizer<outer_product<V1,V2>,V3>{
	typedef vector_scalar_multiply<V1> type;
	
	static type create(outer_product<V1,V2> const& m, typename V3::const_closure_type const& v){
		auto alpha = inner_prod(m.rhs(),v);
		return type(m.lhs(),alpha);
	}
};

}


}}
#endif
