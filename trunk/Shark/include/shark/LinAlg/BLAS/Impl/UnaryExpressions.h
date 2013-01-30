/*!
 *  \brief Meta template transformations and types for generic programming with ublas.
 *
 *  \author O.Krause
 *  \date 2010-2011
 *
 *  \par Copyright (c) 1998-2011:
 *	  Institut f&uuml;r Neuroinformatik<BR>
 *	  Ruhr-Universit&auml;t Bochum<BR>
 *	  D-44780 Bochum, Germany<BR>
 *	  Phone: +49-234-32-25558<BR>
 *	  Fax:   +49-234-32-14209<BR>
 *	  eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *	  www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *	  <BR>
 *
 *
 *  <BR><HR>
 *  This file is part of Shark. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 3, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, see <http://www.gnu.org/licenses/>.
 *
*/
#ifndef SHARK_LINALG_BLAS_IMPL_UNARYEXPRESSIONS_H
#define SHARK_LINALG_BLAS_IMPL_UNARYEXPRESSIONS_H

#include <shark/LinAlg/BLAS/traits/MetaBlas.h>
//ublas has some serious problems concerning the way they handle unary transformations. For example their functors ar eonly allowed to use
//static functions. I think, they do it out of fear, that the compiler does bad things...but seriously this is flawed.

namespace shark{
namespace detail{
///\brief class which allows for vector transformations
///
///transforms a vector Expression e of type E using a Functiof f of type F as an elementwise transformation f(e(i))
///This transformation needs f to be constant, meaning that applying f(x), f(y), f(z) yields the same results independent of the
///order of application. Also F must provide a type F::result_type indicating the result type of the functor.
///F must further provide a boolean flag F::zero_identity which indicates that f(0) = 0. This is needed for correct usage with sparse
///arguments - if f(0) != 0 this expression will be dense!
template<class E, class F>
class VectorUnaryTransformation:
	public blas::vector_expression<VectorUnaryTransformation<E, F> > {
	typedef VectorUnaryTransformation<E, F> self_type;
	typedef E const expression_type;
	typedef typename E::const_iterator const_subiterator_type;
	
public:
	typedef F functor_type;
	typedef typename E::const_closure_type expression_closure_type;
	typedef typename E::size_type size_type;
	typedef typename E::difference_type difference_type;
	typedef typename F::result_type value_type;
	typedef value_type const_reference;
	typedef value_type reference;
	typedef value_type const* const_pointer;
	typedef value_type* pointer;
	
	typedef self_type const const_closure_type;
	typedef self_type closure_type;
	typedef blas::unknown_storage_tag storage_category;

	// Construction and destruction
	// May be used as mutable expression.
	VectorUnaryTransformation (blas::vector_expression<E> const&e, F const& functor):
		m_expression(e()), m_functor(functor) {}

	// Accessors
	size_type size () const {
		return m_expression.size ();
	}

	// Expression accessors
	expression_closure_type const& expression () const {
		return m_expression;
	}

public:
	// Element access
	const_reference operator() (size_type i) const {
		return m_functor(m_expression(i));
	}

	const_reference operator[] (size_type i) const {
		return m_functor(m_expression[i]);
	}

	// Closure comparison
	bool same_closure (VectorUnaryTransformation const& vu) const {
		return (*this).expression ().same_closure (vu.expression ());
	}

	// Iterator types
	
private:
	// Iterator enhances the iterator of the referenced expression
	// with the unary functor.
	
	//this is the version which tries to save the sparse property of the underlying vector expression
	//this is only possible, when the functor maps the 0 on itself, so f(0) = 0.
	//this version is also used when the underlying vector is dense.
	class default_const_iterator:
		public blas::container_const_reference<self_type>,
		public blas::iterator_base_traits<typename const_subiterator_type::iterator_category>::template
					iterator_base<default_const_iterator, value_type>::type {
	public:
		typedef typename const_subiterator_type::iterator_category iterator_category;
		typedef typename VectorUnaryTransformation::difference_type difference_type;
		typedef typename VectorUnaryTransformation::value_type value_type;
		typedef typename VectorUnaryTransformation::const_reference reference;
		typedef typename VectorUnaryTransformation::const_pointer pointer;

		// Construction and destruction
		default_const_iterator ():
		blas::container_const_reference<self_type> (), m_position () {}
		default_const_iterator (self_type const& ref, 
		const_subiterator_type const& it, const_subiterator_type const&, size_type):
		blas::container_const_reference<self_type> (ref), m_position (it) {}

		// Arithmetic
		default_const_iterator &operator ++ () {
			++m_position;
			return *this;
		}
		default_const_iterator &operator -- () {
			-- m_position;
			return *this;
		}
		default_const_iterator &operator += (difference_type n) {
			m_position += n;
			return *this;
		}
		default_const_iterator &operator -= (difference_type n) {
			m_position -= n;
			return *this;
		}
		difference_type operator - (default_const_iterator const& it) const {
			BOOST_UBLAS_CHECK ((*this) ().same_closure (it()), blas::external_logic());
			return m_position - it.m_position;
		}

		// Dereference
		const_reference operator * () const {
			return (*this)().m_functor(*m_position);
		}
		const_reference operator [] (difference_type n) const {
			return *(*this + n);
		}

		// Index
		size_type index () const {
			return m_position.index();
		}

		// Assignment
		default_const_iterator& operator = (default_const_iterator const& it) {
			blas::container_const_reference<self_type>::assign (&it());
			m_position = it.m_position;
			return *this;
		}

		// Comparison
		bool operator == (default_const_iterator const& it) const {
			BOOST_UBLAS_CHECK ((*this) ().same_closure (it ()), blas::external_logic ());
			return m_position == it.m_position;
		}
		bool operator < (default_const_iterator const& it) const {
			BOOST_UBLAS_CHECK ((*this) ().same_closure (it ()), blas::external_logic ());
			return m_position < it.m_position;
		}

	private:
		const_subiterator_type m_position;
	};
	
	///when the underlying container is sparse and Functor does not preserve the identity element, there is no way
	///around densifying the whole range. this way we avoid that empty elements are mapped to 0.
	class densifying_const_iterator:
		public blas::container_const_reference<self_type>,
		public blas::iterator_base_traits<std::bidirectional_iterator_tag>::template
					iterator_base<densifying_const_iterator, value_type>::type {
	public:
		typedef typename std::bidirectional_iterator_tag iterator_category;
		typedef typename VectorUnaryTransformation::difference_type difference_type;
		typedef typename VectorUnaryTransformation::value_type value_type;
		typedef typename VectorUnaryTransformation::const_reference reference;
		typedef typename VectorUnaryTransformation::const_pointer pointer;

		// Construction and destruction
		densifying_const_iterator ():m_index(0) {}
		densifying_const_iterator (
			self_type const& ref, 
			const_subiterator_type const& it, 
			const_subiterator_type const& it_end, 
			size_type index
		):blas::container_const_reference<self_type> (ref), 
		m_position (it), m_end(it_end), m_index(index),
		m_zeroValue(ref.m_functor(value_type(0))){}

		// Arithmetic
		
		//we unfortunately have to solve the issue, that the user might use op-- and op++ together
		//so we can't assume that we only go steps forward but instead that we first going backward (such that m_position.index()<m_index)
		//and than going forward again (such that we first need to find m_position.index()>m_index)
		densifying_const_iterator& operator ++ () {
			while(m_position != m_end && m_index >= m_position.index()){
				++m_position;
			}
			++m_index;
			return *this;
		}
		densifying_const_iterator& operator -- () {
			while(m_position != m_end && m_index <= m_position.index()){
				--m_position;
			}
			--m_index;
			return *this;
		}

		// Dereference
		const_reference operator * () const {
			if(m_position == m_end || m_index != m_position.index())
				return m_zeroValue;
			else
				return (*this)().m_functor(*m_position);
		}

		// Index
		size_type index () const {
			return m_index;
		}

		// Assignment
		densifying_const_iterator& operator = (densifying_const_iterator const& it) {
			blas::container_const_reference<self_type>::assign (&it());
			m_position = it.m_position;
			m_index = it.m_index;
			return *this;
		}

		// Comparison
		bool operator == (densifying_const_iterator const& it) const {
			BOOST_UBLAS_CHECK ((*this) ().same_closure (it ()), blas::external_logic ());
			return m_index == it.m_index;
		}

	private:
		const_subiterator_type m_position;
		const_subiterator_type m_end;
		size_type m_index;
		value_type m_zeroValue;
	};
public:

	//if the Functor does not preserve identity and the underlying expression is sparse, than densify it.
	typedef typename boost::mpl::if_c<
		!functor_type::zero_identity && 
		boost::is_same<typename const_subiterator_type::iterator_category, blas::sparse_bidirectional_iterator_tag>::value,
		densifying_const_iterator,
		default_const_iterator
	>::type const_iterator;
	typedef const_iterator iterator;

	// Element lookup
	const_iterator find (size_type i) const {
		return const_iterator (*this, m_expression.find(i),m_expression.end(),i);
	}

	const_iterator begin () const {
		return find (0); 
	}
	const_iterator end () const {
		return find (size ());
	}

	// Reverse iterator
	typedef blas::reverse_iterator_base<const_iterator> const_reverse_iterator;

	const_reverse_iterator rbegin () const {
		return const_reverse_iterator (end ());
	}
	const_reverse_iterator rend () const {
		return const_reverse_iterator (begin ());
	}

private:
	expression_closure_type m_expression;
	F m_functor;
};

///\brief class which allows for matrix transformations
///
///transforms a matrix expression e of type E using a Functiof f of type F as an elementwise transformation f(e(i,j))
///This transformation needs to leave f constant, meaning that applying f(x), f(y), f(z) yields the same results independent of the
///order of application. Also F must provide a type F::result_type indicating the result type of the functor.
///F must further provide a boolean flag F::zero_identity which indicates that f(0) = 0. This is needed for correct usage with sparse
///arguments - if f(0) != 0 this expression will be dense!
///todo: if densified, this expression is slow!!! try to figure out, how iterators of matrices work.
template<class E, class F>
class MatrixUnaryTransformation:
	public blas::matrix_expression<MatrixUnaryTransformation<E, F> > {
private:
	typedef MatrixUnaryTransformation<E, F> self_type;
	typedef E expression_type;
	typedef typename expression_type::const_iterator1 const_subiterator1_type;
	typedef typename expression_type::const_iterator2 const_subiterator2_type;
	
public:
	typedef typename expression_type::const_closure_type expression_closure_type;

	typedef F functor_type;
	typedef typename functor_type::result_type value_type;
	typedef value_type const_reference;
	typedef const_reference reference;
	typedef value_type const* const_pointer;
	typedef value_type* pointer;
	typedef typename expression_type::size_type size_type;
	typedef typename expression_type::difference_type difference_type;
	
	typedef self_type const const_closure_type;
	typedef const_closure_type closure_type;
	typedef typename E::orientation_category orientation_category;
	typedef blas::unknown_storage_tag storage_category;

	// Construction and destruction
	MatrixUnaryTransformation (blas::matrix_expression<E> const& e, F const& functor): 
		m_expression (e()), m_functor(functor){}

	// Accessors
	size_type size1 () const{
		return m_expression.size1();
	}
	size_type size2 () const{
		return m_expression.size2();
	}

public:
	// Element access
	const_reference operator () (size_type i, size_type j) const {
		return m_functor(m_expression (i, j));
	}

	// Closure comparison
	bool same_closure (MatrixUnaryTransformation const& other) const {
		return m_expression.same_closure(other.m_expression);
	}
private:
	// Iterator types
	class default_const_iterator1;
	class default_const_iterator2;
	class densifying_const_iterator1;
	class densifying_const_iterator2;
	
	//does the matrix need to be densified?
	static const bool densify1 = !functor_type::zero_identity && 
		boost::is_same<typename const_subiterator1_type::iterator_category, blas::sparse_bidirectional_iterator_tag>::value;
	static const bool densify2 = !functor_type::zero_identity && 
		boost::is_same<typename const_subiterator2_type::iterator_category, blas::sparse_bidirectional_iterator_tag>::value;
public:
	typedef typename boost::mpl::if_c<
		densify1,
		densifying_const_iterator1,
		default_const_iterator1
	>::type const_iterator1;
	
	typedef typename boost::mpl::if_c<
		densify2,
		densifying_const_iterator2,
		default_const_iterator2
	>::type const_iterator2;

	typedef const_iterator1 iterator1;
	typedef const_iterator2 iterator2;
	
	typedef blas::reverse_iterator_base1<const_iterator1> const_reverse_iterator1;
	typedef blas::reverse_iterator_base2<const_iterator2> const_reverse_iterator2;

	// Element lookup
	const_iterator1 find1 (int rank, size_type i, size_type j) const {
		const_subiterator1_type it1 (m_expression.find1 (rank, i, j));
		return const_iterator1(*this, it1,i,j);

	}
	const_iterator2 find2 (int rank, size_type i, size_type j) const {
		const_subiterator2_type it2 (m_expression.find2 (rank, i, j));
		return const_iterator2(*this, it2,i,j);
	}
	
private:
	class default_const_iterator1:
		public blas::container_const_reference<MatrixUnaryTransformation>,
		public blas::iterator_base_traits<typename const_subiterator1_type::iterator_category>::template
			iterator_base<default_const_iterator1, value_type>::type {
	public:
		typedef typename const_subiterator1_type::iterator_category iterator_category;
		typedef typename MatrixUnaryTransformation::difference_type difference_type;
		typedef typename MatrixUnaryTransformation::value_type value_type;
		typedef typename MatrixUnaryTransformation::const_reference reference;
		typedef typename MatrixUnaryTransformation::const_pointer pointer;

		typedef const_iterator2 dual_iterator_type;
		typedef const_reverse_iterator2 dual_reverse_iterator_type;

		// Construction and destruction
		default_const_iterator1 ():
			blas::container_const_reference<self_type> (), m_position () {}
		default_const_iterator1 (self_type const& ref, const_subiterator1_type const& it,size_type, size_type):
			blas::container_const_reference<self_type> (ref), m_position (it) {}

		// Arithmetic
		default_const_iterator1& operator ++ () {
			++ m_position;
			return *this;
		}
		default_const_iterator1& operator -- () {
			-- m_position;
			return *this;
		}
		default_const_iterator1& operator += (difference_type n) {
			m_position += n;
			return *this;
		}
		default_const_iterator1& operator -= (difference_type n) {
			m_position -= n;
			return *this;
		}
		difference_type operator - (default_const_iterator1 const& it) const {
			BOOST_UBLAS_CHECK ((*this) ().same_closure (it ()), blas::external_logic ());
			return m_position - it.m_position;
		}

		// Dereference
		const_reference operator * () const {
			return (*this)().m_functor(*m_position);
		}
		const_reference operator [] (difference_type n) const {
			return *(*this + n);
		}

		dual_iterator_type begin () const {
			return (*this)().find2 (1, index1 (), 0);
		}
		dual_iterator_type end () const {
			return (*this)().find2 (1, index1 (), (*this) ().size2 ());
		}
		dual_reverse_iterator_type rbegin () const {
			return const_reverse_iterator2 (end ());
		}
		dual_reverse_iterator_type rend () const {
			return const_reverse_iterator2 (begin ());
		}

		// Indices
		size_type index1 () const {
			return m_position.index1 ();
		}
		size_type index2 () const {
			return m_position.index2 ();
		}

		// Assignment 
		default_const_iterator1& operator = (default_const_iterator1 const& it) {
			blas::container_const_reference<self_type>::assign (&it ());
			m_position = it.m_position;
			return *this;
		}

		// Comparison
		bool operator == (default_const_iterator1 const& it) const {
			BOOST_UBLAS_CHECK ((*this) ().same_closure (it ()), blas::external_logic ());
			return m_position == it.m_position;
		}
		bool operator < (default_const_iterator1 const& it) const {
			BOOST_UBLAS_CHECK ((*this) ().same_closure (it ()), blas::external_logic ());
			return m_position < it.m_position;
		}

	private:
		const_subiterator1_type m_position;
	};

	class default_const_iterator2:
		public blas::container_const_reference<MatrixUnaryTransformation>,
		public blas::iterator_base_traits<typename E::const_iterator2::iterator_category>::template
			iterator_base<default_const_iterator2, value_type>::type {
	public:
		typedef typename E::const_iterator2::iterator_category iterator_category;
		typedef typename MatrixUnaryTransformation::difference_type difference_type;
		typedef typename MatrixUnaryTransformation::value_type value_type;
		typedef typename MatrixUnaryTransformation::const_reference reference;
		typedef typename MatrixUnaryTransformation::const_pointer pointer;

		typedef const_iterator1 dual_iterator_type;
		typedef const_reverse_iterator1 dual_reverse_iterator_type;

		// Construction and destruction
		default_const_iterator2 ():
			blas::container_const_reference<self_type> (), m_position () {}
		default_const_iterator2 (self_type const& ref, const_subiterator2_type const& it,size_type, size_type):
			blas::container_const_reference<self_type> (ref), m_position (it) {}

		// Arithmetic
		default_const_iterator2 &operator ++ () {
			++ m_position;
			return *this;
		}
		default_const_iterator2 &operator -- () {
			-- m_position;
			return *this;
		}
		default_const_iterator2 &operator += (difference_type n) {
			m_position += n;
			return *this;
		}
		default_const_iterator2 &operator -= (difference_type n) {
			m_position -= n;
			return *this;
		}
		difference_type operator - (default_const_iterator2 const& it) const {
			BOOST_UBLAS_CHECK ((*this) ().same_closure (it ()),blas:: external_logic ());
			return m_position - it.m_position;
		}

		// Dereference
		const_reference operator * () const {
			return (*this)().m_functor(*m_position);
		}
		const_reference operator [] (difference_type n) const {
			return *(*this + n);
		}

		dual_iterator_type begin () const {
			return (*this) ().find1 (1, 0, index2 ());
		}

		dual_iterator_type end () const {
			return (*this) ().find1 (1, (*this) ().size1 (), index2 ());
		}

		dual_reverse_iterator_type rbegin () const {
			return const_reverse_iterator1 (end ());
		}
		dual_reverse_iterator_type rend () const {
			return const_reverse_iterator1 (begin ());
		}


		// Indices
		size_type index1 () const {
			return m_position.index1 ();
		}
		size_type index2 () const {
			return m_position.index2 ();
		}

		// Assignment 
		default_const_iterator2 &operator = (default_const_iterator2 const& it) {
		   blas:: container_const_reference<self_type>::assign (&it());
			m_position = it.m_position;
			return *this;
		}

		// Comparison
		bool operator == (default_const_iterator2 const& it) const {
			BOOST_UBLAS_CHECK ((*this) ().same_closure (it ()), blas::external_logic ());
			return m_position == it.m_position;
		}
		bool operator < (default_const_iterator2 const& it) const {
			BOOST_UBLAS_CHECK ((*this) ().same_closure (it ()), blas::external_logic ());
			return m_position < it.m_position;
		}

	private:
		const_subiterator2_type m_position;
	};
	
	class densifying_const_iterator1:
		public blas::container_const_reference<MatrixUnaryTransformation>,
		public blas::iterator_base_traits<std::bidirectional_iterator_tag>::template
			iterator_base<densifying_const_iterator1, value_type>::type {
	public:
		typedef typename std::bidirectional_iterator_tag iterator_category;
		typedef typename MatrixUnaryTransformation::difference_type difference_type;
		typedef typename MatrixUnaryTransformation::value_type value_type;
		typedef typename MatrixUnaryTransformation::const_reference reference;
		typedef typename MatrixUnaryTransformation::const_pointer pointer;

		typedef const_iterator2 dual_iterator_type;
		typedef const_reverse_iterator2 dual_reverse_iterator_type;

		// Construction and destruction
		densifying_const_iterator1 ():
			blas::container_const_reference<self_type> (), m_position1 () {}
		densifying_const_iterator1 (self_type const& ref, const_subiterator1_type const& it,size_type index1, size_type index2):
			blas::container_const_reference<self_type> (ref), m_position1 (it)
			,m_index1(index1),m_index2(index2),
			m_zeroValue(ref.m_functor(value_type(0))) {}

		// Arithmetic
		densifying_const_iterator1& operator ++ () {
//			while(m_index1 >= m_position1.index1()){
//				++m_position1;
//			}
			++m_index1;
			return *this;
		}
		densifying_const_iterator1& operator -- () {
//			while(m_index1 <= m_position1.index1()){
//				--m_position1;
//			}
			--m_index1;
			return *this;
		}

		// Dereference
		const_reference operator *() const {
			return (*this)()(m_index1,m_index2);
//			if(m_index1 == m_position1.index1() && m_index2 == m_position1.index2())
//				return (*this)().m_functor(*m_position1);
//			else
//				return m_zeroValue;
		}

		//dual iterators
		dual_iterator_type begin () const {
			return (*this)().find2 (1, index1 (), 0);
		}
		dual_iterator_type end () const {
			return (*this)().find2 (1, index1 (), (*this) ().size2 ());
		}
		dual_reverse_iterator_type rbegin () const {
			return const_reverse_iterator2 (end ());
		}
		dual_reverse_iterator_type rend () const {
			return const_reverse_iterator2 (begin ());
		}

		// Indices
		size_type index1 () const {
			return m_index1;
		}
		size_type index2 () const {
			return m_index2;
		}

		// Assignment 
		densifying_const_iterator1 &operator = (densifying_const_iterator1 const& it) {
			blas::container_const_reference<self_type>::assign (&it ());
			m_position1 = it.m_position1;
			m_index1 = it.m_index1;
			m_index2 = it.m_index2;
			return *this;
		}

		// Comparison
		bool operator == (densifying_const_iterator1 const& it) const {
			BOOST_UBLAS_CHECK ((*this) ().same_closure (it ()), blas::external_logic ());
			return m_index1 == it.m_index1 && m_index2 == it.m_index2;
		}

	private:
		const_subiterator1_type m_position1;
		size_type m_index1;
		size_type m_index2;
		value_type m_zeroValue;
	};
	
	class densifying_const_iterator2:
		public blas::container_const_reference<MatrixUnaryTransformation>,
		public blas::iterator_base_traits<std::bidirectional_iterator_tag>::template
			iterator_base<densifying_const_iterator2, value_type>::type {
	public:
		typedef typename std::bidirectional_iterator_tag iterator_category;
		typedef typename MatrixUnaryTransformation::difference_type difference_type;
		typedef typename MatrixUnaryTransformation::value_type value_type;
		typedef typename MatrixUnaryTransformation::const_reference reference;
		typedef typename MatrixUnaryTransformation::const_pointer pointer;

		typedef const_iterator1 dual_iterator_type;
		typedef const_reverse_iterator1 dual_reverse_iterator_type;

		// Construction and destruction
		densifying_const_iterator2 ():
			blas::container_const_reference<self_type> (), m_position2 () {}
		densifying_const_iterator2 (self_type const& ref, const_subiterator2_type const& it,size_type index1, size_type index2):
			blas::container_const_reference<self_type> (ref), m_position2 (it)
			,m_index1(index1),m_index2(index2),
			m_zeroValue(ref.m_functor(value_type(0))) {}

		// Arithmetic
		densifying_const_iterator2& operator ++ () {
//			while(m_index2 >= m_position2.index2()){
//				++m_position2;
//			}
			++m_index2;
			return *this;
		}
		densifying_const_iterator2& operator -- () {
//			while(m_index2 <= m_position2.index2()){
//				--m_position2;
//			}
			--m_index2;
			return *this;
		}

		// Dereference
		const_reference operator *() const {
			return (*this)()(m_index1,m_index2);
//			if(m_index2 == m_position2.index2() && m_index1 == m_position2.index1()){
//				std::cout<<m_index1<<" "<<m_index2<<" "<<m_position2.index1()<<" "<<m_position2.index2()<<std::endl;
//				return (*this)().m_functor(*m_position2);
//			}
//			else
//				return m_zeroValue;
		}

		//dual iterators
		dual_iterator_type begin () const {
			return (*this) ().find1 (1, 0, index2 ());
		}

		dual_iterator_type end () const {
			return (*this) ().find1 (1, (*this) ().size1 (), index2 ());
		}
		dual_reverse_iterator_type rbegin () const {
			return const_reverse_iterator2 (end ());
		}
		dual_reverse_iterator_type rend () const {
			return const_reverse_iterator2 (begin ());
		}

		// Indices
		size_type index1 () const {
			return m_index1;
		}
		size_type index2 () const {
			return m_index2;
		}

		// Assignment 
		densifying_const_iterator2 &operator = (densifying_const_iterator2 const& it) {
			blas::container_const_reference<self_type>::assign (&it ());
			m_position2 = it.m_position2;
			m_index1 = it.m_index1;
			m_index2 = it.m_index2;
			return *this;
		}

		// Comparison
		bool operator == (densifying_const_iterator2 const& it) const {
			BOOST_UBLAS_CHECK ((*this) ().same_closure (it ()), blas::external_logic ());
			return m_index1 == it.m_index1 && m_index2 == it.m_index2;
		}

	private:
		const_subiterator2_type m_position2;
		size_type m_index1;
		size_type m_index2;
		value_type m_zeroValue;
	};
	
public:
	const_iterator1 begin1 () const {
		return find1 (0, 0, 0);
	}
	const_iterator1 end1 () const {
		return find1 (0, size1 (), 0);
	}

	const_iterator2 begin2 () const {
		return find2 (0, 0, 0);
	}
	const_iterator2 end2 () const {
		return find2 (0, 0, size2 ());
	}

	// Reverse iterators

	const_reverse_iterator1 rbegin1 () const {
		return const_reverse_iterator1 (end1 ());
	}
	const_reverse_iterator1 rend1 () const {
		return const_reverse_iterator1 (begin1 ());
	}

	const_reverse_iterator2 rbegin2 () const {
		return const_reverse_iterator2 (end2 ());
	}
	const_reverse_iterator2 rend2 () const {
		return const_reverse_iterator2 (begin2 ());
	}

private:
	expression_closure_type m_expression;
	functor_type m_functor;
};

/// This class provides a unification of elementwise transformations for vector and matrix expressions.
template<class E,class Transformation>
struct UnaryTransformationImpl{};
/// Specialisation for vector_expressions for unification of elementwise transformations.
template<class E, class Transformation>
struct UnaryTransformationImpl<blas::vector_expression<E>,Transformation >{
	typedef typename Transformation::template Functor<typename E::value_type> functor_type;
	typedef VectorUnaryTransformation<E, functor_type > type;
};
/// Specialisation for matrix_expressions for unification of elementwise transformations.
template<class E,class Transformation>
struct UnaryTransformationImpl<blas::matrix_expression<E>,Transformation >{
	typedef typename Transformation::template Functor<typename E::value_type> functor_type;
	typedef MatrixUnaryTransformation<E, functor_type > type;
};

///\brief This class provides a unification of blas Transformations of elementwise transformations for vector and matrix expressions.
///
///Based on a expression type, it constructs the correct ublas expression version of the transformation.
///This class needs two arguments. The first is the base expression which is the type of the argument vector/matrix of the transformation.
///The second argument is the type of the Transformation.
///This class defines a typedef called type. It is the expression type of the unary transformation.
///The Functor must be of a special Form:
///struct X{
///    template<class T>
///    struct Functor{
///        typedef ... return_type;
///        typedef static const bool zero_identity = *is f(0) = 0?*;
///        return_type operator(T const& arg){...};
///    };
///};
template<class E,class Transformation>
struct UnaryTransformation : public UnaryTransformationImpl<typename ExpressionType<E>::type,Transformation>{};

}
}

#endif
