/*!
 *  \brief Meta template transformations and types for generic programming with ublas.
 *
 *  \author O.Krause
 *  \date 2012
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
#ifndef SHARK_LINALG_BLAS_IMPL_BINARYEXPRESSIONS_H
#define SHARK_LINALG_BLAS_IMPL_BINARYEXPRESSIONS_H

#include <shark/LinAlg/BLAS/traits/MetaBlas.h>

namespace shark{
namespace detail{
///\brief Binary Expressions using arbitrary functors. 
///
///For two vectors a and b as well as a functor f, defines an resulting expression r(i)=f(a(i),b(i)). 
template<class E1, class E2, class F>
class VectorBinaryElementwiseTransformation:
	public blas::vector_expression<VectorBinaryElementwiseTransformation<E1,E2, F> > {
	typedef VectorBinaryElementwiseTransformation<E1,E2, F> self_type;
	typedef E1 const expression1_type;
	typedef E2 const expression2_type;
	typedef typename E1::const_iterator const_subiterator1_type;
	typedef typename E2::const_iterator const_subiterator2_type;
public:
	typedef F functor_type;
	typedef typename E1::const_closure_type expression_closure1_type;
	typedef typename E2::const_closure_type expression_closure2_type;
	typedef typename blas::promote_traits<
		typename E1::size_type, 
		typename E2::size_type
	>::promote_type size_type;
	typedef typename blas::promote_traits<
		typename E1::difference_type, 
		typename E2::difference_type
	>::promote_type difference_type;
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
	explicit VectorBinaryElementwiseTransformation (
		expression1_type& e1, 
		expression2_type& e2,
		F const& functor
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

public:
	// Element access
	const_reference operator() (size_type i) const {
		return m_functor(m_expression1(i),m_expression2(i));
	}

	const_reference operator[] (size_type i) const {
		return m_functor(m_expression1(i),m_expression2(i));
	}

	// Closure comparison
	bool same_closure (VectorBinaryElementwiseTransformation const& vu) const {
		return expression1 ().same_closure (vu.expression1())
		&& expression2 ().same_closure (vu.expression2());
	}

	// Iterator types
	
	// Iterator enhances the iterator of the referenced expression2
	// with the unary functor.
	
	//this is the iterator of the dense version.
	class const_iterator:
		public blas::container_const_reference<self_type>,
		public blas::iterator_base_traits<
			typename blas::iterator_restrict_traits<
				typename const_subiterator1_type::iterator_category,
				typename const_subiterator2_type::iterator_category
			>::iterator_category
		>::template iterator_base<const_iterator, value_type>::type{
	public:
		typedef typename blas::iterator_restrict_traits<
			typename const_subiterator1_type::iterator_category,
			typename const_subiterator2_type::iterator_category
		>::iterator_category iterator_category;
		typedef typename VectorBinaryElementwiseTransformation::difference_type difference_type;
		typedef typename VectorBinaryElementwiseTransformation::value_type value_type;
		typedef typename VectorBinaryElementwiseTransformation::const_reference reference;
		typedef typename VectorBinaryElementwiseTransformation::const_pointer pointer;

		// Construction and destruction
		const_iterator ():
			blas::container_const_reference<self_type> (), m_index (), 
			m_iterator1 (), m_end1 (), 
			m_iterator2 (), m_end2 () {}
		const_iterator (
			self_type const& ref, size_type i,
			const_subiterator1_type const& it1, const_subiterator1_type const& end1,
			const_subiterator2_type const& it2, const_subiterator2_type const& end2
		):  blas::container_const_reference<self_type>(ref), m_index(i), 
			m_iterator1(it1), m_end1(end1), 
			m_iterator2 (it2), m_end2 (end2) {}

	private: 
		//we need to handle all specializations independently from each other
		//also for packed/sparse we need to check, whether our functor has zero_identity 
		//to change the implementation accordingly
		//the correct choice is handled by ublas.
		
		// Dense specializations are easy
		void increment (blas::dense_random_access_iterator_tag) {
			++ m_index; 
			++ m_iterator1; 
			++ m_iterator2;
		}
		void decrement (blas::dense_random_access_iterator_tag) {
			-- m_index; 
			-- m_iterator1; 
			-- m_iterator2;
		}
		void increment (blas::dense_random_access_iterator_tag, difference_type n) {
			m_index += n; 
			m_iterator1 += n; 
			m_iterator2 += n;
		}
		void decrement (blas::dense_random_access_iterator_tag, difference_type n) {
			m_index -= n; 
			m_iterator1 -= n; 
			m_iterator2 -= n;
		}
		value_type dereference (blas::dense_random_access_iterator_tag) const {
			return (*this)().m_functor(*m_iterator1, *m_iterator2);
		}

		// Packed specializations
		void increment (blas::packed_random_access_iterator_tag) {
			if (m_iterator1 != m_end1)
				if (m_iterator1.index () <= m_index)
					++ m_iterator1;
			if (m_iterator2 != m_end2)
				if (m_iterator2.index () <= m_index)
					++ m_iterator2;
			++ m_index;
		}		
		
		void decrement (blas::packed_random_access_iterator_tag) {
			if (m_iterator1 != m_end1)
				if (m_index <= m_iterator1.index ())
					-- m_iterator1;
			if (m_iterator2 != m_end2)
				if (m_index <= m_iterator2.index ())
					-- m_iterator2;
			-- m_index;
		}
		void increment (blas::packed_random_access_iterator_tag, difference_type n) {
			while (n > 0) {
				increment (blas::packed_random_access_iterator_tag ());
				--n;
			}
			while (n < 0) {
				decrement (blas::packed_random_access_iterator_tag ());
				++n;
			}
		}
		void decrement (blas::packed_random_access_iterator_tag, difference_type n) {
			while (n > 0) {
				decrement (blas::packed_random_access_iterator_tag ());
				--n;
			}
			while (n < 0) {
				increment (blas::packed_random_access_iterator_tag ());
				++n;
			}
		}
		value_type dereference (blas::packed_random_access_iterator_tag) const {
			value_type t1 = value_type/*zero*/();
			if (m_iterator1 != m_end1)
				if (m_iterator1.index () == m_index)
					t1 = *m_iterator1;
			value_type t2 = value_type/*zero*/();
			if (m_iterator2 != m_end2)
				if (m_iterator2.index () == m_index)
					t2 = *m_iterator2;
			return (*this)().m_functor(t1, t2);
		}

		// Sparse specializations
		void increment (blas::sparse_bidirectional_iterator_tag) {
			size_type index1 = (*this) ().size ();
			if (m_iterator1 != m_end1) {
				if  (m_iterator1.index () <= m_index)
					++ m_iterator1;
				if (m_iterator1 != m_end1)
					index1 = m_iterator1.index ();
			}
			size_type index2 = (*this) ().size ();
			if (m_iterator2 != m_end2) {
				if (m_iterator2.index () <= m_index)
					++ m_iterator2;
				if (m_iterator2 != m_end2)
					index2 = m_iterator2.index ();
			}
			m_index = (std::min) (index1, index2);
		}
		void decrement (blas::sparse_bidirectional_iterator_tag) {
			size_type index1 = (*this) ().size ();
			if (m_iterator1 != m_end1) {
				if (m_index <= m_iterator1.index ())
					-- m_iterator1;
				if (m_iterator1 != m_end1)
					index1 = m_iterator1.index ();
			}
			size_type index2 = (*this) ().size ();
			if (m_iterator2 != m_end2) {
				if (m_index <= m_iterator2.index ())
					-- m_iterator2;
				if (m_iterator2 != m_end2)
					index2 = m_iterator2.index ();
			}
			m_index = (std::max) (index1, index2);
		}
		void increment (blas::sparse_bidirectional_iterator_tag, difference_type n) {
			while (n > 0) {
				increment (blas::sparse_bidirectional_iterator_tag ());
				--n;
			}
			while (n < 0) {
				decrement (blas::sparse_bidirectional_iterator_tag ());
				++n;
			}
		}
		void decrement (blas::sparse_bidirectional_iterator_tag, difference_type n) {
			while (n > 0) {
				decrement (blas::sparse_bidirectional_iterator_tag ());
				--n;
			}
			while (n < 0) {
				increment (blas::sparse_bidirectional_iterator_tag ());
				++n;
			}
		}
		value_type dereference (blas::sparse_bidirectional_iterator_tag) const {
			value_type t1 = value_type/*zero*/();
			if (m_iterator1 != m_end1)
				if (m_iterator1.index () == m_index)
					t1 = *m_iterator1;
			value_type t2 = value_type/*zero*/();
			if (m_iterator2 != m_end2)
				if (m_iterator2.index () == m_index)
					t2 = *m_iterator2;
			return (*this)().m_functor(t1, t2);
		}

	public: 
		// Arithmetic
		const_iterator &operator ++ () {
			increment (iterator_category ());
			return *this;
		}
		const_iterator &operator -- () {
			decrement (iterator_category ());
			return *this;
		}
		const_iterator &operator += (difference_type n) {
			increment (iterator_category (), n);
			return *this;
		}
		const_iterator &operator -= (difference_type n) {
			decrement (iterator_category (), n);
			return *this;
		}
		difference_type operator - (const const_iterator &it) const {
			BOOST_UBLAS_CHECK ((*this) ().same_closure (it ()), blas::external_logic ());
			return index () - it.index ();
		}

		// Dereference
		const_reference operator * () const {
			return dereference (iterator_category ());
		}
		const_reference operator [] (difference_type n) const {
			return *(*this + n);
		}

		// Index
		size_type index () const {
			return m_index;
		}

		// Assignment
		const_iterator &operator = (const_iterator const& it) {
			blas::container_const_reference<self_type>::assign (&it ());
			m_index = it.m_index;
			m_iterator1 = it.m_iterator1;
			m_end1 = it.m_end1;
			m_iterator2 = it.m_iterator2;
			m_end2 = it.m_end2;
			return *this;
		}

		// Comparison
		bool operator == (const_iterator const& it) const {
			BOOST_UBLAS_CHECK ((*this) ().same_closure (it ()), blas::external_logic ());
			return index () == it.index ();
		}
		bool operator < (const_iterator const& it) const {
			BOOST_UBLAS_CHECK ((*this) ().same_closure (it ()), blas::external_logic ());
			return index () < it.index ();
		}

	private:
		size_type m_index;
		const_subiterator1_type m_iterator1;
		const_subiterator1_type m_end1;
		const_subiterator2_type m_iterator2;
		const_subiterator2_type m_end2;
	};
	typedef const_iterator iterator;

	// Element lookup
	const_iterator find (size_type i) const {
		return const_iterator (*this,i,
			m_expression1.find(i),m_expression1.end(),
			m_expression2.find(i),m_expression2.end()
		);
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
	expression_closure1_type m_expression1;
	expression_closure2_type m_expression2;
	F m_functor;
};

template<class E1, class E2, class F>
class MatrixBinaryElementwiseTransformation:
	public blas::matrix_expression<MatrixBinaryElementwiseTransformation<E1, E2, F> > {
	
public:
	typedef E1 expression1_type;
	typedef E2 expression2_type;
	typedef typename E1::const_closure_type expression1_closure_type;
	typedef typename E2::const_closure_type expression2_closure_type;

	typedef typename E1::size_type size_type;
	typedef typename E1::difference_type difference_type;
	typedef typename F::result_type value_type;
	typedef value_type const_reference;
	typedef const_reference reference;

	typedef const MatrixBinaryElementwiseTransformation<E1, E2, F> const_closure_type;
	typedef const_closure_type closure_type;
	typedef typename E1::orientation_category orientation_category;
	typedef blas::unknown_storage_tag storage_category;

	typedef F functor_type;

        // Construction and destruction

        MatrixBinaryElementwiseTransformation (
		expression1_type const&e1,  expression2_type const& e2, functor_type functor 
	): m_expression1 (e1), m_expression2 (e2),m_functor(functor) {}

        // Accessors

        size_type size1 () const {
		return m_expression1.size1 ();
        }

        size_type size2 () const {
		return m_expression1.size2 ();
        }

        const_reference operator () (size_type i, size_type j) const {
		return m_functor( m_expression1 (i, j), m_expression2(i,j));
        }

        // Closure comparison
        bool same_closure (MatrixBinaryElementwiseTransformation const&mbs2) const {
		return (*this).m_expression1.same_closure (mbs2.m_expression1) &&
		&m_expression2 == &(mbs2.m_expression2);
        }

        // Iterator types
private:
	typedef typename E2::const_iterator1 const_iterator21_type;
        typedef typename E2::const_iterator2 const_iterator22_type;
public:
	typedef blas::indexed_const_iterator1<const_closure_type, typename const_iterator21_type::iterator_category> const_iterator1;
        typedef const_iterator1 iterator1;
        typedef blas::indexed_const_iterator2<const_closure_type, typename const_iterator22_type::iterator_category> const_iterator2;
        typedef const_iterator2 iterator2;

        typedef blas::reverse_iterator_base1<const_iterator1> const_reverse_iterator1;
        typedef blas::reverse_iterator_base2<const_iterator2> const_reverse_iterator2;

        // Element lookup

        const_iterator1 find1 (int rank, size_type i, size_type j) const {
		const_iterator21_type it21 (m_expression2.find1 (rank, i, j));
		return const_iterator1 (*this, it21.index1 (), it21.index2 ());
        }
        const_iterator2 find2 (int rank, size_type i, size_type j) const {
		const_iterator22_type it22 (m_expression2.find2 (rank, i, j));
		return const_iterator2 (*this, it22.index1 (), it22.index2 ());
        }

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
		return const_reverse_iterator1 (end1());
        }
        const_reverse_iterator1 rend1 () const {
		return const_reverse_iterator1 (begin1());
        }

        const_reverse_iterator2 rbegin2 () const {
		return const_reverse_iterator2 (end2());
        }
        const_reverse_iterator2 rend2 () const {
		return const_reverse_iterator2 (begin2());
        }

private:
	expression1_closure_type m_expression1;
        expression2_closure_type m_expression2;
	functor_type m_functor;
};

/// This class provides a unification of elementwise binary transformations for vector and matrix expressions.
template<class E1, class E2, class Functor>
struct BinaryTransformationImpl{};
/// Specialisation for vector_expressions for unification of elementwise transformations.
template<class E1, class E2, class Functor>
struct BinaryTransformationImpl<
	blas::vector_expression<E1>,
	blas::vector_expression<E2>,
	Functor
>{
	typedef VectorBinaryElementwiseTransformation<E1,E2, Functor > type;
};
//not implemented!!!
/// Specialisation for matrix_expressions for unification of elementwise transformations.
template<class E1, class E2, class Functor>
struct BinaryTransformationImpl<blas::matrix_expression<E1>,blas::matrix_expression<E2>, Functor >{
	typedef MatrixBinaryElementwiseTransformation<E1,E2, Functor > type;
};

///\brief This class provides a unification of blas Transformations of binary elementwise transformations for vector and matrix expressions.
template<class E1,class E2, class Functor>
struct BinaryTransformation : 
public BinaryTransformationImpl<
	typename ExpressionType<E1>::type,
	typename ExpressionType<E2>::type,
	Functor
>{};

}
}

#endif
