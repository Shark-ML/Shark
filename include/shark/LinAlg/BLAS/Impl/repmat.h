/**
*
*  \brief Constructs a matrix as a repetition of a single row vector
*
*  \author O.Krause
*  \date 2010-2011
*
*  \par Copyright (c) 1998-2007:
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
#ifndef SHARK_LINALG_REPMAT_INL
#define SHARK_LINALG_REPMAT_INL

namespace shark{ namespace blas{
template<class V>
class VectorRepeater:public blas::matrix_expression<VectorRepeater<V> > {
private:
	typedef V expression_type;
	typedef VectorRepeater<V> self_type;
	typedef typename V::const_iterator const_subiterator_type;
public:
	typedef typename V::const_closure_type expression_closure_type;
	typedef typename V::size_type size_type;
	typedef typename V::difference_type difference_type;
	typedef typename V::value_type value_type;
	typedef value_type const_reference;
	typedef const_reference reference;
	typedef self_type const const_closure_type;
	typedef const_closure_type closure_type;
	typedef blas::row_major orientation_category;
	typedef blas::unknown_storage_tag storage_category;

	// Construction and destruction
	explicit VectorRepeater (expression_type const& e, std::size_t rows):
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
	const_reference operator() (size_type i, size_type j) const {
		return m_vector(j);
	}

	// Closure comparison
	bool same_closure (const VectorRepeater &other) const {
		return (*this).expression ().same_closure (other.expression ());
	}

	// Iterator types

	class const_iterator1;
	typedef const_iterator1 iterator1;
	class const_iterator2;
	typedef const_iterator2 iterator2;

	typedef blas::reverse_iterator_base1<const_iterator1> const_reverse_iterator1;
	typedef blas::reverse_iterator_base2<const_iterator2> const_reverse_iterator2;

	// Element lookup
	const_iterator1 find1 (int, size_type i, size_type j) const {
		return const_iterator1 (*this,i, m_vector.find(j));
	}
	const_iterator2 find2 (int, size_type i, size_type j) const {
		return const_iterator2 (*this,i, m_vector.find(j));
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
	

	class const_iterator1:
		public blas::container_const_reference<VectorRepeater>,
		public blas::iterator_base_traits<typename const_subiterator_type::iterator_category>::template
			iterator_base<const_iterator1, value_type>::type {
	public:
		typedef typename const_subiterator_type::iterator_category iterator_category;
		typedef typename VectorRepeater::difference_type difference_type;
		typedef typename VectorRepeater::value_type value_type;
		typedef typename VectorRepeater::const_reference reference;
		typedef value_type const* pointer;

		typedef const_iterator2 dual_iterator_type;
		typedef const_reverse_iterator2 dual_reverse_iterator_type;

		// Construction and destruction
		const_iterator1 ():m_row(0){}
		const_iterator1 (const self_type &mu,std::size_t row, const const_subiterator_type &it):
		blas::container_const_reference<self_type> (mu), m_row(row),m_vectorIterator(it) {}

		// Arithmetic
		const_iterator1& operator++ (){
			++m_row;
			return *this;
		}
		const_iterator1& operator-- (){
			-- m_row;
			return *this;
		}
		const_iterator1& operator+= (difference_type n) {
			m_row += n;
			return *this;
		}
		const_iterator1& operator-= (difference_type n) {
			m_row -= n;
			return *this;
		}
		difference_type operator- (const_iterator1 const& it) const{
			return m_row - it.m_row;
		}

		// Dereference
		const_reference operator* () const {
			return *m_vectorIterator;
		}
		const_reference operator[] (difference_type n) const {
			return *(*this + n);
		}

		const_iterator2 begin () const {
			return (*this) ().find2 (1, m_row, 0);
		}
		
		const_iterator2 end () const {
			return (*this) ().find2 (1, m_row,(*this)().size2());
		}

		const_reverse_iterator2 rbegin () const {
			return const_reverse_iterator2 (end());
		}

		const_reverse_iterator2 rend () const {
			return const_reverse_iterator2 (begin());
		}

		// Indices
		size_type index1 () const {
			return m_row;
		}
		size_type index2 () const {
			return m_vectorIterator.index();
		}

		// Assignment 
		const_iterator1 &operator = (const_iterator1 const& it) {
			blas::container_const_reference<self_type>::assign (&it());
			m_vectorIterator = it.m_vectorIterator;
			m_row = it.m_row;
			return *this;
		}

		// Comparison
		bool operator == (const_iterator1 const& it) const {
			return m_row == it.m_row
			&& m_vectorIterator == it.m_vectorIterator;
		}
		bool operator < (const_iterator1 const& it) const {
			return m_row < it.m_row;
		}

	private:
		std::size_t m_row;
		const_subiterator_type m_vectorIterator;
	};

	class const_iterator2:
		public blas::container_const_reference<VectorRepeater>,
		public blas::iterator_base_traits<typename const_subiterator_type::iterator_category>::template
			iterator_base<const_iterator2, value_type>::type 
	{
	public:
		typedef typename const_subiterator_type::iterator_category iterator_category;
		typedef typename VectorRepeater::difference_type difference_type;
		typedef typename VectorRepeater::value_type value_type;
		typedef typename VectorRepeater::const_reference reference;
		typedef value_type const* pointer;

		typedef const_iterator1 dual_iterator_type;
		typedef const_reverse_iterator1 dual_reverse_iterator_type;

		// Construction and destruction
		const_iterator2 ():m_row(0){}
		const_iterator2 (self_type const& mu,std::size_t row, const_subiterator_type const& it):
		blas::container_const_reference<self_type> (mu), m_row(row),m_vectorIterator(it) {}

		// Arithmetic
		const_iterator2& operator++() {
			++m_vectorIterator;
			return *this;
		}
		const_iterator2& operator--() {
			--m_vectorIterator;
			return *this;
		}
		const_iterator2& operator+= (difference_type n) {
			m_vectorIterator+= n;
			return *this;
		}
		const_iterator2& operator-= (difference_type n) {
			m_vectorIterator-= n;
			return *this;
		}
		difference_type operator- (const const_iterator2 &it) const {
			return m_vectorIterator - it.m_vectorIterator;
		}

		// Dereference
		const_reference operator * () const {
			return *m_vectorIterator;
		}
		const_reference operator [] (difference_type n) const {
			return *(m_vectorIterator + n);
		}

		const_iterator1 begin () const {
			return const_iterator1((*this)(),0,m_vectorIterator);
		}
		const_iterator1 end () const {
			return const_iterator1((*this)(),(*this) ().size1 (),m_vectorIterator);
		}
		const_reverse_iterator1 rbegin () const {
			return const_reverse_iterator1 (end());
		}
		const_reverse_iterator1 rend () const {
			return const_reverse_iterator1 (begin());
		}

		// Indices
		size_type index1() const {
			return m_row;
		}
		size_type index2() const {
			return m_vectorIterator.index();
		}

		// Assignment 
		const_iterator2& operator= (const_iterator2 const& it) {
			blas::container_const_reference<self_type>::assign (&it());
			m_vectorIterator = it.m_vectorIterator;
			m_row = it.m_row;
			return *this;
		}

		// Comparison
		bool operator == (const_iterator2 const& it) const {
			return m_vectorIterator == it.m_vectorIterator
			&& m_row == it.m_row;
		}
		bool operator < (const_iterator2 const& it) const {
			return m_vectorIterator < it.m_vectorIterator;
		}

	private:
		std::size_t m_row;
		const_subiterator_type m_vectorIterator;
	};

private:
	expression_closure_type m_vector;
	std::size_t m_rows;
};
}}
#endif
