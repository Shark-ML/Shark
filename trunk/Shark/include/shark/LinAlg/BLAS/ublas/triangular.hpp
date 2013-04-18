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

#ifndef _BOOST_UBLAS_TRIANGULAR_
#define _BOOST_UBLAS_TRIANGULAR_

#include <shark/LinAlg/BLAS/ublas/matrix.hpp>
#include <shark/LinAlg/BLAS/ublas/detail/temporary.hpp>
#include <boost/type_traits/remove_const.hpp>

// Iterators based on ideas of Jeremy Siek

namespace shark {
namespace blas {
// Triangular matrix adaptor class
template<class M, class TRI>
class triangular_adaptor:
	public matrix_expression<triangular_adaptor<M, TRI> > {

	typedef triangular_adaptor<M, TRI> self_type;

public:
#ifdef BOOST_UBLAS_ENABLE_PROXY_SHORTCUTS
	using matrix_expression<self_type>::operator();
#endif
	typedef const M const_matrix_type;
	typedef M matrix_type;
	typedef TRI triangular_type;
	typedef typename M::size_type size_type;
	typedef typename M::difference_type difference_type;
	typedef typename M::value_type value_type;
	typedef typename M::const_reference const_reference;
	typedef typename boost::mpl::if_<boost::is_const<M>,
	        typename M::const_reference,
	        typename M::reference>::type reference;
	typedef typename boost::mpl::if_<boost::is_const<M>,
	        typename M::const_closure_type,
	        typename M::closure_type>::type matrix_closure_type;
	typedef const self_type const_closure_type;
	typedef self_type closure_type;
	// Replaced by _temporary_traits to avoid type requirements on M
	//typedef typename M::vector_temporary_type vector_temporary_type;
	//typedef typename M::matrix_temporary_type matrix_temporary_type;
	typedef typename storage_restrict_traits<typename M::storage_category,
	        packed_proxy_tag>::storage_category storage_category;
	typedef typename M::orientation_category orientation_category;

	// Construction and destruction
	
	triangular_adaptor(matrix_type &data):
		matrix_expression<self_type> (),
		data_(data) {}
	
	triangular_adaptor(const triangular_adaptor &m):
		matrix_expression<self_type> (),
		data_(m.data_) {}

	// Accessors
	
	size_type size1() const {
		return data_.size1();
	}
	
	size_type size2() const {
		return data_.size2();
	}

	// Storage accessors
	
	const matrix_closure_type &data() const {
		return data_;
	}
	
	matrix_closure_type &data() {
		return data_;
	}

	// Element access
#ifndef BOOST_UBLAS_PROXY_CONST_MEMBER
	
	const_reference operator()(size_type i, size_type j) const {
		BOOST_UBLAS_CHECK(i < size1(), bad_index());
		BOOST_UBLAS_CHECK(j < size2(), bad_index());
		if (triangular_type::other(i, j))
			return data()(i, j);
		else if (triangular_type::one(i, j))
			return one_;
		else
			return zero_;
	}
	
	reference operator()(size_type i, size_type j) {
		BOOST_UBLAS_CHECK(i < size1(), bad_index());
		BOOST_UBLAS_CHECK(j < size2(), bad_index());
		if (!triangular_type::other(i, j)) {
			bad_index().raise();
			// NEVER reached
		}
		return data()(i, j);
	}
#else
	
	reference operator()(size_type i, size_type j) const {
		BOOST_UBLAS_CHECK(i < size1(), bad_index());
		BOOST_UBLAS_CHECK(j < size2(), bad_index());
		if (!triangular_type::other(i, j)) {
			bad_index().raise();
			// NEVER reached
		}
		return data()(i, j);
	}
#endif

	// Assignment
	
	triangular_adaptor &operator = (const triangular_adaptor &m) {
		matrix_assign<scalar_assign> (*this, m);
		return *this;
	}
	
	triangular_adaptor &assign_temporary(triangular_adaptor &m) {
		*this = m;
		return *this;
	}
	template<class AE>
	
	triangular_adaptor &operator = (const matrix_expression<AE> &ae) {
		matrix_assign<scalar_assign> (*this, matrix<value_type> (ae));
		return *this;
	}
	template<class AE>
	
	triangular_adaptor &assign(const matrix_expression<AE> &ae) {
		matrix_assign<scalar_assign> (*this, ae);
		return *this;
	}
	template<class AE>
	
	triangular_adaptor &operator += (const matrix_expression<AE> &ae) {
		matrix_assign<scalar_assign> (*this, matrix<value_type> (*this + ae));
		return *this;
	}
	template<class AE>
	
	triangular_adaptor &plus_assign(const matrix_expression<AE> &ae) {
		matrix_assign<scalar_plus_assign> (*this, ae);
		return *this;
	}
	template<class AE>
	
	triangular_adaptor &operator -= (const matrix_expression<AE> &ae) {
		matrix_assign<scalar_assign> (*this, matrix<value_type> (*this - ae));
		return *this;
	}
	template<class AE>
	
	triangular_adaptor &minus_assign(const matrix_expression<AE> &ae) {
		matrix_assign<scalar_minus_assign> (*this, ae);
		return *this;
	}
	template<class AT>
	
	triangular_adaptor &operator *= (const AT &at) {
		matrix_assign_scalar<scalar_multiplies_assign> (*this, at);
		return *this;
	}
	template<class AT>
	
	triangular_adaptor &operator /= (const AT &at) {
		matrix_assign_scalar<scalar_divides_assign> (*this, at);
		return *this;
	}

	// Closure comparison
	
	bool same_closure(const triangular_adaptor &ta) const {
		return (*this).data().same_closure(ta.data());
	}

	// Swapping
	
	void swap(triangular_adaptor &m) {
		if (this != &m)
			matrix_swap<scalar_swap> (*this, m);
	}
	
	friend void swap(triangular_adaptor &m1, triangular_adaptor &m2) {
		m1.swap(m2);
	}

	// Iterator types
private:
	typedef typename M::const_iterator1 const_subiterator1_type;
	typedef typename boost::mpl::if_<boost::is_const<M>,
	        typename M::const_iterator1,
	        typename M::iterator1>::type subiterator1_type;
	typedef typename M::const_iterator2 const_subiterator2_type;
	typedef typename boost::mpl::if_<boost::is_const<M>,
	        typename M::const_iterator2,
	        typename M::iterator2>::type subiterator2_type;

public:
#ifdef BOOST_UBLAS_USE_INDEXED_ITERATOR
	typedef indexed_iterator1<self_type, packed_random_access_iterator_tag> iterator1;
	typedef indexed_iterator2<self_type, packed_random_access_iterator_tag> iterator2;
	typedef indexed_const_iterator1<self_type, packed_random_access_iterator_tag> const_iterator1;
	typedef indexed_const_iterator2<self_type, packed_random_access_iterator_tag> const_iterator2;
#else
	class const_iterator1;
	class iterator1;
	class const_iterator2;
	class iterator2;
#endif
	typedef reverse_iterator_base1<const_iterator1> const_reverse_iterator1;
	typedef reverse_iterator_base1<iterator1> reverse_iterator1;
	typedef reverse_iterator_base2<const_iterator2> const_reverse_iterator2;
	typedef reverse_iterator_base2<iterator2> reverse_iterator2;

	// Element lookup
	
	const_iterator1 find1(int rank, size_type i, size_type j) const {
		if (rank == 1)
			i = triangular_type::restrict1(i, j, size1(), size2());
		if (rank == 0)
			i = triangular_type::global_restrict1(i, size1(), j, size2());
		return const_iterator1(*this, data().find1(rank, i, j));
	}
	
	iterator1 find1(int rank, size_type i, size_type j) {
		if (rank == 1)
			i = triangular_type::mutable_restrict1(i, j, size1(), size2());
		if (rank == 0)
			i = triangular_type::global_mutable_restrict1(i, size1(), j, size2());
		return iterator1(*this, data().find1(rank, i, j));
	}
	
	const_iterator2 find2(int rank, size_type i, size_type j) const {
		if (rank == 1)
			j = triangular_type::restrict2(i, j, size1(), size2());
		if (rank == 0)
			j = triangular_type::global_restrict2(i, size1(), j, size2());
		return const_iterator2(*this, data().find2(rank, i, j));
	}
	
	iterator2 find2(int rank, size_type i, size_type j) {
		if (rank == 1)
			j = triangular_type::mutable_restrict2(i, j, size1(), size2());
		if (rank == 0)
			j = triangular_type::global_mutable_restrict2(i, size1(), j, size2());
		return iterator2(*this, data().find2(rank, i, j));
	}

	// Iterators siboost::mply are indices.

#ifndef BOOST_UBLAS_USE_INDEXED_ITERATOR
	class const_iterator1:
		public container_const_reference<triangular_adaptor>,
		public random_access_iterator_base<typename iterator_restrict_traits<
		typename const_subiterator1_type::iterator_category, packed_random_access_iterator_tag>::iterator_category,
			const_iterator1, value_type> {
	public:
		typedef typename const_subiterator1_type::value_type value_type;
		typedef typename const_subiterator1_type::difference_type difference_type;
		typedef typename const_subiterator1_type::reference reference;
		typedef typename const_subiterator1_type::pointer pointer;

		typedef const_iterator2 dual_iterator_type;
		typedef const_reverse_iterator2 dual_reverse_iterator_type;

		// Construction and destruction
		
		const_iterator1():
			container_const_reference<self_type> (), it1_() {}
		
		const_iterator1(const self_type &m, const const_subiterator1_type &it1):
			container_const_reference<self_type> (m), it1_(it1) {}
		
		const_iterator1(const iterator1 &it):
			container_const_reference<self_type> (it()), it1_(it.it1_) {}

		// Arithmetic
		
		const_iterator1 &operator ++ () {
			++ it1_;
			return *this;
		}
		
		const_iterator1 &operator -- () {
			-- it1_;
			return *this;
		}
		
		const_iterator1 &operator += (difference_type n) {
			it1_ += n;
			return *this;
		}
		
		const_iterator1 &operator -= (difference_type n) {
			it1_ -= n;
			return *this;
		}
		
		difference_type operator - (const const_iterator1 &it) const {
			BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
			return it1_ - it.it1_;
		}

		// Dereference
		
		const_reference operator * () const {
			size_type i = index1();
			size_type j = index2();
			BOOST_UBLAS_CHECK(i < (*this)().size1(), bad_index());
			BOOST_UBLAS_CHECK(j < (*this)().size2(), bad_index());
			if (triangular_type::other(i, j))
				return *it1_;
			else
				return (*this)()(i, j);
		}
		
		const_reference operator [](difference_type n) const {
			return *(*this + n);
		}

#ifndef BOOST_UBLAS_NO_NESTED_CLASS_RELATION
		
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		const_iterator2 begin() const {
			return (*this)().find2(1, index1(), 0);
		}
		
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		const_iterator2 end() const {
			return (*this)().find2(1, index1(), (*this)().size2());
		}
		
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		const_reverse_iterator2 rbegin() const {
			return const_reverse_iterator2(end());
		}
		
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		const_reverse_iterator2 rend() const {
			return const_reverse_iterator2(begin());
		}
#endif

		// Indices
		
		size_type index1() const {
			return it1_.index1();
		}
		
		size_type index2() const {
			return it1_.index2();
		}

		// Assignment
		
		const_iterator1 &operator = (const const_iterator1 &it) {
			container_const_reference<self_type>::assign(&it());
			it1_ = it.it1_;
			return *this;
		}

		// Comparison
		
		bool operator == (const const_iterator1 &it) const {
			BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
			return it1_ == it.it1_;
		}
		
		bool operator < (const const_iterator1 &it) const {
			BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
			return it1_ < it.it1_;
		}

	private:
		const_subiterator1_type it1_;
	};
#endif

	
	const_iterator1 begin1() const {
		return find1(0, 0, 0);
	}
	
	const_iterator1 end1() const {
		return find1(0, size1(), 0);
	}

#ifndef BOOST_UBLAS_USE_INDEXED_ITERATOR
	class iterator1:
		public container_reference<triangular_adaptor>,
		public random_access_iterator_base<typename iterator_restrict_traits<
		typename subiterator1_type::iterator_category, packed_random_access_iterator_tag>::iterator_category,
			iterator1, value_type> {
	public:
		typedef typename subiterator1_type::value_type value_type;
		typedef typename subiterator1_type::difference_type difference_type;
		typedef typename subiterator1_type::reference reference;
		typedef typename subiterator1_type::pointer pointer;

		typedef iterator2 dual_iterator_type;
		typedef reverse_iterator2 dual_reverse_iterator_type;

		// Construction and destruction
		
		iterator1():
			container_reference<self_type> (), it1_() {}
		
		iterator1(self_type &m, const subiterator1_type &it1):
			container_reference<self_type> (m), it1_(it1) {}

		// Arithmetic
		
		iterator1 &operator ++ () {
			++ it1_;
			return *this;
		}
		
		iterator1 &operator -- () {
			-- it1_;
			return *this;
		}
		
		iterator1 &operator += (difference_type n) {
			it1_ += n;
			return *this;
		}
		
		iterator1 &operator -= (difference_type n) {
			it1_ -= n;
			return *this;
		}
		
		difference_type operator - (const iterator1 &it) const {
			BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
			return it1_ - it.it1_;
		}

		// Dereference
		
		reference operator * () const {
			size_type i = index1();
			size_type j = index2();
			BOOST_UBLAS_CHECK(i < (*this)().size1(), bad_index());
			BOOST_UBLAS_CHECK(j < (*this)().size2(), bad_index());
			if (triangular_type::other(i, j))
				return *it1_;
			else
				return (*this)()(i, j);
		}
		
		reference operator [](difference_type n) const {
			return *(*this + n);
		}

#ifndef BOOST_UBLAS_NO_NESTED_CLASS_RELATION
		
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		iterator2 begin() const {
			return (*this)().find2(1, index1(), 0);
		}
		
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		iterator2 end() const {
			return (*this)().find2(1, index1(), (*this)().size2());
		}
		
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		reverse_iterator2 rbegin() const {
			return reverse_iterator2(end());
		}
		
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		reverse_iterator2 rend() const {
			return reverse_iterator2(begin());
		}
#endif

		// Indices
		
		size_type index1() const {
			return it1_.index1();
		}
		
		size_type index2() const {
			return it1_.index2();
		}

		// Assignment
		
		iterator1 &operator = (const iterator1 &it) {
			container_reference<self_type>::assign(&it());
			it1_ = it.it1_;
			return *this;
		}

		// Comparison
		
		bool operator == (const iterator1 &it) const {
			BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
			return it1_ == it.it1_;
		}
		
		bool operator < (const iterator1 &it) const {
			BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
			return it1_ < it.it1_;
		}

	private:
		subiterator1_type it1_;

		friend class const_iterator1;
	};
#endif

	
	iterator1 begin1() {
		return find1(0, 0, 0);
	}
	
	iterator1 end1() {
		return find1(0, size1(), 0);
	}

#ifndef BOOST_UBLAS_USE_INDEXED_ITERATOR
	class const_iterator2:
		public container_const_reference<triangular_adaptor>,
		public random_access_iterator_base<typename iterator_restrict_traits<
		typename const_subiterator1_type::iterator_category, packed_random_access_iterator_tag>::iterator_category,
			const_iterator2, value_type> {
	public:
		typedef typename const_subiterator2_type::value_type value_type;
		typedef typename const_subiterator2_type::difference_type difference_type;
		typedef typename const_subiterator2_type::reference reference;
		typedef typename const_subiterator2_type::pointer pointer;

		typedef const_iterator1 dual_iterator_type;
		typedef const_reverse_iterator1 dual_reverse_iterator_type;

		// Construction and destruction
		
		const_iterator2():
			container_const_reference<self_type> (), it2_() {}
		
		const_iterator2(const self_type &m, const const_subiterator2_type &it2):
			container_const_reference<self_type> (m), it2_(it2) {}
		
		const_iterator2(const iterator2 &it):
			container_const_reference<self_type> (it()), it2_(it.it2_) {}

		// Arithmetic
		
		const_iterator2 &operator ++ () {
			++ it2_;
			return *this;
		}
		
		const_iterator2 &operator -- () {
			-- it2_;
			return *this;
		}
		
		const_iterator2 &operator += (difference_type n) {
			it2_ += n;
			return *this;
		}
		
		const_iterator2 &operator -= (difference_type n) {
			it2_ -= n;
			return *this;
		}
		
		difference_type operator - (const const_iterator2 &it) const {
			BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
			return it2_ - it.it2_;
		}

		// Dereference
		
		const_reference operator * () const {
			size_type i = index1();
			size_type j = index2();
			BOOST_UBLAS_CHECK(i < (*this)().size1(), bad_index());
			BOOST_UBLAS_CHECK(j < (*this)().size2(), bad_index());
			if (triangular_type::other(i, j))
				return *it2_;
			else
				return (*this)()(i, j);
		}
		
		const_reference operator [](difference_type n) const {
			return *(*this + n);
		}

#ifndef BOOST_UBLAS_NO_NESTED_CLASS_RELATION
		
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		const_iterator1 begin() const {
			return (*this)().find1(1, 0, index2());
		}
		
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		const_iterator1 end() const {
			return (*this)().find1(1, (*this)().size1(), index2());
		}
		
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		const_reverse_iterator1 rbegin() const {
			return const_reverse_iterator1(end());
		}
		
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		const_reverse_iterator1 rend() const {
			return const_reverse_iterator1(begin());
		}
#endif

		// Indices
		
		size_type index1() const {
			return it2_.index1();
		}
		
		size_type index2() const {
			return it2_.index2();
		}

		// Assignment
		
		const_iterator2 &operator = (const const_iterator2 &it) {
			container_const_reference<self_type>::assign(&it());
			it2_ = it.it2_;
			return *this;
		}

		// Comparison
		
		bool operator == (const const_iterator2 &it) const {
			BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
			return it2_ == it.it2_;
		}
		
		bool operator < (const const_iterator2 &it) const {
			BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
			return it2_ < it.it2_;
		}

	private:
		const_subiterator2_type it2_;
	};
#endif

	
	const_iterator2 begin2() const {
		return find2(0, 0, 0);
	}
	
	const_iterator2 end2() const {
		return find2(0, 0, size2());
	}

#ifndef BOOST_UBLAS_USE_INDEXED_ITERATOR
	class iterator2:
		public container_reference<triangular_adaptor>,
		public random_access_iterator_base<typename iterator_restrict_traits<
		typename subiterator1_type::iterator_category, packed_random_access_iterator_tag>::iterator_category,
			iterator2, value_type> {
	public:
		typedef typename subiterator2_type::value_type value_type;
		typedef typename subiterator2_type::difference_type difference_type;
		typedef typename subiterator2_type::reference reference;
		typedef typename subiterator2_type::pointer pointer;

		typedef iterator1 dual_iterator_type;
		typedef reverse_iterator1 dual_reverse_iterator_type;

		// Construction and destruction
		
		iterator2():
			container_reference<self_type> (), it2_() {}
		
		iterator2(self_type &m, const subiterator2_type &it2):
			container_reference<self_type> (m), it2_(it2) {}

		// Arithmetic
		
		iterator2 &operator ++ () {
			++ it2_;
			return *this;
		}
		
		iterator2 &operator -- () {
			-- it2_;
			return *this;
		}
		
		iterator2 &operator += (difference_type n) {
			it2_ += n;
			return *this;
		}
		
		iterator2 &operator -= (difference_type n) {
			it2_ -= n;
			return *this;
		}
		
		difference_type operator - (const iterator2 &it) const {
			BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
			return it2_ - it.it2_;
		}

		// Dereference
		
		reference operator * () const {
			size_type i = index1();
			size_type j = index2();
			BOOST_UBLAS_CHECK(i < (*this)().size1(), bad_index());
			BOOST_UBLAS_CHECK(j < (*this)().size2(), bad_index());
			if (triangular_type::other(i, j))
				return *it2_;
			else
				return (*this)()(i, j);
		}
		
		reference operator [](difference_type n) const {
			return *(*this + n);
		}

#ifndef BOOST_UBLAS_NO_NESTED_CLASS_RELATION
		
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		iterator1 begin() const {
			return (*this)().find1(1, 0, index2());
		}
		
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		iterator1 end() const {
			return (*this)().find1(1, (*this)().size1(), index2());
		}
		
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		reverse_iterator1 rbegin() const {
			return reverse_iterator1(end());
		}
		
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		reverse_iterator1 rend() const {
			return reverse_iterator1(begin());
		}
#endif

		// Indices
		
		size_type index1() const {
			return it2_.index1();
		}
		
		size_type index2() const {
			return it2_.index2();
		}

		// Assignment
		
		iterator2 &operator = (const iterator2 &it) {
			container_reference<self_type>::assign(&it());
			it2_ = it.it2_;
			return *this;
		}

		// Comparison
		
		bool operator == (const iterator2 &it) const {
			BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
			return it2_ == it.it2_;
		}
		
		bool operator < (const iterator2 &it) const {
			BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
			return it2_ < it.it2_;
		}

	private:
		subiterator2_type it2_;

		friend class const_iterator2;
	};
#endif

	
	iterator2 begin2() {
		return find2(0, 0, 0);
	}
	
	iterator2 end2() {
		return find2(0, 0, size2());
	}

	// Reverse iterators

	
	const_reverse_iterator1 rbegin1() const {
		return const_reverse_iterator1(end1());
	}
	
	const_reverse_iterator1 rend1() const {
		return const_reverse_iterator1(begin1());
	}

	
	reverse_iterator1 rbegin1() {
		return reverse_iterator1(end1());
	}
	
	reverse_iterator1 rend1() {
		return reverse_iterator1(begin1());
	}

	
	const_reverse_iterator2 rbegin2() const {
		return const_reverse_iterator2(end2());
	}
	
	const_reverse_iterator2 rend2() const {
		return const_reverse_iterator2(begin2());
	}

	
	reverse_iterator2 rbegin2() {
		return reverse_iterator2(end2());
	}
	
	reverse_iterator2 rend2() {
		return reverse_iterator2(begin2());
	}

private:
	matrix_closure_type data_;
	static const value_type zero_;
	static const value_type one_;
};

template<class M, class TRI>
const typename triangular_adaptor<M, TRI>::value_type triangular_adaptor<M, TRI>::zero_ = value_type/*zero*/();
template<class M, class TRI>
const typename triangular_adaptor<M, TRI>::value_type triangular_adaptor<M, TRI>::one_(1);

template <class M, class TRI>
struct vector_temporary_traits< triangular_adaptor<M, TRI> >
		: vector_temporary_traits< typename boost::remove_const<M>::type > {} ;
template <class M, class TRI>
struct vector_temporary_traits< const triangular_adaptor<M, TRI> >
		: vector_temporary_traits< typename boost::remove_const<M>::type > {} ;

template <class M, class TRI>
struct matrix_temporary_traits< triangular_adaptor<M, TRI> >
		: matrix_temporary_traits< typename boost::remove_const<M>::type > {};
template <class M, class TRI>
struct matrix_temporary_traits< const triangular_adaptor<M, TRI> >
		: matrix_temporary_traits< typename boost::remove_const<M>::type > {};




// Operations:
//  n * (n - 1) / 2 + n = n * (n + 1) / 2 multiplications,
//  n * (n - 1) / 2 additions

// Dense (proxy) case
template<class E1, class E2>

void inplace_solve(const matrix_expression<E1> &e1, vector_expression<E2> &e2,
        lower_tag, column_major_tag, dense_proxy_tag) {
	typedef typename E2::size_type size_type;
	typedef typename E2::difference_type difference_type;
	typedef typename E2::value_type value_type;

	BOOST_UBLAS_CHECK(e1().size1() == e1().size2(), bad_size());
	BOOST_UBLAS_CHECK(e1().size2() == e2().size(), bad_size());
	size_type size = e2().size();
	for (size_type n = 0; n < size; ++ n) {
#ifndef BOOST_UBLAS_SINGULAR_CHECK
		BOOST_UBLAS_CHECK(e1()(n, n) != value_type/*zero*/(), singular());
#else
		if (e1()(n, n) == value_type/*zero*/())
			singular().raise();
#endif
		value_type t = e2()(n) /= e1()(n, n);
		if (t != value_type/*zero*/()) {
			for (size_type m = n + 1; m < size; ++ m)
				e2()(m) -= e1()(m, n) * t;
		}
	}
}
// Packed (proxy) case
template<class E1, class E2>

void inplace_solve(const matrix_expression<E1> &e1, vector_expression<E2> &e2,
        lower_tag, column_major_tag, packed_proxy_tag) {
	typedef typename E2::size_type size_type;
	typedef typename E2::difference_type difference_type;
	typedef typename E2::value_type value_type;

	BOOST_UBLAS_CHECK(e1().size1() == e1().size2(), bad_size());
	BOOST_UBLAS_CHECK(e1().size2() == e2().size(), bad_size());
	size_type size = e2().size();
	for (size_type n = 0; n < size; ++ n) {
#ifndef BOOST_UBLAS_SINGULAR_CHECK
		BOOST_UBLAS_CHECK(e1()(n, n) != value_type/*zero*/(), singular());
#else
		if (e1()(n, n) == value_type/*zero*/())
			singular().raise();
#endif
		value_type t = e2()(n) /= e1()(n, n);
		if (t != value_type/*zero*/()) {
			typename E1::const_iterator1 it1e1(e1().find1(1, n + 1, n));
			typename E1::const_iterator1 it1e1_end(e1().find1(1, e1().size1(), n));
			difference_type m(it1e1_end - it1e1);
			while (-- m >= 0)
				e2()(it1e1.index1()) -= *it1e1 * t, ++ it1e1;
		}
	}
}
// Sparse (proxy) case
template<class E1, class E2>

void inplace_solve(const matrix_expression<E1> &e1, vector_expression<E2> &e2,
        lower_tag, column_major_tag, unknown_storage_tag) {
	typedef typename E2::size_type size_type;
	typedef typename E2::difference_type difference_type;
	typedef typename E2::value_type value_type;

	BOOST_UBLAS_CHECK(e1().size1() == e1().size2(), bad_size());
	BOOST_UBLAS_CHECK(e1().size2() == e2().size(), bad_size());
	size_type size = e2().size();
	for (size_type n = 0; n < size; ++ n) {
#ifndef BOOST_UBLAS_SINGULAR_CHECK
		BOOST_UBLAS_CHECK(e1()(n, n) != value_type/*zero*/(), singular());
#else
		if (e1()(n, n) == value_type/*zero*/())
			singular().raise();
#endif
		value_type t = e2()(n) /= e1()(n, n);
		if (t != value_type/*zero*/()) {
			typename E1::const_iterator1 it1e1(e1().find1(1, n + 1, n));
			typename E1::const_iterator1 it1e1_end(e1().find1(1, e1().size1(), n));
			while (it1e1 != it1e1_end)
				e2()(it1e1.index1()) -= *it1e1 * t, ++ it1e1;
		}
	}
}

// Dense (proxy) case
template<class E1, class E2>

void inplace_solve(const matrix_expression<E1> &e1, vector_expression<E2> &e2,
        lower_tag, row_major_tag, dense_proxy_tag) {
	typedef typename E2::size_type size_type;
	typedef typename E2::difference_type difference_type;
	typedef typename E2::value_type value_type;

	BOOST_UBLAS_CHECK(e1().size1() == e1().size2(), bad_size());
	BOOST_UBLAS_CHECK(e1().size2() == e2().size(), bad_size());
	size_type size = e2().size();
	for (size_type n = 0; n < size; ++ n) {
#ifndef BOOST_UBLAS_SINGULAR_CHECK
		BOOST_UBLAS_CHECK(e1()(n, n) != value_type/*zero*/(), singular());
#else
		if (e1()(n, n) == value_type/*zero*/())
			singular().raise();
#endif
		value_type t = e2()(n) /= e1()(n, n);
		if (t != value_type/*zero*/()) {
			for (size_type m = n + 1; m < size; ++ m)
				e2()(m) -= e1()(m, n) * t;
		}
	}
}
// Packed (proxy) case
template<class E1, class E2>

void inplace_solve(const matrix_expression<E1> &e1, vector_expression<E2> &e2,
        lower_tag, row_major_tag, packed_proxy_tag) {
	typedef typename E2::size_type size_type;
	typedef typename E2::difference_type difference_type;
	typedef typename E2::value_type value_type;

	BOOST_UBLAS_CHECK(e1().size1() == e1().size2(), bad_size());
	BOOST_UBLAS_CHECK(e1().size2() == e2().size(), bad_size());
	size_type size = e2().size();
	for (size_type n = 0; n < size; ++ n) {
#ifndef BOOST_UBLAS_SINGULAR_CHECK
		BOOST_UBLAS_CHECK(e1()(n, n) != value_type/*zero*/(), singular());
#else
		if (e1()(n, n) == value_type/*zero*/())
			singular().raise();
#endif
		value_type t = e2()(n);
		typename E1::const_iterator2 it2e1(e1().find2(1, n, 0));
		typename E1::const_iterator2 it2e1_end(e1().find2(1, n, n));
		while (it2e1 != it2e1_end) {
			t -= *it2e1 * e2()(it2e1.index2());
			++ it2e1;
		}
		e2()(n) = t / e1()(n, n);
	}
}
// Sparse (proxy) case
template<class E1, class E2>

void inplace_solve(const matrix_expression<E1> &e1, vector_expression<E2> &e2,
        lower_tag, row_major_tag, unknown_storage_tag) {
	typedef typename E2::size_type size_type;
	typedef typename E2::difference_type difference_type;
	typedef typename E2::value_type value_type;

	BOOST_UBLAS_CHECK(e1().size1() == e1().size2(), bad_size());
	BOOST_UBLAS_CHECK(e1().size2() == e2().size(), bad_size());
	size_type size = e2().size();
	for (size_type n = 0; n < size; ++ n) {
#ifndef BOOST_UBLAS_SINGULAR_CHECK
		BOOST_UBLAS_CHECK(e1()(n, n) != value_type/*zero*/(), singular());
#else
		if (e1()(n, n) == value_type/*zero*/())
			singular().raise();
#endif
		value_type t = e2()(n);
		typename E1::const_iterator2 it2e1(e1().find2(1, n, 0));
		typename E1::const_iterator2 it2e1_end(e1().find2(1, n, n));
		while (it2e1 != it2e1_end) {
			t -= *it2e1 * e2()(it2e1.index2());
			++ it2e1;
		}
		e2()(n) = t / e1()(n, n);
	}
}

// Redirectors :-)
template<class E1, class E2>

void inplace_solve(const matrix_expression<E1> &e1, vector_expression<E2> &e2,
        lower_tag, column_major_tag) {
	typedef typename E1::storage_category storage_category;
	inplace_solve(e1, e2,
	        lower_tag(), column_major_tag(), storage_category());
}
template<class E1, class E2>

void inplace_solve(const matrix_expression<E1> &e1, vector_expression<E2> &e2,
        lower_tag, row_major_tag) {
	typedef typename E1::storage_category storage_category;
	inplace_solve(e1, e2,
	        lower_tag(), row_major_tag(), storage_category());
}
// Dispatcher
template<class E1, class E2>

void inplace_solve(const matrix_expression<E1> &e1, vector_expression<E2> &e2,
        lower_tag) {
	typedef typename E1::orientation_category orientation_category;
	inplace_solve(e1, e2,
	        lower_tag(), orientation_category());
}
template<class E1, class E2>

void inplace_solve(const matrix_expression<E1> &e1, vector_expression<E2> &e2,
        unit_lower_tag) {
	typedef typename E1::orientation_category orientation_category;
	inplace_solve(triangular_adaptor<const E1, unit_lower> (e1()), e2,
	        unit_lower_tag(), orientation_category());
}

// Dense (proxy) case
template<class E1, class E2>

void inplace_solve(const matrix_expression<E1> &e1, vector_expression<E2> &e2,
        upper_tag, column_major_tag, dense_proxy_tag) {
	typedef typename E2::size_type size_type;
	typedef typename E2::difference_type difference_type;
	typedef typename E2::value_type value_type;

	BOOST_UBLAS_CHECK(e1().size1() == e1().size2(), bad_size());
	BOOST_UBLAS_CHECK(e1().size2() == e2().size(), bad_size());
	size_type size = e2().size();
	for (difference_type n = size - 1; n >= 0; -- n) {
#ifndef BOOST_UBLAS_SINGULAR_CHECK
		BOOST_UBLAS_CHECK(e1()(n, n) != value_type/*zero*/(), singular());
#else
		if (e1()(n, n) == value_type/*zero*/())
			singular().raise();
#endif
		value_type t = e2()(n) /= e1()(n, n);
		if (t != value_type/*zero*/()) {
			for (difference_type m = n - 1; m >= 0; -- m)
				e2()(m) -= e1()(m, n) * t;
		}
	}
}
// Packed (proxy) case
template<class E1, class E2>

void inplace_solve(const matrix_expression<E1> &e1, vector_expression<E2> &e2,
        upper_tag, column_major_tag, packed_proxy_tag) {
	typedef typename E2::size_type size_type;
	typedef typename E2::difference_type difference_type;
	typedef typename E2::value_type value_type;

	BOOST_UBLAS_CHECK(e1().size1() == e1().size2(), bad_size());
	BOOST_UBLAS_CHECK(e1().size2() == e2().size(), bad_size());
	size_type size = e2().size();
	for (difference_type n = size - 1; n >= 0; -- n) {
#ifndef BOOST_UBLAS_SINGULAR_CHECK
		BOOST_UBLAS_CHECK(e1()(n, n) != value_type/*zero*/(), singular());
#else
		if (e1()(n, n) == value_type/*zero*/())
			singular().raise();
#endif
		value_type t = e2()(n) /= e1()(n, n);
		if (t != value_type/*zero*/()) {
			typename E1::const_reverse_iterator1 it1e1(e1().find1(1, n, n));
			typename E1::const_reverse_iterator1 it1e1_rend(e1().find1(1, 0, n));
			while (it1e1 != it1e1_rend) {
				e2()(it1e1.index1()) -= *it1e1 * t, ++ it1e1;
			}
		}
	}
}
// Sparse (proxy) case
template<class E1, class E2>

void inplace_solve(const matrix_expression<E1> &e1, vector_expression<E2> &e2,
        upper_tag, column_major_tag, unknown_storage_tag) {
	typedef typename E2::size_type size_type;
	typedef typename E2::difference_type difference_type;
	typedef typename E2::value_type value_type;

	BOOST_UBLAS_CHECK(e1().size1() == e1().size2(), bad_size());
	BOOST_UBLAS_CHECK(e1().size2() == e2().size(), bad_size());
	size_type size = e2().size();
	for (difference_type n = size - 1; n >= 0; -- n) {
#ifndef BOOST_UBLAS_SINGULAR_CHECK
		BOOST_UBLAS_CHECK(e1()(n, n) != value_type/*zero*/(), singular());
#else
		if (e1()(n, n) == value_type/*zero*/())
			singular().raise();
#endif
		value_type t = e2()(n) /= e1()(n, n);
		if (t != value_type/*zero*/()) {
			typename E1::const_reverse_iterator1 it1e1(e1().find1(1, n, n));
			typename E1::const_reverse_iterator1 it1e1_rend(e1().find1(1, 0, n));
			while (it1e1 != it1e1_rend) {
				e2()(it1e1.index1()) -= *it1e1 * t, ++ it1e1;
			}
		}
	}
}

// Dense (proxy) case
template<class E1, class E2>

void inplace_solve(const matrix_expression<E1> &e1, vector_expression<E2> &e2,
        upper_tag, row_major_tag, dense_proxy_tag) {
	typedef typename E2::size_type size_type;
	typedef typename E2::difference_type difference_type;
	typedef typename E2::value_type value_type;

	BOOST_UBLAS_CHECK(e1().size1() == e1().size2(), bad_size());
	BOOST_UBLAS_CHECK(e1().size2() == e2().size(), bad_size());
	size_type size = e1().size1();
	for (difference_type n = size-1; n >=0; -- n) {
#ifndef BOOST_UBLAS_SINGULAR_CHECK
		BOOST_UBLAS_CHECK(e1()(n, n) != value_type/*zero*/(), singular());
#else
		if (e1()(n, n) == value_type/*zero*/())
			singular().raise();
#endif
		value_type t = e2()(n);
		for (std::size_t m = n + 1; m < e1().size2(); ++ m) {
			t -= e1()(n, m)  * e2()(m);
		}
		e2()(n) = t / e1()(n, n);
	}
}
// Packed (proxy) case
template<class E1, class E2>

void inplace_solve(const matrix_expression<E1> &e1, vector_expression<E2> &e2,
        upper_tag, row_major_tag, packed_proxy_tag) {
	typedef typename E2::size_type size_type;
	typedef typename E2::difference_type difference_type;
	typedef typename E2::value_type value_type;

	BOOST_UBLAS_CHECK(e1().size1() == e1().size2(), bad_size());
	BOOST_UBLAS_CHECK(e1().size2() == e2().size(), bad_size());
	size_type size = e1().size1();
	for (difference_type n = size-1; n >=0; -- n) {
#ifndef BOOST_UBLAS_SINGULAR_CHECK
		BOOST_UBLAS_CHECK(e1()(n, n) != value_type/*zero*/(), singular());
#else
		if (e1()(n, n) == value_type/*zero*/())
			singular().raise();
#endif
		value_type t = e2()(n);
		typename E1::const_iterator2 it2e1(e1().find2(1, n, n+1));
		typename E1::const_iterator2 it2e1_end(e1().find2(1, n, e1().size2()));
		while (it2e1 != it2e1_end) {
			t -= *it2e1 * e2()(it2e1.index2());
			++ it2e1;
		}
		e2()(n) = t / e1()(n, n);

	}
}
// Sparse (proxy) case
template<class E1, class E2>

void inplace_solve(const matrix_expression<E1> &e1, vector_expression<E2> &e2,
        upper_tag, row_major_tag, unknown_storage_tag) {
	typedef typename E2::size_type size_type;
	typedef typename E2::difference_type difference_type;
	typedef typename E2::value_type value_type;

	BOOST_UBLAS_CHECK(e1().size1() == e1().size2(), bad_size());
	BOOST_UBLAS_CHECK(e1().size2() == e2().size(), bad_size());
	size_type size = e1().size1();
	for (difference_type n = size-1; n >=0; -- n) {
#ifndef BOOST_UBLAS_SINGULAR_CHECK
		BOOST_UBLAS_CHECK(e1()(n, n) != value_type/*zero*/(), singular());
#else
		if (e1()(n, n) == value_type/*zero*/())
			singular().raise();
#endif
		value_type t = e2()(n);
		typename E1::const_iterator2 it2e1(e1().find2(1, n, n+1));
		typename E1::const_iterator2 it2e1_end(e1().find2(1, n, e1().size2()));
		while (it2e1 != it2e1_end) {
			t -= *it2e1 * e2()(it2e1.index2());
			++ it2e1;
		}
		e2()(n) = t / e1()(n, n);

	}
}

// Redirectors :-)
template<class E1, class E2>

void inplace_solve(const matrix_expression<E1> &e1, vector_expression<E2> &e2,
        upper_tag, column_major_tag) {
	typedef typename E1::storage_category storage_category;
	inplace_solve(e1, e2,
	        upper_tag(), column_major_tag(), storage_category());
}
template<class E1, class E2>

void inplace_solve(const matrix_expression<E1> &e1, vector_expression<E2> &e2,
        upper_tag, row_major_tag) {
	typedef typename E1::storage_category storage_category;
	inplace_solve(e1, e2,
	        upper_tag(), row_major_tag(), storage_category());
}
// Dispatcher
template<class E1, class E2>

void inplace_solve(const matrix_expression<E1> &e1, vector_expression<E2> &e2,
        upper_tag) {
	typedef typename E1::orientation_category orientation_category;
	inplace_solve(e1, e2,
	        upper_tag(), orientation_category());
}
template<class E1, class E2>

void inplace_solve(const matrix_expression<E1> &e1, vector_expression<E2> &e2,
        unit_upper_tag) {
	typedef typename E1::orientation_category orientation_category;
	inplace_solve(triangular_adaptor<const E1, unit_upper> (e1()), e2,
	        unit_upper_tag(), orientation_category());
}



// Redirectors :-)
template<class E1, class E2>

void inplace_solve(vector_expression<E1> &e1, const matrix_expression<E2> &e2,
        lower_tag, row_major_tag) {
	typedef typename E2::storage_category storage_category;
	inplace_solve(trans(e2), e1,
	        upper_tag(), column_major_tag(), storage_category());
}
template<class E1, class E2>

void inplace_solve(vector_expression<E1> &e1, const matrix_expression<E2> &e2,
        lower_tag, column_major_tag) {
	typedef typename E2::storage_category storage_category;
	inplace_solve(trans(e2), e1,
	        upper_tag(), row_major_tag(), storage_category());
}
// Dispatcher
template<class E1, class E2>

void inplace_solve(vector_expression<E1> &e1, const matrix_expression<E2> &e2,
        lower_tag) {
	typedef typename E2::orientation_category orientation_category;
	inplace_solve(e1, e2,
	        lower_tag(), orientation_category());
}
template<class E1, class E2>

void inplace_solve(vector_expression<E1> &e1, const matrix_expression<E2> &e2,
        unit_lower_tag) {
	typedef typename E2::orientation_category orientation_category;
	inplace_solve(e1, triangular_adaptor<const E2, unit_lower> (e2()),
	        unit_lower_tag(), orientation_category());
}


// Redirectors :-)
template<class E1, class E2>

void inplace_solve(vector_expression<E1> &e1, const matrix_expression<E2> &e2,
        upper_tag, row_major_tag) {
	typedef typename E2::storage_category storage_category;
	inplace_solve(trans(e2), e1,
	        lower_tag(), column_major_tag(), storage_category());
}
template<class E1, class E2>

void inplace_solve(vector_expression<E1> &e1, const matrix_expression<E2> &e2,
        upper_tag, column_major_tag) {
	typedef typename E2::storage_category storage_category;
	inplace_solve(trans(e2), e1,
	        lower_tag(), row_major_tag(), storage_category());
}
// Dispatcher
template<class E1, class E2>

void inplace_solve(vector_expression<E1> &e1, const matrix_expression<E2> &e2,
        upper_tag) {
	typedef typename E2::orientation_category orientation_category;
	inplace_solve(e1, e2,
	        upper_tag(), orientation_category());
}
template<class E1, class E2>

void inplace_solve(vector_expression<E1> &e1, const matrix_expression<E2> &e2,
        unit_upper_tag) {
	typedef typename E2::orientation_category orientation_category;
	inplace_solve(e1, triangular_adaptor<const E2, unit_upper> (e2()),
	        unit_upper_tag(), orientation_category());
}

// Operations:
//  k * n * (n - 1) / 2 + k * n = k * n * (n + 1) / 2 multiplications,
//  k * n * (n - 1) / 2 additions

// Dense (proxy) case
template<class E1, class E2>

void inplace_solve(const matrix_expression<E1> &e1, matrix_expression<E2> &e2,
        lower_tag, dense_proxy_tag) {
	typedef typename E2::size_type size_type;
	typedef typename E2::difference_type difference_type;
	typedef typename E2::value_type value_type;

	BOOST_UBLAS_CHECK(e1().size1() == e1().size2(), bad_size());
	BOOST_UBLAS_CHECK(e1().size2() == e2().size1(), bad_size());
	size_type size1 = e2().size1();
	size_type size2 = e2().size2();
	for (size_type n = 0; n < size1; ++ n) {
#ifndef BOOST_UBLAS_SINGULAR_CHECK
		BOOST_UBLAS_CHECK(e1()(n, n) != value_type/*zero*/(), singular());
#else
		if (e1()(n, n) == value_type/*zero*/())
			singular().raise();
#endif
		for (size_type l = 0; l < size2; ++ l) {
			value_type t = e2()(n, l) /= e1()(n, n);
			if (t != value_type/*zero*/()) {
				for (size_type m = n + 1; m < size1; ++ m)
					e2()(m, l) -= e1()(m, n) * t;
			}
		}
	}
}
// Packed (proxy) case
template<class E1, class E2>

void inplace_solve(const matrix_expression<E1> &e1, matrix_expression<E2> &e2,
        lower_tag, packed_proxy_tag) {
	typedef typename E2::size_type size_type;
	typedef typename E2::difference_type difference_type;
	typedef typename E2::value_type value_type;

	BOOST_UBLAS_CHECK(e1().size1() == e1().size2(), bad_size());
	BOOST_UBLAS_CHECK(e1().size2() == e2().size1(), bad_size());
	size_type size1 = e2().size1();
	size_type size2 = e2().size2();
	for (size_type n = 0; n < size1; ++ n) {
#ifndef BOOST_UBLAS_SINGULAR_CHECK
		BOOST_UBLAS_CHECK(e1()(n, n) != value_type/*zero*/(), singular());
#else
		if (e1()(n, n) == value_type/*zero*/())
			singular().raise();
#endif
		for (size_type l = 0; l < size2; ++ l) {
			value_type t = e2()(n, l) /= e1()(n, n);
			if (t != value_type/*zero*/()) {
				typename E1::const_iterator1 it1e1(e1().find1(1, n + 1, n));
				typename E1::const_iterator1 it1e1_end(e1().find1(1, e1().size1(), n));
				difference_type m(it1e1_end - it1e1);
				while (-- m >= 0)
					e2()(it1e1.index1(), l) -= *it1e1 * t, ++ it1e1;
			}
		}
	}
}
// Sparse (proxy) case
template<class E1, class E2>

void inplace_solve(const matrix_expression<E1> &e1, matrix_expression<E2> &e2,
        lower_tag, unknown_storage_tag) {
	typedef typename E2::size_type size_type;
	typedef typename E2::difference_type difference_type;
	typedef typename E2::value_type value_type;

	BOOST_UBLAS_CHECK(e1().size1() == e1().size2(), bad_size());
	BOOST_UBLAS_CHECK(e1().size2() == e2().size1(), bad_size());
	size_type size1 = e2().size1();
	size_type size2 = e2().size2();
	for (size_type n = 0; n < size1; ++ n) {
#ifndef BOOST_UBLAS_SINGULAR_CHECK
		BOOST_UBLAS_CHECK(e1()(n, n) != value_type/*zero*/(), singular());
#else
		if (e1()(n, n) == value_type/*zero*/())
			singular().raise();
#endif
		for (size_type l = 0; l < size2; ++ l) {
			value_type t = e2()(n, l) /= e1()(n, n);
			if (t != value_type/*zero*/()) {
				typename E1::const_iterator1 it1e1(e1().find1(1, n + 1, n));
				typename E1::const_iterator1 it1e1_end(e1().find1(1, e1().size1(), n));
				while (it1e1 != it1e1_end)
					e2()(it1e1.index1(), l) -= *it1e1 * t, ++ it1e1;
			}
		}
	}
}
// Dispatcher
template<class E1, class E2>

void inplace_solve(const matrix_expression<E1> &e1, matrix_expression<E2> &e2,
        lower_tag) {
	typedef typename E1::storage_category dispatch_category;
	inplace_solve(e1, e2,
	        lower_tag(), dispatch_category());
}
template<class E1, class E2>

void inplace_solve(const matrix_expression<E1> &e1, matrix_expression<E2> &e2,
        unit_lower_tag) {
	typedef typename E1::storage_category dispatch_category;
	inplace_solve(triangular_adaptor<const E1, unit_lower> (e1()), e2,
	        unit_lower_tag(), dispatch_category());
}

// Dense (proxy) case
template<class E1, class E2>

void inplace_solve(const matrix_expression<E1> &e1, matrix_expression<E2> &e2,
        upper_tag, dense_proxy_tag) {
	typedef typename E2::size_type size_type;
	typedef typename E2::difference_type difference_type;
	typedef typename E2::value_type value_type;

	BOOST_UBLAS_CHECK(e1().size1() == e1().size2(), bad_size());
	BOOST_UBLAS_CHECK(e1().size2() == e2().size1(), bad_size());
	size_type size1 = e2().size1();
	size_type size2 = e2().size2();
	for (difference_type n = size1 - 1; n >= 0; -- n) {
#ifndef BOOST_UBLAS_SINGULAR_CHECK
		BOOST_UBLAS_CHECK(e1()(n, n) != value_type/*zero*/(), singular());
#else
		if (e1()(n, n) == value_type/*zero*/())
			singular().raise();
#endif
		for (difference_type l = size2 - 1; l >= 0; -- l) {
			value_type t = e2()(n, l) /= e1()(n, n);
			if (t != value_type/*zero*/()) {
				for (difference_type m = n - 1; m >= 0; -- m)
					e2()(m, l) -= e1()(m, n) * t;
			}
		}
	}
}
// Packed (proxy) case
template<class E1, class E2>

void inplace_solve(const matrix_expression<E1> &e1, matrix_expression<E2> &e2,
        upper_tag, packed_proxy_tag) {
	typedef typename E2::size_type size_type;
	typedef typename E2::difference_type difference_type;
	typedef typename E2::value_type value_type;

	BOOST_UBLAS_CHECK(e1().size1() == e1().size2(), bad_size());
	BOOST_UBLAS_CHECK(e1().size2() == e2().size1(), bad_size());
	size_type size1 = e2().size1();
	size_type size2 = e2().size2();
	for (difference_type n = size1 - 1; n >= 0; -- n) {
#ifndef BOOST_UBLAS_SINGULAR_CHECK
		BOOST_UBLAS_CHECK(e1()(n, n) != value_type/*zero*/(), singular());
#else
		if (e1()(n, n) == value_type/*zero*/())
			singular().raise();
#endif
		for (difference_type l = size2 - 1; l >= 0; -- l) {
			value_type t = e2()(n, l) /= e1()(n, n);
			if (t != value_type/*zero*/()) {
				typename E1::const_reverse_iterator1 it1e1(e1().find1(1, n, n));
				typename E1::const_reverse_iterator1 it1e1_rend(e1().find1(1, 0, n));
				difference_type m(it1e1_rend - it1e1);
				while (-- m >= 0)
					e2()(it1e1.index1(), l) -= *it1e1 * t, ++ it1e1;
			}
		}
	}
}
// Sparse (proxy) case
template<class E1, class E2>

void inplace_solve(const matrix_expression<E1> &e1, matrix_expression<E2> &e2,
        upper_tag, unknown_storage_tag) {
	typedef typename E2::size_type size_type;
	typedef typename E2::difference_type difference_type;
	typedef typename E2::value_type value_type;

	BOOST_UBLAS_CHECK(e1().size1() == e1().size2(), bad_size());
	BOOST_UBLAS_CHECK(e1().size2() == e2().size1(), bad_size());
	size_type size1 = e2().size1();
	size_type size2 = e2().size2();
	for (difference_type n = size1 - 1; n >= 0; -- n) {
#ifndef BOOST_UBLAS_SINGULAR_CHECK
		BOOST_UBLAS_CHECK(e1()(n, n) != value_type/*zero*/(), singular());
#else
		if (e1()(n, n) == value_type/*zero*/())
			singular().raise();
#endif
		for (difference_type l = size2 - 1; l >= 0; -- l) {
			value_type t = e2()(n, l) /= e1()(n, n);
			if (t != value_type/*zero*/()) {
				typename E1::const_reverse_iterator1 it1e1(e1().find1(1, n, n));
				typename E1::const_reverse_iterator1 it1e1_rend(e1().find1(1, 0, n));
				while (it1e1 != it1e1_rend)
					e2()(it1e1.index1(), l) -= *it1e1 * t, ++ it1e1;
			}
		}
	}
}
// Dispatcher
template<class E1, class E2>

void inplace_solve(const matrix_expression<E1> &e1, matrix_expression<E2> &e2,
        upper_tag) {
	typedef typename E1::storage_category dispatch_category;
	inplace_solve(e1, e2,
	        upper_tag(), dispatch_category());
}
template<class E1, class E2>

void inplace_solve(const matrix_expression<E1> &e1, matrix_expression<E2> &e2,
        unit_upper_tag) {
	typedef typename E1::storage_category dispatch_category;
	inplace_solve(triangular_adaptor<const E1, unit_upper> (e1()), e2,
	        unit_upper_tag(), dispatch_category());
}

}
}

#endif
