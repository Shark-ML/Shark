//
//  Copyright (c) 2000-2010
//  Joerg Walter, Mathias Koch, Gunter Winkler, David Bellot
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  GeNeSys mbH & Co. KG in producing this work.
//

#ifndef _BOOST_UBLAS_MATRIX_
#define _BOOST_UBLAS_MATRIX_

#include <shark/LinAlg/BLAS/ublas/vector.hpp>
#include <shark/LinAlg/BLAS/ublas/matrix_expression.hpp>
#include <shark/LinAlg/BLAS/ublas/detail/matrix_assign.hpp>
#include <boost/serialization/collection_size_type.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/nvp.hpp>

// Iterators based on ideas of Jeremy Siek

namespace shark {
namespace blas {


namespace detail {
using namespace shark::blas;

// Matrix resizing algorithm
template <class L, class M>
BOOST_UBLAS_INLINE
void matrix_resize_preserve(M &m, M &temporary) {
	typedef L layout_type;
	typedef typename M::size_type size_type;
	const size_type msize1(m.size1());          // original size
	const size_type msize2(m.size2());
	const size_type size1(temporary.size1());      // new size is specified by temporary
	const size_type size2(temporary.size2());
	// Common elements to preserve
	const size_type size1_min = (std::min)(size1, msize1);
	const size_type size2_min = (std::min)(size2, msize2);
	// Order for major and minor sizes
	const size_type major_size = layout_type::size_M(size1_min, size2_min);
	const size_type minor_size = layout_type::size_m(size1_min, size2_min);
	// Indexing copy over major
	for (size_type major = 0; major != major_size; ++major) {
		for (size_type minor = 0; minor != minor_size; ++minor) {
			// find indexes - use invertability of element_ functions
			const size_type i1 = layout_type::index_M(major, minor);
			const size_type i2 = layout_type::index_m(major, minor);
			temporary.data() [layout_type::element(i1, size1, i2, size2)] =
			    m.data() [layout_type::element(i1, msize1, i2, msize2)];
		}
	}
	m.assign_temporary(temporary);
}
}

/** \brief A dense matrix of values of type \c T.
 *
 * For a \f$(m \times n)\f$-dimensional matrix and \f$ 0 \leq i < m, 0 \leq j < n\f$, every element \f$ m_{i,j} \f$ is mapped to
 * the \f$(i.n + j)\f$-th element of the container for row major orientation or the \f$ (i + j.m) \f$-th element of
 * the container for column major orientation. In a dense matrix all elements are represented in memory in a
 * contiguous chunk of memory by definition.
 *
 * Orientation and storage can also be specified, otherwise a \c row_major and \c unbounded_array are used. It is \b not
 * required by the storage to initialize elements of the matrix.
 *
 * \tparam T the type of object stored in the matrix (like double, float, complex, etc...)
 * \tparam L the storage organization. It can be either \c row_major or \c column_major. Default is \c row_major
 * \tparam A the type of Storage array. Default is \c unbounded_array
 */
template<class T, class L, class A>
class matrix:
	public matrix_container<matrix<T, L, A> > {

	typedef T *pointer;
	typedef L layout_type;
	typedef matrix<T, L, A> self_type;
public:
#ifdef BOOST_UBLAS_ENABLE_PROXY_SHORTCUTS
	using matrix_container<self_type>::operator();
#endif
	typedef typename A::size_type size_type;
	typedef typename A::difference_type difference_type;
	typedef T value_type;
	typedef const T &const_reference;
	typedef T &reference;
	typedef A array_type;
	typedef const matrix_reference<const self_type> const_closure_type;
	typedef matrix_reference<self_type> closure_type;
	typedef vector<T, A> vector_temporary_type;
	typedef self_type matrix_temporary_type;
	typedef dense_tag storage_category;
	// This could be better for performance,
	// typedef typename unknown_orientation_tag orientation_category;
	// but others depend on the orientation information...
	typedef typename L::orientation_category orientation_category;

	// Construction and destruction

	/// Default dense matrix constructor. Make a dense matrix of size (0,0)
	BOOST_UBLAS_INLINE
	matrix():
		matrix_container<self_type> (),
		size1_(0), size2_(0), data_() {}

	/** Dense matrix constructor with defined size
	 * \param size1 number of rows
	 * \param size2 number of columns
	 */
	BOOST_UBLAS_INLINE
	matrix(size_type size1, size_type size2):
		matrix_container<self_type> (),
		size1_(size1), size2_(size2), data_(layout_type::storage_size(size1, size2)) {
	}

	/** Dense matrix constructor with defined size a initial value for all the matrix elements
	 * \param size1 number of rows
	 * \param size2 number of columns
	 * \param init initial value assigned to all elements
	 */
	matrix(size_type size1, size_type size2, const value_type &init):
		matrix_container<self_type> (),
		size1_(size1), size2_(size2), data_(layout_type::storage_size(size1, size2), init) {
	}

	/** Dense matrix constructor with defined size and an initial data array
	 * \param size1 number of rows
	 * \param size2 number of columns
	 * \param data array to copy into the matrix. Must have the same dimension as the matrix
	 */
	BOOST_UBLAS_INLINE
	matrix(size_type size1, size_type size2, const array_type &data):
		matrix_container<self_type> (),
		size1_(size1), size2_(size2), data_(data) {}

	/** Copy-constructor of a dense matrix
	 * \param m is a dense matrix
	 */
	BOOST_UBLAS_INLINE
	matrix(const matrix &m):
		matrix_container<self_type> (),
		size1_(m.size1_), size2_(m.size2_), data_(m.data_) {}

	/** Copy-constructor of a dense matrix from a matrix expression
	 * \param ae is a matrix expression
	 */
	template<class AE>
	BOOST_UBLAS_INLINE
	matrix(const matrix_expression<AE> &ae):
		matrix_container<self_type> (),
		size1_(ae().size1()), size2_(ae().size2()), data_(layout_type::storage_size(size1_, size2_)) {
		matrix_assign<scalar_assign> (*this, ae);
	}

	// Accessors
	/** Return the number of rows of the matrix
	 * You can also use the free size<>() function in operation/size.hpp as size<1>(m) where m is a matrix
	 */
	BOOST_UBLAS_INLINE
	size_type size1() const {
		return size1_;
	}

	/** Return the number of colums of the matrix
	 * You can also use the free size<>() function in operation/size.hpp as size<2>(m) where m is a matrix
	 */
	BOOST_UBLAS_INLINE
	size_type size2() const {
		return size2_;
	}

	// Storage accessors
	/** Return a constant reference to the internal storage of a dense matrix, i.e. the raw data
	 * It's type depends on the type used by the matrix to store its data
	 */
	BOOST_UBLAS_INLINE
	const array_type &data() const {
		return data_;
	}
	/** Return a reference to the internal storage of a dense matrix, i.e. the raw data
	 * It's type depends on the type used by the matrix to store its data
	 */
	BOOST_UBLAS_INLINE
	array_type &data() {
		return data_;
	}

	// Resizing
	/** Resize a matrix to new dimensions
	 * If data are preserved, then if the size if bigger at least on one dimension, extra values are filled with zeros.
	 * If data are not preserved, then nothing has to be assumed regarding the content of the matrix after resizing.
	 * \param size1 the new number of rows
	 * \param size2 the new number of colums
	 * \param preserve a boolean to say if one wants the data to be preserved during the resizing. Default is true.
	 */
	BOOST_UBLAS_INLINE
	void resize(size_type size1, size_type size2, bool preserve = true) {
		if (preserve) {
			self_type temporary(size1, size2);
			detail::matrix_resize_preserve<layout_type> (*this, temporary);
		} else {
			data().resize(layout_type::storage_size(size1, size2));
			size1_ = size1;
			size2_ = size2;
		}
	}

	// Element access
	BOOST_UBLAS_INLINE
	const_reference operator()(size_type i, size_type j) const {
		return data() [layout_type::element(i, size1_, j, size2_)];
	}
	BOOST_UBLAS_INLINE
	reference at_element(size_type i, size_type j) {
		return data() [layout_type::element(i, size1_, j, size2_)];
	}
	BOOST_UBLAS_INLINE
	reference operator()(size_type i, size_type j) {
		return at_element(i, j);
	}

	// Element assignment
	BOOST_UBLAS_INLINE
	reference insert_element(size_type i, size_type j, const_reference t) {
		return (at_element(i, j) = t);
	}
	void erase_element(size_type i, size_type j) {
		at_element(i, j) = value_type/*zero*/();
	}

	// Zeroing
	BOOST_UBLAS_INLINE
	void clear() {
		std::fill(data().begin(), data().end(), value_type/*zero*/());
	}

	// Assignment
#ifdef BOOST_UBLAS_MOVE_SEMANTICS

	/*! @note "pass by value" the key idea to enable move semantics */
	BOOST_UBLAS_INLINE
	matrix &operator = (matrix m) {
		assign_temporary(m);
		return *this;
	}
#else
	BOOST_UBLAS_INLINE
	matrix &operator = (const matrix &m) {
		size1_ = m.size1_;
		size2_ = m.size2_;
		data() = m.data();
		return *this;
	}
#endif
	template<class C>          // Container assignment without temporary
	BOOST_UBLAS_INLINE
	matrix &operator = (const matrix_container<C> &m) {
		resize(m().size1(), m().size2(), false);
		assign(m);
		return *this;
	}
	BOOST_UBLAS_INLINE
	matrix &assign_temporary(matrix &m) {
		swap(m);
		return *this;
	}
	template<class AE>
	BOOST_UBLAS_INLINE
	matrix &operator = (const matrix_expression<AE> &ae) {
		self_type temporary(ae);
		return assign_temporary(temporary);
	}
	template<class AE>
	BOOST_UBLAS_INLINE
	matrix &assign(const matrix_expression<AE> &ae) {
		matrix_assign<scalar_assign> (*this, ae);
		return *this;
	}
	template<class AE>
	BOOST_UBLAS_INLINE
	matrix &operator += (const matrix_expression<AE> &ae) {
		self_type temporary(*this + ae);
		return assign_temporary(temporary);
	}
	template<class C>          // Container assignment without temporary
	BOOST_UBLAS_INLINE
	matrix &operator += (const matrix_container<C> &m) {
		plus_assign(m);
		return *this;
	}
	template<class AE>
	BOOST_UBLAS_INLINE
	matrix &plus_assign(const matrix_expression<AE> &ae) {
		matrix_assign<scalar_plus_assign> (*this, ae);
		return *this;
	}
	template<class AE>
	BOOST_UBLAS_INLINE
	matrix &operator -= (const matrix_expression<AE> &ae) {
		self_type temporary(*this - ae);
		return assign_temporary(temporary);
	}
	template<class C>          // Container assignment without temporary
	BOOST_UBLAS_INLINE
	matrix &operator -= (const matrix_container<C> &m) {
		minus_assign(m);
		return *this;
	}
	template<class AE>
	BOOST_UBLAS_INLINE
	matrix &minus_assign(const matrix_expression<AE> &ae) {
		matrix_assign<scalar_minus_assign> (*this, ae);
		return *this;
	}
	template<class AT>
	BOOST_UBLAS_INLINE
	matrix &operator *= (const AT &at) {
		matrix_assign_scalar<scalar_multiplies_assign> (*this, at);
		return *this;
	}
	template<class AT>
	BOOST_UBLAS_INLINE
	matrix &operator /= (const AT &at) {
		matrix_assign_scalar<scalar_divides_assign> (*this, at);
		return *this;
	}

	// Swapping
	BOOST_UBLAS_INLINE
	void swap(matrix &m) {
		if (this != &m) {
			std::swap(size1_, m.size1_);
			std::swap(size2_, m.size2_);
			data().swap(m.data());
		}
	}
	BOOST_UBLAS_INLINE
	friend void swap(matrix &m1, matrix &m2) {
		m1.swap(m2);
	}

	// Iterator types
private:
	// Use the storage array iterator
	typedef typename A::const_iterator const_subiterator_type;
	typedef typename A::iterator subiterator_type;

public:
#ifdef BOOST_UBLAS_USE_INDEXED_ITERATOR
	typedef indexed_iterator1<self_type, dense_random_access_iterator_tag> iterator1;
	typedef indexed_iterator2<self_type, dense_random_access_iterator_tag> iterator2;
	typedef indexed_const_iterator1<self_type, dense_random_access_iterator_tag> const_iterator1;
	typedef indexed_const_iterator2<self_type, dense_random_access_iterator_tag> const_iterator2;
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
	BOOST_UBLAS_INLINE
	const_iterator1 find1(int /* rank */, size_type i, size_type j) const {
#ifdef BOOST_UBLAS_USE_INDEXED_ITERATOR
		return const_iterator1(*this, i, j);
#else
		return const_iterator1(*this, data().begin() + layout_type::address(i, size1_, j, size2_));
#endif
	}
	BOOST_UBLAS_INLINE
	iterator1 find1(int /* rank */, size_type i, size_type j) {
#ifdef BOOST_UBLAS_USE_INDEXED_ITERATOR
		return iterator1(*this, i, j);
#else
		return iterator1(*this, data().begin() + layout_type::address(i, size1_, j, size2_));
#endif
	}
	BOOST_UBLAS_INLINE
	const_iterator2 find2(int /* rank */, size_type i, size_type j) const {
#ifdef BOOST_UBLAS_USE_INDEXED_ITERATOR
		return const_iterator2(*this, i, j);
#else
		return const_iterator2(*this, data().begin() + layout_type::address(i, size1_, j, size2_));
#endif
	}
	BOOST_UBLAS_INLINE
	iterator2 find2(int /* rank */, size_type i, size_type j) {
#ifdef BOOST_UBLAS_USE_INDEXED_ITERATOR
		return iterator2(*this, i, j);
#else
		return iterator2(*this, data().begin() + layout_type::address(i, size1_, j, size2_));
#endif
	}


#ifndef BOOST_UBLAS_USE_INDEXED_ITERATOR
	class const_iterator1:
		public container_const_reference<matrix>,
		public random_access_iterator_base<dense_random_access_iterator_tag,
			const_iterator1, value_type> {
	public:
		typedef typename matrix::value_type value_type;
		typedef typename matrix::difference_type difference_type;
		typedef typename matrix::const_reference reference;
		typedef const typename matrix::pointer pointer;

		typedef const_iterator2 dual_iterator_type;
		typedef const_reverse_iterator2 dual_reverse_iterator_type;

		// Construction and destruction
		BOOST_UBLAS_INLINE
		const_iterator1():
			container_const_reference<self_type> (), it_() {}
		BOOST_UBLAS_INLINE
		const_iterator1(const self_type &m, const const_subiterator_type &it):
			container_const_reference<self_type> (m), it_(it) {}
		BOOST_UBLAS_INLINE
		const_iterator1(const iterator1 &it):
			container_const_reference<self_type> (it()), it_(it.it_) {}

		// Arithmetic
		BOOST_UBLAS_INLINE
		const_iterator1 &operator ++ () {
			layout_type::increment_i(it_, (*this)().size1(), (*this)().size2());
			return *this;
		}
		BOOST_UBLAS_INLINE
		const_iterator1 &operator -- () {
			layout_type::decrement_i(it_, (*this)().size1(), (*this)().size2());
			return *this;
		}
		BOOST_UBLAS_INLINE
		const_iterator1 &operator += (difference_type n) {
			layout_type::increment_i(it_, n, (*this)().size1(), (*this)().size2());
			return *this;
		}
		BOOST_UBLAS_INLINE
		const_iterator1 &operator -= (difference_type n) {
			layout_type::decrement_i(it_, n, (*this)().size1(), (*this)().size2());
			return *this;
		}
		BOOST_UBLAS_INLINE
		difference_type operator - (const const_iterator1 &it) const {
			BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
			return layout_type::distance_i(it_ - it.it_, (*this)().size1(), (*this)().size2());
		}

		// Dereference
		BOOST_UBLAS_INLINE
		const_reference operator * () const {
			BOOST_UBLAS_CHECK(index1() < (*this)().size1(), bad_index());
			BOOST_UBLAS_CHECK(index2() < (*this)().size2(), bad_index());
			return *it_;
		}
		BOOST_UBLAS_INLINE
		const_reference operator [](difference_type n) const {
			return *(*this + n);
		}

#ifndef BOOST_UBLAS_NO_NESTED_CLASS_RELATION
		BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		const_iterator2 begin() const {
			const self_type &m = (*this)();
			return m.find2(1, index1(), 0);
		}
		BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		const_iterator2 end() const {
			const self_type &m = (*this)();
			return m.find2(1, index1(), m.size2());
		}
		BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		const_reverse_iterator2 rbegin() const {
			return const_reverse_iterator2(end());
		}
		BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		const_reverse_iterator2 rend() const {
			return const_reverse_iterator2(begin());
		}
#endif

		// Indices
		BOOST_UBLAS_INLINE
		size_type index1() const {
			const self_type &m = (*this)();
			return layout_type::index_i(it_ - m.begin1().it_, m.size1(), m.size2());
		}
		BOOST_UBLAS_INLINE
		size_type index2() const {
			const self_type &m = (*this)();
			return layout_type::index_j(it_ - m.begin1().it_, m.size1(), m.size2());
		}

		// Assignment
		BOOST_UBLAS_INLINE
		const_iterator1 &operator = (const const_iterator1 &it) {
			container_const_reference<self_type>::assign(&it());
			it_ = it.it_;
			return *this;
		}

		// Comparison
		BOOST_UBLAS_INLINE
		bool operator == (const const_iterator1 &it) const {
			BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
			return it_ == it.it_;
		}
		BOOST_UBLAS_INLINE
		bool operator < (const const_iterator1 &it) const {
			BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
			return it_ < it.it_;
		}

	private:
		const_subiterator_type it_;

		friend class iterator1;
	};
#endif

	BOOST_UBLAS_INLINE
	const_iterator1 begin1() const {
		return find1(0, 0, 0);
	}
	BOOST_UBLAS_INLINE
	const_iterator1 end1() const {
		return find1(0, size1_, 0);
	}

#ifndef BOOST_UBLAS_USE_INDEXED_ITERATOR
	class iterator1:
		public container_reference<matrix>,
		public random_access_iterator_base<dense_random_access_iterator_tag,
			iterator1, value_type> {
	public:
		typedef typename matrix::value_type value_type;
		typedef typename matrix::difference_type difference_type;
		typedef typename matrix::reference reference;
		typedef typename matrix::pointer pointer;

		typedef iterator2 dual_iterator_type;
		typedef reverse_iterator2 dual_reverse_iterator_type;

		// Construction and destruction
		BOOST_UBLAS_INLINE
		iterator1():
			container_reference<self_type> (), it_() {}
		BOOST_UBLAS_INLINE
		iterator1(self_type &m, const subiterator_type &it):
			container_reference<self_type> (m), it_(it) {}

		// Arithmetic
		BOOST_UBLAS_INLINE
		iterator1 &operator ++ () {
			layout_type::increment_i(it_, (*this)().size1(), (*this)().size2());
			return *this;
		}
		BOOST_UBLAS_INLINE
		iterator1 &operator -- () {
			layout_type::decrement_i(it_, (*this)().size1(), (*this)().size2());
			return *this;
		}
		BOOST_UBLAS_INLINE
		iterator1 &operator += (difference_type n) {
			layout_type::increment_i(it_, n, (*this)().size1(), (*this)().size2());
			return *this;
		}
		BOOST_UBLAS_INLINE
		iterator1 &operator -= (difference_type n) {
			layout_type::decrement_i(it_, n, (*this)().size1(), (*this)().size2());
			return *this;
		}
		BOOST_UBLAS_INLINE
		difference_type operator - (const iterator1 &it) const {
			BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
			return layout_type::distance_i(it_ - it.it_, (*this)().size1(), (*this)().size2());
		}

		// Dereference
		BOOST_UBLAS_INLINE
		reference operator * () const {
			BOOST_UBLAS_CHECK(index1() < (*this)().size1(), bad_index());
			BOOST_UBLAS_CHECK(index2() < (*this)().size2(), bad_index());
			return *it_;
		}
		BOOST_UBLAS_INLINE
		reference operator [](difference_type n) const {
			return *(*this + n);
		}

#ifndef BOOST_UBLAS_NO_NESTED_CLASS_RELATION
		BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		iterator2 begin() const {
			self_type &m = (*this)();
			return m.find2(1, index1(), 0);
		}
		BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		iterator2 end() const {
			self_type &m = (*this)();
			return m.find2(1, index1(), m.size2());
		}
		BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		reverse_iterator2 rbegin() const {
			return reverse_iterator2(end());
		}
		BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		reverse_iterator2 rend() const {
			return reverse_iterator2(begin());
		}
#endif

		// Indices
		BOOST_UBLAS_INLINE
		size_type index1() const {
			self_type &m = (*this)();
			return layout_type::index_i(it_ - m.begin1().it_, m.size1(), m.size2());
		}
		BOOST_UBLAS_INLINE
		size_type index2() const {
			self_type &m = (*this)();
			return layout_type::index_j(it_ - m.begin1().it_, m.size1(), m.size2());
		}

		// Assignment
		BOOST_UBLAS_INLINE
		iterator1 &operator = (const iterator1 &it) {
			container_reference<self_type>::assign(&it());
			it_ = it.it_;
			return *this;
		}

		// Comparison
		BOOST_UBLAS_INLINE
		bool operator == (const iterator1 &it) const {
			BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
			return it_ == it.it_;
		}
		BOOST_UBLAS_INLINE
		bool operator < (const iterator1 &it) const {
			BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
			return it_ < it.it_;
		}

	private:
		subiterator_type it_;

		friend class const_iterator1;
	};
#endif

	BOOST_UBLAS_INLINE
	iterator1 begin1() {
		return find1(0, 0, 0);
	}
	BOOST_UBLAS_INLINE
	iterator1 end1() {
		return find1(0, size1_, 0);
	}

#ifndef BOOST_UBLAS_USE_INDEXED_ITERATOR
	class const_iterator2:
		public container_const_reference<matrix>,
		public random_access_iterator_base<dense_random_access_iterator_tag,
			const_iterator2, value_type> {
	public:
		typedef typename matrix::value_type value_type;
		typedef typename matrix::difference_type difference_type;
		typedef typename matrix::const_reference reference;
		typedef const typename matrix::pointer pointer;

		typedef const_iterator1 dual_iterator_type;
		typedef const_reverse_iterator1 dual_reverse_iterator_type;

		// Construction and destruction
		BOOST_UBLAS_INLINE
		const_iterator2():
			container_const_reference<self_type> (), it_() {}
		BOOST_UBLAS_INLINE
		const_iterator2(const self_type &m, const const_subiterator_type &it):
			container_const_reference<self_type> (m), it_(it) {}
		BOOST_UBLAS_INLINE
		const_iterator2(const iterator2 &it):
			container_const_reference<self_type> (it()), it_(it.it_) {}

		// Arithmetic
		BOOST_UBLAS_INLINE
		const_iterator2 &operator ++ () {
			layout_type::increment_j(it_, (*this)().size1(), (*this)().size2());
			return *this;
		}
		BOOST_UBLAS_INLINE
		const_iterator2 &operator -- () {
			layout_type::decrement_j(it_, (*this)().size1(), (*this)().size2());
			return *this;
		}
		BOOST_UBLAS_INLINE
		const_iterator2 &operator += (difference_type n) {
			layout_type::increment_j(it_, n, (*this)().size1(), (*this)().size2());
			return *this;
		}
		BOOST_UBLAS_INLINE
		const_iterator2 &operator -= (difference_type n) {
			layout_type::decrement_j(it_, n, (*this)().size1(), (*this)().size2());
			return *this;
		}
		BOOST_UBLAS_INLINE
		difference_type operator - (const const_iterator2 &it) const {
			BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
			return layout_type::distance_j(it_ - it.it_, (*this)().size1(), (*this)().size2());
		}

		// Dereference
		BOOST_UBLAS_INLINE
		const_reference operator * () const {
			BOOST_UBLAS_CHECK(index1() < (*this)().size1(), bad_index());
			BOOST_UBLAS_CHECK(index2() < (*this)().size2(), bad_index());
			return *it_;
		}
		BOOST_UBLAS_INLINE
		const_reference operator [](difference_type n) const {
			return *(*this + n);
		}

#ifndef BOOST_UBLAS_NO_NESTED_CLASS_RELATION
		BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		const_iterator1 begin() const {
			const self_type &m = (*this)();
			return m.find1(1, 0, index2());
		}
		BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		const_iterator1 end() const {
			const self_type &m = (*this)();
			return m.find1(1, m.size1(), index2());
		}
		BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		const_reverse_iterator1 rbegin() const {
			return const_reverse_iterator1(end());
		}
		BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		const_reverse_iterator1 rend() const {
			return const_reverse_iterator1(begin());
		}
#endif

		// Indices
		BOOST_UBLAS_INLINE
		size_type index1() const {
			const self_type &m = (*this)();
			return layout_type::index_i(it_ - m.begin2().it_, m.size1(), m.size2());
		}
		BOOST_UBLAS_INLINE
		size_type index2() const {
			const self_type &m = (*this)();
			return layout_type::index_j(it_ - m.begin2().it_, m.size1(), m.size2());
		}

		// Assignment
		BOOST_UBLAS_INLINE
		const_iterator2 &operator = (const const_iterator2 &it) {
			container_const_reference<self_type>::assign(&it());
			it_ = it.it_;
			return *this;
		}

		// Comparison
		BOOST_UBLAS_INLINE
		bool operator == (const const_iterator2 &it) const {
			BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
			return it_ == it.it_;
		}
		BOOST_UBLAS_INLINE
		bool operator < (const const_iterator2 &it) const {
			BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
			return it_ < it.it_;
		}

	private:
		const_subiterator_type it_;

		friend class iterator2;
	};
#endif

	BOOST_UBLAS_INLINE
	const_iterator2 begin2() const {
		return find2(0, 0, 0);
	}
	BOOST_UBLAS_INLINE
	const_iterator2 end2() const {
		return find2(0, 0, size2_);
	}

#ifndef BOOST_UBLAS_USE_INDEXED_ITERATOR
	class iterator2:
		public container_reference<matrix>,
		public random_access_iterator_base<dense_random_access_iterator_tag,
			iterator2, value_type> {
	public:
		typedef typename matrix::value_type value_type;
		typedef typename matrix::difference_type difference_type;
		typedef typename matrix::reference reference;
		typedef typename matrix::pointer pointer;

		typedef iterator1 dual_iterator_type;
		typedef reverse_iterator1 dual_reverse_iterator_type;

		// Construction and destruction
		BOOST_UBLAS_INLINE
		iterator2():
			container_reference<self_type> (), it_() {}
		BOOST_UBLAS_INLINE
		iterator2(self_type &m, const subiterator_type &it):
			container_reference<self_type> (m), it_(it) {}

		// Arithmetic
		BOOST_UBLAS_INLINE
		iterator2 &operator ++ () {
			layout_type::increment_j(it_, (*this)().size1(), (*this)().size2());
			return *this;
		}
		BOOST_UBLAS_INLINE
		iterator2 &operator -- () {
			layout_type::decrement_j(it_, (*this)().size1(), (*this)().size2());
			return *this;
		}
		BOOST_UBLAS_INLINE
		iterator2 &operator += (difference_type n) {
			layout_type::increment_j(it_, n, (*this)().size1(), (*this)().size2());
			return *this;
		}
		BOOST_UBLAS_INLINE
		iterator2 &operator -= (difference_type n) {
			layout_type::decrement_j(it_, n, (*this)().size1(), (*this)().size2());
			return *this;
		}
		BOOST_UBLAS_INLINE
		difference_type operator - (const iterator2 &it) const {
			BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
			return layout_type::distance_j(it_ - it.it_, (*this)().size1(), (*this)().size2());
		}

		// Dereference
		BOOST_UBLAS_INLINE
		reference operator * () const {
			BOOST_UBLAS_CHECK(index1() < (*this)().size1(), bad_index());
			BOOST_UBLAS_CHECK(index2() < (*this)().size2(), bad_index());
			return *it_;
		}
		BOOST_UBLAS_INLINE
		reference operator [](difference_type n) const {
			return *(*this + n);
		}

#ifndef BOOST_UBLAS_NO_NESTED_CLASS_RELATION
		BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		iterator1 begin() const {
			self_type &m = (*this)();
			return m.find1(1, 0, index2());
		}
		BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		iterator1 end() const {
			self_type &m = (*this)();
			return m.find1(1, m.size1(), index2());
		}
		BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		reverse_iterator1 rbegin() const {
			return reverse_iterator1(end());
		}
		BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		reverse_iterator1 rend() const {
			return reverse_iterator1(begin());
		}
#endif

		// Indices
		BOOST_UBLAS_INLINE
		size_type index1() const {
			self_type &m = (*this)();
			return layout_type::index_i(it_ - m.begin2().it_, m.size1(), m.size2());
		}
		BOOST_UBLAS_INLINE
		size_type index2() const {
			self_type &m = (*this)();
			return layout_type::index_j(it_ - m.begin2().it_, m.size1(), m.size2());
		}

		// Assignment
		BOOST_UBLAS_INLINE
		iterator2 &operator = (const iterator2 &it) {
			container_reference<self_type>::assign(&it());
			it_ = it.it_;
			return *this;
		}

		// Comparison
		BOOST_UBLAS_INLINE
		bool operator == (const iterator2 &it) const {
			BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
			return it_ == it.it_;
		}
		BOOST_UBLAS_INLINE
		bool operator < (const iterator2 &it) const {
			BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
			return it_ < it.it_;
		}

	private:
		subiterator_type it_;

		friend class const_iterator2;
	};
#endif

	BOOST_UBLAS_INLINE
	iterator2 begin2() {
		return find2(0, 0, 0);
	}
	BOOST_UBLAS_INLINE
	iterator2 end2() {
		return find2(0, 0, size2_);
	}

	// Reverse iterators

	BOOST_UBLAS_INLINE
	const_reverse_iterator1 rbegin1() const {
		return const_reverse_iterator1(end1());
	}
	BOOST_UBLAS_INLINE
	const_reverse_iterator1 rend1() const {
		return const_reverse_iterator1(begin1());
	}

	BOOST_UBLAS_INLINE
	reverse_iterator1 rbegin1() {
		return reverse_iterator1(end1());
	}
	BOOST_UBLAS_INLINE
	reverse_iterator1 rend1() {
		return reverse_iterator1(begin1());
	}

	BOOST_UBLAS_INLINE
	const_reverse_iterator2 rbegin2() const {
		return const_reverse_iterator2(end2());
	}
	BOOST_UBLAS_INLINE
	const_reverse_iterator2 rend2() const {
		return const_reverse_iterator2(begin2());
	}

	BOOST_UBLAS_INLINE
	reverse_iterator2 rbegin2() {
		return reverse_iterator2(end2());
	}
	BOOST_UBLAS_INLINE
	reverse_iterator2 rend2() {
		return reverse_iterator2(begin2());
	}

	// Serialization
	template<class Archive>
	void serialize(Archive &ar, const unsigned int /* file_version */) {

		// we need to copy to a collection_size_type to get a portable
		// and efficient boost::serialization
		boost::serialization::collection_size_type s1(size1_);
		boost::serialization::collection_size_type s2(size2_);

		// serialize the sizes
		ar &boost::serialization::make_nvp("size1",s1)
		& boost::serialization::make_nvp("size2",s2);

		// copy the values back if loading
		if (Archive::is_loading::value) {
			size1_ = s1;
			size2_ = s2;
		}
		ar &boost::serialization::make_nvp("data",data_);
	}

private:
	size_type size1_;
	size_type size2_;
	array_type data_;
};
/** \brief An identity matrix with values of type \c T
 *
 * Elements or cordinates \f$(i,i)\f$ are equal to 1 (one) and all others to 0 (zero).
 * Changing values does not affect the matrix, however assigning it to a normal matrix will
 * make the matrix equal to an identity matrix. All accesses are constant du to the trivial values.
 *
 * \tparam T the type of object stored in the matrix (like double, float, complex, etc...)
 * \tparam ALLOC an allocator for storing the zeros and one elements. By default, a standar allocator is used.
 */
template<class T>
class identity_matrix:
	public matrix_container<identity_matrix<T> > {

	typedef const T *const_pointer;
	typedef identity_matrix<T> self_type;
public:
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;
	typedef T value_type;
	typedef const T &const_reference;
	typedef T &reference;
	typedef const matrix_reference<const self_type> const_closure_type;
	typedef matrix_reference<self_type> closure_type;
	typedef sparse_tag storage_category;
	typedef unknown_orientation_tag orientation_category;

	// Construction and destruction
	BOOST_UBLAS_INLINE
	identity_matrix():
		matrix_container<self_type> (),
		size1_(0), size2_(0), size_common_(0) {}
	BOOST_UBLAS_INLINE
	identity_matrix(size_type size):
		matrix_container<self_type> (),
		size1_(size), size2_(size), size_common_((std::min)(size1_, size2_)) {}
	BOOST_UBLAS_INLINE
	identity_matrix(size_type size1, size_type size2):
		matrix_container<self_type> (),
		size1_(size1), size2_(size2), size_common_((std::min)(size1_, size2_)) {}
	BOOST_UBLAS_INLINE
	identity_matrix(const identity_matrix &m):
		matrix_container<self_type> (),
		size1_(m.size1_), size2_(m.size2_), size_common_((std::min)(size1_, size2_)) {}

	// Accessors
	BOOST_UBLAS_INLINE
	size_type size1() const {
		return size1_;
	}
	BOOST_UBLAS_INLINE
	size_type size2() const {
		return size2_;
	}

	// Resizing
	BOOST_UBLAS_INLINE
	void resize(size_type size, bool preserve = true) {
		size1_ = size;
		size2_ = size;
		size_common_ = ((std::min)(size1_, size2_));
	}
	BOOST_UBLAS_INLINE
	void resize(size_type size1, size_type size2, bool /*preserve*/ = true) {
		size1_ = size1;
		size2_ = size2;
		size_common_ = ((std::min)(size1_, size2_));
	}

	// Element access
	BOOST_UBLAS_INLINE
	const_reference operator()(size_type i, size_type j) const {
		if (i == j)
			return one_;
		else
			return zero_;
	}

	// Assignment
	BOOST_UBLAS_INLINE
	identity_matrix &operator = (const identity_matrix &m) {
		size1_ = m.size1_;
		size2_ = m.size2_;
		size_common_ = m.size_common_;
		return *this;
	}
	BOOST_UBLAS_INLINE
	identity_matrix &assign_temporary(identity_matrix &m) {
		swap(m);
		return *this;
	}

	// Swapping
	BOOST_UBLAS_INLINE
	void swap(identity_matrix &m) {
		if (this != &m) {
			std::swap(size1_, m.size1_);
			std::swap(size2_, m.size2_);
			std::swap(size_common_, m.size_common_);
		}
	}
	BOOST_UBLAS_INLINE
	friend void swap(identity_matrix &m1, identity_matrix &m2) {
		m1.swap(m2);
	}

	// Iterator types
private:
	// Use an index
	typedef size_type const_subiterator_type;

public:
	class const_iterator1;
	class const_iterator2;
	typedef reverse_iterator_base1<const_iterator1> const_reverse_iterator1;
	typedef reverse_iterator_base2<const_iterator2> const_reverse_iterator2;

	// Element lookup
	BOOST_UBLAS_INLINE
	const_iterator1 find1(int rank, size_type i, size_type j) const {
		if (rank == 1) {
			i = (std::max)(i, j);
			i = (std::min)(i, j + 1);
		}
		return const_iterator1(*this, i);
	}
	BOOST_UBLAS_INLINE
	const_iterator2 find2(int rank, size_type i, size_type j) const {
		if (rank == 1) {
			j = (std::max)(j, i);
			j = (std::min)(j, i + 1);
		}
		return const_iterator2(*this, j);
	}


	class const_iterator1:
		public container_const_reference<identity_matrix>,
		public bidirectional_iterator_base<sparse_bidirectional_iterator_tag,
			const_iterator1, value_type> {
	public:
		typedef typename identity_matrix::value_type value_type;
		typedef typename identity_matrix::difference_type difference_type;
		typedef typename identity_matrix::const_reference reference;
		typedef typename identity_matrix::const_pointer pointer;

		typedef const_iterator2 dual_iterator_type;
		typedef const_reverse_iterator2 dual_reverse_iterator_type;

		// Construction and destruction
		BOOST_UBLAS_INLINE
		const_iterator1():
			container_const_reference<self_type> (), it_() {}
		BOOST_UBLAS_INLINE
		const_iterator1(const self_type &m, const const_subiterator_type &it):
			container_const_reference<self_type> (m), it_(it) {}

		// Arithmetic
		BOOST_UBLAS_INLINE
		const_iterator1 &operator ++ () {
			BOOST_UBLAS_CHECK(it_ < (*this)().size1(), bad_index());
			++it_;
			return *this;
		}
		BOOST_UBLAS_INLINE
		const_iterator1 &operator -- () {
			BOOST_UBLAS_CHECK(it_ > 0, bad_index());
			--it_;
			return *this;
		}

		// Dereference
		BOOST_UBLAS_INLINE
		const_reference operator * () const {
			return one_;
		}

#ifndef BOOST_UBLAS_NO_NESTED_CLASS_RELATION
		BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		const_iterator2 begin() const {
			return const_iterator2((*this)(), it_);
		}
		BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		const_iterator2 end() const {
			return const_iterator2((*this)(), it_ + 1);
		}
		BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		const_reverse_iterator2 rbegin() const {
			return const_reverse_iterator2(end());
		}
		BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		const_reverse_iterator2 rend() const {
			return const_reverse_iterator2(begin());
		}
#endif

		// Indices
		BOOST_UBLAS_INLINE
		size_type index1() const {
			return it_;
		}
		BOOST_UBLAS_INLINE
		size_type index2() const {
			return it_;
		}

		// Assignment
		BOOST_UBLAS_INLINE
		const_iterator1 &operator = (const const_iterator1 &it) {
			container_const_reference<self_type>::assign(&it());
			it_ = it.it_;
			return *this;
		}

		// Comparison
		BOOST_UBLAS_INLINE
		bool operator == (const const_iterator1 &it) const {
			BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
			return it_ == it.it_;
		}

	private:
		const_subiterator_type it_;
	};

	typedef const_iterator1 iterator1;

	BOOST_UBLAS_INLINE
	const_iterator1 begin1() const {
		return const_iterator1(*this, 0);
	}
	BOOST_UBLAS_INLINE
	const_iterator1 end1() const {
		return const_iterator1(*this, size_common_);
	}

	class const_iterator2:
		public container_const_reference<identity_matrix>,
		public bidirectional_iterator_base<sparse_bidirectional_iterator_tag,
			const_iterator2, value_type> {
	public:
		typedef typename identity_matrix::value_type value_type;
		typedef typename identity_matrix::difference_type difference_type;
		typedef typename identity_matrix::const_reference reference;
		typedef typename identity_matrix::const_pointer pointer;

		typedef const_iterator1 dual_iterator_type;
		typedef const_reverse_iterator1 dual_reverse_iterator_type;

		// Construction and destruction
		BOOST_UBLAS_INLINE
		const_iterator2():
			container_const_reference<self_type> (), it_() {}
		BOOST_UBLAS_INLINE
		const_iterator2(const self_type &m, const const_subiterator_type &it):
			container_const_reference<self_type> (m), it_(it) {}

		// Arithmetic
		BOOST_UBLAS_INLINE
		const_iterator2 &operator ++ () {
			BOOST_UBLAS_CHECK(it_ < (*this)().size_common_, bad_index());
			++it_;
			return *this;
		}
		BOOST_UBLAS_INLINE
		const_iterator2 &operator -- () {
			BOOST_UBLAS_CHECK(it_ > 0, bad_index());
			--it_;
			return *this;
		}

		// Dereference
		BOOST_UBLAS_INLINE
		const_reference operator * () const {
			return one_;
		}

#ifndef BOOST_UBLAS_NO_NESTED_CLASS_RELATION
		BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		const_iterator1 begin() const {
			return const_iterator1((*this)(), it_);
		}
		BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		const_iterator1 end() const {
			return const_iterator1((*this)(), it_ + 1);
		}
		BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		const_reverse_iterator1 rbegin() const {
			return const_reverse_iterator1(end());
		}
		BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
		typename self_type::
#endif
		const_reverse_iterator1 rend() const {
			return const_reverse_iterator1(begin());
		}
#endif

		// Indices
		BOOST_UBLAS_INLINE
		size_type index1() const {
			return it_;
		}
		BOOST_UBLAS_INLINE
		size_type index2() const {
			return it_;
		}

		// Assignment
		BOOST_UBLAS_INLINE
		const_iterator2 &operator = (const const_iterator2 &it) {
			container_const_reference<self_type>::assign(&it());
			it_ = it.it_;
			return *this;
		}

		// Comparison
		BOOST_UBLAS_INLINE
		bool operator == (const const_iterator2 &it) const {
			BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
			return it_ == it.it_;
		}

	private:
		const_subiterator_type it_;
	};

	typedef const_iterator2 iterator2;

	BOOST_UBLAS_INLINE
	const_iterator2 begin2() const {
		return const_iterator2(*this, 0);
	}
	BOOST_UBLAS_INLINE
	const_iterator2 end2() const {
		return const_iterator2(*this, size_common_);
	}

	// Reverse iterators

	BOOST_UBLAS_INLINE
	const_reverse_iterator1 rbegin1() const {
		return const_reverse_iterator1(end1());
	}
	BOOST_UBLAS_INLINE
	const_reverse_iterator1 rend1() const {
		return const_reverse_iterator1(begin1());
	}

	BOOST_UBLAS_INLINE
	const_reverse_iterator2 rbegin2() const {
		return const_reverse_iterator2(end2());
	}
	BOOST_UBLAS_INLINE
	const_reverse_iterator2 rend2() const {
		return const_reverse_iterator2(begin2());
	}

	// Serialization
	template<class Archive>
	void serialize(Archive &ar, const unsigned int /* file_version */) {

		// we need to copy to a collection_size_type to get a portable
		// and efficient boost::serialization
		boost::serialization::collection_size_type s1(size1_);
		boost::serialization::collection_size_type s2(size2_);

		// serialize the sizes
		ar &boost::serialization::make_nvp("size1",s1)
		& boost::serialization::make_nvp("size2",s2);

		// copy the values back if loading
		if (Archive::is_loading::value) {
			size1_ = s1;
			size2_ = s2;
			size_common_ = ((std::min)(size1_, size2_));
		}
	}

private:
	size_type size1_;
	size_type size2_;
	size_type size_common_;
	static const value_type zero_;
	static const value_type one_;
};

template<class T>
const typename identity_matrix<T>::value_type identity_matrix<T>::zero_ = T(/*zero*/);
template<class T>
const typename identity_matrix<T>::value_type identity_matrix<T>::one_(1);  // ISSUE: need 'one'-traits here


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

	typedef const T *const_pointer;
	typedef scalar_matrix<T> self_type;
public:
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;
	typedef T value_type;
	typedef const T &const_reference;
	typedef T &reference;
	typedef const matrix_reference<const self_type> const_closure_type;
	typedef matrix_reference<self_type> closure_type;
	typedef dense_tag storage_category;
	typedef unknown_orientation_tag orientation_category;

	// Construction and destruction
	scalar_matrix():
		matrix_container<self_type> (),
		size1_(0), size2_(0), value_() {}
	scalar_matrix(size_type size1, size_type size2, const value_type &value = value_type(1)):
		matrix_container<self_type> (),
		size1_(size1), size2_(size2), value_(value) {}
	scalar_matrix(const scalar_matrix &m):
		matrix_container<self_type> (),
		size1_(m.size1_), size2_(m.size2_), value_(m.value_) {}

	// Accessors
	size_type size1() const {
		return size1_;
	}
	size_type size2() const {
		return size2_;
	}

	// Resizing
	void resize(size_type size1, size_type size2, bool /*preserve*/ = true) {
		size1_ = size1;
		size2_ = size2;
	}

	// Element access
	BOOST_UBLAS_INLINE
	const_reference operator()(size_type /*i*/, size_type /*j*/) const {
		return value_;
	}

	// Assignment

	scalar_matrix &operator = (const scalar_matrix &m) {
		size1_ = m.size1_;
		size2_ = m.size2_;
		value_ = m.value_;
		return *this;
	}
	
	//Iterators
	typedef indexed_const_iterator1<self_type, dense_random_access_iterator_tag> const_iterator1;
	typedef indexed_const_iterator2<self_type, dense_random_access_iterator_tag> const_iterator2;
	typedef const_iterator1 iterator1;
	typedef const_iterator1 iterator2;
	typedef reverse_iterator_base1<const_iterator1> const_reverse_iterator1;
	typedef reverse_iterator_base2<const_iterator2> const_reverse_iterator2;
	typedef const_reverse_iterator1 reverse_iterator1;
	typedef const_reverse_iterator2 reverse_iterator2;

	const_iterator1 find1(int /*rank*/, size_type i, size_type j) const {
		return const_iterator1(*this, i, j);
	}
	const_iterator2 find2(int /*rank*/, size_type i, size_type j) const {
		return const_iterator2(*this, i, j);
	}

	const_iterator1 begin1() const {
		return find1(0, 0, 0);
	}
	const_iterator1 end1() const {
		return find1(0, size1_, 0);
	}

	const_iterator2 begin2() const {
		return find2(0, 0, 0);
	}
	const_iterator2 end2() const {
		return find2(0, 0, size2_);
	}

	// Reverse iterators
	const_reverse_iterator1 rbegin1() const {
		return const_reverse_iterator1(end1());
	}
	const_reverse_iterator1 rend1() const {
		return const_reverse_iterator1(begin1());
	}
	const_reverse_iterator2 rbegin2() const {
		return const_reverse_iterator2(end2());
	}
	const_reverse_iterator2 rend2() const {
		return const_reverse_iterator2(begin2());
	}

private:
	size_type size1_;
	size_type size2_;
	value_type value_;
};



/** \brief A matrix with all values of type \c T equal to zero
 *
 * Changing values does not affect the matrix, however assigning it to a normal matrix will put zero
 * everywhere in the target matrix. All accesses are constant time, due to the trivial value.
 *
 * \tparam T the type of object stored in the matrix (like double, float, complex, etc...)
 */
template<class T>
struct zero_matrix:public scalar_matrix<T> {
	zero_matrix(std::size_t size1, std::size_t size2):scalar_matrix<T>(size1,size2,T()) {}
};


}
}

#endif
