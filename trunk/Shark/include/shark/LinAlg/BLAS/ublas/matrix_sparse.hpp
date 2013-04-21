//
//  Copyright (c) 2000-2007
//  Joerg Walter, Mathias Koch, Gunter Winkler
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  GeNeSys mbH & Co. KG in producing this work.
//

#ifndef _BOOST_UBLAS_MATRIX_SPARSE_
#define _BOOST_UBLAS_MATRIX_SPARSE_

#include <shark/LinAlg/BLAS/ublas/vector_sparse.hpp>
#include <shark/LinAlg/BLAS/ublas/matrix_expression.hpp>
#include <shark/LinAlg/BLAS/ublas/detail/matrix_assign.hpp>

namespace shark {
namespace blas {

// Comperssed array based sparse matrix class
// Thanks to Kresimir Fresl for extending this to cover different index bases.
template<class T, class L, std::size_t IB, class IA, class TA>
class compressed_matrix:
	public matrix_container<compressed_matrix<T, L, IB, IA, TA> > {

	typedef T &true_reference;
	typedef T *pointer;
	typedef const T *const_pointer;
	typedef L layout_type;
	typedef compressed_matrix<T, L, IB, IA, TA> self_type;
public:
	// ISSUE require type consistency check
	// is_convertable (IA::size_type, TA::size_type)
	typedef typename IA::value_type size_type;
	// size_type for the data arrays.
	typedef typename IA::size_type array_size_type;
	// FIXME difference type for sparse storage iterators should it be in the container?
	typedef typename IA::difference_type difference_type;
	typedef T value_type;
	typedef const T &const_reference;
	typedef T &reference;
	typedef IA index_array_type;
	typedef TA value_array_type;
	typedef const matrix_reference<const self_type> const_closure_type;
	typedef matrix_reference<self_type> closure_type;
	typedef compressed_vector<T, IB, IA, TA> vector_temporary_type;
	typedef self_type matrix_temporary_type;
	typedef sparse_tag storage_category;
	typedef typename L::orientation_category orientation_category;

	// Construction and destruction

	compressed_matrix():
		matrix_container<self_type> (),
		size1_(0), size2_(0), capacity_(restrict_capacity(0)),
		filled1_(1), filled2_(0),
		index1_data_(layout_type::size_M(size1_, size2_) + 1), index2_data_(capacity_), value_data_(capacity_) {
		index1_data_ [filled1_ - 1] = k_based(filled2_);
		storage_invariants();
	}

	compressed_matrix(size_type size1, size_type size2, size_type non_zeros = 0):
		matrix_container<self_type> (),
		size1_(size1), size2_(size2), capacity_(restrict_capacity(non_zeros)),
		filled1_(1), filled2_(0),
		index1_data_(layout_type::size_M(size1_, size2_) + 1), index2_data_(capacity_), value_data_(capacity_) {
		index1_data_ [filled1_ - 1] = k_based(filled2_);
		storage_invariants();
	}

	compressed_matrix(const compressed_matrix &m):
		matrix_container<self_type> (),
		size1_(m.size1_), size2_(m.size2_), capacity_(m.capacity_),
		filled1_(m.filled1_), filled2_(m.filled2_),
		index1_data_(m.index1_data_), index2_data_(m.index2_data_), value_data_(m.value_data_) {
		storage_invariants();
	}

	template<class AE>

	compressed_matrix(const matrix_expression<AE> &ae, size_type non_zeros = 0):
		matrix_container<self_type> (),
		size1_(ae().size1()), size2_(ae().size2()), capacity_(restrict_capacity(non_zeros)),
		filled1_(1), filled2_(0),
		index1_data_(layout_type::size_M(ae().size1(), ae().size2()) + 1),
		index2_data_(capacity_), value_data_(capacity_) {
		index1_data_ [filled1_ - 1] = k_based(filled2_);
		storage_invariants();
		matrix_assign<scalar_assign> (*this, ae);
	}

	// Accessors
	size_type size1() const {
		return size1_;
	}
	size_type size2() const {
		return size2_;
	}

	size_type nnz_capacity() const {
		return capacity_;
	}
	size_type nnz() const {
		return filled2_;
	}

	// Storage accessors

	static size_type index_base() {
		return IB;
	}

	array_size_type filled1() const {
		return filled1_;
	}

	array_size_type filled2() const {
		return filled2_;
	}

	const index_array_type &index1_data() const {
		return index1_data_;
	}

	const index_array_type &index2_data() const {
		return index2_data_;
	}

	const value_array_type &value_data() const {
		return value_data_;
	}

	void set_filled(const array_size_type &filled1, const array_size_type &filled2) {
		filled1_ = filled1;
		filled2_ = filled2;
		storage_invariants();
	}

	index_array_type &index1_data() {
		return index1_data_;
	}

	index_array_type &index2_data() {
		return index2_data_;
	}

	value_array_type &value_data() {
		return value_data_;
	}

	void complete_index1_data() {
		while (filled1_ <= layout_type::size_M(size1_, size2_)) {
			this->index1_data_ [filled1_] = k_based(filled2_);
			++ this->filled1_;
		}
	}

	// Resizing
private:

	size_type restrict_capacity(size_type non_zeros) const {
		non_zeros = (std::max)(non_zeros, (std::min)(size1_, size2_));
		// Guarding against overflow - Thanks to Alexei Novakov for the hint.
		// non_zeros = (std::min) (non_zeros, size1_ * size2_);
		if (size1_ > 0 && non_zeros / size1_ >= size2_)
			non_zeros = size1_ * size2_;
		return non_zeros;
	}
public:

	void resize(size_type size1, size_type size2, bool preserve = true) {
		// FIXME preserve uniboost::mplemented
		BOOST_UBLAS_CHECK(!preserve, internal_logic());
		size1_ = size1;
		size2_ = size2;
		capacity_ = restrict_capacity(capacity_);
		filled1_ = 1;
		filled2_ = 0;
		index1_data_.resize(layout_type::size_M(size1_, size2_) + 1);
		index2_data_.resize(capacity_);
		value_data_.resize(capacity_);
		index1_data_ [filled1_ - 1] = k_based(filled2_);
		storage_invariants();
	}

	// Reserving

	void reserve(size_type non_zeros, bool preserve = true) {
		capacity_ = restrict_capacity(non_zeros);
		if (preserve) {
			index2_data_.resize(capacity_, size_type());
			value_data_.resize(capacity_, value_type());
			filled2_ = (std::min)(capacity_, filled2_);
		} else {
			index2_data_.resize(capacity_);
			value_data_.resize(capacity_);
			filled1_ = 1;
			filled2_ = 0;
			index1_data_ [filled1_ - 1] = k_based(filled2_);
		}
		storage_invariants();
	}

	// Element support

	pointer find_element(size_type i, size_type j) {
		return const_cast<pointer>(const_cast<const self_type &>(*this).find_element(i, j));
	}

	const_pointer find_element(size_type i, size_type j) const {
		size_type element1(layout_type::index_M(i, j));
		size_type element2(layout_type::index_m(i, j));
		if (filled1_ <= element1 + 1)
			return 0;
		vector_const_subiterator_type itv(index1_data_.begin() + element1);
		const_subiterator_type it_begin(index2_data_.begin() + zero_based(*itv));
		const_subiterator_type it_end(index2_data_.begin() + zero_based(*(itv + 1)));
		const_subiterator_type it(detail::lower_bound(it_begin, it_end, k_based(element2), std::less<size_type> ()));
		if (it == it_end || *it != k_based(element2))
			return 0;
		return &value_data_ [it - index2_data_.begin()];
	}

	// Element access

	const_reference operator()(size_type i, size_type j) const {
		const_pointer p = find_element(i, j);
		if (p)
			return *p;
		else
			return zero_;
	}

	reference operator()(size_type i, size_type j) {
		size_type element1(layout_type::index_M(i, j));
		if (filled1_ <= element1 + 1)
			return insert_element(i, j, value_type/*zero*/());
		pointer p = find_element(i, j);
		if (p)
			return *p;
		else
			return insert_element(i, j, value_type/*zero*/());
	}

	// Element assignment

	true_reference insert_element(size_type i, size_type j, const_reference t) {
		BOOST_UBLAS_CHECK(!find_element(i, j), bad_index());           // duplicate element
		if (filled2_ >= capacity_)
			reserve(2 * filled2_, true);
		BOOST_UBLAS_CHECK(filled2_ < capacity_, internal_logic());
		size_type element1 = layout_type::index_M(i, j);
		size_type element2 = layout_type::index_m(i, j);
		while (filled1_ <= element1 + 1) {
			index1_data_ [filled1_] = k_based(filled2_);
			++ filled1_;
		}
		vector_subiterator_type itv(index1_data_.begin() + element1);
		subiterator_type it_begin(index2_data_.begin() + zero_based(*itv));
		subiterator_type it_end(index2_data_.begin() + zero_based(*(itv + 1)));
		subiterator_type it(detail::lower_bound(it_begin, it_end, k_based(element2), std::less<size_type> ()));
		typename std::iterator_traits<subiterator_type>::difference_type n = it - index2_data_.begin();
		BOOST_UBLAS_CHECK(it == it_end || *it != k_based(element2), internal_logic());      // duplicate bound by lower_bound
		++ filled2_;
		it = index2_data_.begin() + n;
		std::copy_backward(it, index2_data_.begin() + filled2_ - 1, index2_data_.begin() + filled2_);
		*it = k_based(element2);
		typename value_array_type::iterator itt(value_data_.begin() + n);
		std::copy_backward(itt, value_data_.begin() + filled2_ - 1, value_data_.begin() + filled2_);
		*itt = t;
		while (element1 + 1 < filled1_) {
			++ index1_data_ [element1 + 1];
			++ element1;
		}
		storage_invariants();
		return *itt;
	}

	void erase_element(size_type i, size_type j) {
		size_type element1 = layout_type::index_M(i, j);
		size_type element2 = layout_type::index_m(i, j);
		if (element1 + 1 >= filled1_)
			return;
		vector_subiterator_type itv(index1_data_.begin() + element1);
		subiterator_type it_begin(index2_data_.begin() + zero_based(*itv));
		subiterator_type it_end(index2_data_.begin() + zero_based(*(itv + 1)));
		subiterator_type it(detail::lower_bound(it_begin, it_end, k_based(element2), std::less<size_type> ()));
		if (it != it_end && *it == k_based(element2)) {
			typename std::iterator_traits<subiterator_type>::difference_type n = it - index2_data_.begin();
			std::copy(it + 1, index2_data_.begin() + filled2_, it);
			typename value_array_type::iterator itt(value_data_.begin() + n);
			std::copy(itt + 1, value_data_.begin() + filled2_, itt);
			-- filled2_;
			while (index1_data_ [filled1_ - 2] > k_based(filled2_)) {
				index1_data_ [filled1_ - 1] = 0;
				-- filled1_;
			}
			while (element1 + 1 < filled1_) {
				-- index1_data_ [element1 + 1];
				++ element1;
			}
		}
		storage_invariants();
	}

	// Zeroing

	void clear() {
		filled1_ = 1;
		filled2_ = 0;
		index1_data_ [filled1_ - 1] = k_based(filled2_);
		storage_invariants();
	}

	// Assignment

	compressed_matrix &operator = (const compressed_matrix &m) {
		if (this != &m) {
			size1_ = m.size1_;
			size2_ = m.size2_;
			capacity_ = m.capacity_;
			filled1_ = m.filled1_;
			filled2_ = m.filled2_;
			index1_data_ = m.index1_data_;
			index2_data_ = m.index2_data_;
			value_data_ = m.value_data_;
		}
		storage_invariants();
		return *this;
	}
	template<class C>          // Container assignment without temporary

	compressed_matrix &operator = (const matrix_container<C> &m) {
		resize(m().size1(), m().size2(), false);
		assign(m);
		return *this;
	}

	compressed_matrix &assign_temporary(compressed_matrix &m) {
		swap(m);
		return *this;
	}
	template<class AE>

	compressed_matrix &operator = (const matrix_expression<AE> &ae) {
		self_type temporary(ae, capacity_);
		return assign_temporary(temporary);
	}
	template<class AE>

	compressed_matrix &assign(const matrix_expression<AE> &ae) {
		matrix_assign<scalar_assign> (*this, ae);
		return *this;
	}
	template<class AE>

	compressed_matrix &operator += (const matrix_expression<AE> &ae) {
		self_type temporary(*this + ae, capacity_);
		return assign_temporary(temporary);
	}
	template<class C>          // Container assignment without temporary

	compressed_matrix &operator += (const matrix_container<C> &m) {
		plus_assign(m);
		return *this;
	}
	template<class AE>

	compressed_matrix &plus_assign(const matrix_expression<AE> &ae) {
		matrix_assign<scalar_plus_assign> (*this, ae);
		return *this;
	}
	template<class AE>

	compressed_matrix &operator -= (const matrix_expression<AE> &ae) {
		self_type temporary(*this - ae, capacity_);
		return assign_temporary(temporary);
	}
	template<class C>          // Container assignment without temporary

	compressed_matrix &operator -= (const matrix_container<C> &m) {
		minus_assign(m);
		return *this;
	}
	template<class AE>

	compressed_matrix &minus_assign(const matrix_expression<AE> &ae) {
		matrix_assign<scalar_minus_assign> (*this, ae);
		return *this;
	}
	template<class AT>

	compressed_matrix &operator *= (const AT &at) {
		matrix_assign_scalar<scalar_multiplies_assign> (*this, at);
		return *this;
	}
	template<class AT>

	compressed_matrix &operator /= (const AT &at) {
		matrix_assign_scalar<scalar_divides_assign> (*this, at);
		return *this;
	}

	// Swapping

	void swap(compressed_matrix &m) {
		if (this != &m) {
			std::swap(size1_, m.size1_);
			std::swap(size2_, m.size2_);
			std::swap(capacity_, m.capacity_);
			std::swap(filled1_, m.filled1_);
			std::swap(filled2_, m.filled2_);
			index1_data_.swap(m.index1_data_);
			index2_data_.swap(m.index2_data_);
			value_data_.swap(m.value_data_);
		}
		storage_invariants();
	}

	friend void swap(compressed_matrix &m1, compressed_matrix &m2) {
		m1.swap(m2);
	}

	// Back element insertion and erasure

	void push_back(size_type i, size_type j, const_reference t) {
		if (filled2_ >= capacity_)
			reserve(2 * filled2_, true);
		BOOST_UBLAS_CHECK(filled2_ < capacity_, internal_logic());
		size_type element1 = layout_type::index_M(i, j);
		size_type element2 = layout_type::index_m(i, j);
		while (filled1_ < element1 + 2) {
			index1_data_ [filled1_] = k_based(filled2_);
			++ filled1_;
		}
		// must maintain sort order
		BOOST_UBLAS_CHECK((filled1_ == element1 + 2 &&
		        (filled2_ == zero_based(index1_data_ [filled1_ - 2]) ||
		                index2_data_ [filled2_ - 1] < k_based(element2))), external_logic());
		++ filled2_;
		index1_data_ [filled1_ - 1] = k_based(filled2_);
		index2_data_ [filled2_ - 1] = k_based(element2);
		value_data_ [filled2_ - 1] = t;
		storage_invariants();
	}

	void pop_back() {
		BOOST_UBLAS_CHECK(filled1_ > 0 && filled2_ > 0, external_logic());
		-- filled2_;
		while (index1_data_ [filled1_ - 2] > k_based(filled2_)) {
			index1_data_ [filled1_ - 1] = 0;
			-- filled1_;
		}
		-- index1_data_ [filled1_ - 1];
		storage_invariants();
	}

	// Iterator types
private:
	// Use index array iterator
	typedef typename IA::const_iterator vector_const_subiterator_type;
	typedef typename IA::iterator vector_subiterator_type;
	typedef typename IA::const_iterator const_subiterator_type;
	typedef typename IA::iterator subiterator_type;


	true_reference at_element(size_type i, size_type j) {
		pointer p = find_element(i, j);
		BOOST_UBLAS_CHECK(p, bad_index());
		return *p;
	}

public:
	class const_iterator1;
	class iterator1;
	class const_iterator2;
	class iterator2;
	typedef reverse_iterator_base1<const_iterator1> const_reverse_iterator1;
	typedef reverse_iterator_base1<iterator1> reverse_iterator1;
	typedef reverse_iterator_base2<const_iterator2> const_reverse_iterator2;
	typedef reverse_iterator_base2<iterator2> reverse_iterator2;

	// Element lookup
	//  This function seems to be big. So we do not let the compiler inline it.
	const_iterator1 find1(int rank, size_type i, size_type j, int direction = 1) const {
		for (;;) {
			array_size_type address1(layout_type::index_M(i, j));
			array_size_type address2(layout_type::index_m(i, j));
			vector_const_subiterator_type itv(index1_data_.begin() + (std::min)(filled1_ - 1, address1));
			if (filled1_ <= address1 + 1)
				return const_iterator1(*this, rank, i, j, itv, index2_data_.begin() + filled2_);

			const_subiterator_type it_begin(index2_data_.begin() + zero_based(*itv));
			const_subiterator_type it_end(index2_data_.begin() + zero_based(*(itv + 1)));

			const_subiterator_type it(detail::lower_bound(it_begin, it_end, k_based(address2), std::less<size_type> ()));
			if (rank == 0)
				return const_iterator1(*this, rank, i, j, itv, it);
			if (it != it_end && zero_based(*it) == address2)
				return const_iterator1(*this, rank, i, j, itv, it);
			if (direction > 0) {
				if (layout_type::fast_i()) {
					if (it == it_end)
						return const_iterator1(*this, rank, i, j, itv, it);
					i = zero_based(*it);
				} else {
					if (i >= size1_)
						return const_iterator1(*this, rank, i, j, itv, it);
					++ i;
				}
			} else { /* if (direction < 0)  */
				if (layout_type::fast_i()) {
					if (it == index2_data_.begin() + zero_based(*itv))
						return const_iterator1(*this, rank, i, j, itv, it);
					i = zero_based(*(it - 1));
				} else {
					if (i == 0)
						return const_iterator1(*this, rank, i, j, itv, it);
					-- i;
				}
			}
		}
	}
	//  This function seems to be big. So we do not let the compiler inline it.
	iterator1 find1(int rank, size_type i, size_type j, int direction = 1) {
		for (;;) {
			array_size_type address1(layout_type::index_M(i, j));
			array_size_type address2(layout_type::index_m(i, j));
			vector_subiterator_type itv(index1_data_.begin() + (std::min)(filled1_ - 1, address1));
			if (filled1_ <= address1 + 1)
				return iterator1(*this, rank, i, j, itv, index2_data_.begin() + filled2_);

			subiterator_type it_begin(index2_data_.begin() + zero_based(*itv));
			subiterator_type it_end(index2_data_.begin() + zero_based(*(itv + 1)));

			subiterator_type it(detail::lower_bound(it_begin, it_end, k_based(address2), std::less<size_type> ()));
			if (rank == 0)
				return iterator1(*this, rank, i, j, itv, it);
			if (it != it_end && zero_based(*it) == address2)
				return iterator1(*this, rank, i, j, itv, it);
			if (direction > 0) {
				if (layout_type::fast_i()) {
					if (it == it_end)
						return iterator1(*this, rank, i, j, itv, it);
					i = zero_based(*it);
				} else {
					if (i >= size1_)
						return iterator1(*this, rank, i, j, itv, it);
					++ i;
				}
			} else { /* if (direction < 0)  */
				if (layout_type::fast_i()) {
					if (it == index2_data_.begin() + zero_based(*itv))
						return iterator1(*this, rank, i, j, itv, it);
					i = zero_based(*(it - 1));
				} else {
					if (i == 0)
						return iterator1(*this, rank, i, j, itv, it);
					-- i;
				}
			}
		}
	}
	//  This function seems to be big. So we do not let the compiler inline it.
	const_iterator2 find2(int rank, size_type i, size_type j, int direction = 1) const {
		for (;;) {
			array_size_type address1(layout_type::index_M(i, j));
			array_size_type address2(layout_type::index_m(i, j));
			vector_const_subiterator_type itv(index1_data_.begin() + (std::min)(filled1_ - 1, address1));
			if (filled1_ <= address1 + 1)
				return const_iterator2(*this, rank, i, j, itv, index2_data_.begin() + filled2_);

			const_subiterator_type it_begin(index2_data_.begin() + zero_based(*itv));
			const_subiterator_type it_end(index2_data_.begin() + zero_based(*(itv + 1)));

			const_subiterator_type it(detail::lower_bound(it_begin, it_end, k_based(address2), std::less<size_type> ()));
			if (rank == 0)
				return const_iterator2(*this, rank, i, j, itv, it);
			if (it != it_end && zero_based(*it) == address2)
				return const_iterator2(*this, rank, i, j, itv, it);
			if (direction > 0) {
				if (layout_type::fast_j()) {
					if (it == it_end)
						return const_iterator2(*this, rank, i, j, itv, it);
					j = zero_based(*it);
				} else {
					if (j >= size2_)
						return const_iterator2(*this, rank, i, j, itv, it);
					++ j;
				}
			} else { /* if (direction < 0)  */
				if (layout_type::fast_j()) {
					if (it == index2_data_.begin() + zero_based(*itv))
						return const_iterator2(*this, rank, i, j, itv, it);
					j = zero_based(*(it - 1));
				} else {
					if (j == 0)
						return const_iterator2(*this, rank, i, j, itv, it);
					-- j;
				}
			}
		}
	}
	//  This function seems to be big. So we do not let the compiler inline it.
	iterator2 find2(int rank, size_type i, size_type j, int direction = 1) {
		for (;;) {
			array_size_type address1(layout_type::index_M(i, j));
			array_size_type address2(layout_type::index_m(i, j));
			vector_subiterator_type itv(index1_data_.begin() + (std::min)(filled1_ - 1, address1));
			if (filled1_ <= address1 + 1)
				return iterator2(*this, rank, i, j, itv, index2_data_.begin() + filled2_);

			subiterator_type it_begin(index2_data_.begin() + zero_based(*itv));
			subiterator_type it_end(index2_data_.begin() + zero_based(*(itv + 1)));

			subiterator_type it(detail::lower_bound(it_begin, it_end, k_based(address2), std::less<size_type> ()));
			if (rank == 0)
				return iterator2(*this, rank, i, j, itv, it);
			if (it != it_end && zero_based(*it) == address2)
				return iterator2(*this, rank, i, j, itv, it);
			if (direction > 0) {
				if (layout_type::fast_j()) {
					if (it == it_end)
						return iterator2(*this, rank, i, j, itv, it);
					j = zero_based(*it);
				} else {
					if (j >= size2_)
						return iterator2(*this, rank, i, j, itv, it);
					++ j;
				}
			} else { /* if (direction < 0)  */
				if (layout_type::fast_j()) {
					if (it == index2_data_.begin() + zero_based(*itv))
						return iterator2(*this, rank, i, j, itv, it);
					j = zero_based(*(it - 1));
				} else {
					if (j == 0)
						return iterator2(*this, rank, i, j, itv, it);
					-- j;
				}
			}
		}
	}


	class const_iterator1:
		public container_const_reference<compressed_matrix>,
		public bidirectional_iterator_base<sparse_bidirectional_iterator_tag,
			const_iterator1, value_type> {
	public:
		typedef typename compressed_matrix::value_type value_type;
		typedef typename compressed_matrix::difference_type difference_type;
		typedef typename compressed_matrix::const_reference reference;
		typedef const typename compressed_matrix::pointer pointer;

		typedef const_iterator2 dual_iterator_type;
		typedef const_reverse_iterator2 dual_reverse_iterator_type;

		// Construction and destruction

		const_iterator1():
			container_const_reference<self_type> (), rank_(), i_(), j_(), itv_(), it_() {}

		const_iterator1(const self_type &m, int rank, size_type i, size_type j, const vector_const_subiterator_type &itv, const const_subiterator_type &it):
			container_const_reference<self_type> (m), rank_(rank), i_(i), j_(j), itv_(itv), it_(it) {}

		const_iterator1(const iterator1 &it):
			container_const_reference<self_type> (it()), rank_(it.rank_), i_(it.i_), j_(it.j_), itv_(it.itv_), it_(it.it_) {}

		// Arithmetic

		const_iterator1 &operator ++ () {
			if (rank_ == 1 && layout_type::fast_i())
				++ it_;
			else {
				i_ = index1() + 1;
				if (rank_ == 1)
					*this = (*this)().find1(rank_, i_, j_, 1);
			}
			return *this;
		}

		const_iterator1 &operator -- () {
			if (rank_ == 1 && layout_type::fast_i())
				-- it_;
			else {
				--i_;
				if (rank_ == 1)
					*this = (*this)().find1(rank_, i_, j_, -1);
			}
			return *this;
		}

		// Dereference

		const_reference operator * () const {
			BOOST_UBLAS_CHECK(index1() < (*this)().size1(), bad_index());
			BOOST_UBLAS_CHECK(index2() < (*this)().size2(), bad_index());
			if (rank_ == 1) {
				return (*this)().value_data_ [it_ - (*this)().index2_data_.begin()];
			} else {
				return (*this)()(i_, j_);
			}
		}

		const_iterator2 begin() const {
			const self_type &m = (*this)();
			return m.find2(1, index1(), 0);
		}

		const_iterator2 end() const {
			const self_type &m = (*this)();
			return m.find2(1, index1(), m.size2());
		}

		const_reverse_iterator2 rbegin() const {
			return const_reverse_iterator2(end());
		}

		const_reverse_iterator2 rend() const {
			return const_reverse_iterator2(begin());
		}

		// Indices

		size_type index1() const {
			BOOST_UBLAS_CHECK(*this != (*this)().find1(0, (*this)().size1(), j_), bad_index());
			if (rank_ == 1) {
				BOOST_UBLAS_CHECK(layout_type::index_M(itv_ - (*this)().index1_data_.begin(), (*this)().zero_based(*it_)) < (*this)().size1(), bad_index());
				return layout_type::index_M(itv_ - (*this)().index1_data_.begin(), (*this)().zero_based(*it_));
			} else {
				return i_;
			}
		}

		size_type index2() const {
			if (rank_ == 1) {
				BOOST_UBLAS_CHECK(layout_type::index_m(itv_ - (*this)().index1_data_.begin(), (*this)().zero_based(*it_)) < (*this)().size2(), bad_index());
				return layout_type::index_m(itv_ - (*this)().index1_data_.begin(), (*this)().zero_based(*it_));
			} else {
				return j_;
			}
		}

		// Assignment

		const_iterator1 &operator = (const const_iterator1 &it) {
			container_const_reference<self_type>::assign(&it());
			rank_ = it.rank_;
			i_ = it.i_;
			j_ = it.j_;
			itv_ = it.itv_;
			it_ = it.it_;
			return *this;
		}

		// Comparison

		bool operator == (const const_iterator1 &it) const {
			BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
			// BOOST_UBLAS_CHECK (rank_ == it.rank_, internal_logic ());
			if (rank_ == 1 || it.rank_ == 1) {
				return it_ == it.it_;
			} else {
				return i_ == it.i_ && j_ == it.j_;
			}
		}

	private:
		int rank_;
		size_type i_;
		size_type j_;
		vector_const_subiterator_type itv_;
		const_subiterator_type it_;
	};


	const_iterator1 begin1() const {
		return find1(0, 0, 0);
	}

	const_iterator1 end1() const {
		return find1(0, size1_, 0);
	}

	class iterator1:
		public container_reference<compressed_matrix>,
		public bidirectional_iterator_base<sparse_bidirectional_iterator_tag,
			iterator1, value_type> {
	public:
		typedef typename compressed_matrix::value_type value_type;
		typedef typename compressed_matrix::difference_type difference_type;
		typedef typename compressed_matrix::true_reference reference;
		typedef typename compressed_matrix::pointer pointer;

		typedef iterator2 dual_iterator_type;
		typedef reverse_iterator2 dual_reverse_iterator_type;

		// Construction and destruction

		iterator1():
			container_reference<self_type> (), rank_(), i_(), j_(), itv_(), it_() {}

		iterator1(self_type &m, int rank, size_type i, size_type j, const vector_subiterator_type &itv, const subiterator_type &it):
			container_reference<self_type> (m), rank_(rank), i_(i), j_(j), itv_(itv), it_(it) {}

		// Arithmetic

		iterator1 &operator ++ () {
			if (rank_ == 1 && layout_type::fast_i())
				++ it_;
			else {
				i_ = index1() + 1;
				if (rank_ == 1)
					*this = (*this)().find1(rank_, i_, j_, 1);
			}
			return *this;
		}

		iterator1 &operator -- () {
			if (rank_ == 1 && layout_type::fast_i())
				-- it_;
			else {
				--i_;
				if (rank_ == 1)
					*this = (*this)().find1(rank_, i_, j_, -1);
			}
			return *this;
		}

		// Dereference

		reference operator * () const {
			BOOST_UBLAS_CHECK(index1() < (*this)().size1(), bad_index());
			BOOST_UBLAS_CHECK(index2() < (*this)().size2(), bad_index());
			if (rank_ == 1) {
				return (*this)().value_data_ [it_ - (*this)().index2_data_.begin()];
			} else {
				return (*this)().at_element(i_, j_);
			}
		}

		iterator2 begin() const {
			self_type &m = (*this)();
			return m.find2(1, index1(), 0);
		}
		iterator2 end() const {
			self_type &m = (*this)();
			return m.find2(1, index1(), m.size2());
		}
		reverse_iterator2 rbegin() const {
			return reverse_iterator2(end());
		}
		reverse_iterator2 rend() const {
			return reverse_iterator2(begin());
		}

		// Indices

		size_type index1() const {
			BOOST_UBLAS_CHECK(*this != (*this)().find1(0, (*this)().size1(), j_), bad_index());
			if (rank_ == 1) {
				BOOST_UBLAS_CHECK(layout_type::index_M(itv_ - (*this)().index1_data_.begin(), (*this)().zero_based(*it_)) < (*this)().size1(), bad_index());
				return layout_type::index_M(itv_ - (*this)().index1_data_.begin(), (*this)().zero_based(*it_));
			} else {
				return i_;
			}
		}

		size_type index2() const {
			if (rank_ == 1) {
				BOOST_UBLAS_CHECK(layout_type::index_m(itv_ - (*this)().index1_data_.begin(), (*this)().zero_based(*it_)) < (*this)().size2(), bad_index());
				return layout_type::index_m(itv_ - (*this)().index1_data_.begin(), (*this)().zero_based(*it_));
			} else {
				return j_;
			}
		}

		// Assignment

		iterator1 &operator = (const iterator1 &it) {
			container_reference<self_type>::assign(&it());
			rank_ = it.rank_;
			i_ = it.i_;
			j_ = it.j_;
			itv_ = it.itv_;
			it_ = it.it_;
			return *this;
		}

		// Comparison

		bool operator == (const iterator1 &it) const {
			BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
			// BOOST_UBLAS_CHECK (rank_ == it.rank_, internal_logic ());
			if (rank_ == 1 || it.rank_ == 1) {
				return it_ == it.it_;
			} else {
				return i_ == it.i_ && j_ == it.j_;
			}
		}

	private:
		int rank_;
		size_type i_;
		size_type j_;
		vector_subiterator_type itv_;
		subiterator_type it_;

		friend class const_iterator1;
	};


	iterator1 begin1() {
		return find1(0, 0, 0);
	}

	iterator1 end1() {
		return find1(0, size1_, 0);
	}

	class const_iterator2:
		public container_const_reference<compressed_matrix>,
		public bidirectional_iterator_base<sparse_bidirectional_iterator_tag,
			const_iterator2, value_type> {
	public:
		typedef typename compressed_matrix::value_type value_type;
		typedef typename compressed_matrix::difference_type difference_type;
		typedef typename compressed_matrix::const_reference reference;
		typedef const typename compressed_matrix::pointer pointer;

		typedef const_iterator1 dual_iterator_type;
		typedef const_reverse_iterator1 dual_reverse_iterator_type;

		// Construction and destruction

		const_iterator2():
			container_const_reference<self_type> (), rank_(), i_(), j_(), itv_(), it_() {}

		const_iterator2(const self_type &m, int rank, size_type i, size_type j, const vector_const_subiterator_type itv, const const_subiterator_type &it):
			container_const_reference<self_type> (m), rank_(rank), i_(i), j_(j), itv_(itv), it_(it) {}

		const_iterator2(const iterator2 &it):
			container_const_reference<self_type> (it()), rank_(it.rank_), i_(it.i_), j_(it.j_), itv_(it.itv_), it_(it.it_) {}

		// Arithmetic

		const_iterator2 &operator ++ () {
			if (rank_ == 1 && layout_type::fast_j())
				++ it_;
			else {
				j_ = index2() + 1;
				if (rank_ == 1)
					*this = (*this)().find2(rank_, i_, j_, 1);
			}
			return *this;
		}

		const_iterator2 &operator -- () {
			if (rank_ == 1 && layout_type::fast_j())
				-- it_;
			else {
				--j_;
				if (rank_ == 1)
					*this = (*this)().find2(rank_, i_, j_, -1);
			}
			return *this;
		}

		// Dereference

		const_reference operator * () const {
			BOOST_UBLAS_CHECK(index1() < (*this)().size1(), bad_index());
			BOOST_UBLAS_CHECK(index2() < (*this)().size2(), bad_index());
			if (rank_ == 1) {
				return (*this)().value_data_ [it_ - (*this)().index2_data_.begin()];
			} else {
				return (*this)()(i_, j_);
			}
		}

		const_iterator1 begin() const {
			const self_type &m = (*this)();
			return m.find1(1, 0, index2());
		}
		const_iterator1 end() const {
			const self_type &m = (*this)();
			return m.find1(1, m.size1(), index2());
		}
		const_reverse_iterator1 rbegin() const {
			return const_reverse_iterator1(end());
		}
		const_reverse_iterator1 rend() const {
			return const_reverse_iterator1(begin());
		}
		// Indices
		size_type index1() const {
			if (rank_ == 1) {
				BOOST_UBLAS_CHECK(layout_type::index_M(itv_ - (*this)().index1_data_.begin(), (*this)().zero_based(*it_)) < (*this)().size1(), bad_index());
				return layout_type::index_M(itv_ - (*this)().index1_data_.begin(), (*this)().zero_based(*it_));
			} else {
				return i_;
			}
		}
		size_type index2() const {
			BOOST_UBLAS_CHECK(*this != (*this)().find2(0, i_, (*this)().size2()), bad_index());
			if (rank_ == 1) {
				BOOST_UBLAS_CHECK(layout_type::index_m(itv_ - (*this)().index1_data_.begin(), (*this)().zero_based(*it_)) < (*this)().size2(), bad_index());
				return layout_type::index_m(itv_ - (*this)().index1_data_.begin(), (*this)().zero_based(*it_));
			} else {
				return j_;
			}
		}

		// Assignment
		const_iterator2 &operator = (const const_iterator2 &it) {
			container_const_reference<self_type>::assign(&it());
			rank_ = it.rank_;
			i_ = it.i_;
			j_ = it.j_;
			itv_ = it.itv_;
			it_ = it.it_;
			return *this;
		}

		// Comparison
		bool operator == (const const_iterator2 &it) const {
			BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
			// BOOST_UBLAS_CHECK (rank_ == it.rank_, internal_logic ());
			if (rank_ == 1 || it.rank_ == 1) {
				return it_ == it.it_;
			} else {
				return i_ == it.i_ && j_ == it.j_;
			}
		}

	private:
		int rank_;
		size_type i_;
		size_type j_;
		vector_const_subiterator_type itv_;
		const_subiterator_type it_;
	};


	const_iterator2 begin2() const {
		return find2(0, 0, 0);
	}

	const_iterator2 end2() const {
		return find2(0, 0, size2_);
	}

	class iterator2:
		public container_reference<compressed_matrix>,
		public bidirectional_iterator_base<sparse_bidirectional_iterator_tag,
			iterator2, value_type> {
	public:
		typedef typename compressed_matrix::value_type value_type;
		typedef typename compressed_matrix::difference_type difference_type;
		typedef typename compressed_matrix::true_reference reference;
		typedef typename compressed_matrix::pointer pointer;

		typedef iterator1 dual_iterator_type;
		typedef reverse_iterator1 dual_reverse_iterator_type;

		// Construction and destruction

		iterator2():
			container_reference<self_type> (), rank_(), i_(), j_(), itv_(), it_() {}

		iterator2(self_type &m, int rank, size_type i, size_type j, const vector_subiterator_type &itv, const subiterator_type &it):
			container_reference<self_type> (m), rank_(rank), i_(i), j_(j), itv_(itv), it_(it) {}

		// Arithmetic

		iterator2 &operator ++ () {
			if (rank_ == 1 && layout_type::fast_j())
				++ it_;
			else {
				j_ = index2() + 1;
				if (rank_ == 1)
					*this = (*this)().find2(rank_, i_, j_, 1);
			}
			return *this;
		}

		iterator2 &operator -- () {
			if (rank_ == 1 && layout_type::fast_j())
				-- it_;
			else {
				--j_;
				if (rank_ == 1)
					*this = (*this)().find2(rank_, i_, j_, -1);
			}
			return *this;
		}

		// Dereference

		reference operator * () const {
			BOOST_UBLAS_CHECK(index1() < (*this)().size1(), bad_index());
			BOOST_UBLAS_CHECK(index2() < (*this)().size2(), bad_index());
			if (rank_ == 1) {
				return (*this)().value_data_ [it_ - (*this)().index2_data_.begin()];
			} else {
				return (*this)().at_element(i_, j_);
			}
		}

		iterator1 begin() const {
			self_type &m = (*this)();
			return m.find1(1, 0, index2());
		}
		iterator1 end() const {
			self_type &m = (*this)();
			return m.find1(1, m.size1(), index2());
		}
		reverse_iterator1 rbegin() const {
			return reverse_iterator1(end());
		}
		reverse_iterator1 rend() const {
			return reverse_iterator1(begin());
		}

		// Indices

		size_type index1() const {
			if (rank_ == 1) {
				BOOST_UBLAS_CHECK(layout_type::index_M(itv_ - (*this)().index1_data_.begin(), (*this)().zero_based(*it_)) < (*this)().size1(), bad_index());
				return layout_type::index_M(itv_ - (*this)().index1_data_.begin(), (*this)().zero_based(*it_));
			} else {
				return i_;
			}
		}

		size_type index2() const {
			BOOST_UBLAS_CHECK(*this != (*this)().find2(0, i_, (*this)().size2()), bad_index());
			if (rank_ == 1) {
				BOOST_UBLAS_CHECK(layout_type::index_m(itv_ - (*this)().index1_data_.begin(), (*this)().zero_based(*it_)) < (*this)().size2(), bad_index());
				return layout_type::index_m(itv_ - (*this)().index1_data_.begin(), (*this)().zero_based(*it_));
			} else {
				return j_;
			}
		}

		// Assignment

		iterator2 &operator = (const iterator2 &it) {
			container_reference<self_type>::assign(&it());
			rank_ = it.rank_;
			i_ = it.i_;
			j_ = it.j_;
			itv_ = it.itv_;
			it_ = it.it_;
			return *this;
		}

		// Comparison

		bool operator == (const iterator2 &it) const {
			BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
			// BOOST_UBLAS_CHECK (rank_ == it.rank_, internal_logic ());
			if (rank_ == 1 || it.rank_ == 1) {
				return it_ == it.it_;
			} else {
				return i_ == it.i_ && j_ == it.j_;
			}
		}

	private:
		int rank_;
		size_type i_;
		size_type j_;
		vector_subiterator_type itv_;
		subiterator_type it_;

		friend class const_iterator2;
	};


	iterator2 begin2() {
		return find2(0, 0, 0);
	}

	iterator2 end2() {
		return find2(0, 0, size2_);
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

	// Serialization
	template<class Archive>
	void serialize(Archive &ar, const unsigned int /* file_version */) {
		boost::serialization::collection_size_type s1(size1_);
		boost::serialization::collection_size_type s2(size2_);
		ar &boost::serialization::make_nvp("size1",s1);
		ar &boost::serialization::make_nvp("size2",s2);
		if (Archive::is_loading::value) {
			size1_ = s1;
			size2_ = s2;
		}
		ar &boost::serialization::make_nvp("capacity", capacity_);
		ar &boost::serialization::make_nvp("filled1", filled1_);
		ar &boost::serialization::make_nvp("filled2", filled2_);
		ar &boost::serialization::make_nvp("index1_data", index1_data_);
		ar &boost::serialization::make_nvp("index2_data", index2_data_);
		ar &boost::serialization::make_nvp("value_data", value_data_);
		storage_invariants();
	}

private:
	void storage_invariants() const {
		BOOST_UBLAS_CHECK(layout_type::size_M(size1_, size2_) + 1 == index1_data_.size(), internal_logic());
		BOOST_UBLAS_CHECK(capacity_ == index2_data_.size(), internal_logic());
		BOOST_UBLAS_CHECK(capacity_ == value_data_.size(), internal_logic());
		BOOST_UBLAS_CHECK(filled1_ > 0 && filled1_ <= layout_type::size_M(size1_, size2_) + 1, internal_logic());
		BOOST_UBLAS_CHECK(filled2_ <= capacity_, internal_logic());
		BOOST_UBLAS_CHECK(index1_data_ [filled1_ - 1] == k_based(filled2_), internal_logic());
	}

	size_type size1_;
	size_type size2_;
	array_size_type capacity_;
	array_size_type filled1_;
	array_size_type filled2_;
	index_array_type index1_data_;
	index_array_type index2_data_;
	value_array_type value_data_;
	static const value_type zero_;


	static size_type zero_based(size_type k_based_index) {
		return k_based_index - IB;
	}

	static size_type k_based(size_type zero_based_index) {
		return zero_based_index + IB;
	}

	friend class iterator1;
	friend class iterator2;
	friend class const_iterator1;
	friend class const_iterator2;
};

template<class T, class L, std::size_t IB, class IA, class TA>
const typename compressed_matrix<T, L, IB, IA, TA>::value_type compressed_matrix<T, L, IB, IA, TA>::zero_ = value_type/*zero*/();

}
}

#endif
