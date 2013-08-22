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

#ifndef _BOOST_UBLAS_MATRIX_EXPRESSION_
#define _BOOST_UBLAS_MATRIX_EXPRESSION_

#include <shark/LinAlg/BLAS/ublas/vector_expression.hpp>

// Expression templates based on ideas of Todd Veldhuizen and Geoffrey Furnish
// Iterators based on ideas of Jeremy Siek
//
// Classes that model the Matrix Expression concept

namespace shark {
namespace blas {

template<class E>
class matrix_reference:
	public matrix_expression<matrix_reference<E> > {

	typedef matrix_reference<E> self_type;
public:
	typedef typename E::size_type size_type;
	typedef typename E::difference_type difference_type;
	typedef typename E::value_type value_type;
	typedef typename E::const_reference const_reference;
	typedef typename boost::mpl::if_<boost::is_const<E>,
	        typename E::const_reference,
	        typename E::reference>::type reference;
	typedef E referred_type;
	typedef const self_type const_closure_type;
	typedef self_type closure_type;
	typedef typename E::orientation_category orientation_category;
	typedef typename E::storage_category storage_category;

	// Construction and destruction
	
	explicit matrix_reference(referred_type &e):
		e_(e) {}

	// Accessors
	
	size_type size1() const {
		return e_.size1();
	}
	
	size_type size2() const {
		return e_.size2();
	}

public:
	// Expression accessors - const correct
	
	const referred_type &expression() const {
		return e_;
	}
	
	referred_type &expression() {
		return e_;
	}

public:
	// Element access
#ifndef BOOST_UBLAS_REFERENCE_CONST_MEMBER
	
	const_reference operator()(size_type i, size_type j) const {
		return expression()(i, j);
	}
	
	reference operator()(size_type i, size_type j) {
		return expression()(i, j);
	}
#else
	
	reference operator()(size_type i, size_type j) const {
		return expression()(i, j);
	}
#endif

	// Assignment
	
	matrix_reference &operator = (const matrix_reference &m) {
		expression().operator = (m);
		return *this;
	}
	template<class AE>
	
	matrix_reference &operator = (const matrix_expression<AE> &ae) {
		expression().operator = (ae);
		return *this;
	}
	template<class AE>
	
	matrix_reference &assign(const matrix_expression<AE> &ae) {
		expression().assign(ae);
		return *this;
	}
	template<class AE>
	
	matrix_reference &operator += (const matrix_expression<AE> &ae) {
		expression().operator += (ae);
		return *this;
	}
	template<class AE>
	
	matrix_reference &plus_assign(const matrix_expression<AE> &ae) {
		expression().plus_assign(ae);
		return *this;
	}
	template<class AE>
	
	matrix_reference &operator -= (const matrix_expression<AE> &ae) {
		expression().operator -= (ae);
		return *this;
	}
	template<class AE>
	
	matrix_reference &minus_assign(const matrix_expression<AE> &ae) {
		expression().minus_assign(ae);
		return *this;
	}
	template<class AT>
	
	matrix_reference &operator *= (const AT &at) {
		expression().operator *= (at);
		return *this;
	}
	template<class AT>
	
	matrix_reference &operator /= (const AT &at) {
		expression().operator /= (at);
		return *this;
	}

	// Swapping
	
	void swap(matrix_reference &m) {
		expression().swap(m.expression());
	}

	// Closure comparison
	
	bool same_closure(const matrix_reference &mr) const {
		return &(*this).e_ == &mr.e_;
	}

	// Iterator types
	typedef typename E::const_iterator1 const_iterator1;
	typedef typename boost::mpl::if_<boost::is_const<E>,
	        typename E::const_iterator1,
	        typename E::iterator1>::type iterator1;
	typedef typename E::const_iterator2 const_iterator2;
	typedef typename boost::mpl::if_<boost::is_const<E>,
	        typename E::const_iterator2,
	        typename E::iterator2>::type iterator2;

	// Element lookup
	
	const_iterator1 find1(int rank, size_type i, size_type j) const {
		return expression().find1(rank, i, j);
	}
	
	iterator1 find1(int rank, size_type i, size_type j) {
		return expression().find1(rank, i, j);
	}
	
	const_iterator2 find2(int rank, size_type i, size_type j) const {
		return expression().find2(rank, i, j);
	}
	
	iterator2 find2(int rank, size_type i, size_type j) {
		return expression().find2(rank, i, j);
	}

	// Iterators are the iterators of the referenced expression.

	
	const_iterator1 begin1() const {
		return expression().begin1();
	}
	
	const_iterator1 end1() const {
		return expression().end1();
	}

	
	iterator1 begin1() {
		return expression().begin1();
	}
	
	iterator1 end1() {
		return expression().end1();
	}

	
	const_iterator2 begin2() const {
		return expression().begin2();
	}
	
	const_iterator2 end2() const {
		return expression().end2();
	}

	
	iterator2 begin2() {
		return expression().begin2();
	}
	
	iterator2 end2() {
		return expression().end2();
	}

	// Reverse iterators
	typedef reverse_iterator_base1<const_iterator1> const_reverse_iterator1;
	typedef reverse_iterator_base1<iterator1> reverse_iterator1;

	
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

	typedef reverse_iterator_base2<const_iterator2> const_reverse_iterator2;
	typedef reverse_iterator_base2<iterator2> reverse_iterator2;

	
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
	referred_type &e_;
};


template<class E1, class E2, class F>
class vector_matrix_binary:
	public matrix_expression<vector_matrix_binary<E1, E2, F> > {

	typedef E1 expression1_type;
	typedef E2 expression2_type;
public:
	typedef typename E1::const_closure_type expression1_closure_type;
	typedef typename E2::const_closure_type expression2_closure_type;
private:
	typedef vector_matrix_binary<E1, E2, F> self_type;
public:
	typedef F functor_type;
	typedef typename promote_traits<typename E1::size_type, typename E2::size_type>::promote_type size_type;
	typedef typename promote_traits<typename E1::difference_type, typename E2::difference_type>::promote_type difference_type;
	typedef typename F::result_type value_type;
	typedef value_type const_reference;
	typedef const_reference reference;
	typedef const self_type const_closure_type;
	typedef const_closure_type closure_type;
	typedef unknown_orientation_tag orientation_category;
	typedef unknown_storage_tag storage_category;

	// Construction and destruction
	
	vector_matrix_binary(const expression1_type &e1, const expression2_type &e2):
		m_expression1(e1), m_expression2(e2) {}

	// Accessors
	
	size_type size1() const {
		return m_expression1.size();
	}
	
	size_type size2() const {
		return m_expression2.size();
	}

public:
	// Expression accessors
	
	const expression1_closure_type &expression1() const {
		return m_expression1;
	}
	
	const expression2_closure_type &expression2() const {
		return m_expression2;
	}

public:
	// Element access
	
	const_reference operator()(size_type i, size_type j) const {
		functor_type f;
		return f(m_expression1(i), m_expression2(j));
	}

	// Closure comparison
	
	bool same_closure(const vector_matrix_binary &vmb) const {
		return (*this).expression1().same_closure(vmb.expression1()) &&
		       (*this).expression2().same_closure(vmb.expression2());
	}

	// Iterator types
private:
	typedef typename E1::const_iterator const_subiterator1_type;
	typedef typename E2::const_iterator const_subiterator2_type;
	typedef const value_type *const_pointer;

public:
#ifdef BOOST_UBLAS_USE_INDEXED_ITERATOR
	typedef typename iterator_restrict_traits<typename const_subiterator1_type::iterator_category,
	        typename const_subiterator2_type::iterator_category>::iterator_category iterator_category;
	typedef indexed_const_iterator1<const_closure_type, iterator_category> const_iterator1;
	typedef const_iterator1 iterator1;
	typedef indexed_const_iterator2<const_closure_type, iterator_category> const_iterator2;
	typedef const_iterator2 iterator2;
#else
	class const_iterator1;
	typedef const_iterator1 iterator1;
	class const_iterator2;
	typedef const_iterator2 iterator2;
#endif
	typedef reverse_iterator_base1<const_iterator1> const_reverse_iterator1;
	typedef reverse_iterator_base2<const_iterator2> const_reverse_iterator2;

	// Element lookup
	
	const_iterator1 find1(int rank, size_type i, size_type j) const {
		const_subiterator1_type it1(m_expression1.find(i));
		const_subiterator1_type it1_end(m_expression1.find(size1()));
		const_subiterator2_type it2(m_expression2.find(j));
		const_subiterator2_type it2_end(m_expression2.find(size2()));
		if (it2 == it2_end || (rank == 1 && (it2.index() != j || *it2 == value_type/*zero*/()))) {
			it1 = it1_end;
			it2 = it2_end;
		}
#ifdef BOOST_UBLAS_USE_INDEXED_ITERATOR
		return const_iterator1(*this, it1.index(), it2.index());
#else
#ifdef BOOST_UBLAS_USE_INVARIANT_HOISTING
		return const_iterator1(*this, it1, it2, it2 != it2_end ? *it2 : value_type/*zero*/());
#else
		return const_iterator1(*this, it1, it2);
#endif
#endif
	}
	
	const_iterator2 find2(int rank, size_type i, size_type j) const {
		const_subiterator2_type it2(m_expression2.find(j));
		const_subiterator2_type it2_end(m_expression2.find(size2()));
		const_subiterator1_type it1(m_expression1.find(i));
		const_subiterator1_type it1_end(m_expression1.find(size1()));
		if (it1 == it1_end || (rank == 1 && (it1.index() != i || *it1 == value_type/*zero*/()))) {
			it2 = it2_end;
			it1 = it1_end;
		}
#ifdef BOOST_UBLAS_USE_INDEXED_ITERATOR
		return const_iterator2(*this, it1.index(), it2.index());
#else
#ifdef BOOST_UBLAS_USE_INVARIANT_HOISTING
		return const_iterator2(*this, it1, it2, it1 != it1_end ? *it1 : value_type/*zero*/());
#else
		return const_iterator2(*this, it1, it2);
#endif
#endif
	}

	// Iterators enhance the iterators of the referenced expressions
	// with the binary functor.

#ifndef BOOST_UBLAS_USE_INDEXED_ITERATOR
	class const_iterator1:
		public container_const_reference<vector_matrix_binary>,
		public iterator_base_traits<typename iterator_restrict_traits<typename E1::const_iterator::iterator_category,
		typename E2::const_iterator::iterator_category>::iterator_category>::template
			iterator_base<const_iterator1, value_type>::type {
    public:
	    typedef typename iterator_restrict_traits<typename E1::const_iterator::iterator_category,
	    typename E2::const_iterator::iterator_category>::iterator_category iterator_category;
	    typedef typename vector_matrix_binary::difference_type difference_type;
	    typedef typename vector_matrix_binary::value_type value_type;
	    typedef typename vector_matrix_binary::const_reference reference;
	    typedef typename vector_matrix_binary::const_pointer pointer;

	    typedef const_iterator2 dual_iterator_type;
	    typedef const_reverse_iterator2 dual_reverse_iterator_type;

	    // Construction and destruction
#ifdef BOOST_UBLAS_USE_INVARIANT_HOISTING
	    
	    const_iterator1():
		    container_const_reference<self_type> (), it1_(), it2_(), t2_() {}
	
	const_iterator1(const self_type &vmb, const const_subiterator1_type &it1, const const_subiterator2_type &it2, value_type t2):
		container_const_reference<self_type> (vmb), it1_(it1), it2_(it2), t2_(t2) {}
#else
	    
	    const_iterator1():
		    container_const_reference<self_type> (), it1_(), it2_() {}
	
	const_iterator1(const self_type &vmb, const const_subiterator1_type &it1, const const_subiterator2_type &it2):
		container_const_reference<self_type> (vmb), it1_(it1), it2_(it2) {}
#endif

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
		BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());
		BOOST_UBLAS_CHECK(it2_ == it.it2_, external_logic());
		return it1_ - it.it1_;
	}

	// Dereference
	
	const_reference operator * () const {
		functor_type f;
		return f(*it1_, *it2_);
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
		return it1_.index();
	}
	
	size_type  index2() const {
		return it2_.index();
	}

	// Assignment
	
	const_iterator1 &operator = (const const_iterator1 &it) {
		container_const_reference<self_type>::assign(&it());
		it1_ = it.it1_;
		it2_ = it.it2_;
#ifdef BOOST_UBLAS_USE_INVARIANT_HOISTING
		t2_ = it.t2_;
#endif
		return *this;
	}

	// Comparison
	
	bool operator == (const const_iterator1 &it) const {
		BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());
		BOOST_UBLAS_CHECK(it2_ == it.it2_, external_logic());
		return it1_ == it.it1_;
	}
	
	bool operator < (const const_iterator1 &it) const {
		BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());
		BOOST_UBLAS_CHECK(it2_ == it.it2_, external_logic());
		return it1_ < it.it1_;
	}

private:
#ifdef BOOST_UBLAS_USE_INVARIANT_HOISTING
	const_subiterator1_type it1_;
	// Mutable due to assignment
	/* const */
	const_subiterator2_type it2_;
	value_type t2_;
#else
	const_subiterator1_type it1_;
	const_subiterator2_type it2_;
#endif
	};
#endif

	
	const_iterator1 begin1() const {
		return find1(0, 0, 0);
	}
	
	const_iterator1 end1() const {
		return find1(0, size1(), 0);
	}

#ifndef BOOST_UBLAS_USE_INDEXED_ITERATOR
	class const_iterator2:
		public container_const_reference<vector_matrix_binary>,
		public iterator_base_traits<typename iterator_restrict_traits<typename E1::const_iterator::iterator_category,
		typename E2::const_iterator::iterator_category>::iterator_category>::template
			iterator_base<const_iterator2, value_type>::type {
    public:
	    typedef typename iterator_restrict_traits<typename E1::const_iterator::iterator_category,
	    typename E2::const_iterator::iterator_category>::iterator_category iterator_category;
	    typedef typename vector_matrix_binary::difference_type difference_type;
	    typedef typename vector_matrix_binary::value_type value_type;
	    typedef typename vector_matrix_binary::const_reference reference;
	    typedef typename vector_matrix_binary::const_pointer pointer;

	    typedef const_iterator1 dual_iterator_type;
	    typedef const_reverse_iterator1 dual_reverse_iterator_type;

	    // Construction and destruction
#ifdef BOOST_UBLAS_USE_INVARIANT_HOISTING
	    
	    const_iterator2():
		    container_const_reference<self_type> (), it1_(), it2_(), t1_() {}
	
	const_iterator2(const self_type &vmb, const const_subiterator1_type &it1, const const_subiterator2_type &it2, value_type t1):
		container_const_reference<self_type> (vmb), it1_(it1), it2_(it2), t1_(t1) {}
#else
	    
	    const_iterator2():
		    container_const_reference<self_type> (), it1_(), it2_() {}
	
	const_iterator2(const self_type &vmb, const const_subiterator1_type &it1, const const_subiterator2_type &it2):
		container_const_reference<self_type> (vmb), it1_(it1), it2_(it2) {}
#endif

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
		BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());
		BOOST_UBLAS_CHECK(it1_ == it.it1_, external_logic());
		return it2_ - it.it2_;
	}

	// Dereference
	
	const_reference operator * () const {
		functor_type f;
		return f(*it1_, *it2_);

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
		return it1_.index();
	}
	
	size_type  index2() const {
		return it2_.index();
	}

	// Assignment
	
	const_iterator2 &operator = (const const_iterator2 &it) {
		container_const_reference<self_type>::assign(&it());
		it1_ = it.it1_;
		it2_ = it.it2_;
#ifdef BOOST_UBLAS_USE_INVARIANT_HOISTING
		t1_ = it.t1_;
#endif
		return *this;
	}

	// Comparison
	
	bool operator == (const const_iterator2 &it) const {
		BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());
		BOOST_UBLAS_CHECK(it1_ == it.it1_, external_logic());
		return it2_ == it.it2_;
	}
	
	bool operator < (const const_iterator2 &it) const {
		BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());
		BOOST_UBLAS_CHECK(it1_ == it.it1_, external_logic());
		return it2_ < it.it2_;
	}

private:
#ifdef BOOST_UBLAS_USE_INVARIANT_HOISTING
	// Mutable due to assignment
	/* const */ const_subiterator1_type it1_;
	const_subiterator2_type it2_;
	value_type t1_;
#else
	const_subiterator1_type it1_;
	const_subiterator2_type it2_;
#endif
	};
#endif

	
	const_iterator2 begin2() const {
		return find2(0, 0, 0);
	}
	
	const_iterator2 end2() const {
		return find2(0, 0, size2());
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
	expression1_closure_type m_expression1;
	expression2_closure_type m_expression2;
};

template<class E1, class E2, class F>
struct vector_matrix_binary_traits {
	typedef vector_matrix_binary<E1, E2, F> expression_type;
#ifndef BOOST_UBLAS_SIMPLE_ET_DEBUG
	typedef expression_type result_type;
#else
	// ISSUE matrix is arbitary temporary type
	typedef matrix<typename F::value_type> result_type;
#endif
};

// (outer_prod (v1, v2)) [i] [j] = v1 [i] * v2 [j]
template<class E1, class E2>

typename vector_matrix_binary_traits<E1, E2, scalar_binary_multiply<typename E1::value_type, typename E2::value_type> >::result_type
outer_prod(const vector_expression<E1> &e1,
        const vector_expression<E2> &e2) {
	typedef typename vector_matrix_binary_traits<E1, E2, scalar_binary_multiply<typename E1::value_type, typename E2::value_type> >::expression_type expression_type;
	return expression_type(e1(), e2());
}

///\brief class which allows for matrix transformations
///
///transforms a matrix expression e of type E using a Functiof f of type F as an elementwise transformation f(e(i,j))
///This transformation needs to leave f constant, meaning that applying f(x), f(y), f(z) yields the same results independent of the
///order of application. Also F must provide a type F::result_type indicating the result type of the functor.
///F must further provide a boolean flag F::zero_identity which indicates that f(0) = 0. This is needed for correct usage with sparse
///arguments - if f(0) != 0 this expression will be dense!
///todo: desification is not implemented
template<class E, class F>
class matrix_unary1:
	public blas::matrix_expression<matrix_unary1<E, F> > {
private:
	typedef matrix_unary1<E, F> self_type;
	typedef E expression_type;
	typedef typename expression_type::const_iterator1 const_subiterator1_type;
	typedef typename expression_type::const_iterator2 const_subiterator2_type;

public:
	typedef typename expression_type::const_closure_type expression_closure_type;

	typedef F functor_type;
	typedef typename functor_type::result_type value_type;
	typedef value_type const_reference;
	typedef const_reference reference;
	typedef value_type const *const_pointer;
	typedef value_type *pointer;
	typedef typename expression_type::size_type size_type;
	typedef typename expression_type::difference_type difference_type;

	typedef self_type const const_closure_type;
	typedef const_closure_type closure_type;
	typedef typename E::orientation_category orientation_category;
	typedef blas::unknown_storage_tag storage_category;

	// Construction and destruction
	matrix_unary1(blas::matrix_expression<E> const &e, F const &functor):
		m_expression(e()), m_functor(functor) {}

	// Accessors
	size_type size1() const {
		return m_expression.size1();
	}
	size_type size2() const {
		return m_expression.size2();
	}

public:
	// Element access
	const_reference operator()(size_type i, size_type j) const {
		return m_functor(m_expression(i, j));
	}

	// Closure comparison
	bool same_closure(matrix_unary1 const &other) const {
		return m_expression.same_closure(other.m_expression);
	}

	// Iterator types
	class const_iterator1;
	class const_iterator2;
	typedef const_iterator1 iterator1;
	typedef const_iterator2 iterator2;

	typedef blas::reverse_iterator_base1<const_iterator1> const_reverse_iterator1;
	typedef blas::reverse_iterator_base2<const_iterator2> const_reverse_iterator2;
	typedef const_reverse_iterator1 reverse_iterator1;
	typedef const_reverse_iterator2 reverse_iterator2;

	// Element lookup
	const_iterator1 find1(int rank, size_type i, size_type j) const {
		const_subiterator1_type it1(m_expression.find1(rank, i, j));
		return const_iterator1(*this, it1,i,j);

	}
	const_iterator2 find2(int rank, size_type i, size_type j) const {
		const_subiterator2_type it2(m_expression.find2(rank, i, j));
		return const_iterator2(*this, it2,i,j);
	}

	class const_iterator1:
		public blas::container_const_reference<matrix_unary1>,
		public blas::iterator_base_traits<typename const_subiterator1_type::iterator_category>::template
			iterator_base<const_iterator1, value_type>::type {
	public:
		typedef typename const_subiterator1_type::iterator_category iterator_category;
		typedef typename matrix_unary1::difference_type difference_type;
		typedef typename matrix_unary1::value_type value_type;
		typedef typename matrix_unary1::const_reference reference;
		typedef typename matrix_unary1::const_pointer pointer;

		typedef const_iterator2 dual_iterator_type;
		typedef const_reverse_iterator2 dual_reverse_iterator_type;

		// Construction and destruction
		const_iterator1():
			blas::container_const_reference<self_type> (), m_position() {}
		const_iterator1(self_type const &ref, const_subiterator1_type const &it,size_type, size_type):
			blas::container_const_reference<self_type> (ref), m_position(it) {}

		// Arithmetic
		const_iterator1 &operator ++ () {
			++ m_position;
			return *this;
		}
		const_iterator1 &operator -- () {
			-- m_position;
			return *this;
		}
		const_iterator1 &operator += (difference_type n) {
			m_position += n;
			return *this;
		}
		const_iterator1 &operator -= (difference_type n) {
			m_position -= n;
			return *this;
		}
		difference_type operator - (const_iterator1 const &it) const {
			BOOST_UBLAS_CHECK((*this)().same_closure(it()), blas::external_logic());
			return m_position - it.m_position;
		}

		// Dereference
		const_reference operator * () const {
			return (*this)().m_functor(*m_position);
		}
		const_reference operator [](difference_type n) const {
			return *(*this + n);
		}

		dual_iterator_type begin() const {
			return (*this)().find2(1, index1(), 0);
		}
		dual_iterator_type end() const {
			return (*this)().find2(1, index1(), (*this)().size2());
		}
		dual_reverse_iterator_type rbegin() const {
			return const_reverse_iterator2(end());
		}
		dual_reverse_iterator_type rend() const {
			return const_reverse_iterator2(begin());
		}

		// Indices
		size_type index1() const {
			return m_position.index1();
		}
		size_type index2() const {
			return m_position.index2();
		}

		// Assignment
		const_iterator1 &operator = (const_iterator1 const &it) {
			blas::container_const_reference<self_type>::assign(&it());
			m_position = it.m_position;
			return *this;
		}

		// Comparison
		bool operator == (const_iterator1 const &it) const {
			BOOST_UBLAS_CHECK((*this)().same_closure(it()), blas::external_logic());
			return m_position == it.m_position;
		}
		bool operator < (const_iterator1 const &it) const {
			BOOST_UBLAS_CHECK((*this)().same_closure(it()), blas::external_logic());
			return m_position < it.m_position;
		}

	private:
		const_subiterator1_type m_position;
	};

	class const_iterator2:
		public blas::container_const_reference<matrix_unary1>,
		public blas::iterator_base_traits<typename E::const_iterator2::iterator_category>::template
			iterator_base<const_iterator2, value_type>::type {
	public:
		typedef typename E::const_iterator2::iterator_category iterator_category;
		typedef typename matrix_unary1::difference_type difference_type;
		typedef typename matrix_unary1::value_type value_type;
		typedef typename matrix_unary1::const_reference reference;
		typedef typename matrix_unary1::const_pointer pointer;

		typedef const_iterator1 dual_iterator_type;
		typedef const_reverse_iterator1 dual_reverse_iterator_type;

		// Construction and destruction
		const_iterator2():
			blas::container_const_reference<self_type> (), m_position() {}
		const_iterator2(self_type const &ref, const_subiterator2_type const &it,size_type, size_type):
			blas::container_const_reference<self_type> (ref), m_position(it) {}

		// Arithmetic
		const_iterator2 &operator ++ () {
			++ m_position;
			return *this;
		}
		const_iterator2 &operator -- () {
			-- m_position;
			return *this;
		}
		const_iterator2 &operator += (difference_type n) {
			m_position += n;
			return *this;
		}
		const_iterator2 &operator -= (difference_type n) {
			m_position -= n;
			return *this;
		}
		difference_type operator - (const_iterator2 const &it) const {
			BOOST_UBLAS_CHECK((*this)().same_closure(it()),blas:: external_logic());
			return m_position - it.m_position;
		}

		// Dereference
		const_reference operator * () const {
			return (*this)().m_functor(*m_position);
		}
		const_reference operator [](difference_type n) const {
			return *(*this + n);
		}

		dual_iterator_type begin() const {
			return (*this)().find1(1, 0, index2());
		}

		dual_iterator_type end() const {
			return (*this)().find1(1, (*this)().size1(), index2());
		}

		dual_reverse_iterator_type rbegin() const {
			return const_reverse_iterator1(end());
		}
		dual_reverse_iterator_type rend() const {
			return const_reverse_iterator1(begin());
		}


		// Indices
		size_type index1() const {
			return m_position.index1();
		}
		size_type index2() const {
			return m_position.index2();
		}

		// Assignment
		const_iterator2 &operator = (const_iterator2 const &it) {
			blas:: container_const_reference<self_type>::assign(&it());
			m_position = it.m_position;
			return *this;
		}

		// Comparison
		bool operator == (const_iterator2 const &it) const {
			BOOST_UBLAS_CHECK((*this)().same_closure(it()), blas::external_logic());
			return m_position == it.m_position;
		}
		bool operator < (const_iterator2 const &it) const {
			BOOST_UBLAS_CHECK((*this)().same_closure(it()), blas::external_logic());
			return m_position < it.m_position;
		}

	private:
		const_subiterator2_type m_position;
	};

public:
	const_iterator1 begin1() const {
		return find1(0, 0, 0);
	}
	const_iterator1 end1() const {
		return find1(0, size1(), 0);
	}

	const_iterator2 begin2() const {
		return find2(0, 0, 0);
	}
	const_iterator2 end2() const {
		return find2(0, 0, size2());
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
	expression_closure_type m_expression;
	functor_type m_functor;
};

#define SHARK_UNARY_MATRIX_TRANSFORMATION(name, F)\
template<class E>\
matrix_unary1<E,F<typename E::value_type> >\
name(matrix_expression<E> const& e){\
	typedef F<typename E::value_type> functor_type;\
	return matrix_unary1<E, functor_type>(e, functor_type());\
}
SHARK_UNARY_MATRIX_TRANSFORMATION(operator-, scalar_negate)
SHARK_UNARY_MATRIX_TRANSFORMATION(conj, scalar_conj)
SHARK_UNARY_MATRIX_TRANSFORMATION(real, scalar_real)
SHARK_UNARY_MATRIX_TRANSFORMATION(imag, scalar_imag)
SHARK_UNARY_MATRIX_TRANSFORMATION(abs, scalar_abs)
SHARK_UNARY_MATRIX_TRANSFORMATION(log, scalar_log)
SHARK_UNARY_MATRIX_TRANSFORMATION(exp, scalar_exp)
SHARK_UNARY_MATRIX_TRANSFORMATION(tanh,scalar_tanh)
SHARK_UNARY_MATRIX_TRANSFORMATION(sqr, scalar_sqr)
SHARK_UNARY_MATRIX_TRANSFORMATION(sqrt, scalar_sqrt)
SHARK_UNARY_MATRIX_TRANSFORMATION(sigmoid, scalar_sigmoid)
SHARK_UNARY_MATRIX_TRANSFORMATION(softPlus, scalar_soft_plus)
#undef SHARK_UNARY_MATRIX_TRANSFORMATION

#define SHARK_MATRIX_SCALAR_TRANSFORMATION(name, F)\
template<class E> \
matrix_unary1<E,F<typename E::value_type> > \
name (matrix_expression<E> const& e, typename E::value_type scalar){ \
	typedef F<typename E::value_type> functor_type; \
	return matrix_unary1<E, functor_type>(e, functor_type(scalar)); \
}
SHARK_MATRIX_SCALAR_TRANSFORMATION(operator*, scalar_multiply2)
SHARK_MATRIX_SCALAR_TRANSFORMATION(operator/, scalar_divide)
SHARK_MATRIX_SCALAR_TRANSFORMATION(operator<, scalar_less_than)
SHARK_MATRIX_SCALAR_TRANSFORMATION(operator<=, scalar_less_equal_than)
SHARK_MATRIX_SCALAR_TRANSFORMATION(operator>, scalar_bigger_than)
SHARK_MATRIX_SCALAR_TRANSFORMATION(operator>=, scalar_bigger_equal_than)
SHARK_MATRIX_SCALAR_TRANSFORMATION(operator==, scalar_equal)
SHARK_MATRIX_SCALAR_TRANSFORMATION(operator!=, scalar_not_equal)
#undef SHARK_MATRIX_SCALAR_TRANSFORMATION

// (t * v) [i] = t * v [i]
template<class T, class E>
typename boost::enable_if<
boost::is_convertible<T, typename E::value_type >,
      matrix_unary1<E,scalar_multiply1<typename E::value_type> >
      >::type
operator * (T scalar, matrix_expression<E> const &e) {
	typedef scalar_multiply1<typename E::value_type> functor_type;
	return matrix_unary1<E, functor_type>(e, functor_type(scalar));
}

// pow(v,t)[i,j]= pow(v[i,j],t)
template<class E, class U>
matrix_unary1<E,scalar_pow<typename E::value_type, U> >
pow (matrix_expression<E> const& e, U exponent){
	typedef scalar_pow<typename E::value_type, U> functor_type;
	return matrix_unary1<E, functor_type>(e, functor_type(exponent));
}

template<class E, class F>
class matrix_unary2:public matrix_expression<matrix_unary2<E, F> > {
public:

	typedef typename boost::mpl::if_<boost::is_same<F, scalar_identity<typename E::value_type> >,
	        E,
	        const E>::type expression_type;
	typedef F functor_type;

	typedef typename boost::mpl::if_<boost::is_const<expression_type>,
	        typename E::const_closure_type,
	        typename E::closure_type>::type expression_closure_type;
private:
	typedef matrix_unary2<E, F> self_type;
public:
	typedef typename E::size_type size_type;
	typedef typename E::difference_type difference_type;
	typedef typename F::result_type value_type;
	typedef value_type const_reference;
	typedef typename boost::mpl::if_<boost::is_same<F, scalar_identity<value_type> >,
	        typename E::reference,
	        value_type>::type reference;

	typedef const self_type const_closure_type;
	typedef self_type closure_type;
	typedef typename boost::mpl::if_<boost::is_same<typename E::orientation_category,
	        row_major_tag>,
	        column_major_tag,
	        typename boost::mpl::if_<boost::is_same<typename E::orientation_category,
	        column_major_tag>,
	        row_major_tag,
	        typename E::orientation_category>::type>::type orientation_category;
	typedef typename E::storage_category storage_category;

	// Construction and destruction
	
	// matrix_unary2 may be used as mutable expression -
	// this is the only non const expression constructor
	explicit matrix_unary2(expression_type &e):
		e_(e) {}

	// Accessors
	
	size_type size1() const {
		return e_.size2();
	}
	
	size_type size2() const {
		return e_.size1();
	}

public:
	// Expression accessors
	
	const expression_closure_type &expression() const {
		return e_;
	}

public:
	// Element access
	
	const_reference operator()(size_type i, size_type j) const {
		functor_type f;
		return f(e_(j, i));
	}
	
	reference operator()(size_type i, size_type j) {
		BOOST_STATIC_ASSERT((boost::is_same<functor_type, scalar_identity<value_type > >::value));
		return e_(j, i);
	}

	// Closure comparison
	
	bool same_closure(const matrix_unary2 &mu2) const {
		return (*this).expression().same_closure(mu2.expression());
	}

	// Iterator types
private:
	typedef typename E::const_iterator1 const_subiterator2_type;
	typedef typename E::const_iterator2 const_subiterator1_type;
	typedef const value_type *const_pointer;

public:
#ifdef BOOST_UBLAS_USE_INDEXED_ITERATOR
	typedef indexed_const_iterator1<const_closure_type, typename const_subiterator1_type::iterator_category> const_iterator1;
	typedef const_iterator1 iterator1;
	typedef indexed_const_iterator2<const_closure_type, typename const_subiterator2_type::iterator_category> const_iterator2;
	typedef const_iterator2 iterator2;
#else
	class const_iterator1;
	typedef const_iterator1 iterator1;
	class const_iterator2;
	typedef const_iterator2 iterator2;
#endif
	typedef reverse_iterator_base1<const_iterator1> const_reverse_iterator1;
	typedef reverse_iterator_base2<const_iterator2> const_reverse_iterator2;

	// Element lookup
	
	const_iterator1 find1(int rank, size_type i, size_type j) const {
		const_subiterator1_type it1(e_.find2(rank, j, i));
#ifdef BOOST_UBLAS_USE_INDEXED_ITERATOR
		return const_iterator1(*this, it1.index2(), it1.index1());
#else
		return const_iterator1(*this, it1);
#endif
	}
	
	const_iterator2 find2(int rank, size_type i, size_type j) const {
		const_subiterator2_type it2(e_.find1(rank, j, i));
#ifdef BOOST_UBLAS_USE_INDEXED_ITERATOR
		return const_iterator2(*this, it2.index2(), it2.index1());
#else
		return const_iterator2(*this, it2);
#endif
	}

	// Iterators enhance the iterators of the referenced expression
	// with the unary functor.

#ifndef BOOST_UBLAS_USE_INDEXED_ITERATOR
	class const_iterator1:
		public container_const_reference<matrix_unary2>,
		public iterator_base_traits<typename E::const_iterator2::iterator_category>::template
			iterator_base<const_iterator1, value_type>::type {
    public:
	    typedef typename E::const_iterator2::iterator_category iterator_category;
	    typedef typename matrix_unary2::difference_type difference_type;
	    typedef typename matrix_unary2::value_type value_type;
	    typedef typename matrix_unary2::const_reference reference;
	    typedef typename matrix_unary2::const_pointer pointer;

	    typedef const_iterator2 dual_iterator_type;
	    typedef const_reverse_iterator2 dual_reverse_iterator_type;

	    // Construction and destruction
	    
	    const_iterator1():
		    container_const_reference<self_type> (), it_() {}
	
	const_iterator1(const self_type &mu, const const_subiterator1_type &it):
		container_const_reference<self_type> (mu), it_(it) {}

	// Arithmetic
	
	const_iterator1 &operator ++ () {
		++ it_;
		return *this;
	}
	
	const_iterator1 &operator -- () {
		-- it_;
		return *this;
	}
	
	const_iterator1 &operator += (difference_type n) {
		it_ += n;
		return *this;
	}
	
	const_iterator1 &operator -= (difference_type n) {
		it_ -= n;
		return *this;
	}
	
	difference_type operator - (const const_iterator1 &it) const {
		BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());
		return it_ - it.it_;
	}

	// Dereference
	
	const_reference operator * () const {
		functor_type f;
		return f(*it_);
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
		return it_.index2();
	}
	
	size_type index2() const {
		return it_.index1();
	}

	// Assignment
	
	const_iterator1 &operator = (const const_iterator1 &it) {
		container_const_reference<self_type>::assign(&it());
		it_ = it.it_;
		return *this;
	}

	// Comparison
	
	bool operator == (const const_iterator1 &it) const {
		BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());
		return it_ == it.it_;
	}
	
	bool operator < (const const_iterator1 &it) const {
		BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());
		return it_ < it.it_;
	}

private:
	const_subiterator1_type it_;
	};
#endif

	
	const_iterator1 begin1() const {
		return find1(0, 0, 0);
	}
	
	const_iterator1 end1() const {
		return find1(0, size1(), 0);
	}

#ifndef BOOST_UBLAS_USE_INDEXED_ITERATOR
	class const_iterator2:
		public container_const_reference<matrix_unary2>,
		public iterator_base_traits<typename E::const_iterator1::iterator_category>::template
			iterator_base<const_iterator2, value_type>::type {
    public:
	    typedef typename E::const_iterator1::iterator_category iterator_category;
	    typedef typename matrix_unary2::difference_type difference_type;
	    typedef typename matrix_unary2::value_type value_type;
	    typedef typename matrix_unary2::const_reference reference;
	    typedef typename matrix_unary2::const_pointer pointer;

	    typedef const_iterator1 dual_iterator_type;
	    typedef const_reverse_iterator1 dual_reverse_iterator_type;

	    // Construction and destruction
	    
	    const_iterator2():
		    container_const_reference<self_type> (), it_() {}
	
	const_iterator2(const self_type &mu, const const_subiterator2_type &it):
		container_const_reference<self_type> (mu), it_(it) {}

	// Arithmetic
	
	const_iterator2 &operator ++ () {
		++ it_;
		return *this;
	}
	
	const_iterator2 &operator -- () {
		-- it_;
		return *this;
	}
	
	const_iterator2 &operator += (difference_type n) {
		it_ += n;
		return *this;
	}
	
	const_iterator2 &operator -= (difference_type n) {
		it_ -= n;
		return *this;
	}
	
	difference_type operator - (const const_iterator2 &it) const {
		BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());
		return it_ - it.it_;
	}

	// Dereference
	
	const_reference operator * () const {
		functor_type f;
		return f(*it_);
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
		return it_.index2();
	}
	
	size_type index2() const {
		return it_.index1();
	}

	// Assignment
	
	const_iterator2 &operator = (const const_iterator2 &it) {
		container_const_reference<self_type>::assign(&it());
		it_ = it.it_;
		return *this;
	}

	// Comparison
	
	bool operator == (const const_iterator2 &it) const {
		BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());
		return it_ == it.it_;
	}
	
	bool operator < (const const_iterator2 &it) const {
		BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());
		return it_ < it.it_;
	}

private:
	const_subiterator2_type it_;
	};
#endif

	
	const_iterator2 begin2() const {
		return find2(0, 0, 0);
	}
	
	const_iterator2 end2() const {
		return find2(0, 0, size2());
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
	expression_closure_type e_;
};

template<class E, class F>
struct matrix_unary2_traits {
	typedef matrix_unary2<E, F> expression_type;
#ifndef BOOST_UBLAS_SIMPLE_ET_DEBUG
	typedef expression_type result_type;
#else
	typedef typename E::matrix_temporary_type result_type;
#endif
};

template<class E>
struct matrix_transpose: public matrix_expression<matrix_transpose<E> >{
private:
	typedef matrix_unary2<E, scalar_identity<typename E::value_type> > Transpose;
	typedef typename Transpose::closure_type TransposeClosure;
public:

	typedef typename Transpose::size_type size_type;
	typedef typename Transpose::difference_type difference_type;
	typedef typename Transpose::value_type value_type;
	typedef typename Transpose::const_reference const_reference;
	typedef typename Transpose::reference reference;

	typedef matrix_transpose<E> const const_closure_type;
	typedef matrix_transpose<E> closure_type;
	typedef typename Transpose::orientation_category orientation_category;
	typedef typename Transpose::storage_category storage_category;

	typedef typename Transpose::expression_type expression_type;
	typedef typename Transpose::expression_closure_type expression_closure_type;

	matrix_transpose(expression_type& expression):m_expression(expression){}
		
	// Accessors
	size_type size1() const {
		return m_expression.size1();
	}
	size_type size2() const {
		return m_expression.size2();
	}

	// Expression accessors
	expression_closure_type const& expression() const {
		return m_expression.expression();
	}
	//~ expression_closure_type& expression(){
		//~ return m_expression.expression();
	//~ }
	
	// Element access
	const_reference operator()(size_type i, size_type j) const{
		return m_expression(i, j);
	}
	reference operator()(size_type i, size_type j) {
		return m_expression(i, j);
	}

	// Closure comparison
	bool same_closure(matrix_transpose const& mu2) const {
		return (*this).expression().same_closure(mu2.expression());
	}

	// Iterator types
	typedef typename Transpose::const_iterator1 const_iterator1;
	typedef typename Transpose::const_iterator1 iterator1;
	typedef typename Transpose::const_iterator2 const_iterator2;
	typedef typename Transpose::const_iterator2 iterator2;
	
	// Element lookup
	const_iterator1 find1(int rank, size_type i, size_type j) const {
		SIZE_CHECK(i <= size1());
		SIZE_CHECK(j <= size2());
		return m_expression.find1(rank, i, j);
	}
	
	const_iterator2 find2(int rank, size_type i, size_type j) const {
		SIZE_CHECK(i <= size1());
		SIZE_CHECK(j <= size2());
		return m_expression.find2(rank, i, j);
	}
	
	//Iterators
	const_iterator1 begin1() const {
		return find1(0, 0, 0);
	}
	const_iterator1 end1() const {
		return find1(0, size1(), 0);
	}
	const_iterator2 begin2() const {
		return find2(0, 0, 0);
	}
	const_iterator2 end2() const {
		return find2(0, 0, size2());
	}
private:
	TransposeClosure m_expression;
};

// (trans m) [i] [j] = m [j] [i]
template<class E>
matrix_transpose<const E> trans(const matrix_expression<E> &e) {
	return matrix_transpose<const E>(e());
}
template<class E>
matrix_transpose<E> trans(matrix_expression<E> &e) {
	return matrix_transpose<E>(e());
}

// (herm m) [i] [j] = conj (m [j] [i])
template<class E>

typename matrix_unary2_traits<E, scalar_conj<typename E::value_type> >::result_type
herm(const matrix_expression<E> &e) {
	typedef typename matrix_unary2_traits<E, scalar_conj<typename E::value_type> >::expression_type expression_type;
	return expression_type(e());
}

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
	typedef value_type const_reference;
	typedef const_reference reference;

	typedef const matrix_binary<E1, E2, F> const_closure_type;
	typedef const_closure_type closure_type;
	typedef typename E1::orientation_category orientation_category;
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

        const_reference operator () (size_type i, size_type j) const {
		return m_functor( m_expression1 (i, j), m_expression2(i,j));
        }

        // Closure comparison
        bool same_closure (matrix_binary const&mbs2) const {
		return m_expression1.same_closure (mbs2.m_expression1) ||
		m_expression2.same_closure (mbs2.m_expression2);
        }

	// Iterator types
private:
	typedef typename E1::const_iterator1 const_iterator11_type;
	typedef typename E1::const_iterator2 const_iterator12_type;
	typedef typename E2::const_iterator1 const_iterator21_type;
	typedef typename E2::const_iterator2 const_iterator22_type;
	typedef const value_type *const_pointer;

public:
	class const_iterator1;
	typedef const_iterator1 iterator1;
	class const_iterator2;
	typedef const_iterator2 iterator2;

	typedef reverse_iterator_base1<const_iterator1> const_reverse_iterator1;
	typedef reverse_iterator_base2<const_iterator2> const_reverse_iterator2;

	// Element lookup
	const_iterator1 find1(int rank, size_type i, size_type j) const {
		const_iterator11_type it11(m_expression1.find1(rank, i, j));
		const_iterator11_type it11_end(m_expression1.find1(rank, size1(), j));
		const_iterator21_type it21(m_expression2.find1(rank, i, j));
		const_iterator21_type it21_end(m_expression2.find1(rank, size1(), j));
		BOOST_UBLAS_CHECK(rank == 0 || it11 == it11_end || it11.index2() == j, internal_logic())
		BOOST_UBLAS_CHECK(rank == 0 || it21 == it21_end || it21.index2() == j, internal_logic())
		i = (std::min)(it11 != it11_end ? it11.index1() : size1(),
		        it21 != it21_end ? it21.index1() : size1());
		return const_iterator1(*this, i, j, it11, it11_end, it21, it21_end);
	}
	
	const_iterator2 find2(int rank, size_type i, size_type j) const {
		const_iterator12_type it12(m_expression1.find2(rank, i, j));
		const_iterator12_type it12_end(m_expression1.find2(rank, i, size2()));
		const_iterator22_type it22(m_expression2.find2(rank, i, j));
		const_iterator22_type it22_end(m_expression2.find2(rank, i, size2()));
		BOOST_UBLAS_CHECK(rank == 0 || it12 == it12_end || it12.index1() == i, internal_logic())
		BOOST_UBLAS_CHECK(rank == 0 || it22 == it22_end || it22.index1() == i, internal_logic())
		j = (std::min)(it12 != it12_end ? it12.index2() : size2(),
		        it22 != it22_end ? it22.index2() : size2());
		return const_iterator2(*this, i, j, it12, it12_end, it22, it22_end);
	}

	// Iterators enhance the iterators of the referenced expression
	// with the binary functor.

	class const_iterator1:
		public container_const_reference<matrix_binary>,
		public iterator_base_traits<typename iterator_restrict_traits<typename E1::const_iterator1::iterator_category,
		typename E2::const_iterator1::iterator_category>::iterator_category>::template
			iterator_base<const_iterator1, value_type>::type {
	public:
		typedef typename iterator_restrict_traits<typename E1::const_iterator1::iterator_category,
		typename E2::const_iterator1::iterator_category>::iterator_category iterator_category;
		typedef typename matrix_binary::difference_type difference_type;
		typedef typename matrix_binary::value_type value_type;
		typedef typename matrix_binary::const_reference reference;
		typedef typename matrix_binary::const_pointer pointer;

		typedef const_iterator2 dual_iterator_type;
		typedef const_reverse_iterator2 dual_reverse_iterator_type;

		    // Construction and destruction
		const_iterator1(): i_(), j_(), it1_(), it1_end_(), it2_(), it2_end_() {}
		const_iterator1(const self_type &mb, size_type i, size_type j,
			const const_iterator11_type &it1, const const_iterator11_type &it1_end,
			const const_iterator21_type &it2, const const_iterator21_type &it2_end):
			container_const_reference<self_type> (mb), i_(i), j_(j), 
			it1_(it1), it1_end_(it1_end), it2_(it2), it2_end_(it2_end) {}

	private:
		// Dense specializations
		void increment(dense_random_access_iterator_tag) {
			++ i_;
			++ it1_;
			++ it2_;
		}
		void decrement(dense_random_access_iterator_tag) {
			-- i_;
			-- it1_;
			-- it2_;
		}
		void increment(dense_random_access_iterator_tag, difference_type n) {
			i_ += n;
			it1_ += n;
			it2_ += n;
		}
		void decrement(dense_random_access_iterator_tag, difference_type n) {
			i_ -= n;
			it1_ -= n;
			it2_ -= n;
		}
		value_type dereference(dense_random_access_iterator_tag) const {
			return (*this)().m_functor(*it1_, *it2_);
		}

		// Sparse specializations
		void increment(sparse_bidirectional_iterator_tag) {
			size_type index1 = (*this)().size1();
			if (it1_ != it1_end_) {
				if (it1_.index1() <= i_)
					++ it1_;
				if (it1_ != it1_end_)
					index1 = it1_.index1();
			}
			size_type index2 = (*this)().size1();
			if (it2_ != it2_end_)
				if (it2_.index1() <= i_)
					++ it2_;
			if (it2_ != it2_end_) {
				index2 = it2_.index1();
			}
			i_ = (std::min)(index1, index2);
		}
		void decrement(sparse_bidirectional_iterator_tag) {
			size_type index1 = (*this)().size1();
			if (it1_ != it1_end_) {
				if (i_ <= it1_.index1())
					-- it1_;
				if (it1_ != it1_end_)
					index1 = it1_.index1();
			}
			size_type index2 = (*this)().size1();
			if (it2_ != it2_end_) {
				if (i_ <= it2_.index1())
					-- it2_;
				if (it2_ != it2_end_)
					index2 = it2_.index1();
			}
			i_ = (std::max)(index1, index2);
		}
		void increment(sparse_bidirectional_iterator_tag, difference_type n) {
			while (n > 0) {
				increment(sparse_bidirectional_iterator_tag());
				--n;
			}
			while (n < 0) {
				decrement(sparse_bidirectional_iterator_tag());
				++n;
			}
		}
		void decrement(sparse_bidirectional_iterator_tag, difference_type n) {
			while (n > 0) {
				decrement(sparse_bidirectional_iterator_tag());
				--n;
			}
			while (n < 0) {
				increment(sparse_bidirectional_iterator_tag());
				++n;
			}
		}
		value_type dereference(sparse_bidirectional_iterator_tag) const {
			value_type t1 = value_type/*zero*/();
			if (it1_ != it1_end_) {
				BOOST_UBLAS_CHECK(it1_.index2() == j_, internal_logic());
				if (it1_.index1() == i_)
					t1 = *it1_;
			}
			value_type t2 = value_type/*zero*/();
			if (it2_ != it2_end_) {
				BOOST_UBLAS_CHECK(it2_.index2() == j_, internal_logic());
				if (it2_.index1() == i_)
					t2 = *it2_;
			}
			return (*this)().m_functor(t1, t2);
		}

		public:
		// Arithmetic
		const_iterator1 &operator ++ () {
			increment(iterator_category());
			return *this;
		}
		const_iterator1 &operator -- () {
			decrement(iterator_category());
			return *this;
		}
		const_iterator1 &operator += (difference_type n) {
			increment(iterator_category(), n);
			return *this;
		}
		const_iterator1 &operator -= (difference_type n) {
			decrement(iterator_category(), n);
			return *this;
		}
		difference_type operator - (const const_iterator1 &it) const {
			BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());
			BOOST_UBLAS_CHECK(index2() == it.index2(), external_logic());
			return index1() - it.index1();
		}

		// Dereference
		const_reference operator * () const {
			return dereference(iterator_category());
		}
		const_reference operator [](difference_type n) const {
			return *(*this + n);
		}

		const_iterator2 begin() const {
			return (*this)().find2(1, index1(), 0);
		}
		const_iterator2 end() const {
			return (*this)().find2(1, index1(), (*this)().size2());
		}
		const_reverse_iterator2 rbegin() const {
			return const_reverse_iterator2(end());
		}
		const_reverse_iterator2 rend() const {
			return const_reverse_iterator2(begin());
		}

		// Indices
		size_type index1() const {
			return i_;
		}
		size_type index2() const {
			return j_;
		}

		// Assignment
		const_iterator1 &operator = (const const_iterator1 &it) {
			container_const_reference<self_type>::assign(&it());
			i_ = it.i_;
			j_ = it.j_;
			it1_ = it.it1_;
			it1_end_ = it.it1_end_;
			it2_ = it.it2_;
			it2_end_ = it.it2_end_;
			return *this;
		}

		// Comparison
		bool operator == (const const_iterator1 &it) const {
			BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());
			BOOST_UBLAS_CHECK(index2() == it.index2(), external_logic());
			return index1() == it.index1();
		}
		bool operator < (const const_iterator1 &it) const {
			BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());
			BOOST_UBLAS_CHECK(index2() == it.index2(), external_logic());
			return index1() < it.index1();
		}

	private:
		size_type i_;
		size_type j_;
		const_iterator11_type it1_;
		const_iterator11_type it1_end_;
		const_iterator21_type it2_;
		const_iterator21_type it2_end_;
	};


	const_iterator1 begin1() const {
		return find1(0, 0, 0);
	}
	const_iterator1 end1() const {
		return find1(0, size1(), 0);
	}


	class const_iterator2:
		public container_const_reference<matrix_binary>,
		public iterator_base_traits<typename iterator_restrict_traits<typename E1::const_iterator2::iterator_category,
		typename E2::const_iterator2::iterator_category>::iterator_category>::template
			iterator_base<const_iterator2, value_type>::type {
	public:
		typedef typename iterator_restrict_traits<typename E1::const_iterator2::iterator_category,
		typename E2::const_iterator2::iterator_category>::iterator_category iterator_category;
		typedef typename matrix_binary::difference_type difference_type;
		typedef typename matrix_binary::value_type value_type;
		typedef typename matrix_binary::const_reference reference;
		typedef typename matrix_binary::const_pointer pointer;

		typedef const_iterator1 dual_iterator_type;
		typedef const_reverse_iterator1 dual_reverse_iterator_type;

		// Construction and destruction
		const_iterator2():
			i_(), j_(), it1_(), it1_end_(), it2_(), it2_end_() {}
		const_iterator2(
			const self_type &mb, size_type i, size_type j,
			const const_iterator12_type &it1, const const_iterator12_type &it1_end,
			const const_iterator22_type &it2, const const_iterator22_type &it2_end
		):
		container_const_reference<self_type> (mb), i_(i), j_(j), it1_(it1), it1_end_(it1_end), it2_(it2), it2_end_(it2_end) {}

	private:
		// Dense access specializations
		void increment(dense_random_access_iterator_tag) {
			++ j_;
			++ it1_;
			++ it2_;
		}
		void decrement(dense_random_access_iterator_tag) {
			-- j_;
			-- it1_;
			-- it2_;
		}
		void increment(dense_random_access_iterator_tag, difference_type n) {
			j_ += n;
			it1_ += n;
			it2_ += n;
		}
		void decrement(dense_random_access_iterator_tag, difference_type n) {
			j_ -= n;
			it1_ -= n;
			it2_ -= n;
		}
		value_type dereference(dense_random_access_iterator_tag) const {
			return (*this)().m_functor(*it1_, *it2_);
		}
		// Sparse specializations
		void increment(sparse_bidirectional_iterator_tag) {
			size_type index1 = (*this)().size2();
			if (it1_ != it1_end_) {
				if (it1_.index2() <= j_)
					++ it1_;
				if (it1_ != it1_end_)
					index1 = it1_.index2();
			}
			size_type index2 = (*this)().size2();
			if (it2_ != it2_end_) {
				if (it2_.index2() <= j_)
					++ it2_;
				if (it2_ != it2_end_)
					index2 = it2_.index2();
			}
			j_ = (std::min)(index1, index2);
		}
		void decrement(sparse_bidirectional_iterator_tag) {
			size_type index1 = (*this)().size2();
			if (it1_ != it1_end_) {
				if (j_ <= it1_.index2())
					-- it1_;
				if (it1_ != it1_end_)
					index1 = it1_.index2();
			}
			size_type index2 = (*this)().size2();
			if (it2_ != it2_end_) {
				if (j_ <= it2_.index2())
					-- it2_;
				if (it2_ != it2_end_)
					index2 = it2_.index2();
			}
			j_ = (std::max)(index1, index2);
		}
		void increment(sparse_bidirectional_iterator_tag, difference_type n) {
			while (n > 0) {
				increment(sparse_bidirectional_iterator_tag());
				--n;
			}
			while (n < 0) {
				decrement(sparse_bidirectional_iterator_tag());
				++n;
			}
		}
		void decrement(sparse_bidirectional_iterator_tag, difference_type n) {
			while (n > 0) {
				decrement(sparse_bidirectional_iterator_tag());
				--n;
			}
			while (n < 0) {
				increment(sparse_bidirectional_iterator_tag());
				++n;
			}
		}
		value_type dereference(sparse_bidirectional_iterator_tag) const {
			value_type t1 = value_type/*zero*/();
			if (it1_ != it1_end_) {
				BOOST_UBLAS_CHECK(it1_.index1() == i_, internal_logic());
				if (it1_.index2() == j_)
					t1 = *it1_;
			}
			value_type t2 = value_type/*zero*/();
			if (it2_ != it2_end_) {
				BOOST_UBLAS_CHECK(it2_.index1() == i_, internal_logic());
				if (it2_.index2() == j_)
					t2 = *it2_;
			}
			return (*this)().m_functor(t1, t2);
		}

	public:
		// Arithmetic
		const_iterator2 &operator ++ () {
			increment(iterator_category());
			return *this;
		}
		const_iterator2 &operator -- () {
			decrement(iterator_category());
			return *this;
		}
		const_iterator2 &operator += (difference_type n) {
			increment(iterator_category(), n);
			return *this;
		}
		const_iterator2 &operator -= (difference_type n) {
			decrement(iterator_category(), n);
			return *this;
		}
		difference_type operator - (const const_iterator2 &it) const {
			BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());
			BOOST_UBLAS_CHECK(index1() == it.index1(), external_logic());
			return index2() - it.index2();
		}

		// Dereference
		const_reference operator * () const {
			return dereference(iterator_category());
		}
		const_reference operator [](difference_type n) const {
			return *(*this + n);
		}

		const_iterator1 begin() const {
			return (*this)().find1(1, 0, index2());
		}
		const_iterator1 end() const {
			return (*this)().find1(1, (*this)().size1(), index2());
		}
		
		const_reverse_iterator1 rbegin() const {
			return const_reverse_iterator1(end());
		}
		const_reverse_iterator1 rend() const {
			return const_reverse_iterator1(begin());
		}

		// Indices
		size_type index1() const {
			return i_;
		}
		size_type index2() const {
			return j_;
		}

		// Assignment
		const_iterator2 &operator = (const const_iterator2 &it) {
			container_const_reference<self_type>::assign(&it());
			i_ = it.i_;
			j_ = it.j_;
			it1_ = it.it1_;
			it1_end_ = it.it1_end_;
			it2_ = it.it2_;
			it2_end_ = it.it2_end_;
			return *this;
		}

		// Comparison
		bool operator == (const const_iterator2 &it) const {
			BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());
			BOOST_UBLAS_CHECK(index1() == it.index1(), external_logic());
			return index2() == it.index2();
		}
		bool operator < (const const_iterator2 &it) const {
			BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());
			BOOST_UBLAS_CHECK(index1() == it.index1(), external_logic());
			return index2() < it.index2();
		}

	private:
		size_type i_;
		size_type j_;
		const_iterator12_type it1_;
		const_iterator12_type it1_end_;
		const_iterator22_type it2_;
		const_iterator22_type it2_end_;
	};

	const_iterator2 begin2() const {
		return find2(0, 0, 0);
	}
	const_iterator2 end2() const {
		return find2(0, 0, size2());
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
safeDiv(
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
	typedef value_type const_reference;
	typedef const_reference reference;
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

public:
	// Expression accessors
	
	const expression1_closure_type &expression1() const {
		return m_expression1;
	}
	
	const expression2_closure_type &expression2() const {
		return m_expression2;
	}

public:
	// Element access
	
	const_reference operator()(size_type i) const {
		return functor_type::apply(m_expression1, m_expression2, i);
	}

	// Closure comparison
	
	bool same_closure(const matrix_vector_binary1 &mvb1) const {
		return (*this).expression1().same_closure(mvb1.expression1()) &&
		       (*this).expression2().same_closure(mvb1.expression2());
	}

	// Iterator types
private:
	typedef typename E1::const_iterator1 const_subiterator1_type;
public:
	typedef indexed_const_iterator<const_closure_type, typename const_subiterator1_type::iterator_category> const_iterator;
	typedef const_iterator iterator;

	// Element lookup
	const_iterator find(size_type i) const {
		return const_iterator(*this,i);
	}
	
	const_iterator begin() const {
		return find(0);
	}
	
	const_iterator end() const {
		return find(size());
	}

	// Reverse iterator
	typedef reverse_iterator_base<const_iterator> const_reverse_iterator;

	const_reverse_iterator rbegin() const {
		return const_reverse_iterator(end());
	}
	
	const_reverse_iterator rend() const {
		return const_reverse_iterator(begin());
	}

private:
	expression1_closure_type m_expression1;
	expression2_closure_type m_expression2;
};

template<class T1, class E1, class T2, class E2>
struct matrix_vector_binary1_traits {
	typedef unknown_storage_tag storage_category;
	typedef row_major_tag orientation_category;
	typedef typename promote_traits<T1, T2>::promote_type promote_type;
	typedef matrix_vector_binary1<E1, E2, matrix_vector_prod1<E1, E2, promote_type> > expression_type;
#ifndef BOOST_UBLAS_SIMPLE_ET_DEBUG
	typedef expression_type result_type;
#else
	typedef typename E1::vector_temporary_type result_type;
#endif
};

template<class E1, class E2>
typename matrix_vector_binary1_traits<typename E1::value_type, E1,
         typename E2::value_type, E2>::result_type
         prod(const matrix_expression<E1> &e1,
                 const vector_expression<E2> &e2,
                 unknown_storage_tag,
row_major_tag) {
	typedef typename matrix_vector_binary1_traits<typename E1::value_type, E1,
	        typename E2::value_type, E2>::expression_type expression_type;
	return expression_type(e1(), e2());
}

// Dispatcher
template<class E1, class E2>
typename matrix_vector_binary1_traits<typename E1::value_type, E1,
         typename E2::value_type, E2>::result_type
         prod(const matrix_expression<E1> &e1,
const vector_expression<E2> &e2) {
	typedef typename matrix_vector_binary1_traits<typename E1::value_type, E1,
	        typename E2::value_type, E2>::storage_category storage_category;
	typedef typename matrix_vector_binary1_traits<typename E1::value_type, E1,
	        typename E2::value_type, E2>::orientation_category orientation_category;
	return prod(e1, e2, storage_category(), orientation_category());
}

//~ template<class E1, class E2>

//~ typename matrix_vector_binary1_traits<typename type_traits<typename E1::value_type>::precision_type, E1,
         //~ typename type_traits<typename E2::value_type>::precision_type, E2>::result_type
         //~ prec_prod(const matrix_expression<E1> &e1,
                 //~ const vector_expression<E2> &e2,
                 //~ unknown_storage_tag,
//~ row_major_tag) {
	//~ typedef typename matrix_vector_binary1_traits<typename type_traits<typename E1::value_type>::precision_type, E1,
	        //~ typename type_traits<typename E2::value_type>::precision_type, E2>::expression_type expression_type;
	//~ return expression_type(e1(), e2());
//~ }

//~ // Dispatcher
//~ template<class E1, class E2>

//~ typename matrix_vector_binary1_traits<typename type_traits<typename E1::value_type>::precision_type, E1,
         //~ typename type_traits<typename E2::value_type>::precision_type, E2>::result_type
         //~ prec_prod(const matrix_expression<E1> &e1,
//~ const vector_expression<E2> &e2) {
	//~ typedef typename matrix_vector_binary1_traits<typename type_traits<typename E1::value_type>::precision_type, E1,
	        //~ typename type_traits<typename E2::value_type>::precision_type, E2>::storage_category storage_category;
	//~ typedef typename matrix_vector_binary1_traits<typename type_traits<typename E1::value_type>::precision_type, E1,
	        //~ typename type_traits<typename E2::value_type>::precision_type, E2>::orientation_category orientation_category;
	//~ return prec_prod(e1, e2, storage_category(), orientation_category());
//~ }

template<class V, class E1, class E2>
V & prod(const matrix_expression<E1> &e1,
     const vector_expression<E2> &e2,
     V &v) {
	return v.assign(prod(e1, e2));
}

//~ template<class V, class E1, class E2>

//~ V &
//~ prec_prod(const matrix_expression<E1> &e1,
        //~ const vector_expression<E2> &e2,
        //~ V &v) {
	//~ return v.assign(prec_prod(e1, e2));
//~ }

template<class V, class E1, class E2>
V prod(const matrix_expression<E1> &e1,
     const vector_expression<E2> &e2) {
	return V(prod(e1, e2));
}

//~ template<class V, class E1, class E2>

//~ V
//~ prec_prod(const matrix_expression<E1> &e1,
        //~ const vector_expression<E2> &e2) {
	//~ return V(prec_prod(e1, e2));
//~ }

template<class E1, class E2, class F>
class matrix_vector_binary2:
	public vector_expression<matrix_vector_binary2<E1, E2, F> > {

	typedef E1 expression1_type;
	typedef E2 expression2_type;
	typedef F functor_type;
public:
	typedef typename E1::const_closure_type expression1_closure_type;
	typedef typename E2::const_closure_type expression2_closure_type;
private:
	typedef matrix_vector_binary2<E1, E2, F> self_type;
public:
#ifdef BOOST_UBLAS_ENABLE_PROXY_SHORTCUTS
	using vector_expression<self_type>::operator();
#endif
	typedef typename promote_traits<typename E1::size_type, typename E2::size_type>::promote_type size_type;
	typedef typename promote_traits<typename E1::difference_type, typename E2::difference_type>::promote_type difference_type;
	typedef typename F::result_type value_type;
	typedef value_type const_reference;
	typedef const_reference reference;
	typedef const self_type const_closure_type;
	typedef const_closure_type closure_type;
	typedef unknown_storage_tag storage_category;

	// Construction and destruction
	
	matrix_vector_binary2(const expression1_type &e1, const expression2_type &e2):
		m_expression1(e1), m_expression2(e2) {}

	// Accessors
	
	size_type size() const {
		return m_expression2.size2();
	}

public:
	// Expression accessors
	
	const expression1_closure_type &expression1() const {
		return m_expression1;
	}
	
	const expression2_closure_type &expression2() const {
		return m_expression2;
	}
public:

	// Element access
	
	const_reference operator()(size_type j) const {
		return functor_type::apply(m_expression1, m_expression2, j);
	}

	// Closure comparison
	
	bool same_closure(const matrix_vector_binary2 &mvb2) const {
		return (*this).expression1().same_closure(mvb2.expression1()) &&
		       (*this).expression2().same_closure(mvb2.expression2());
	}

	// Iterator types
private:
	typedef typename E1::const_iterator const_subiterator1_type;
	typedef typename E2::const_iterator2 const_subiterator2_type;
	typedef const value_type *const_pointer;

public:
#ifdef BOOST_UBLAS_USE_INDEXED_ITERATOR
	typedef indexed_const_iterator<const_closure_type, typename const_subiterator2_type::iterator_category> const_iterator;
	typedef const_iterator iterator;
#else
	class const_iterator;
	typedef const_iterator iterator;
#endif

	// Element lookup
	
	const_iterator find(size_type j) const {
#ifdef BOOST_UBLAS_USE_INDEXED_ITERATOR
		const_subiterator2_type it2(m_expression2.find2(0, 0, j));
		return const_iterator(*this, it2.index2());
#else
		return const_iterator(*this, m_expression2.find2(0, 0, j));
#endif
	}


#ifndef BOOST_UBLAS_USE_INDEXED_ITERATOR
	class const_iterator:
		public container_const_reference<matrix_vector_binary2>,
		public iterator_base_traits<typename iterator_restrict_traits<typename E1::const_iterator::iterator_category,
		typename E2::const_iterator2::iterator_category>::iterator_category>::template
			iterator_base<const_iterator, value_type>::type {
    public:
	    typedef typename iterator_restrict_traits<typename E1::const_iterator::iterator_category,
	    typename E2::const_iterator2::iterator_category>::iterator_category iterator_category;
	    typedef typename matrix_vector_binary2::difference_type difference_type;
	    typedef typename matrix_vector_binary2::value_type value_type;
	    typedef typename matrix_vector_binary2::const_reference reference;
	    typedef typename matrix_vector_binary2::const_pointer pointer;

	    // Construction and destruction
#ifdef BOOST_UBLAS_USE_INVARIANT_HOISTING
	    
	    const_iterator():
		    container_const_reference<self_type> (), it2_(), e1_begin_(), e1_end_() {}
	
	const_iterator(const self_type &mvb, const const_subiterator2_type &it2):
		container_const_reference<self_type> (mvb), it2_(it2), e1_begin_(mvb.expression1().begin()), e1_end_(mvb.expression1().end()) {}
#else
	    
	    const_iterator():
		    container_const_reference<self_type> (), it2_() {}
	
	const_iterator(const self_type &mvb, const const_subiterator2_type &it2):
		container_const_reference<self_type> (mvb), it2_(it2) {}
#endif

private:
	// Dense random access specialization
	
	value_type dereference(dense_random_access_iterator_tag) const {
		const self_type &mvb = (*this)();
#ifdef BOOST_UBLAS_USE_INDEXING
		return mvb(index());
#elif BOOST_UBLAS_USE_ITERATING
		difference_type size = BOOST_UBLAS_SAME(mvb.expression2().size1(), mvb.expression1().size());
#ifdef BOOST_UBLAS_USE_INVARIANT_HOISTING
		return functor_type::apply(size, e1_begin_, it2_.begin());
#else
		return functor_type::apply(size, mvb.expression1().begin(), it2_.begin());
#endif
#else
		difference_type size = BOOST_UBLAS_SAME(mvb.expression2().size1(), mvb.expression1().size());
		if (size >= BOOST_UBLAS_ITERATOR_THRESHOLD)
#ifdef BOOST_UBLAS_USE_INVARIANT_HOISTING
			return functor_type::apply(size, e1_begin_, it2_.begin());
#else
			return functor_type::apply(size, mvb.expression1().begin(), it2_.begin());
#endif
		else
			return mvb(index());
#endif
	}

	// Packed bidirectional specialization
	
	value_type dereference(packed_random_access_iterator_tag) const {
#ifdef BOOST_UBLAS_USE_INVARIANT_HOISTING
		return functor_type::apply(e1_begin_, e1_end_, it2_.begin(), it2_.end());
#else
		const self_type &mvb = (*this)();
#ifndef BOOST_UBLAS_NO_NESTED_CLASS_RELATION
		return functor_type::apply(mvb.expression1().begin(), mvb.expression1().end(),
		        it2_.begin(), it2_.end());
#else
		return functor_type::apply(mvb.expression1().begin(), mvb.expression1().end(),
		        shark::blas::begin(it2_, iterator2_tag()),
		        shark::blas::end(it2_, iterator2_tag()));
#endif
#endif
	}

	// Sparse bidirectional specialization
	
	value_type dereference(sparse_bidirectional_iterator_tag) const {
#ifdef BOOST_UBLAS_USE_INVARIANT_HOISTING
		return functor_type::apply(e1_begin_, e1_end_, it2_.begin(), it2_.end(), sparse_bidirectional_iterator_tag());
#else
		const self_type &mvb = (*this)();
#ifndef BOOST_UBLAS_NO_NESTED_CLASS_RELATION
		return functor_type::apply(mvb.expression1().begin(), mvb.expression1().end(),
		        it2_.begin(), it2_.end(), sparse_bidirectional_iterator_tag());
#else
		return functor_type::apply(mvb.expression1().begin(), mvb.expression1().end(),
		        shark::blas::begin(it2_, iterator2_tag()),
		        shark::blas::end(it2_, iterator2_tag()), sparse_bidirectional_iterator_tag());
#endif
#endif
	}

public:
	// Arithmetic
	
	const_iterator &operator ++ () {
		++ it2_;
		return *this;
	}
	
	const_iterator &operator -- () {
		-- it2_;
		return *this;
	}
	
	const_iterator &operator += (difference_type n) {
		it2_ += n;
		return *this;
	}
	
	const_iterator &operator -= (difference_type n) {
		it2_ -= n;
		return *this;
	}
	
	difference_type operator - (const const_iterator &it) const {
		BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());
		return it2_ - it.it2_;
	}

	// Dereference
	
	const_reference operator * () const {
		return dereference(iterator_category());
	}
	
	const_reference operator [](difference_type n) const {
		return *(*this + n);
	}

	// Index
	
	size_type index() const {
		return it2_.index2();
	}

	// Assignment
	
	const_iterator &operator = (const const_iterator &it) {
		container_const_reference<self_type>::assign(&it());
		it2_ = it.it2_;
#ifdef BOOST_UBLAS_USE_INVARIANT_HOISTING
		e1_begin_ = it.e1_begin_;
		e1_end_ = it.e1_end_;
#endif
		return *this;
	}

	// Comparison
	
	bool operator == (const const_iterator &it) const {
		BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());
		return it2_ == it.it2_;
	}
	
	bool operator < (const const_iterator &it) const {
		BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());
		return it2_ < it.it2_;
	}

private:
	const_subiterator2_type it2_;
#ifdef BOOST_UBLAS_USE_INVARIANT_HOISTING
	// Mutable due to assignment
	/* const */ const_subiterator1_type e1_begin_;
	/* const */
	const_subiterator1_type e1_end_;
#endif
	};
#endif

	
	const_iterator begin() const {
		return find(0);
	}
	
	const_iterator end() const {
		return find(size());
	}

	// Reverse iterator
	typedef reverse_iterator_base<const_iterator> const_reverse_iterator;

	
	const_reverse_iterator rbegin() const {
		return const_reverse_iterator(end());
	}
	
	const_reverse_iterator rend() const {
		return const_reverse_iterator(begin());
	}

private:
	expression1_closure_type m_expression1;
	expression2_closure_type m_expression2;
};

template<class T1, class E1, class T2, class E2>
struct matrix_vector_binary2_traits {
	typedef unknown_storage_tag storage_category;
	typedef column_major_tag orientation_category;
	typedef typename promote_traits<T1, T2>::promote_type promote_type;
	typedef matrix_vector_binary2<E1, E2, matrix_vector_prod2<E1, E2, promote_type> > expression_type;
#ifndef BOOST_UBLAS_SIMPLE_ET_DEBUG
	typedef expression_type result_type;
#else
	typedef typename E2::vector_temporary_type result_type;
#endif
};

template<class E1, class E2>
typename matrix_vector_binary2_traits<typename E1::value_type, E1,
         typename E2::value_type, E2>::result_type
         prod(const vector_expression<E1> &e1,
                 const matrix_expression<E2> &e2,
                 unknown_storage_tag,
column_major_tag) {
	typedef typename matrix_vector_binary2_traits<typename E1::value_type, E1,
	        typename E2::value_type, E2>::expression_type expression_type;
	return expression_type(e1(), e2());
}

// Dispatcher
template<class E1, class E2>
typename matrix_vector_binary2_traits<typename E1::value_type, E1,
         typename E2::value_type, E2>::result_type
         prod(const vector_expression<E1> &e1,
const matrix_expression<E2> &e2) {
	typedef typename matrix_vector_binary2_traits<typename E1::value_type, E1,
	        typename E2::value_type, E2>::storage_category storage_category;
	typedef typename matrix_vector_binary2_traits<typename E1::value_type, E1,
	        typename E2::value_type, E2>::orientation_category orientation_category;
	return prod(e1, e2, storage_category(), orientation_category());
}

//~ template<class E1, class E2>

//~ typename matrix_vector_binary2_traits<typename type_traits<typename E1::value_type>::precision_type, E1,
         //~ typename type_traits<typename E2::value_type>::precision_type, E2>::result_type
         //~ prec_prod(const vector_expression<E1> &e1,
                 //~ const matrix_expression<E2> &e2,
                 //~ unknown_storage_tag,
//~ column_major_tag) {
	//~ typedef typename matrix_vector_binary2_traits<typename type_traits<typename E1::value_type>::precision_type, E1,
	        //~ typename type_traits<typename E2::value_type>::precision_type, E2>::expression_type expression_type;
	//~ return expression_type(e1(), e2());
//~ }

// Dispatcher
//~ template<class E1, class E2>

//~ typename matrix_vector_binary2_traits<typename type_traits<typename E1::value_type>::precision_type, E1,
         //~ typename type_traits<typename E2::value_type>::precision_type, E2>::result_type
         //~ prec_prod(const vector_expression<E1> &e1,
//~ const matrix_expression<E2> &e2) {
	//~ typedef typename matrix_vector_binary2_traits<typename type_traits<typename E1::value_type>::precision_type, E1,
	        //~ typename type_traits<typename E2::value_type>::precision_type, E2>::storage_category storage_category;
	//~ typedef typename matrix_vector_binary2_traits<typename type_traits<typename E1::value_type>::precision_type, E1,
	        //~ typename type_traits<typename E2::value_type>::precision_type, E2>::orientation_category orientation_category;
	//~ return prec_prod(e1, e2, storage_category(), orientation_category());
//~ }

template<class V, class E1, class E2>
V & prod(const vector_expression<E1> &e1,
     const matrix_expression<E2> &e2,
     V &v) {
	return v.assign(prod(e1, e2));
}

//~ template<class V, class E1, class E2>
//~ V &
//~ prec_prod(const vector_expression<E1> &e1,
        //~ const matrix_expression<E2> &e2,
        //~ V &v) {
	//~ return v.assign(prec_prod(e1, e2));
//~ }

template<class V, class E1, class E2>
V prod(const vector_expression<E1> &e1,
     const matrix_expression<E2> &e2) {
	return V(prod(e1, e2));
}

//~ template<class V, class E1, class E2>
//~ V
//~ prec_prod(const vector_expression<E1> &e1,
        //~ const matrix_expression<E2> &e2) {
	//~ return V(prec_prod(e1, e2));
//~ }

template<class E1, class E2, class F>
class matrix_matrix_binary:
	public matrix_expression<matrix_matrix_binary<E1, E2, F> > {

public:
	typedef E1 expression1_type;
	typedef E2 expression2_type;
private:
	typedef F functor_type;
public:
	typedef typename E1::const_closure_type expression1_closure_type;
	typedef typename E2::const_closure_type expression2_closure_type;
private:
	typedef matrix_matrix_binary<E1, E2, F> self_type;
public:
#ifdef BOOST_UBLAS_ENABLE_PROXY_SHORTCUTS
	using matrix_expression<self_type>::operator();
#endif
	typedef typename promote_traits<typename E1::size_type, typename E2::size_type>::promote_type size_type;
	typedef typename promote_traits<typename E1::difference_type, typename E2::difference_type>::promote_type difference_type;
	typedef typename F::result_type value_type;
	typedef value_type const_reference;
	typedef const_reference reference;
	typedef const self_type const_closure_type;
	typedef const_closure_type closure_type;
	typedef unknown_orientation_tag orientation_category;
	typedef unknown_storage_tag storage_category;

	// Construction and destruction
	
	matrix_matrix_binary(const expression1_type &e1, const expression2_type &e2):
		m_expression1(e1), m_expression2(e2) {}

	// Accessors
	
	size_type size1() const {
		return m_expression1.size1();
	}
	
	size_type size2() const {
		return m_expression2.size2();
	}

public:
	// Expression accessors
	
	const expression1_closure_type &expression1() const {
		return m_expression1;
	}
	
	const expression2_closure_type &expression2() const {
		return m_expression2;
	}

public:
	// Element access
	
	const_reference operator()(size_type i, size_type j) const {
		return functor_type::apply(m_expression1, m_expression2, i, j);
	}

	// Closure comparison
	
	bool same_closure(const matrix_matrix_binary &mmb) const {
		return (*this).expression1().same_closure(mmb.expression1()) &&
		       (*this).expression2().same_closure(mmb.expression2());
	}

	// Iterator types
private:
	typedef typename E1::const_iterator1 const_iterator11_type;
	typedef typename E1::const_iterator2 const_iterator12_type;
	typedef typename E2::const_iterator1 const_iterator21_type;
	typedef typename E2::const_iterator2 const_iterator22_type;
	typedef const value_type *const_pointer;

public:
#ifdef BOOST_UBLAS_USE_INDEXED_ITERATOR
	typedef typename iterator_restrict_traits<typename const_iterator11_type::iterator_category,
	        typename const_iterator22_type::iterator_category>::iterator_category iterator_category;
	typedef indexed_const_iterator1<const_closure_type, iterator_category> const_iterator1;
	typedef const_iterator1 iterator1;
	typedef indexed_const_iterator2<const_closure_type, iterator_category> const_iterator2;
	typedef const_iterator2 iterator2;
#else
	class const_iterator1;
	typedef const_iterator1 iterator1;
	class const_iterator2;
	typedef const_iterator2 iterator2;
#endif
	typedef reverse_iterator_base1<const_iterator1> const_reverse_iterator1;
	typedef reverse_iterator_base2<const_iterator2> const_reverse_iterator2;

	// Element lookup
	
	const_iterator1 find1(int /* rank */, size_type i, size_type j) const {
		// FIXME sparse matrix tests fail!
		// const_iterator11_type it11 (m_expression1.find1 (rank, i, 0));
		const_iterator11_type it11(m_expression1.find1(0, i, 0));
#ifdef BOOST_UBLAS_USE_INDEXED_ITERATOR
		return const_iterator1(*this, it11.index1(), j);
#else
		// FIXME sparse matrix tests fail!
		// const_iterator22_type it22 (m_expression2.find2 (rank, 0, j));
		const_iterator22_type it22(m_expression2.find2(0, 0, j));
		return const_iterator1(*this, it11, it22);
#endif
	}
	
	const_iterator2 find2(int /* rank */, size_type i, size_type j) const {
		// FIXME sparse matrix tests fail!
		// const_iterator22_type it22 (m_expression2.find2 (rank, 0, j));
		const_iterator22_type it22(m_expression2.find2(0, 0, j));
#ifdef BOOST_UBLAS_USE_INDEXED_ITERATOR
		return const_iterator2(*this, i, it22.index2());
#else
		// FIXME sparse matrix tests fail!
		// const_iterator11_type it11 (m_expression1.find1 (rank, i, 0));
		const_iterator11_type it11(m_expression1.find1(0, i, 0));
		return const_iterator2(*this, it11, it22);
#endif
	}


#ifndef BOOST_UBLAS_USE_INDEXED_ITERATOR
	class const_iterator1:
		public container_const_reference<matrix_matrix_binary>,
		public iterator_base_traits<typename iterator_restrict_traits<typename E1::const_iterator1::iterator_category,
		typename E2::const_iterator2::iterator_category>::iterator_category>::template
			iterator_base<const_iterator1, value_type>::type {
    public:
	    typedef typename iterator_restrict_traits<typename E1::const_iterator1::iterator_category,
	    typename E2::const_iterator2::iterator_category>::iterator_category iterator_category;
	    typedef typename matrix_matrix_binary::difference_type difference_type;
	    typedef typename matrix_matrix_binary::value_type value_type;
	    typedef typename matrix_matrix_binary::const_reference reference;
	    typedef typename matrix_matrix_binary::const_pointer pointer;

	    typedef const_iterator2 dual_iterator_type;
	    typedef const_reverse_iterator2 dual_reverse_iterator_type;

	    // Construction and destruction
#ifdef BOOST_UBLAS_USE_INVARIANT_HOISTING
	    
	    const_iterator1():
		    container_const_reference<self_type> (), it1_(), it2_(), it2_begin_(), it2_end_() {}
	
	const_iterator1(const self_type &mmb, const const_iterator11_type &it1, const const_iterator22_type &it2):
		container_const_reference<self_type> (mmb), it1_(it1), it2_(it2), it2_begin_(it2.begin()), it2_end_(it2.end()) {}
#else
	    
	    const_iterator1():
		    container_const_reference<self_type> (), it1_(), it2_() {}
	
	const_iterator1(const self_type &mmb, const const_iterator11_type &it1, const const_iterator22_type &it2):
		container_const_reference<self_type> (mmb), it1_(it1), it2_(it2) {}
#endif

private:
	// Random access specialization
	
	value_type dereference(dense_random_access_iterator_tag) const {
		const self_type &mmb = (*this)();
#ifdef BOOST_UBLAS_USE_INDEXING
		return mmb(index1(), index2());
#elif BOOST_UBLAS_USE_ITERATING
		difference_type size = BOOST_UBLAS_SAME(mmb.expression1().size2(), mmb.expression2().size1());
#ifdef BOOST_UBLAS_USE_INVARIANT_HOISTING
		return functor_type::apply(size, it1_.begin(), it2_begin_);
#else
		return functor_type::apply(size, it1_.begin(), it2_.begin());
#endif
#else
		difference_type size = BOOST_UBLAS_SAME(mmb.expression1().size2(), mmb.expression2().size1());
		if (size >= BOOST_UBLAS_ITERATOR_THRESHOLD)
#ifdef BOOST_UBLAS_USE_INVARIANT_HOISTING
			return functor_type::apply(size, it1_.begin(), it2_begin_);
#else
			return functor_type::apply(size, it1_.begin(), it2_.begin());
#endif
		else
			return mmb(index1(), index2());
#endif
	}

	// Packed bidirectional specialization
	
	value_type dereference(packed_random_access_iterator_tag) const {
#ifdef BOOST_UBLAS_USE_INVARIANT_HOISTING
		return functor_type::apply(it1_.begin(), it1_.end(),
		        it2_begin_, it2_end_, packed_random_access_iterator_tag());
#else
#ifndef BOOST_UBLAS_NO_NESTED_CLASS_RELATION
		return functor_type::apply(it1_.begin(), it1_.end(),
		        it2_.begin(), it2_.end(), packed_random_access_iterator_tag());
#else
		return functor_type::apply(shark::blas::begin(it1_, iterator1_tag()),
		        shark::blas::end(it1_, iterator1_tag()),
		        shark::blas::begin(it2_, iterator2_tag()),
		        shark::blas::end(it2_, iterator2_tag()), packed_random_access_iterator_tag());
#endif
#endif
	}

	// Sparse bidirectional specialization
	
	value_type dereference(sparse_bidirectional_iterator_tag) const {
#ifdef BOOST_UBLAS_USE_INVARIANT_HOISTING
		return functor_type::apply(it1_.begin(), it1_.end(),
		        it2_begin_, it2_end_, sparse_bidirectional_iterator_tag());
#else
#ifndef BOOST_UBLAS_NO_NESTED_CLASS_RELATION
		return functor_type::apply(it1_.begin(), it1_.end(),
		        it2_.begin(), it2_.end(), sparse_bidirectional_iterator_tag());
#else
		return functor_type::apply(shark::blas::begin(it1_, iterator1_tag()),
		        shark::blas::end(it1_, iterator1_tag()),
		        shark::blas::begin(it2_, iterator2_tag()),
		        shark::blas::end(it2_, iterator2_tag()), sparse_bidirectional_iterator_tag());
#endif
#endif
	}

public:
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
		BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());
		BOOST_UBLAS_CHECK(it2_ == it.it2_, external_logic());
		return it1_ - it.it1_;
	}

	// Dereference
	
	const_reference operator * () const {
		return dereference(iterator_category());
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
		return it2_.index2();
	}

	// Assignment
	
	const_iterator1 &operator = (const const_iterator1 &it) {
		container_const_reference<self_type>::assign(&it());
		it1_ = it.it1_;
		it2_ = it.it2_;
#ifdef BOOST_UBLAS_USE_INVARIANT_HOISTING
		it2_begin_ = it.it2_begin_;
		it2_end_ = it.it2_end_;
#endif
		return *this;
	}

	// Comparison
	
	bool operator == (const const_iterator1 &it) const {
		BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());
		BOOST_UBLAS_CHECK(it2_ == it.it2_, external_logic());
		return it1_ == it.it1_;
	}
	
	bool operator < (const const_iterator1 &it) const {
		BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());
		BOOST_UBLAS_CHECK(it2_ == it.it2_, external_logic());
		return it1_ < it.it1_;
	}

private:
	const_iterator11_type it1_;
	// Mutable due to assignment
	/* const */
	const_iterator22_type it2_;
#ifdef BOOST_UBLAS_USE_INVARIANT_HOISTING
	/* const */ const_iterator21_type it2_begin_;
	/* const */
	const_iterator21_type it2_end_;
#endif
	};
#endif

	
	const_iterator1 begin1() const {
		return find1(0, 0, 0);
	}
	
	const_iterator1 end1() const {
		return find1(0, size1(), 0);
	}

#ifndef BOOST_UBLAS_USE_INDEXED_ITERATOR
	class const_iterator2:
		public container_const_reference<matrix_matrix_binary>,
		public iterator_base_traits<typename iterator_restrict_traits<typename E1::const_iterator1::iterator_category,
		typename E2::const_iterator2::iterator_category>::iterator_category>::template
			iterator_base<const_iterator2, value_type>::type {
    public:
	    typedef typename iterator_restrict_traits<typename E1::const_iterator1::iterator_category,
	    typename E2::const_iterator2::iterator_category>::iterator_category iterator_category;
	    typedef typename matrix_matrix_binary::difference_type difference_type;
	    typedef typename matrix_matrix_binary::value_type value_type;
	    typedef typename matrix_matrix_binary::const_reference reference;
	    typedef typename matrix_matrix_binary::const_pointer pointer;

	    typedef const_iterator1 dual_iterator_type;
	    typedef const_reverse_iterator1 dual_reverse_iterator_type;

	    // Construction and destruction
#ifdef BOOST_UBLAS_USE_INVARIANT_HOISTING
	    
	    const_iterator2():
		    container_const_reference<self_type> (), it1_(), it2_(), it1_begin_(), it1_end_() {}
	
	const_iterator2(const self_type &mmb, const const_iterator11_type &it1, const const_iterator22_type &it2):
		container_const_reference<self_type> (mmb), it1_(it1), it2_(it2), it1_begin_(it1.begin()), it1_end_(it1.end()) {}
#else
	    
	    const_iterator2():
		    container_const_reference<self_type> (), it1_(), it2_() {}
	
	const_iterator2(const self_type &mmb, const const_iterator11_type &it1, const const_iterator22_type &it2):
		container_const_reference<self_type> (mmb), it1_(it1), it2_(it2) {}
#endif

private:
	// Random access specialization
	
	value_type dereference(dense_random_access_iterator_tag) const {
		const self_type &mmb = (*this)();
#ifdef BOOST_UBLAS_USE_INDEXING
		return mmb(index1(), index2());
#elif BOOST_UBLAS_USE_ITERATING
		difference_type size = BOOST_UBLAS_SAME(mmb.expression1().size2(), mmb.expression2().size1());
#ifdef BOOST_UBLAS_USE_INVARIANT_HOISTING
		return functor_type::apply(size, it1_begin_, it2_.begin());
#else
		return functor_type::apply(size, it1_.begin(), it2_.begin());
#endif
#else
		difference_type size = BOOST_UBLAS_SAME(mmb.expression1().size2(), mmb.expression2().size1());
		if (size >= BOOST_UBLAS_ITERATOR_THRESHOLD)
#ifdef BOOST_UBLAS_USE_INVARIANT_HOISTING
			return functor_type::apply(size, it1_begin_, it2_.begin());
#else
			return functor_type::apply(size, it1_.begin(), it2_.begin());
#endif
		else
			return mmb(index1(), index2());
#endif
	}

	// Packed bidirectional specialization
	
	value_type dereference(packed_random_access_iterator_tag) const {
#ifdef BOOST_UBLAS_USE_INVARIANT_HOISTING
		return functor_type::apply(it1_begin_, it1_end_,
		        it2_.begin(), it2_.end(), packed_random_access_iterator_tag());
#else
#ifndef BOOST_UBLAS_NO_NESTED_CLASS_RELATION
		return functor_type::apply(it1_.begin(), it1_.end(),
		        it2_.begin(), it2_.end(), packed_random_access_iterator_tag());
#else
		return functor_type::apply(shark::blas::begin(it1_, iterator1_tag()),
		        shark::blas::end(it1_, iterator1_tag()),
		        shark::blas::begin(it2_, iterator2_tag()),
		        shark::blas::end(it2_, iterator2_tag()), packed_random_access_iterator_tag());
#endif
#endif
	}

	// Sparse bidirectional specialization
	
	value_type dereference(sparse_bidirectional_iterator_tag) const {
#ifdef BOOST_UBLAS_USE_INVARIANT_HOISTING
		return functor_type::apply(it1_begin_, it1_end_,
		        it2_.begin(), it2_.end(), sparse_bidirectional_iterator_tag());
#else
#ifndef BOOST_UBLAS_NO_NESTED_CLASS_RELATION
		return functor_type::apply(it1_.begin(), it1_.end(),
		        it2_.begin(), it2_.end(), sparse_bidirectional_iterator_tag());
#else
		return functor_type::apply(shark::blas::begin(it1_, iterator1_tag()),
		        shark::blas::end(it1_, iterator1_tag()),
		        shark::blas::begin(it2_, iterator2_tag()),
		        shark::blas::end(it2_, iterator2_tag()), sparse_bidirectional_iterator_tag());
#endif
#endif
	}

public:
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
		BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());
		BOOST_UBLAS_CHECK(it1_ == it.it1_, external_logic());
		return it2_ - it.it2_;
	}

	// Dereference
	
	const_reference operator * () const {
		return dereference(iterator_category());
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
		return it1_.index1();
	}
	
	size_type index2() const {
		return it2_.index2();
	}

	// Assignment
	
	const_iterator2 &operator = (const const_iterator2 &it) {
		container_const_reference<self_type>::assign(&it());
		it1_ = it.it1_;
		it2_ = it.it2_;
#ifdef BOOST_UBLAS_USE_INVARIANT_HOISTING
		it1_begin_ = it.it1_begin_;
		it1_end_ = it.it1_end_;
#endif
		return *this;
	}

	// Comparison
	
	bool operator == (const const_iterator2 &it) const {
		BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());
		BOOST_UBLAS_CHECK(it1_ == it.it1_, external_logic());
		return it2_ == it.it2_;
	}
	
	bool operator < (const const_iterator2 &it) const {
		BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());
		BOOST_UBLAS_CHECK(it1_ == it.it1_, external_logic());
		return it2_ < it.it2_;
	}

private:
	// Mutable due to assignment
	/* const */
	const_iterator11_type it1_;
	const_iterator22_type it2_;
#ifdef BOOST_UBLAS_USE_INVARIANT_HOISTING
	/* const */ const_iterator12_type it1_begin_;
	/* const */
	const_iterator12_type it1_end_;
#endif
	};
#endif

	
	const_iterator2 begin2() const {
		return find2(0, 0, 0);
	}
	
	const_iterator2 end2() const {
		return find2(0, 0, size2());
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
	expression1_closure_type m_expression1;
	expression2_closure_type m_expression2;
};

template<class T1, class E1, class T2, class E2>
struct matrix_matrix_binary_traits {
	typedef unknown_storage_tag storage_category;
	typedef unknown_orientation_tag orientation_category;
	typedef typename promote_traits<T1, T2>::promote_type promote_type;
	typedef matrix_matrix_binary<E1, E2, matrix_matrix_prod<E1, E2, promote_type> > expression_type;
#ifndef BOOST_UBLAS_SIMPLE_ET_DEBUG
	typedef expression_type result_type;
#else
	typedef typename E1::matrix_temporary_type result_type;
#endif
};

template<class E1, class E2>
typename matrix_matrix_binary_traits<typename E1::value_type, E1,
         typename E2::value_type, E2>::result_type
         prod(const matrix_expression<E1> &e1,
                 const matrix_expression<E2> &e2,
                 unknown_storage_tag,
unknown_orientation_tag) {
	typedef typename matrix_matrix_binary_traits<typename E1::value_type, E1,
	        typename E2::value_type, E2>::expression_type expression_type;
	return expression_type(e1(), e2());
}

// Dispatcher
template<class E1, class E2>
typename matrix_matrix_binary_traits<typename E1::value_type, E1,
         typename E2::value_type, E2>::result_type
         prod(const matrix_expression<E1> &e1,
const matrix_expression<E2> &e2) {
	typedef typename matrix_matrix_binary_traits<typename E1::value_type, E1,
	        typename E2::value_type, E2>::storage_category storage_category;
	typedef typename matrix_matrix_binary_traits<typename E1::value_type, E1,
	        typename E2::value_type, E2>::orientation_category orientation_category;
	return prod(e1, e2, storage_category(), orientation_category());
}

//~ template<class E1, class E2>
//~ typename matrix_matrix_binary_traits<typename type_traits<typename E1::value_type>::precision_type, E1,
         //~ typename type_traits<typename E2::value_type>::precision_type, E2>::result_type
         //~ prec_prod(const matrix_expression<E1> &e1,
                 //~ const matrix_expression<E2> &e2,
                 //~ unknown_storage_tag,
//~ unknown_orientation_tag) {
	//~ typedef typename matrix_matrix_binary_traits<typename type_traits<typename E1::value_type>::precision_type, E1,
	        //~ typename type_traits<typename E2::value_type>::precision_type, E2>::expression_type expression_type;
	//~ return expression_type(e1(), e2());
//~ }

// Dispatcher
//~ template<class E1, class E2>

//~ typename matrix_matrix_binary_traits<typename type_traits<typename E1::value_type>::precision_type, E1,
         //~ typename type_traits<typename E2::value_type>::precision_type, E2>::result_type
         //~ prec_prod(const matrix_expression<E1> &e1,
//~ const matrix_expression<E2> &e2) {
	//~ typedef typename matrix_matrix_binary_traits<typename type_traits<typename E1::value_type>::precision_type, E1,
	        //~ typename type_traits<typename E2::value_type>::precision_type, E2>::storage_category storage_category;
	//~ typedef typename matrix_matrix_binary_traits<typename type_traits<typename E1::value_type>::precision_type, E1,
	        //~ typename type_traits<typename E2::value_type>::precision_type, E2>::orientation_category orientation_category;
	//~ return prec_prod(e1, e2, storage_category(), orientation_category());
//~ }

template<class M, class E1, class E2>
M & prod(const matrix_expression<E1> &e1,
     const matrix_expression<E2> &e2,
     M &m) {
	return m.assign(prod(e1, e2));
}

//~ template<class M, class E1, class E2>
//~ M & prec_prod(const matrix_expression<E1> &e1,
        //~ const matrix_expression<E2> &e2,
        //~ M &m) {
	//~ return m.assign(prec_prod(e1, e2));
//~ }

template<class M, class E1, class E2>
M prod(const matrix_expression<E1> &e1,
     const matrix_expression<E2> &e2) {
	return M(prod(e1, e2));
}

//~ template<class M, class E1, class E2>
//~ M prec_prod(const matrix_expression<E1> &e1,
        //~ const matrix_expression<E2> &e2) {
	//~ return M(prec_prod(e1, e2));
//~ }


template<class E>
typename matrix_norm_1<E>::result_type
norm_1(const matrix_expression<E> &e) {
	return matrix_norm_1<E>::apply(e());
}

template<class E>
typename matrix_norm_frobenius<E>::result_type
norm_frobenius(const matrix_expression<E> &e) {
	return matrix_norm_frobenius<E>::apply(e());
}

template<class E>
typename matrix_norm_inf<E>::result_type
norm_inf(const matrix_expression<E> &e) {
	return matrix_norm_inf<E>::apply(e());
}

}
}

#endif
