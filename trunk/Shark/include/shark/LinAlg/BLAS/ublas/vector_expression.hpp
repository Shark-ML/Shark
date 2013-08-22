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

#ifndef _BOOST_UBLAS_VECTOR_EXPRESSION_
#define _BOOST_UBLAS_VECTOR_EXPRESSION_

#include <shark/LinAlg/BLAS/ublas/expression_types.hpp>


// Expression templates based on ideas of Todd Veldhuizen and Geoffrey Furnish
// Iterators based on ideas of Jeremy Siek
//
// Classes that model the Vector Expression concept

namespace shark {
namespace blas {

template<class E>
class vector_reference:
	public vector_expression<vector_reference<E> > {

	typedef vector_reference<E> self_type;
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
	typedef typename E::storage_category storage_category;

	// Construction and destruction
	
	explicit vector_reference(referred_type &e):
		e_(e) {}

	// Accessors
	
	size_type size() const {
		return expression().size();
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
	
	const_reference operator()(size_type i) const {
		return expression()(i);
	}
	reference operator()(size_type i) {
		return expression()(i);
	}

	
	const_reference operator [](size_type i) const {
		return expression() [i];
	}
	reference operator [](size_type i) {
		return expression() [i];
	}

	// Assignment
	vector_reference &operator = (const vector_reference &v) {
		expression().operator = (v);
		return *this;
	}
	template<class AE>
	vector_reference &operator = (const vector_expression<AE> &ae) {
		expression().operator = (ae);
		return *this;
	}
	template<class AE>
	vector_reference &assign(const vector_expression<AE> &ae) {
		expression().assign(ae);
		return *this;
	}
	template<class AE>
	vector_reference &operator += (const vector_expression<AE> &ae) {
		expression().operator += (ae);
		return *this;
	}
	template<class AE>
	vector_reference &plus_assign(const vector_expression<AE> &ae) {
		expression().plus_assign(ae);
		return *this;
	}
	template<class AE>
	vector_reference &operator -= (const vector_expression<AE> &ae) {
		expression().operator -= (ae);
		return *this;
	}
	template<class AE>
	
	vector_reference &minus_assign(const vector_expression<AE> &ae) {
		expression().minus_assign(ae);
		return *this;
	}
	template<class AT>
	vector_reference &operator *= (const AT &at) {
		expression().operator *= (at);
		return *this;
	}
	template<class AT>
	vector_reference &operator /= (const AT &at) {
		expression().operator /= (at);
		return *this;
	}

	// Swapping
	void swap(vector_reference &v) {
		expression().swap(v.expression());
	}

	// Closure comparison
	bool same_closure(const vector_reference &vr) const {
		return &(*this).e_ == &vr.e_;
	}

	// Iterator types
	typedef typename E::const_iterator const_iterator;
	typedef typename boost::mpl::if_<boost::is_const<E>,
	        typename E::const_iterator,
	        typename E::iterator>::type iterator;

	// Element lookup
	const_iterator find(size_type i) const {
		return expression().find(i);
	}
	iterator find(size_type i) {
		return expression().find(i);
	}

	// Iterator is the iterator of the referenced expression.
	const_iterator begin() const {
		return expression().begin();
	}
	const_iterator end() const {
		return expression().end();
	}
	iterator begin() {
		return expression().begin();
	}
	iterator end() {
		return expression().end();
	}

	// Reverse iterator
	typedef reverse_iterator_base<const_iterator> const_reverse_iterator;
	typedef reverse_iterator_base<iterator> reverse_iterator;

	const_reverse_iterator rbegin() const {
		return const_reverse_iterator(end());
	}
	const_reverse_iterator rend() const {
		return const_reverse_iterator(begin());
	}
	reverse_iterator rbegin() {
		return reverse_iterator(end());
	}
	reverse_iterator rend() {
		return reverse_iterator(begin());
	}

private:
	referred_type &e_;
};


///\brief class which allows for vector transformations
///
///transforms a vector Expression e of type E using a Function f of type F as an elementwise transformation f(e(i))
///This transformation needs f to be constant, meaning that applying f(x), f(y), f(z) yields the same results independent of the
///order of application. Also F must provide a type F::result_type indicating the result type of the functor.
///F must further provide a boolean flag F::zero_identity which indicates that f(0) = 0. This is needed for correct usage with sparse
///arguments - if f(0) != 0 this expression will be dense!
template<class E, class F>
class vector_unary:
	public vector_expression<vector_unary<E, F> > {
	typedef vector_unary<E, F> self_type;
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
	typedef value_type const *const_pointer;
	typedef value_type *pointer;

	typedef self_type const const_closure_type;
	typedef self_type closure_type;
	typedef unknown_storage_tag storage_category;

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

public:
	// Element access
	const_reference operator()(size_type i) const {
		return m_functor(m_expression(i));
	}

	const_reference operator[](size_type i) const {
		return m_functor(m_expression[i]);
	}

	// Closure comparison
	bool same_closure(vector_unary const &vu) const {
		return (*this).expression().same_closure(vu.expression());
	}

	// Iterator types

private:
	// Iterator enhances the iterator of the referenced expression
	// with the unary functor.

	//this is the version which tries to save the sparse property of the underlying vector expression
	//this is only possible, when the functor maps the 0 on itself, so f(0) = 0.
	//this version is also used when the underlying vector is dense.
	class default_const_iterator:
		public container_const_reference<self_type>,
		public iterator_base_traits<typename const_subiterator_type::iterator_category>::template
			iterator_base<default_const_iterator, value_type>::type {
	public:
		typedef typename const_subiterator_type::iterator_category iterator_category;
		typedef typename vector_unary::difference_type difference_type;
		typedef typename vector_unary::value_type value_type;
		typedef typename vector_unary::const_reference reference;
		typedef typename vector_unary::const_pointer pointer;

		// Construction and destruction
		default_const_iterator():container_const_reference<self_type> (), m_position() {}
		default_const_iterator(self_type const &ref,const_subiterator_type const &it, const_subiterator_type const &, size_type):
		container_const_reference<self_type> (ref), m_position(it) {}

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
		difference_type operator - (default_const_iterator const &it) const {
			BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());
			return m_position - it.m_position;
		}

		// Dereference
		const_reference operator * () const {
			return (*this)().m_functor(*m_position);
		}
		const_reference operator [](difference_type n) const {
			return *(*this + n);
		}

		// Index
		size_type index() const {
			return m_position.index();
		}

		// Assignment
		default_const_iterator &operator = (default_const_iterator const &it) {
			container_const_reference<self_type>::assign(&it());
			m_position = it.m_position;
			return *this;
		}

		// Comparison
		bool operator == (default_const_iterator const &it) const {
			BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());
			return m_position == it.m_position;
		}
		bool operator < (default_const_iterator const &it) const {
			BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());
			return m_position < it.m_position;
		}

	private:
		const_subiterator_type m_position;
	};

	///when the underlying container is sparse and Functor does not preserve the identity element, there is no way
	///around densifying the whole range. this way we avoid that empty elements are mapped to 0.
	class densifying_const_iterator:
		public container_const_reference<self_type>,
		public iterator_base_traits<std::bidirectional_iterator_tag>::template
			iterator_base<densifying_const_iterator, value_type>::type {
	public:
		typedef typename std::bidirectional_iterator_tag iterator_category;
		typedef typename vector_unary::difference_type difference_type;
		typedef typename vector_unary::value_type value_type;
		typedef typename vector_unary::const_reference reference;
		typedef typename vector_unary::const_pointer pointer;

		// Construction and destruction
		densifying_const_iterator():m_index(0) {}
		densifying_const_iterator(
			self_type const &ref,
			const_subiterator_type const &it,
			const_subiterator_type const &it_end,
			size_type index
		):container_const_reference<self_type> (ref),
		m_position(it), m_end(it_end), m_index(index),
		m_zeroValue(ref.m_functor(value_type(0))) {}

		// Arithmetic

		//we unfortunately have to solve the issue, that the user might use op-- and op++ together
		//so we can't assume that we only go steps forward but instead that we first going backward (such that m_position.index()<m_index)
		//and than going forward again (such that we first need to find m_position.index()>m_index)
		densifying_const_iterator &operator ++ () {
			while (m_position != m_end && m_index >= m_position.index()) {
				++m_position;
			}
			++m_index;
			return *this;
		}
		densifying_const_iterator &operator -- () {
			while (m_position != m_end && m_index <= m_position.index()) {
				--m_position;
			}
			--m_index;
			return *this;
		}

		// Dereference
		const_reference operator * () const {
			if (m_position == m_end || m_index != m_position.index())
				return m_zeroValue;
			else
				return (*this)().m_functor(*m_position);
		}

		// Index
		size_type index() const {
			return m_index;
		}

		// Assignment
		densifying_const_iterator &operator = (densifying_const_iterator const &it) {
			container_const_reference<self_type>::assign(&it());
			m_position = it.m_position;
			m_index = it.m_index;
			return *this;
		}

		// Comparison
		bool operator == (densifying_const_iterator const &it) const {
			BOOST_UBLAS_CHECK((*this)().same_closure(it()), external_logic());
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
	boost::is_same<typename const_subiterator_type::iterator_category, sparse_bidirectional_iterator_tag>::value,
	      densifying_const_iterator,
	      default_const_iterator
	      >::type const_iterator;
	typedef const_iterator iterator;

	// Element lookup
	const_iterator find(size_type i) const {
		return const_iterator(*this, m_expression.find(i),m_expression.end(),i);
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
SHARK_UNARY_VECTOR_TRANSFORMATION(operator-, scalar_negate)
SHARK_UNARY_VECTOR_TRANSFORMATION(conj, scalar_conj)
SHARK_UNARY_VECTOR_TRANSFORMATION(real, scalar_real)
SHARK_UNARY_VECTOR_TRANSFORMATION(imag, scalar_imag)
SHARK_UNARY_VECTOR_TRANSFORMATION(abs, scalar_abs)
SHARK_UNARY_VECTOR_TRANSFORMATION(log, scalar_log)
SHARK_UNARY_VECTOR_TRANSFORMATION(exp, scalar_exp)
SHARK_UNARY_VECTOR_TRANSFORMATION(tanh,scalar_tanh)
SHARK_UNARY_VECTOR_TRANSFORMATION(sqr, scalar_sqr)
SHARK_UNARY_VECTOR_TRANSFORMATION(abs_sqr, scalar_abs_sqr)
SHARK_UNARY_VECTOR_TRANSFORMATION(sqrt, scalar_sqrt)
SHARK_UNARY_VECTOR_TRANSFORMATION(sigmoid, scalar_sigmoid)
SHARK_UNARY_VECTOR_TRANSFORMATION(softPlus, scalar_soft_plus)
#undef SHARK_UNARY_VECTOR_TRANSFORMATION


//operations of the form op(v,t)[i] = op(v[i],t)
#define SHARK_VECTOR_SCALAR_TRANSFORMATION(name, F)\
template<class E> \
vector_unary<E,F<typename E::value_type> > \
name (vector_expression<E> const& e, typename E::value_type scalar){ \
	typedef F<typename E::value_type> functor_type; \
	return vector_unary<E, functor_type>(e, functor_type(scalar)); \
}
SHARK_VECTOR_SCALAR_TRANSFORMATION(operator*, scalar_multiply2)
SHARK_VECTOR_SCALAR_TRANSFORMATION(operator/, scalar_divide)
SHARK_VECTOR_SCALAR_TRANSFORMATION(operator<, scalar_less_than)
SHARK_VECTOR_SCALAR_TRANSFORMATION(operator<=, scalar_less_equal_than)
SHARK_VECTOR_SCALAR_TRANSFORMATION(operator>, scalar_bigger_than)
SHARK_VECTOR_SCALAR_TRANSFORMATION(operator>=, scalar_bigger_equal_than)
SHARK_VECTOR_SCALAR_TRANSFORMATION(operator==, scalar_equal)
SHARK_VECTOR_SCALAR_TRANSFORMATION(operator!=, scalar_not_equal)
#undef SHARK_VECTOR_SCALAR_TRANSFORMATION

// (t * v) [i] = t * v [i]
template<class T, class E>
typename boost::enable_if< 
	boost::is_convertible<T, typename E::value_type >,
        vector_unary<E,scalar_multiply1<typename E::value_type> >
>::type
operator * (T scalar, vector_expression<E> const& e){
	typedef scalar_multiply1<typename E::value_type> functor_type;
	return vector_unary<E, functor_type>(e, functor_type(scalar));
}

// pow(v,t)[i]= pow(v[i],t)
template<class E, class U>
vector_unary<E,scalar_pow<typename E::value_type, U> >
pow (vector_expression<E> const& e, U exponent){
	typedef scalar_pow<typename E::value_type, U> functor_type;
	return vector_unary<E, functor_type>(e, functor_type(exponent));
}


template<class E1, class E2, class F>
class vector_binary:
	public vector_expression<vector_binary<E1,E2, F> > {
	typedef vector_binary<E1,E2, F> self_type;
	typedef E1 const expression1_type;
	typedef E2 const expression2_type;
	typedef typename E1::const_iterator const_subiterator1_type;
	typedef typename E2::const_iterator const_subiterator2_type;
public:
	typedef F functor_type;
	typedef typename E1::const_closure_type expression_closure1_type;
	typedef typename E2::const_closure_type expression_closure2_type;
	typedef typename promote_traits<
		typename E1::size_type, 
		typename E2::size_type
	>::promote_type size_type;
	typedef typename promote_traits<
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
	typedef unknown_storage_tag storage_category;

	// Construction and destruction
	explicit vector_binary (
		vector_expression<E1> const& e1, 
		vector_expression<E2> const& e2,
		F functor
	):m_expression1(e1()),m_expression2(e2()), m_functor(functor) {
		//SIZE_CHECK(e1().size() == e2().size());
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
	const_reference operator() (size_type i) const {
		return m_functor(m_expression1(i),m_expression2(i));
	}

	const_reference operator[] (size_type i) const {
		return m_functor(m_expression1(i),m_expression2(i));
	}

	// Closure comparison
	bool same_closure (vector_binary const& vu) const {
		return expression1 ().same_closure (vu.expression1())
		&& expression2 ().same_closure (vu.expression2());
	}

	// Iterator types
	
	// Iterator enhances the iterator of the referenced expression2
	// with the unary functor.
	class const_iterator:
		public container_const_reference<self_type>,
		public iterator_base_traits<
			typename iterator_restrict_traits<
				typename const_subiterator1_type::iterator_category,
				typename const_subiterator2_type::iterator_category
			>::iterator_category
		>::template iterator_base<const_iterator, value_type>::type{
	public:
		typedef typename iterator_restrict_traits<
			typename const_subiterator1_type::iterator_category,
			typename const_subiterator2_type::iterator_category
		>::iterator_category iterator_category;
		typedef typename vector_binary::difference_type difference_type;
		typedef typename vector_binary::value_type value_type;
		typedef typename vector_binary::const_reference reference;
		typedef typename vector_binary::const_pointer pointer;

		// Construction and destruction
		const_iterator ():
			container_const_reference<self_type> (), m_index (), 
			m_iterator1 (), m_end1 (), 
			m_iterator2 (), m_end2 () {}
		const_iterator (
			self_type const& ref, size_type i,
			const_subiterator1_type const& it1, const_subiterator1_type const& end1,
			const_subiterator2_type const& it2, const_subiterator2_type const& end2
		):  container_const_reference<self_type>(ref), m_index(i), 
			m_iterator1(it1), m_end1(end1), 
			m_iterator2 (it2), m_end2 (end2) {}

	private: 
		//we need to handle all specializations independently from each other
		//also for packed/sparse we need to check, whether our functor has zero_identity 
		//to change the implementation accordingly
		//the correct choice is handled by ublas.
		
		// Dense specializations are easy
		void increment (dense_random_access_iterator_tag) {
			++ m_index; 
			++ m_iterator1; 
			++ m_iterator2;
		}
		void decrement (dense_random_access_iterator_tag) {
			-- m_index; 
			-- m_iterator1; 
			-- m_iterator2;
		}
		void increment (dense_random_access_iterator_tag, difference_type n) {
			m_index += n; 
			m_iterator1 += n; 
			m_iterator2 += n;
		}
		void decrement (dense_random_access_iterator_tag, difference_type n) {
			m_index -= n; 
			m_iterator1 -= n; 
			m_iterator2 -= n;
		}
		value_type dereference (dense_random_access_iterator_tag) const {
			return (*this)().m_functor(*m_iterator1, *m_iterator2);
		}

		// Sparse specializations
		void increment (sparse_bidirectional_iterator_tag) {
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
		void decrement (sparse_bidirectional_iterator_tag) {
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
		void increment (sparse_bidirectional_iterator_tag, difference_type n) {
			while (n > 0) {
				increment (sparse_bidirectional_iterator_tag ());
				--n;
			}
			while (n < 0) {
				decrement (sparse_bidirectional_iterator_tag ());
				++n;
			}
		}
		void decrement (sparse_bidirectional_iterator_tag, difference_type n) {
			while (n > 0) {
				decrement (sparse_bidirectional_iterator_tag ());
				--n;
			}
			while (n < 0) {
				increment (sparse_bidirectional_iterator_tag ());
				++n;
			}
		}
		value_type dereference (sparse_bidirectional_iterator_tag) const {
			value_type t1 = value_type/*zero*/();
			if (m_iterator1 != m_end1 && m_iterator1.index () == m_index)
				t1 = *m_iterator1;
			value_type t2 = value_type/*zero*/();
			if (m_iterator2 != m_end2 && m_iterator2.index () == m_index)
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
			BOOST_UBLAS_CHECK ((*this) ().same_closure (it ()), external_logic ());
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
			container_const_reference<self_type>::assign (&it ());
			m_index = it.m_index;
			m_iterator1 = it.m_iterator1;
			m_end1 = it.m_end1;
			m_iterator2 = it.m_iterator2;
			m_end2 = it.m_end2;
			return *this;
		}

		// Comparison
		bool operator == (const_iterator const& it) const {
			BOOST_UBLAS_CHECK ((*this) ().same_closure (it ()), external_logic ());
			return index () == it.index ();
		}
		bool operator < (const_iterator const& it) const {
			BOOST_UBLAS_CHECK ((*this) ().same_closure (it ()), external_logic ());
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
	typedef reverse_iterator_base<const_iterator> const_reverse_iterator;

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

#define SHARK_BINARY_VECTOR_EXPRESSION(name, F)\
template<class E1, class E2>\
vector_binary<E1, E2, F<typename E1::value_type, typename E2::value_type> >\
name(vector_expression<E1> const& e1, vector_expression<E2> const& e2){\
	typedef F<typename E1::value_type, typename E2::value_type> functor_type;\
	return vector_binary<E1, E2, functor_type>(e1,e2, functor_type());\
}
SHARK_BINARY_VECTOR_EXPRESSION(operator+, scalar_binary_plus)
SHARK_BINARY_VECTOR_EXPRESSION(operator-, scalar_binary_minus)
SHARK_BINARY_VECTOR_EXPRESSION(operator*, scalar_binary_multiply)
SHARK_BINARY_VECTOR_EXPRESSION(element_prod, scalar_binary_multiply)
SHARK_BINARY_VECTOR_EXPRESSION(operator/, scalar_binary_divide)
SHARK_BINARY_VECTOR_EXPRESSION(element_div, scalar_binary_divide)
#undef SHARK_BINARY_VECTOR_EXPRESSION

template<class E1, class E2>
vector_binary<E1, E2, 
	scalar_binary_safe_divide<typename E1::value_type, typename E2::value_type> 
>
safeDiv(
	vector_expression<E1> const& e1, 
	vector_expression<E2> const& e2, 
	typename promote_traits<
		typename E1::value_type, 
		typename E2::value_type
	>::promote_type defaultValue
){
	typedef scalar_binary_safe_divide<typename E1::value_type, typename E2::value_type> functor_type;
	return vector_binary<E1, E2, functor_type>(e1,e2, functor_type(defaultValue));
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

template<class E>
std::size_t arg_min(const vector_expression<E> &e) {
	SIZE_CHECK(e().size() > 0);
	return arg_max(-e);
}

////implement all the norms based on sum!

/// \brief norm_1 v = sum_i |v_i|
template<class E>
typename real_traits<typename E::value_type >::type
norm_1(const vector_expression<E> &e) {
	return sum(abs(e));
}

/// \brief norm_2 v = sum_i |v_i|^2
template<class E>
typename real_traits<typename E::value_type >::type
norm_sqr(const vector_expression<E> &e) {
	return sum(abs_sqr(e));
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
	return max(abs(e));
}

/// \brief index_norm_inf v = arg max_i |v_i|
template<class E>
std::size_t index_norm_inf(vector_expression<E> const &e){
	return arg_max(abs(e));
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
	vector_inner_prod<value_type> kernel;
	return kernel(e1,e2);
}

}
}

#endif
