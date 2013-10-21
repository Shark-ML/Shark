//
//  Copyright (c) 2000-2002
//  Joerg Walter, Mathias Koch
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  GeNeSys mbH & Co. KG in producing this work.s
//

#ifndef _BOOST_UBLAS_MATRIX_PROXY_
#define _BOOST_UBLAS_MATRIX_PROXY_

#include <shark/LinAlg/BLAS/ublas/matrix_expression.hpp>
#include <shark/LinAlg/BLAS/ublas/detail/vector_assign.hpp>
#include <shark/LinAlg/BLAS/ublas/detail/matrix_assign.hpp>
#include <shark/LinAlg/BLAS/ublas/detail/temporary.hpp>

// Iterators based on ideas of Jeremy Siek

namespace shark{ namespace blas{

	/// \brief matrox transpose
//~ template<class E>
//~ class matrix_transpose: public matrix_expression<matrix_transpose<E> > {
//~ public:
	//~ typedef typename boost::mpl::if_<boost::is_const<E>,
	        //~ typename E::const_closure_type,
	        //~ typename E::closure_type
	//~ >::type expression_closure_type;

	//~ typedef typename E::size_type size_type;
	//~ typedef typename E::difference_type difference_type;
	//~ typedef typename E::value_type value_type;
	//~ typedef typename E::const_reference const_reference;
	//~ typedef typename boost::mpl::if_<
		//~ boost::is_const<E>,
	        //~ typename E::const_reference,
	        //~ typename E::reference
	//~ >::type reference;

	//~ typedef const matrix_transpose<E> const_closure_type;
	//~ typedef matrix_transpose<E> closure_type;
	//~ typedef typename boost::mpl::if_<
		//~ boost::is_same<typename E::orientation_category,row_major_tag>,
	        //~ column_major_tag,
	        //~ typename boost::mpl::if_<
			//~ boost::is_same<typename E::orientation_category,column_major_tag>,
			//~ row_major_tag,
			//~ typename E::orientation_category
		//~ >::type
	//~ >::type orientation_category;
	//~ typedef typename E::storage_category storage_category;

	//~ // Construction and destruction
	//~ template<class AE>
	//~ explicit matrix_transpose(matrix_expression<AE>& e):
		//~ m_expression(e()) {}

	//~ template<class AE>
	//~ explicit matrix_transpose(matrix_expression<AE> const& e):
		//~ m_expression(e()) {}

	//~ // Accessors
	//~ size_type size1() const {
		//~ return m_expression.size2();
	//~ }
	//~ size_type size2() const {
		//~ return m_expression.size1();
	//~ }

	//~ // Expression accessors
	//~ expression_closure_type const& expression() const {
		//~ return m_expression;
	//~ }
	//~ expression_closure_type& expression(){
		//~ return m_expression;
	//~ }

	//~ // Element access
	//~ const_reference operator()(size_type i, size_type j) const{
		//~ return m_expression(j,i);
	//~ }
	//~ reference operator()(size_type i, size_type j) {
		//~ return m_expression(j, i);
	//~ }

	//~ // Closure comparison
	//~ bool same_closure(matrix_transpose const& mu2) const {
		//~ return (*this).expression().same_closure(mu2.expression());
	//~ }

	//~ // Iterator types
	//~ typedef typename expression_closure_type::const_iterator2 const_iterator1;
	//~ typedef typename expression_closure_type::iterator2 iterator1;
	//~ typedef typename expression_closure_type::const_iterator1 const_iterator2;
	//~ typedef typename expression_closure_type::iterator1 iterator2;

	//~ // Element lookup
	//~ const_iterator1 find1(int rank, size_type i, size_type j) const {
		//~ SIZE_CHECK(i <= size1());
		//~ SIZE_CHECK(j <= size2());
		//~ return m_expression.find2(rank, j, i);
	//~ }

	//~ const_iterator2 find2(int rank, size_type i, size_type j) const {
		//~ SIZE_CHECK(i <= size1());
		//~ SIZE_CHECK(j <= size2());
		//~ return m_expression.find1(rank, j, i);
	//~ }

	//~ iterator1 find1(int rank, size_type i, size_type j){
		//~ SIZE_CHECK(i <= size1());
		//~ SIZE_CHECK(j <= size2());
		//~ return m_expression.find2(rank, j, i);
	//~ }

	//~ iterator2 find2(int rank, size_type i, size_type j){
		//~ SIZE_CHECK(i <= size1());
		//~ SIZE_CHECK(j <= size2());
		//~ return m_expression.find1(rank, j, i);
	//~ }


	//~ //Iterators
	//~ const_iterator1 begin1() const {
		//~ return find1(0, 0, 0);
	//~ }
	//~ const_iterator1 end1() const {
		//~ return find1(0, size1(), 0);
	//~ }
	//~ const_iterator2 begin2() const {
		//~ return find2(0, 0, 0);
	//~ }
	//~ const_iterator2 end2() const {
		//~ return find2(0, 0, size2());
	//~ }

	//~ iterator1 begin1(){
		//~ return find1(0, 0, 0);
	//~ }
	//~ iterator1 end1(){
		//~ return find1(0, size1(), 0);
	//~ }
	//~ iterator2 begin2(){
		//~ return find2(0, 0, 0);
	//~ }
	//~ iterator2 end2(){
		//~ return find2(0, 0, size2());
	//~ }
//~ private:
	//~ expression_closure_type m_expression;
//~ };

// (trans m) [i] [j] = m [j] [i]
//~ template<class E>
//~ matrix_transpose<E const> trans(matrix_expression<E> const& e){
	//~ return matrix_transpose<E const>(e);
//~ }
//~ template<class E>
//~ matrix_transpose<E> trans(matrix_expression<E>& e){
	//~ return matrix_transpose<E>(e);
//~ }

    /** \brief
     */
    template<class M>
    class matrix_row:
        public vector_expression<matrix_row<M> > {

        typedef matrix_row<M> self_type;
    public:
#ifdef BOOST_UBLAS_ENABLE_PROXY_SHORTCUTS
        using vector_expression<self_type>::operator ();
#endif
        typedef M matrix_type;
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
        typedef typename storage_restrict_traits<typename M::storage_category,
                                                 dense_proxy_tag>::storage_category storage_category;

        // Construction and destruction

        matrix_row (matrix_type &data, size_type i):
            data_ (data), i_ (i) {
            // Early checking of preconditions here.
            // BOOST_UBLAS_CHECK (i_ < data_.size1 (), bad_index ());
        }

        // Accessors

        size_type size () const {
            return data_.size2 ();
        }

        size_type index () const {
            return i_;
        }

        // Storage accessors

        const matrix_closure_type &data () const {
            return data_;
        }

        matrix_closure_type &data () {
            return data_;
        }

        // Element access
#ifndef BOOST_UBLAS_PROXY_CONST_MEMBER

        const_reference operator () (size_type j) const {
            return data_ (i_, j);
        }

        reference operator () (size_type j) {
            return data_ (i_, j);
        }


        const_reference operator [] (size_type j) const {
            return (*this) (j);
        }

        reference operator [] (size_type j) {
            return (*this) (j);
        }
#else

        reference operator () (size_type j) const {
            return data_ (i_, j);
        }


        reference operator [] (size_type j) const {
            return (*this) (j);
        }
#endif

        // Assignment

        matrix_row &operator = (const matrix_row &mr) {
            // ISSUE need a temporary, proxy can be overlaping alias
            vector_assign<scalar_assign> (*this, typename vector_temporary_traits<M>::type (mr));
            return *this;
        }

        matrix_row &assign_temporary (matrix_row &mr) {
            // assign elements, proxied container remains the same
            vector_assign<scalar_assign> (*this, mr);
            return *this;
        }
        template<class AE>

        matrix_row &operator = (const vector_expression<AE> &ae) {
            vector_assign<scalar_assign> (*this, typename vector_temporary_traits<M>::type (ae));
            return *this;
        }
        template<class AE>

        matrix_row &assign (const vector_expression<AE> &ae) {
            vector_assign<scalar_assign> (*this, ae);
            return *this;
        }
        template<class AE>

        matrix_row &operator += (const vector_expression<AE> &ae) {
            vector_assign<scalar_assign> (*this, typename vector_temporary_traits<M>::type (*this + ae));
            return *this;
        }
        template<class AE>

        matrix_row &plus_assign (const vector_expression<AE> &ae) {
            vector_assign<scalar_plus_assign> (*this, ae);
            return *this;
        }
        template<class AE>

        matrix_row &operator -= (const vector_expression<AE> &ae) {
            vector_assign<scalar_assign> (*this, typename vector_temporary_traits<M>::type (*this - ae));
            return *this;
        }
        template<class AE>

        matrix_row &minus_assign (const vector_expression<AE> &ae) {
            vector_assign<scalar_minus_assign> (*this, ae);
            return *this;
        }
        template<class AT>

        matrix_row &operator *= (const AT &at) {
            vector_assign_scalar<scalar_multiplies_assign> (*this, at);
            return *this;
        }
        template<class AT>

        matrix_row &operator /= (const AT &at) {
            vector_assign_scalar<scalar_divides_assign> (*this, at);
            return *this;
        }

        // Closure comparison

        bool same_closure (const matrix_row &mr) const {
            return (*this).data_.same_closure (mr.data_);
        }

        // Comparison

        bool operator == (const matrix_row &mr) const {
            return (*this).data_ == mr.data_ && index () == mr.index ();
        }

        // Swapping

        void swap (matrix_row mr) {
            if (this != &mr) {
                BOOST_UBLAS_CHECK (size () == mr.size (), bad_size ());
                // Sparse ranges may be nonconformant now.
                // std::swap_ranges (begin (), end (), mr.begin ());
                vector_swap<scalar_swap> (*this, mr);
            }
        }

        friend void swap (matrix_row mr1, matrix_row mr2) {
            mr1.swap (mr2);
        }

        // Iterator types
    private:
        typedef typename M::const_iterator2 const_subiterator_type;
        typedef typename boost::mpl::if_<boost::is_const<M>,
                                          typename M::const_iterator2,
                                          typename M::iterator2>::type subiterator_type;

    public:
#ifdef BOOST_UBLAS_USE_INDEXED_ITERATOR
        typedef indexed_iterator<matrix_row<matrix_type>,
                                 typename subiterator_type::iterator_category> iterator;
        typedef indexed_const_iterator<matrix_row<matrix_type>,
                                       typename const_subiterator_type::iterator_category> const_iterator;
#else
        class const_iterator;
        class iterator;
#endif

        // Element lookup

        const_iterator find (size_type j) const {
            const_subiterator_type it2 (data_.find2 (1, i_, j));
#ifdef BOOST_UBLAS_USE_INDEXED_ITERATOR
            return const_iterator (*this, it2.index2 ());
#else
            return const_iterator (*this, it2);
#endif
        }

        iterator find (size_type j) {
            subiterator_type it2 (data_.find2 (1, i_, j));
#ifdef BOOST_UBLAS_USE_INDEXED_ITERATOR
            return iterator (*this, it2.index2 ());
#else
            return iterator (*this, it2);
#endif
        }

#ifndef BOOST_UBLAS_USE_INDEXED_ITERATOR
        class const_iterator:
            public container_const_reference<matrix_row>,
            public iterator_base_traits<typename const_subiterator_type::iterator_category>::template
                        iterator_base<const_iterator, value_type>::type {
        public:
            typedef typename const_subiterator_type::value_type value_type;
            typedef typename const_subiterator_type::difference_type difference_type;
            typedef typename const_subiterator_type::reference reference;
            typedef typename const_subiterator_type::pointer pointer;

            // Construction and destruction

            const_iterator ():
                container_const_reference<self_type> (), it_ () {}

            const_iterator (const self_type &mr, const const_subiterator_type &it):
                container_const_reference<self_type> (mr), it_ (it) {}

            const_iterator (const typename self_type::iterator &it):  // ISSUE self_type:: stops VC8 using std::iterator here
                container_const_reference<self_type> (it ()), it_ (it.it_) {}

            // Arithmetic

            const_iterator &operator ++ () {
                ++ it_;
                return *this;
            }

            const_iterator &operator -- () {
                -- it_;
                return *this;
            }

            const_iterator &operator += (difference_type n) {
                it_ += n;
                return *this;
            }

            const_iterator &operator -= (difference_type n) {
                it_ -= n;
                return *this;
            }

            difference_type operator - (const const_iterator &it) const {
                BOOST_UBLAS_CHECK ((*this) ().same_closure  (it ()), external_logic ());
                return it_ - it.it_;
            }

            // Dereference

            const_reference operator * () const {
                BOOST_UBLAS_CHECK (index () < (*this) ().size (), bad_index ());
                return *it_;
            }

            const_reference operator [] (difference_type n) const {
                return *(*this + n);
            }

            // Index

            size_type index () const {
                return it_.index2 ();
            }

            // Assignment

            const_iterator &operator = (const const_iterator &it) {
                container_const_reference<self_type>::assign (&it ());
                it_ = it.it_;
                return *this;
            }

            // Comparison

            bool operator == (const const_iterator &it) const {
                BOOST_UBLAS_CHECK ((*this) ().same_closure (it ()), external_logic ());
                return it_ == it.it_;
            }

            bool operator < (const const_iterator &it) const {
                BOOST_UBLAS_CHECK ((*this) ().same_closure (it ()), external_logic ());
                return it_ < it.it_;
            }

        private:
            const_subiterator_type it_;
        };
#endif


        const_iterator begin () const {
            return find (0);
        }

        const_iterator end () const {
            return find (size ());
        }

#ifndef BOOST_UBLAS_USE_INDEXED_ITERATOR
        class iterator:
            public container_reference<matrix_row>,
            public iterator_base_traits<typename subiterator_type::iterator_category>::template
                        iterator_base<iterator, value_type>::type {
        public:
            typedef typename subiterator_type::value_type value_type;
            typedef typename subiterator_type::difference_type difference_type;
            typedef typename subiterator_type::reference reference;
            typedef typename subiterator_type::pointer pointer;

            // Construction and destruction

            iterator ():
                container_reference<self_type> (), it_ () {}

            iterator (self_type &mr, const subiterator_type &it):
                container_reference<self_type> (mr), it_ (it) {}

            // Arithmetic

            iterator &operator ++ () {
                ++ it_;
                return *this;
            }

            iterator &operator -- () {
                -- it_;
                return *this;
            }

            iterator &operator += (difference_type n) {
                it_ += n;
                return *this;
            }

            iterator &operator -= (difference_type n) {
                it_ -= n;
                return *this;
            }

            difference_type operator - (const iterator &it) const {
                BOOST_UBLAS_CHECK ((*this) ().same_closure  (it ()), external_logic ());
                return it_ - it.it_;
            }

            // Dereference

            reference operator * () const {
                BOOST_UBLAS_CHECK (index () < (*this) ().size (), bad_index ());
                return *it_;
            }

            reference operator [] (difference_type n) const {
                return *(*this + n);
            }

            // Index

            size_type index () const {
                return it_.index2 ();
            }

            // Assignment

            iterator &operator = (const iterator &it) {
                container_reference<self_type>::assign (&it ());
                it_ = it.it_;
                return *this;
            }

            // Comparison

            bool operator == (const iterator &it) const {
                BOOST_UBLAS_CHECK ((*this) ().same_closure (it ()), external_logic ());
                return it_ == it.it_;
            }

            bool operator < (const iterator &it) const {
                BOOST_UBLAS_CHECK ((*this) ().same_closure (it ()), external_logic ());
                return it_ < it.it_;
            }

        private:
            subiterator_type it_;

            friend class const_iterator;
        };
#endif


        iterator begin () {
            return find (0);
        }

        iterator end () {
            return find (size ());
        }

        // Reverse iterator
        typedef reverse_iterator_base<const_iterator> const_reverse_iterator;
        typedef reverse_iterator_base<iterator> reverse_iterator;


        const_reverse_iterator rbegin () const {
            return const_reverse_iterator (end ());
        }

        const_reverse_iterator rend () const {
            return const_reverse_iterator (begin ());
        }

        reverse_iterator rbegin () {
            return reverse_iterator (end ());
        }

        reverse_iterator rend () {
            return reverse_iterator (begin ());
        }

    private:
        matrix_closure_type data_;
        size_type i_;
    };

    // Projections
    template<class M>

    matrix_row<M> row (M &data, typename M::size_type i) {
        return matrix_row<M> (data, i);
    }
    template<class M>

    const matrix_row<const M> row (const M &data, typename M::size_type i) {
        return matrix_row<const M> (data, i);
    }

    // Specialize temporary
    template <class M>
    struct vector_temporary_traits< matrix_row<M> >
    : vector_temporary_traits< M > {} ;
    template <class M>
    struct vector_temporary_traits< const matrix_row<M> >
    : vector_temporary_traits< M > {} ;

    // Matrix based column vector class
    template<class M>
    class matrix_column:
        public vector_expression<matrix_column<M> > {

        typedef matrix_column<M> self_type;
    public:
#ifdef BOOST_UBLAS_ENABLE_PROXY_SHORTCUTS
        using vector_expression<self_type>::operator ();
#endif
        typedef M matrix_type;
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
        typedef typename storage_restrict_traits<typename M::storage_category,
                                                 dense_proxy_tag>::storage_category storage_category;

        // Construction and destruction

        matrix_column (matrix_type &data, size_type j):
            data_ (data), j_ (j) {
            // Early checking of preconditions here.
            // BOOST_UBLAS_CHECK (j_ < data_.size2 (), bad_index ());
        }

        // Accessors

        size_type size () const {
            return data_.size1 ();
        }

        size_type index () const {
            return j_;
        }

        // Storage accessors

        const matrix_closure_type &data () const {
            return data_;
        }

        matrix_closure_type &data () {
            return data_;
        }

        // Element access
#ifndef BOOST_UBLAS_PROXY_CONST_MEMBER

        const_reference operator () (size_type i) const {
            return data_ (i, j_);
        }

        reference operator () (size_type i) {
            return data_ (i, j_);
        }


        const_reference operator [] (size_type i) const {
            return (*this) (i);
        }

        reference operator [] (size_type i) {
            return (*this) (i);
        }
#else

        reference operator () (size_type i) const {
            return data_ (i, j_);
        }


        reference operator [] (size_type i) const {
            return (*this) (i);
        }
#endif

        // Assignment

        matrix_column &operator = (const matrix_column &mc) {
            // ISSUE need a temporary, proxy can be overlaping alias
            vector_assign<scalar_assign> (*this, typename vector_temporary_traits<M>::type (mc));
            return *this;
        }

        matrix_column &assign_temporary (matrix_column &mc) {
            // assign elements, proxied container remains the same
            vector_assign<scalar_assign> (*this, mc);
            return *this;
        }
        template<class AE>

        matrix_column &operator = (const vector_expression<AE> &ae) {
            vector_assign<scalar_assign> (*this, typename vector_temporary_traits<M>::type (ae));
            return *this;
        }
        template<class AE>

        matrix_column &assign (const vector_expression<AE> &ae) {
            vector_assign<scalar_assign> (*this, ae);
            return *this;
        }
        template<class AE>

        matrix_column &operator += (const vector_expression<AE> &ae) {
            vector_assign<scalar_assign> (*this, typename vector_temporary_traits<M>::type (*this + ae));
            return *this;
        }
        template<class AE>

        matrix_column &plus_assign (const vector_expression<AE> &ae) {
            vector_assign<scalar_plus_assign> (*this, ae);
            return *this;
        }
        template<class AE>

        matrix_column &operator -= (const vector_expression<AE> &ae) {
            vector_assign<scalar_assign> (*this, typename vector_temporary_traits<M>::type (*this - ae));
            return *this;
        }
        template<class AE>

        matrix_column &minus_assign (const vector_expression<AE> &ae) {
            vector_assign<scalar_minus_assign> (*this, ae);
            return *this;
        }
        template<class AT>

        matrix_column &operator *= (const AT &at) {
            vector_assign_scalar<scalar_multiplies_assign> (*this, at);
            return *this;
        }
        template<class AT>

        matrix_column &operator /= (const AT &at) {
            vector_assign_scalar<scalar_divides_assign> (*this, at);
            return *this;
        }

        // Closure comparison

        bool same_closure (const matrix_column &mc) const {
            return (*this).data_.same_closure (mc.data_);
        }

        // Comparison

        bool operator == (const matrix_column &mc) const {
            return (*this).data_ == mc.data_ && index () == mc.index ();
        }

        // Swapping

        void swap (matrix_column mc) {
            if (this != &mc) {
                BOOST_UBLAS_CHECK (size () == mc.size (), bad_size ());
                // Sparse ranges may be nonconformant now.
                // std::swap_ranges (begin (), end (), mc.begin ());
                vector_swap<scalar_swap> (*this, mc);
            }
        }

        friend void swap (matrix_column mc1, matrix_column mc2) {
            mc1.swap (mc2);
        }

        // Iterator types
    private:
        typedef typename M::const_iterator1 const_subiterator_type;
        typedef typename boost::mpl::if_<boost::is_const<M>,
                                          typename M::const_iterator1,
                                          typename M::iterator1>::type subiterator_type;

    public:
#ifdef BOOST_UBLAS_USE_INDEXED_ITERATOR
        typedef indexed_iterator<matrix_column<matrix_type>,
                                 typename subiterator_type::iterator_category> iterator;
        typedef indexed_const_iterator<matrix_column<matrix_type>,
                                       typename const_subiterator_type::iterator_category> const_iterator;
#else
        class const_iterator;
        class iterator;
#endif

        // Element lookup

        const_iterator find (size_type i) const {
            const_subiterator_type it1 (data_.find1 (1, i, j_));
#ifdef BOOST_UBLAS_USE_INDEXED_ITERATOR
            return const_iterator (*this, it1.index1 ());
#else
            return const_iterator (*this, it1);
#endif
        }

        iterator find (size_type i) {
            subiterator_type it1 (data_.find1 (1, i, j_));
#ifdef BOOST_UBLAS_USE_INDEXED_ITERATOR
            return iterator (*this, it1.index1 ());
#else
            return iterator (*this, it1);
#endif
        }

#ifndef BOOST_UBLAS_USE_INDEXED_ITERATOR
        class const_iterator:
            public container_const_reference<matrix_column>,
            public iterator_base_traits<typename const_subiterator_type::iterator_category>::template
                        iterator_base<const_iterator, value_type>::type {
        public:
            typedef typename const_subiterator_type::value_type value_type;
            typedef typename const_subiterator_type::difference_type difference_type;
            typedef typename const_subiterator_type::reference reference;
            typedef typename const_subiterator_type::pointer pointer;

            // Construction and destruction

            const_iterator ():
                container_const_reference<self_type> (), it_ () {}

            const_iterator (const self_type &mc, const const_subiterator_type &it):
                container_const_reference<self_type> (mc), it_ (it) {}

            const_iterator (const typename self_type::iterator &it):  // ISSUE self_type:: stops VC8 using std::iterator here
                container_const_reference<self_type> (it ()), it_ (it.it_) {}

            // Arithmetic

            const_iterator &operator ++ () {
                ++ it_;
                return *this;
            }

            const_iterator &operator -- () {
                -- it_;
                return *this;
            }

            const_iterator &operator += (difference_type n) {
                it_ += n;
                return *this;
            }

            const_iterator &operator -= (difference_type n) {
                it_ -= n;
                return *this;
            }

            difference_type operator - (const const_iterator &it) const {
                BOOST_UBLAS_CHECK ((*this) ().same_closure  (it ()), external_logic ());
                return it_ - it.it_;
            }

            // Dereference

            const_reference operator * () const {
                BOOST_UBLAS_CHECK (index () < (*this) ().size (), bad_index ());
                return *it_;
            }

            const_reference operator [] (difference_type n) const {
                return *(*this + n);
            }

            // Index

            size_type index () const {
                return it_.index1 ();
            }

            // Assignment

            const_iterator &operator = (const const_iterator &it) {
                container_const_reference<self_type>::assign (&it ());
                it_ = it.it_;
                return *this;
            }

            // Comparison

            bool operator == (const const_iterator &it) const {
                BOOST_UBLAS_CHECK ((*this) ().same_closure (it ()), external_logic ());
                return it_ == it.it_;
            }

            bool operator < (const const_iterator &it) const {
                BOOST_UBLAS_CHECK ((*this) ().same_closure (it ()), external_logic ());
                return it_ < it.it_;
            }

        private:
            const_subiterator_type it_;
        };
#endif


        const_iterator begin () const {
            return find (0);
        }

        const_iterator end () const {
            return find (size ());
        }

#ifndef BOOST_UBLAS_USE_INDEXED_ITERATOR
        class iterator:
            public container_reference<matrix_column>,
            public iterator_base_traits<typename subiterator_type::iterator_category>::template
                        iterator_base<iterator, value_type>::type {
        public:
            typedef typename subiterator_type::value_type value_type;
            typedef typename subiterator_type::difference_type difference_type;
            typedef typename subiterator_type::reference reference;
            typedef typename subiterator_type::pointer pointer;

            // Construction and destruction

            iterator ():
                container_reference<self_type> (), it_ () {}

            iterator (self_type &mc, const subiterator_type &it):
                container_reference<self_type> (mc), it_ (it) {}

            // Arithmetic

            iterator &operator ++ () {
                ++ it_;
                return *this;
            }

            iterator &operator -- () {
                -- it_;
                return *this;
            }

            iterator &operator += (difference_type n) {
                it_ += n;
                return *this;
            }

            iterator &operator -= (difference_type n) {
                it_ -= n;
                return *this;
            }

            difference_type operator - (const iterator &it) const {
                BOOST_UBLAS_CHECK ((*this) ().same_closure  (it ()), external_logic ());
                return it_ - it.it_;
            }

            // Dereference

            reference operator * () const {
                BOOST_UBLAS_CHECK (index () < (*this) ().size (), bad_index ());
                return *it_;
            }

            reference operator [] (difference_type n) const {
                return *(*this + n);
            }

            // Index

            size_type index () const {
                return it_.index1 ();
            }

            // Assignment

            iterator &operator = (const iterator &it) {
                container_reference<self_type>::assign (&it ());
                it_ = it.it_;
                return *this;
            }

            // Comparison

            bool operator == (const iterator &it) const {
                BOOST_UBLAS_CHECK ((*this) ().same_closure (it ()), external_logic ());
                return it_ == it.it_;
            }

            bool operator < (const iterator &it) const {
                BOOST_UBLAS_CHECK ((*this) ().same_closure (it ()), external_logic ());
                return it_ < it.it_;
            }

        private:
            subiterator_type it_;

            friend class const_iterator;
        };
#endif


        iterator begin () {
            return find (0);
        }

        iterator end () {
            return find (size ());
        }

        // Reverse iterator
        typedef reverse_iterator_base<const_iterator> const_reverse_iterator;
        typedef reverse_iterator_base<iterator> reverse_iterator;


        const_reverse_iterator rbegin () const {
            return const_reverse_iterator (end ());
        }

        const_reverse_iterator rend () const {
            return const_reverse_iterator (begin ());
        }
        reverse_iterator rbegin () {
            return reverse_iterator (end ());
        }

        reverse_iterator rend () {
            return reverse_iterator (begin ());
        }

    private:
        matrix_closure_type data_;
        size_type j_;
    };

    // Projections
    template<class M>

    matrix_column<M> column (M &data, typename M::size_type j) {
        return matrix_column<M> (data, j);
    }
    template<class M>

    const matrix_column<const M> column (const M &data, typename M::size_type j) {
        return matrix_column<const M> (data, j);
    }

    // Specialize temporary
    template <class M>
    struct vector_temporary_traits< matrix_column<M> >
    : vector_temporary_traits< M > {} ;
    template <class M>
    struct vector_temporary_traits< const matrix_column<M> >
    : vector_temporary_traits< M > {} ;

    // Matrix based vector range class
    template<class M>
    class matrix_vector_range:
        public vector_expression<matrix_vector_range<M> > {

        typedef matrix_vector_range<M> self_type;
    public:
#ifdef BOOST_UBLAS_ENABLE_PROXY_SHORTCUTS
        using vector_expression<self_type>::operator ();
#endif
        typedef M matrix_type;
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
        typedef basic_range<size_type, difference_type> range_type;
        typedef const self_type const_closure_type;
        typedef self_type closure_type;
        typedef typename storage_restrict_traits<typename M::storage_category,
                                                 dense_proxy_tag>::storage_category storage_category;

        // Construction and destruction

        matrix_vector_range (matrix_type &data, const range_type &r1, const range_type &r2):
            data_ (data), r1_ (r1.preprocess (data.size1 ())), r2_ (r2.preprocess (data.size2 ())) {
            // Early checking of preconditions here.
            // BOOST_UBLAS_CHECK (r1_.start () <= data_.size1 () &&
            //                     r1_.start () + r1_.size () <= data_.size1 (), bad_index ());
            // BOOST_UBLAS_CHECK (r2_.start () <= data_.size2 () &&
            //                   r2_.start () + r2_.size () <= data_.size2 (), bad_index ());
            // BOOST_UBLAS_CHECK (r1_.size () == r2_.size (), bad_size ());
        }

        // Accessors

        size_type start1 () const {
            return r1_.start ();
        }

        size_type start2 () const {
            return r2_.start ();
        }

        size_type size () const {
            return BOOST_UBLAS_SAME (r1_.size (), r2_.size ());
        }

        // Storage accessors

        const matrix_closure_type &data () const {
            return data_;
        }

        matrix_closure_type &data () {
            return data_;
        }

        // Element access
#ifndef BOOST_UBLAS_PROXY_CONST_MEMBER

        const_reference operator () (size_type i) const {
            return data_ (r1_ (i), r2_ (i));
        }

        reference operator () (size_type i) {
            return data_ (r1_ (i), r2_ (i));
        }


        const_reference operator [] (size_type i) const {
            return (*this) (i);
        }

        reference operator [] (size_type i) {
            return (*this) (i);
        }
#else

        reference operator () (size_type i) const {
            return data_ (r1_ (i), r2_ (i));
        }


        reference operator [] (size_type i) const {
            return (*this) (i);
        }
#endif

        // Assignment

        matrix_vector_range &operator = (const matrix_vector_range &mvr) {
            // ISSUE need a temporary, proxy can be overlaping alias
            vector_assign<scalar_assign> (*this, typename vector_temporary_traits<M>::type (mvr));
            return *this;
        }

        matrix_vector_range &assign_temporary (matrix_vector_range &mvr) {
            // assign elements, proxied container remains the same
            vector_assign<scalar_assign> (*this, mvr);
            return *this;
        }
        template<class AE>

        matrix_vector_range &operator = (const vector_expression<AE> &ae) {
            vector_assign<scalar_assign> (*this, typename vector_temporary_traits<M>::type (ae));
            return *this;
        }
        template<class AE>

        matrix_vector_range &assign (const vector_expression<AE> &ae) {
            vector_assign<scalar_assign> (*this, ae);
            return *this;
        }
        template<class AE>

        matrix_vector_range &operator += (const vector_expression<AE> &ae) {
            vector_assign<scalar_assign> (*this, typename vector_temporary_traits<M>::type (*this + ae));
            return *this;
        }
        template<class AE>

        matrix_vector_range &plus_assign (const vector_expression<AE> &ae) {
            vector_assign<scalar_plus_assign> (*this, ae);
            return *this;
        }
        template<class AE>

        matrix_vector_range &operator -= (const vector_expression<AE> &ae) {
            vector_assign<scalar_assign> (*this, typename vector_temporary_traits<M>::type (*this - ae));
            return *this;
        }
        template<class AE>

        matrix_vector_range &minus_assign (const vector_expression<AE> &ae) {
            vector_assign<scalar_minus_assign> (*this, ae);
            return *this;
        }
        template<class AT>

        matrix_vector_range &operator *= (const AT &at) {
            vector_assign_scalar<scalar_multiplies_assign> (*this, at);
            return *this;
        }
        template<class AT>

        matrix_vector_range &operator /= (const AT &at) {
            vector_assign_scalar<scalar_divides_assign> (*this, at);
            return *this;
        }

        // Closure comparison

        bool same_closure (const matrix_vector_range &mvr) const {
            return (*this).data_.same_closure (mvr.data_);
        }

        // Comparison

        bool operator == (const matrix_vector_range &mvr) const {
            return (*this).data_ == mvr.data_ && r1_ == mvr.r1_ && r2_ == mvr.r2_;
        }

        // Swapping

        void swap (matrix_vector_range mvr) {
            if (this != &mvr) {
                BOOST_UBLAS_CHECK (size () == mvr.size (), bad_size ());
                // Sparse ranges may be nonconformant now.
                // std::swap_ranges (begin (), end (), mvr.begin ());
                vector_swap<scalar_swap> (*this, mvr);
            }
        }

        friend void swap (matrix_vector_range mvr1, matrix_vector_range mvr2) {
            mvr1.swap (mvr2);
        }

        // Iterator types
    private:
        // Use range as an index - FIXME this fails for packed assignment
        typedef typename range_type::const_iterator const_subiterator1_type;
        typedef typename range_type::const_iterator subiterator1_type;
        typedef typename range_type::const_iterator const_subiterator2_type;
        typedef typename range_type::const_iterator subiterator2_type;

    public:
        class const_iterator;
        class iterator;

        // Element lookup

        const_iterator find (size_type i) const {
            return const_iterator (*this, r1_.begin () + i, r2_.begin () + i);
        }

        iterator find (size_type i) {
            return iterator (*this, r1_.begin () + i, r2_.begin () + i);
        }

        class const_iterator:
            public container_const_reference<matrix_vector_range>,
            public iterator_base_traits<typename M::const_iterator1::iterator_category>::template
                        iterator_base<const_iterator, value_type>::type {
        public:
            // FIXME Iterator can never be different code was:
            // typename iterator_restrict_traits<typename M::const_iterator1::iterator_category, typename M::const_iterator2::iterator_category>::iterator_category>
            #ifndef DOXYGEN_SHOULD_SKIP_THIS
                BOOST_STATIC_ASSERT ((boost::is_same<typename M::const_iterator1::iterator_category, typename M::const_iterator2::iterator_category>::value ));
            #endif /* DOXYGEN_SHOULD_SKIP_THIS */

            typedef typename matrix_vector_range::value_type value_type;
            typedef typename matrix_vector_range::difference_type difference_type;
            typedef typename matrix_vector_range::const_reference reference;
            typedef const typename matrix_vector_range::value_type *pointer;

            // Construction and destruction

            const_iterator ():
                container_const_reference<self_type> (), it1_ (), it2_ () {}

            const_iterator (const self_type &mvr, const const_subiterator1_type &it1, const const_subiterator2_type &it2):
                container_const_reference<self_type> (mvr), it1_ (it1), it2_ (it2) {}

            const_iterator (const typename self_type::iterator &it):  // ISSUE self_type:: stops VC8 using std::iterator here
                container_const_reference<self_type> (it ()), it1_ (it.it1_), it2_ (it.it2_) {}

            // Arithmetic

            const_iterator &operator ++ () {
                ++ it1_;
                ++ it2_;
                return *this;
            }

            const_iterator &operator -- () {
                -- it1_;
                -- it2_;
                return *this;
            }

            const_iterator &operator += (difference_type n) {
                it1_ += n;
                it2_ += n;
                return *this;
            }

            const_iterator &operator -= (difference_type n) {
                it1_ -= n;
                it2_ -= n;
                return *this;
            }

            difference_type operator - (const const_iterator &it) const {
                BOOST_UBLAS_CHECK ((*this) ().same_closure  (it ()), external_logic ());
                return BOOST_UBLAS_SAME (it1_ - it.it1_, it2_ - it.it2_);
            }

            // Dereference

            const_reference operator * () const {
                // FIXME replace find with at_element
                return (*this) ().data_ (*it1_, *it2_);
            }

            const_reference operator [] (difference_type n) const {
                return *(*this + n);
            }

            // Index

            size_type  index () const {
                return BOOST_UBLAS_SAME (it1_.index (), it2_.index ());
            }

            // Assignment

            const_iterator &operator = (const const_iterator &it) {
                container_const_reference<self_type>::assign (&it ());
                it1_ = it.it1_;
                it2_ = it.it2_;
                return *this;
            }

            // Comparison

            bool operator == (const const_iterator &it) const {
                BOOST_UBLAS_CHECK ((*this) ().same_closure (it ()), external_logic ());
                return it1_ == it.it1_ && it2_ == it.it2_;
            }

            bool operator < (const const_iterator &it) const {
                BOOST_UBLAS_CHECK ((*this) ().same_closure (it ()), external_logic ());
                return it1_ < it.it1_ && it2_ < it.it2_;
            }

        private:
            const_subiterator1_type it1_;
            const_subiterator2_type it2_;
        };


        const_iterator begin () const {
            return find (0);
        }

        const_iterator end () const {
            return find (size ());
        }

        class iterator:
            public container_reference<matrix_vector_range>,
            public iterator_base_traits<typename M::iterator1::iterator_category>::template
                        iterator_base<iterator, value_type>::type {
        public:
            // FIXME Iterator can never be different code was:
            // typename iterator_restrict_traits<typename M::const_iterator1::iterator_category, typename M::const_iterator2::iterator_category>::iterator_category>
            BOOST_STATIC_ASSERT ((boost::is_same<typename M::const_iterator1::iterator_category, typename M::const_iterator2::iterator_category>::value ));

            typedef typename matrix_vector_range::value_type value_type;
            typedef typename matrix_vector_range::difference_type difference_type;
            typedef typename matrix_vector_range::reference reference;
            typedef typename matrix_vector_range::value_type *pointer;

            // Construction and destruction

            iterator ():
                container_reference<self_type> (), it1_ (), it2_ () {}

            iterator (self_type &mvr, const subiterator1_type &it1, const subiterator2_type &it2):
                container_reference<self_type> (mvr), it1_ (it1), it2_ (it2) {}

            // Arithmetic

            iterator &operator ++ () {
                ++ it1_;
                ++ it2_;
                return *this;
            }

            iterator &operator -- () {
                -- it1_;
                -- it2_;
                return *this;
            }

            iterator &operator += (difference_type n) {
                it1_ += n;
                it2_ += n;
                return *this;
            }

            iterator &operator -= (difference_type n) {
                it1_ -= n;
                it2_ -= n;
                return *this;
            }

            difference_type operator - (const iterator &it) const {
                BOOST_UBLAS_CHECK ((*this) ().same_closure  (it ()), external_logic ());
                return BOOST_UBLAS_SAME (it1_ - it.it1_, it2_ - it.it2_);
            }

            // Dereference

            reference operator * () const {
                // FIXME replace find with at_element
                return (*this) ().data_ (*it1_, *it2_);
            }

            reference operator [] (difference_type n) const {
                return *(*this + n);
            }

            // Index

            size_type index () const {
                return BOOST_UBLAS_SAME (it1_.index (), it2_.index ());
            }

            // Assignment

            iterator &operator = (const iterator &it) {
                container_reference<self_type>::assign (&it ());
                it1_ = it.it1_;
                it2_ = it.it2_;
                return *this;
            }

            // Comparison

            bool operator == (const iterator &it) const {
                BOOST_UBLAS_CHECK ((*this) ().same_closure (it ()), external_logic ());
                return it1_ == it.it1_ && it2_ == it.it2_;
            }

            bool operator < (const iterator &it) const {
                BOOST_UBLAS_CHECK ((*this) ().same_closure (it ()), external_logic ());
                return it1_ < it.it1_ && it2_ < it.it2_;
            }

        private:
            subiterator1_type it1_;
            subiterator2_type it2_;

            friend class const_iterator;
        };


        iterator begin () {
            return find (0);
        }

        iterator end () {
            return find (size ());
        }

        // Reverse iterator
        typedef reverse_iterator_base<const_iterator> const_reverse_iterator;
        typedef reverse_iterator_base<iterator> reverse_iterator;


        const_reverse_iterator rbegin () const {
            return const_reverse_iterator (end ());
        }

        const_reverse_iterator rend () const {
            return const_reverse_iterator (begin ());
        }

        reverse_iterator rbegin () {
            return reverse_iterator (end ());
        }

        reverse_iterator rend () {
            return reverse_iterator (begin ());
        }

    private:
        matrix_closure_type data_;
        range_type r1_;
        range_type r2_;
    };

    // Specialize temporary
    template <class M>
    struct vector_temporary_traits< matrix_vector_range<M> >
    : vector_temporary_traits< M > {} ;
    template <class M>
    struct vector_temporary_traits< const matrix_vector_range<M> >
    : vector_temporary_traits< M > {} ;

    // Matrix based range class
    template<class M>
    class matrix_range:
        public matrix_expression<matrix_range<M> > {

        typedef matrix_range<M> self_type;
    public:
#ifdef BOOST_UBLAS_ENABLE_PROXY_SHORTCUTS
        using matrix_expression<self_type>::operator ();
#endif
        typedef M matrix_type;
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
        typedef basic_range<size_type, difference_type> range_type;
        typedef const self_type const_closure_type;
        typedef self_type closure_type;
        typedef typename storage_restrict_traits<typename M::storage_category,
                                                 dense_proxy_tag>::storage_category storage_category;
        typedef typename M::orientation_category orientation_category;

        // Construction and destruction

        matrix_range (matrix_type &data, const range_type &r1, const range_type &r2):
            data_ (data), r1_ (r1.preprocess (data.size1 ())), r2_ (r2.preprocess (data.size2 ())) {
            // Early checking of preconditions here.
            // BOOST_UBLAS_CHECK (r1_.start () <= data_.size1 () &&
            //                    r1_.start () + r1_.size () <= data_.size1 (), bad_index ());
            // BOOST_UBLAS_CHECK (r2_.start () <= data_.size2 () &&
            //                    r2_.start () + r2_.size () <= data_.size2 (), bad_index ());
        }

        matrix_range (const matrix_closure_type &data, const range_type &r1, const range_type &r2, int):
            data_ (data), r1_ (r1.preprocess (data.size1 ())), r2_ (r2.preprocess (data.size2 ())) {
            // Early checking of preconditions here.
            // BOOST_UBLAS_CHECK (r1_.start () <= data_.size1 () &&
            //                    r1_.start () + r1_.size () <= data_.size1 (), bad_index ());
            // BOOST_UBLAS_CHECK (r2_.start () <= data_.size2 () &&
            //                    r2_.start () + r2_.size () <= data_.size2 (), bad_index ());
        }

        // Accessors

        size_type start1 () const {
            return r1_.start ();
        }

        size_type size1 () const {
            return r1_.size ();
        }

        size_type start2() const {
            return r2_.start ();
        }

        size_type size2 () const {
            return r2_.size ();
        }

        // Storage accessors

        const matrix_closure_type &data () const {
            return data_;
        }

        matrix_closure_type &data () {
            return data_;
        }

        // Element access
#ifndef BOOST_UBLAS_PROXY_CONST_MEMBER

        const_reference operator () (size_type i, size_type j) const {
            return data_ (r1_ (i), r2_ (j));
        }

        reference operator () (size_type i, size_type j) {
            return data_ (r1_ (i), r2_ (j));
        }
#else

        reference operator () (size_type i, size_type j) const {
            return data_ (r1_ (i), r2_ (j));
        }
#endif

        // ISSUE can this be done in free project function?
        // Although a const function can create a non-const proxy to a non-const object
        // Critical is that matrix_type and data_ (vector_closure_type) are const correct

        matrix_range<matrix_type> project (const range_type &r1, const range_type &r2) const {
            return matrix_range<matrix_type>  (data_, r1_.compose (r1.preprocess (data_.size1 ())), r2_.compose (r2.preprocess (data_.size2 ())), 0);
        }

        // Assignment

        matrix_range &operator = (const matrix_range &mr) {
            matrix_assign<scalar_assign> (*this, mr);
            return *this;
        }

        matrix_range &assign_temporary (matrix_range &mr) {
            return *this = mr;
        }
        template<class AE>

        matrix_range &operator = (const matrix_expression<AE> &ae) {
            matrix_assign<scalar_assign> (*this, typename matrix_temporary_traits<M>::type (ae));
            return *this;
        }
        template<class AE>

        matrix_range &assign (const matrix_expression<AE> &ae) {
            matrix_assign<scalar_assign> (*this, ae);
            return *this;
        }
        template<class AE>

        matrix_range& operator += (const matrix_expression<AE> &ae) {
            matrix_assign<scalar_assign> (*this, typename matrix_temporary_traits<M>::type (*this + ae));
            return *this;
        }
        template<class AE>

        matrix_range &plus_assign (const matrix_expression<AE> &ae) {
            matrix_assign<scalar_plus_assign> (*this, ae);
            return *this;
        }
        template<class AE>

        matrix_range& operator -= (const matrix_expression<AE> &ae) {
            matrix_assign<scalar_assign> (*this, typename matrix_temporary_traits<M>::type (*this - ae));
            return *this;
        }
        template<class AE>

        matrix_range &minus_assign (const matrix_expression<AE> &ae) {
            matrix_assign<scalar_minus_assign> (*this, ae);
            return *this;
        }
        template<class AT>

        matrix_range& operator *= (const AT &at) {
            matrix_assign_scalar<scalar_multiplies_assign> (*this, at);
            return *this;
        }
        template<class AT>

        matrix_range& operator /= (const AT &at) {
            matrix_assign_scalar<scalar_divides_assign> (*this, at);
            return *this;
        }

        // Closure comparison

        bool same_closure (const matrix_range &mr) const {
            return (*this).data_.same_closure (mr.data_);
        }

        // Comparison

        bool operator == (const matrix_range &mr) const {
            return (*this).data_ == (mr.data_) && r1_ == mr.r1_ && r2_ == mr.r2_;
        }

        // Swapping

        void swap (matrix_range mr) {
            if (this != &mr) {
                BOOST_UBLAS_CHECK (size1 () == mr.size1 (), bad_size ());
                BOOST_UBLAS_CHECK (size2 () == mr.size2 (), bad_size ());
                matrix_swap<scalar_swap> (*this, mr);
            }
        }

        friend void swap (matrix_range mr1, matrix_range mr2) {
            mr1.swap (mr2);
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
        typedef indexed_iterator1<matrix_range<matrix_type>,
                                  typename subiterator1_type::iterator_category> iterator1;
        typedef indexed_iterator2<matrix_range<matrix_type>,
                                  typename subiterator2_type::iterator_category> iterator2;
        typedef indexed_const_iterator1<matrix_range<matrix_type>,
                                        typename const_subiterator1_type::iterator_category> const_iterator1;
        typedef indexed_const_iterator2<matrix_range<matrix_type>,
                                        typename const_subiterator2_type::iterator_category> const_iterator2;
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

        const_iterator1 find1 (int rank, size_type i, size_type j) const {
            const_subiterator1_type it1 (data_.find1 (rank, start1 () + i, start2 () + j));
#ifdef BOOST_UBLAS_USE_INDEXED_ITERATOR
            return const_iterator1 (*this, it1.index1 (), it1.index2 ());
#else
            return const_iterator1 (*this, it1);
#endif
        }

        iterator1 find1 (int rank, size_type i, size_type j) {
            subiterator1_type it1 (data_.find1 (rank, start1 () + i, start2 () + j));
#ifdef BOOST_UBLAS_USE_INDEXED_ITERATOR
            return iterator1 (*this, it1.index1 (), it1.index2 ());
#else
            return iterator1 (*this, it1);
#endif
        }

        const_iterator2 find2 (int rank, size_type i, size_type j) const {
            const_subiterator2_type it2 (data_.find2 (rank, start1 () + i, start2 () + j));
#ifdef BOOST_UBLAS_USE_INDEXED_ITERATOR
            return const_iterator2 (*this, it2.index1 (), it2.index2 ());
#else
            return const_iterator2 (*this, it2);
#endif
        }

        iterator2 find2 (int rank, size_type i, size_type j) {
            subiterator2_type it2 (data_.find2 (rank, start1 () + i, start2 () + j));
#ifdef BOOST_UBLAS_USE_INDEXED_ITERATOR
            return iterator2 (*this, it2.index1 (), it2.index2 ());
#else
            return iterator2 (*this, it2);
#endif
        }


#ifndef BOOST_UBLAS_USE_INDEXED_ITERATOR
        class const_iterator1:
            public container_const_reference<matrix_range>,
            public iterator_base_traits<typename const_subiterator1_type::iterator_category>::template
                        iterator_base<const_iterator1, value_type>::type {
        public:
            typedef typename const_subiterator1_type::value_type value_type;
            typedef typename const_subiterator1_type::difference_type difference_type;
            typedef typename const_subiterator1_type::reference reference;
            typedef typename const_subiterator1_type::pointer pointer;
            typedef const_iterator2 dual_iterator_type;
            typedef const_reverse_iterator2 dual_reverse_iterator_type;

            // Construction and destruction

            const_iterator1 ():
                container_const_reference<self_type> (), it_ () {}

            const_iterator1 (const self_type &mr, const const_subiterator1_type &it):
                container_const_reference<self_type> (mr), it_ (it) {}

            const_iterator1 (const iterator1 &it):
                container_const_reference<self_type> (it ()), it_ (it.it_) {}

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
                BOOST_UBLAS_CHECK ((*this) ().same_closure (it ()), external_logic ());
                return it_ - it.it_;
            }

            // Dereference

            const_reference operator * () const {
                return *it_;
            }

            const_reference operator [] (difference_type n) const {
                return *(*this + n);
            }

#ifndef BOOST_UBLAS_NO_NESTED_CLASS_RELATION

#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
            typename self_type::
#endif
            const_iterator2 begin () const {
                const self_type &mr = (*this) ();
                return mr.find2 (1, index1 (), 0);
            }

#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
            typename self_type::
#endif
            const_iterator2 end () const {
                const self_type &mr = (*this) ();
                return mr.find2 (1, index1 (), mr.size2 ());
            }

#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
            typename self_type::
#endif
            const_reverse_iterator2 rbegin () const {
                return const_reverse_iterator2 (end ());
            }

#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
            typename self_type::
#endif
            const_reverse_iterator2 rend () const {
                return const_reverse_iterator2 (begin ());
            }
#endif

            // Indices

            size_type index1 () const {
                return it_.index1 () - (*this) ().start1 ();
            }

            size_type index2 () const {
                return it_.index2 () - (*this) ().start2 ();
            }

            // Assignment

            const_iterator1 &operator = (const const_iterator1 &it) {
                container_const_reference<self_type>::assign (&it ());
                it_ = it.it_;
                return *this;
            }

            // Comparison

            bool operator == (const const_iterator1 &it) const {
                BOOST_UBLAS_CHECK ((*this) ().same_closure (it ()), external_logic ());
                return it_ == it.it_;
            }

            bool operator < (const const_iterator1 &it) const {
                BOOST_UBLAS_CHECK ((*this) ().same_closure (it ()), external_logic ());
                return it_ < it.it_;
            }

        private:
            const_subiterator1_type it_;
        };
#endif


        const_iterator1 begin1 () const {
            return find1 (0, 0, 0);
        }

        const_iterator1 end1 () const {
            return find1 (0, size1 (), 0);
        }

#ifndef BOOST_UBLAS_USE_INDEXED_ITERATOR
        class iterator1:
            public container_reference<matrix_range>,
            public iterator_base_traits<typename subiterator1_type::iterator_category>::template
                        iterator_base<iterator1, value_type>::type {
        public:
            typedef typename subiterator1_type::value_type value_type;
            typedef typename subiterator1_type::difference_type difference_type;
            typedef typename subiterator1_type::reference reference;
            typedef typename subiterator1_type::pointer pointer;
            typedef iterator2 dual_iterator_type;
            typedef reverse_iterator2 dual_reverse_iterator_type;

            // Construction and destruction

            iterator1 ():
                container_reference<self_type> (), it_ () {}

            iterator1 (self_type &mr, const subiterator1_type &it):
                container_reference<self_type> (mr), it_ (it) {}

            // Arithmetic

            iterator1 &operator ++ () {
                ++ it_;
                return *this;
            }

            iterator1 &operator -- () {
                -- it_;
                return *this;
            }

            iterator1 &operator += (difference_type n) {
                it_ += n;
                return *this;
            }

            iterator1 &operator -= (difference_type n) {
                it_ -= n;
                return *this;
            }

            difference_type operator - (const iterator1 &it) const {
                BOOST_UBLAS_CHECK ((*this) ().same_closure  (it ()), external_logic ());
                return it_ - it.it_;
            }

            // Dereference

            reference operator * () const {
                return *it_;
            }

            reference operator [] (difference_type n) const {
                return *(*this + n);
            }

#ifndef BOOST_UBLAS_NO_NESTED_CLASS_RELATION

#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
            typename self_type::
#endif
            iterator2 begin () const {
                self_type &mr = (*this) ();
                return mr.find2 (1, index1 (), 0);
            }

#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
            typename self_type::
#endif
            iterator2 end () const {
                self_type &mr = (*this) ();
                return mr.find2 (1, index1 (), mr.size2 ());
            }

#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
            typename self_type::
#endif
            reverse_iterator2 rbegin () const {
                return reverse_iterator2 (end ());
            }

#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
            typename self_type::
#endif
            reverse_iterator2 rend () const {
                return reverse_iterator2 (begin ());
            }
#endif

            // Indices

            size_type index1 () const {
                return it_.index1 () - (*this) ().start1 ();
            }

            size_type index2 () const {
                return it_.index2 () - (*this) ().start2 ();
            }

            // Assignment

            iterator1 &operator = (const iterator1 &it) {
                container_reference<self_type>::assign (&it ());
                it_ = it.it_;
                return *this;
            }

            // Comparison

            bool operator == (const iterator1 &it) const {
                BOOST_UBLAS_CHECK ((*this) ().same_closure (it ()), external_logic ());
                return it_ == it.it_;
            }

            bool operator < (const iterator1 &it) const {
                BOOST_UBLAS_CHECK ((*this) ().same_closure (it ()), external_logic ());
                return it_ < it.it_;
            }

        private:
            subiterator1_type it_;

            friend class const_iterator1;
        };
#endif


        iterator1 begin1 () {
            return find1 (0, 0, 0);
        }

        iterator1 end1 () {
            return find1 (0, size1 (), 0);
        }

#ifndef BOOST_UBLAS_USE_INDEXED_ITERATOR
        class const_iterator2:
            public container_const_reference<matrix_range>,
            public iterator_base_traits<typename const_subiterator2_type::iterator_category>::template
                        iterator_base<const_iterator2, value_type>::type {
        public:
            typedef typename const_subiterator2_type::value_type value_type;
            typedef typename const_subiterator2_type::difference_type difference_type;
            typedef typename const_subiterator2_type::reference reference;
            typedef typename const_subiterator2_type::pointer pointer;
            typedef const_iterator1 dual_iterator_type;
            typedef const_reverse_iterator1 dual_reverse_iterator_type;

            // Construction and destruction

            const_iterator2 ():
                container_const_reference<self_type> (), it_ () {}

            const_iterator2 (const self_type &mr, const const_subiterator2_type &it):
                container_const_reference<self_type> (mr), it_ (it) {}

            const_iterator2 (const iterator2 &it):
                container_const_reference<self_type> (it ()), it_ (it.it_) {}

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
                BOOST_UBLAS_CHECK ((*this) ().same_closure (it ()), external_logic ());
                return it_ - it.it_;
            }

            // Dereference

            const_reference operator * () const {
                return *it_;
            }

            const_reference operator [] (difference_type n) const {
                return *(*this + n);
            }

#ifndef BOOST_UBLAS_NO_NESTED_CLASS_RELATION

#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
            typename self_type::
#endif
            const_iterator1 begin () const {
                const self_type &mr = (*this) ();
                return mr.find1 (1, 0, index2 ());
            }

#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
            typename self_type::
#endif
            const_iterator1 end () const {
                const self_type &mr = (*this) ();
                return mr.find1 (1, mr.size1 (), index2 ());
            }

#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
            typename self_type::
#endif
            const_reverse_iterator1 rbegin () const {
                return const_reverse_iterator1 (end ());
            }

#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
            typename self_type::
#endif
            const_reverse_iterator1 rend () const {
                return const_reverse_iterator1 (begin ());
            }
#endif

            // Indices

            size_type index1 () const {
                return it_.index1 () - (*this) ().start1 ();
            }

            size_type index2 () const {
                return it_.index2 () - (*this) ().start2 ();
            }

            // Assignment

            const_iterator2 &operator = (const const_iterator2 &it) {
                container_const_reference<self_type>::assign (&it ());
                it_ = it.it_;
                return *this;
            }

            // Comparison

            bool operator == (const const_iterator2 &it) const {
                BOOST_UBLAS_CHECK ((*this) ().same_closure  (it ()), external_logic ());
                return it_ == it.it_;
            }

            bool operator < (const const_iterator2 &it) const {
                BOOST_UBLAS_CHECK ((*this) ().same_closure  (it ()), external_logic ());
                return it_ < it.it_;
            }

        private:
            const_subiterator2_type it_;
        };
#endif


        const_iterator2 begin2 () const {
            return find2 (0, 0, 0);
        }

        const_iterator2 end2 () const {
            return find2 (0, 0, size2 ());
        }

#ifndef BOOST_UBLAS_USE_INDEXED_ITERATOR
        class iterator2:
            public container_reference<matrix_range>,
            public iterator_base_traits<typename subiterator2_type::iterator_category>::template
                        iterator_base<iterator2, value_type>::type {
        public:
            typedef typename subiterator2_type::value_type value_type;
            typedef typename subiterator2_type::difference_type difference_type;
            typedef typename subiterator2_type::reference reference;
            typedef typename subiterator2_type::pointer pointer;
            typedef iterator1 dual_iterator_type;
            typedef reverse_iterator1 dual_reverse_iterator_type;

            // Construction and destruction

            iterator2 ():
                container_reference<self_type> (), it_ () {}

            iterator2 (self_type &mr, const subiterator2_type &it):
                container_reference<self_type> (mr), it_ (it) {}

            // Arithmetic

            iterator2 &operator ++ () {
                ++ it_;
                return *this;
            }

            iterator2 &operator -- () {
                -- it_;
                return *this;
            }

            iterator2 &operator += (difference_type n) {
                it_ += n;
                return *this;
            }

            iterator2 &operator -= (difference_type n) {
                it_ -= n;
                return *this;
            }

            difference_type operator - (const iterator2 &it) const {
               BOOST_UBLAS_CHECK ((*this) ().same_closure  (it ()), external_logic ());
                return it_ - it.it_;
            }

            // Dereference

            reference operator * () const {
                return *it_;
            }

            reference operator [] (difference_type n) const {
                return *(*this + n);
            }

#ifndef BOOST_UBLAS_NO_NESTED_CLASS_RELATION

#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
            typename self_type::
#endif
            iterator1 begin () const {
                self_type &mr = (*this) ();
                return mr.find1 (1, 0, index2 ());
            }

#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
            typename self_type::
#endif
            iterator1 end () const {
                self_type &mr = (*this) ();
                return mr.find1 (1, mr.size1 (), index2 ());
            }

#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
            typename self_type::
#endif
            reverse_iterator1 rbegin () const {
                return reverse_iterator1 (end ());
            }

#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
            typename self_type::
#endif
            reverse_iterator1 rend () const {
                return reverse_iterator1 (begin ());
            }
#endif

            // Indices

            size_type index1 () const {
                return it_.index1 () - (*this) ().start1 ();
            }

            size_type index2 () const {
                return it_.index2 () - (*this) ().start2 ();
            }

            // Assignment

            iterator2 &operator = (const iterator2 &it) {
                container_reference<self_type>::assign (&it ());
                it_ = it.it_;
                return *this;
            }

            // Comparison

            bool operator == (const iterator2 &it) const {
                BOOST_UBLAS_CHECK ((*this) ().same_closure  (it ()), external_logic ());
                return it_ == it.it_;
            }

            bool operator < (const iterator2 &it) const {
                BOOST_UBLAS_CHECK ((*this) ().same_closure  (it ()), external_logic ());
                return it_ < it.it_;
            }

        private:
            subiterator2_type it_;

            friend class const_iterator2;
        };
#endif


        iterator2 begin2 () {
            return find2 (0, 0, 0);
        }

        iterator2 end2 () {
            return find2 (0, 0, size2 ());
        }

        // Reverse iterators


        const_reverse_iterator1 rbegin1 () const {
            return const_reverse_iterator1 (end1 ());
        }

        const_reverse_iterator1 rend1 () const {
            return const_reverse_iterator1 (begin1 ());
        }


        reverse_iterator1 rbegin1 () {
            return reverse_iterator1 (end1 ());
        }

        reverse_iterator1 rend1 () {
            return reverse_iterator1 (begin1 ());
        }


        const_reverse_iterator2 rbegin2 () const {
            return const_reverse_iterator2 (end2 ());
        }

        const_reverse_iterator2 rend2 () const {
            return const_reverse_iterator2 (begin2 ());
        }


        reverse_iterator2 rbegin2 () {
            return reverse_iterator2 (end2 ());
        }

        reverse_iterator2 rend2 () {
            return reverse_iterator2 (begin2 ());
        }

    private:
        matrix_closure_type data_;
        range_type r1_;
        range_type r2_;
    };

    // Siboost::mple Projections
    template<class M>

    matrix_range<M> subrange (M &data, typename M::size_type start1, typename M::size_type stop1, typename M::size_type start2, typename M::size_type stop2) {
        typedef basic_range<typename M::size_type, typename M::difference_type> range_type;
        return matrix_range<M> (data, range_type (start1, stop1), range_type (start2, stop2));
    }
    template<class M>

    matrix_range<const M> subrange (const M &data, typename M::size_type start1, typename M::size_type stop1, typename M::size_type start2, typename M::size_type stop2) {
        typedef basic_range<typename M::size_type, typename M::difference_type> range_type;
        return matrix_range<const M> (data, range_type (start1, stop1), range_type (start2, stop2));
    }

     template<class M>
    matrix_range<M> subrange (M &data,
	basic_range<typename M::size_type, typename M::difference_type> range1,
	basic_range<typename M::size_type, typename M::difference_type> range2
    ) {
        return matrix_range<M> (data, range1, range2);
    }
    template<class M>
    matrix_range<M const> subrange (M const & data,
	basic_range<typename M::size_type, typename M::difference_type> range1,
	basic_range<typename M::size_type, typename M::difference_type> range2
    ) {
        return matrix_range<M const> (data, range1, range2);
    }

    // Specialization of temporary_traits
    template <class M>
    struct matrix_temporary_traits< matrix_range<M> >
    : matrix_temporary_traits< M > {} ;
    template <class M>
    struct matrix_temporary_traits< const matrix_range<M> >
    : matrix_temporary_traits< M > {} ;

    template <class M>
    struct vector_temporary_traits< matrix_range<M> >
    : vector_temporary_traits< M > {} ;
    template <class M>
    struct vector_temporary_traits< const matrix_range<M> >
    : vector_temporary_traits< M > {} ;
}}

#endif
