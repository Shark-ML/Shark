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

#ifndef _BOOST_UBLAS_VECTOR_PROXY_
#define _BOOST_UBLAS_VECTOR_PROXY_

#include <shark/LinAlg/BLAS/ublas/vector_expression.hpp>
#include <shark/LinAlg/BLAS/ublas/detail/vector_assign.hpp>
#include <shark/LinAlg/BLAS/ublas/detail/temporary.hpp>

// Iterators based on ideas of Jeremy Siek

namespace shark{ namespace blas{

    /** \brief A vector referencing a continuous subvector of elements of vector \c v containing all elements specified by \c range.
     *
     * A vector range can be used as a normal vector in any expression. 
     * If the specified range falls outside that of the index range of the vector, then
     * the \c vector_range is not a well formed vector expression and access to an 
     * element outside of index range of the vector is \b undefined.
     *
     * \tparam V the type of vector referenced (for exaboost::mple \c vector<double>)
     */
    template<class V>
    class vector_range:
        public vector_expression<vector_range<V> > {

        typedef vector_range<V> self_type;
    public:
#ifdef BOOST_UBLAS_ENABLE_PROXY_SHORTCUTS
        using vector_expression<self_type>::operator ();
#endif
        typedef const V const_vector_type;
        typedef V vector_type;
        typedef typename V::size_type size_type;
        typedef typename V::difference_type difference_type;
        typedef typename V::value_type value_type;
        typedef typename V::const_reference const_reference;
        typedef typename boost::mpl::if_<boost::is_const<V>,
                                          typename V::const_reference,
                                          typename V::reference>::type reference;
        typedef typename boost::mpl::if_<boost::is_const<V>,
                                          typename V::const_closure_type,
                                          typename V::closure_type>::type vector_closure_type;
        typedef basic_range<size_type, difference_type> range_type;
        typedef const self_type const_closure_type;
        typedef self_type closure_type;
        typedef typename storage_restrict_traits<typename V::storage_category,
                                                 dense_proxy_tag>::storage_category storage_category;

        // Construction and destruction
        
        vector_range (vector_type &data, const range_type &r):
            data_ (data), r_ (r.preprocess (data.size ())) {
            // Early checking of preconditions here.
            // BOOST_UBLAS_CHECK (r_.start () <= data_.size () &&
            //                   r_.start () + r_.size () <= data_.size (), bad_index ());
        }
        
        vector_range (const vector_closure_type &data, const range_type &r, bool):
            data_ (data), r_ (r.preprocess (data.size ())) {
            // Early checking of preconditions here.
            // BOOST_UBLAS_CHECK (r_.start () <= data_.size () &&
            //                    r_.start () + r_.size () <= data_.size (), bad_index ());
        }

        // Accessors
        
        size_type start () const {
            return r_.start ();
        }
        
        size_type size () const {
            return r_.size ();
        }

        // Storage accessors
        
        const vector_closure_type &data () const {
            return data_;
        }
        
        vector_closure_type &data () {
            return data_;
        }

        // Element access
#ifndef BOOST_UBLAS_PROXY_CONST_MEMBER
        
        const_reference operator () (size_type i) const {
            return data_ (r_ (i));
        }
        
        reference operator () (size_type i) {
            return data_ (r_ (i));
        }

        
        const_reference operator [] (size_type i) const {
            return (*this) (i);
        }
        
        reference operator [] (size_type i) {
            return (*this) (i);
        }
#else
        
        reference operator () (size_type i) const {
            return data_ (r_ (i));
        }

        
        reference operator [] (size_type i) const {
            return (*this) (i);
        }
#endif

        // ISSUE can this be done in free project function?
        // Although a const function can create a non-const proxy to a non-const object
        // Critical is that vector_type and data_ (vector_closure_type) are const correct
        
        vector_range<vector_type> project (const range_type &r) const {
            return vector_range<vector_type> (data_, r_.compose (r.preprocess (data_.size ())), false);
        }

        // Assignment
        
        vector_range &operator = (const vector_range &vr) {
            // ISSUE need a temporary, proxy can be overlaping alias
            vector_assign<scalar_assign> (*this, typename vector_temporary_traits<V>::type (vr));
            return *this;
        }
        
        vector_range &assign_temporary (vector_range &vr) {
            // assign elements, proxied container remains the same
            vector_assign<scalar_assign> (*this, vr);
            return *this;
        }
        template<class AE>
        
        vector_range &operator = (const vector_expression<AE> &ae) {
            vector_assign<scalar_assign> (*this, typename vector_temporary_traits<V>::type (ae));
            return *this;
        }
        template<class AE>
        
        vector_range &assign (const vector_expression<AE> &ae) {
            vector_assign<scalar_assign> (*this, ae);
            return *this;
        }
        template<class AE>
        
        vector_range &operator += (const vector_expression<AE> &ae) {
            vector_assign<scalar_assign> (*this, typename vector_temporary_traits<V>::type (*this + ae));
            return *this;
        }
        template<class AE>
        
        vector_range &plus_assign (const vector_expression<AE> &ae) {
            vector_assign<scalar_plus_assign> (*this, ae);
            return *this;
        }
        template<class AE>
        
        vector_range &operator -= (const vector_expression<AE> &ae) {
            vector_assign<scalar_assign> (*this, typename vector_temporary_traits<V>::type (*this - ae));
            return *this;
        }
        template<class AE>
        
        vector_range &minus_assign (const vector_expression<AE> &ae) {
            vector_assign<scalar_minus_assign> (*this, ae);
            return *this;
        }
        template<class AT>
        
        vector_range &operator *= (const AT &at) {
            vector_assign_scalar<scalar_multiplies_assign> (*this, at);
            return *this;
        }
        template<class AT>
        
        vector_range &operator /= (const AT &at) {
            vector_assign_scalar<scalar_divides_assign> (*this, at);
            return *this;
        }

        // Closure comparison
        
        bool same_closure (const vector_range &vr) const {
            return (*this).data_.same_closure (vr.data_);
        }

        // Comparison
        
        bool operator == (const vector_range &vr) const {
            return (*this).data_ == vr.data_ && r_ == vr.r_;
        }

        // Swapping
        
        void swap (vector_range vr) {
            if (this != &vr) {
                BOOST_UBLAS_CHECK (size () == vr.size (), bad_size ());
                // Sparse ranges may be nonconformant now.
                // std::swap_ranges (begin (), end (), vr.begin ());
                vector_swap<scalar_swap> (*this, vr);
            }
        }
        
        friend void swap (vector_range vr1, vector_range vr2) {
            vr1.swap (vr2);
        }

        // Iterator types
    private:
        typedef typename V::const_iterator const_subiterator_type;
        typedef typename boost::mpl::if_<boost::is_const<V>,
                                          typename V::const_iterator,
                                          typename V::iterator>::type subiterator_type;

    public:
#ifdef BOOST_UBLAS_USE_INDEXED_ITERATOR
        typedef indexed_iterator<vector_range<vector_type>,
                                 typename subiterator_type::iterator_category> iterator;
        typedef indexed_const_iterator<vector_range<vector_type>,
                                       typename const_subiterator_type::iterator_category> const_iterator;
#else
        class const_iterator;
        class iterator;
#endif

        // Element lookup
        
        const_iterator find (size_type i) const {
            const_subiterator_type it (data_.find (start () + i));
#ifdef BOOST_UBLAS_USE_INDEXED_ITERATOR
            return const_iterator (*this, it.index ());
#else
            return const_iterator (*this, it);
#endif
        }
        
        iterator find (size_type i) {
            subiterator_type it (data_.find (start () + i));
#ifdef BOOST_UBLAS_USE_INDEXED_ITERATOR
            return iterator (*this, it.index ());
#else
            return iterator (*this, it);
#endif
        }

#ifndef BOOST_UBLAS_USE_INDEXED_ITERATOR
        class const_iterator:
            public container_const_reference<vector_range>,
            public iterator_base_traits<typename const_subiterator_type::iterator_category>::template
                        iterator_base<const_iterator, value_type>::type {
        public:
            typedef typename const_subiterator_type::difference_type difference_type;
            typedef typename const_subiterator_type::value_type value_type;
            typedef typename const_subiterator_type::reference reference;
            typedef typename const_subiterator_type::pointer pointer;

            // Construction and destruction
            
            const_iterator ():
                container_const_reference<self_type> (), it_ () {}
            
            const_iterator (const self_type &vr, const const_subiterator_type &it):
                container_const_reference<self_type> (vr), it_ (it) {}
            
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
                BOOST_UBLAS_CHECK ((*this) ().same_closure (it ()), external_logic ());
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
                return it_.index () - (*this) ().start ();
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
            public container_reference<vector_range>,
            public iterator_base_traits<typename subiterator_type::iterator_category>::template
                        iterator_base<iterator, value_type>::type {
        public:
            typedef typename subiterator_type::difference_type difference_type;
            typedef typename subiterator_type::value_type value_type;
            typedef typename subiterator_type::reference reference;
            typedef typename subiterator_type::pointer pointer;

            // Construction and destruction
            
            iterator ():
                container_reference<self_type> (), it_ () {}
            
            iterator (self_type &vr, const subiterator_type &it):
                container_reference<self_type> (vr), it_ (it) {}

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
                BOOST_UBLAS_CHECK ((*this) ().same_closure (it ()), external_logic ());
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
                return it_.index () - (*this) ().start ();
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
        vector_closure_type data_;
        range_type r_;
    };

    // ------------------
    // Siboost::mple Projections
    // ------------------

    /** \brief Return a \c vector_range on a specified vector, a start and stop index.
     * Return a \c vector_range on a specified vector, a start and stop index. The resulting \c vector_range can be manipulated like a normal vector.
     * If the specified range falls outside that of of the index range of the vector, then the resulting \c vector_range is not a well formed
     * Vector Expression and access to an element outside of index range of the vector is \b undefined.
     */
    template<class V>
    
    vector_range<V> subrange (V &data, typename V::size_type start, typename V::size_type stop) {
        typedef basic_range<typename V::size_type, typename V::difference_type> range_type;
        return vector_range<V> (data, range_type (start, stop));
    }

    
    template<class V>
    
    vector_range<V> subrange (V &data, basic_range<typename V::size_type, typename V::difference_type> range) {
        return vector_range<V> (data, range);
    }
    /** \brief Return a \c const \c vector_range on a specified vector, a start and stop index.
     * Return a \c const \c vector_range on a specified vector, a start and stop index. The resulting \c const \c vector_range can be manipulated like a normal vector.
     *If the specified range falls outside that of of the index range of the vector, then the resulting \c vector_range is not a well formed
     * Vector Expression and access to an element outside of index range of the vector is \b undefined.
     */
    template<class V>
    
    vector_range<const V> subrange (const V &data, typename V::size_type start, typename V::size_type stop) {
        typedef basic_range<typename V::size_type, typename V::difference_type> range_type;
        return vector_range<const V> (data, range_type (start, stop));
    }
    
    template<class V>
    
    vector_range<const V> subrange (const V &data, basic_range<typename V::size_type, typename V::difference_type> range) {
        return vector_range<const V> (data, range);
    }

    // -------------------
    // Generic Projections
    // -------------------
    
    /** \brief Return a \c const \c vector_range on a specified vector and \c range
     * Return a \c const \c vector_range on a specified vector and \c range. The resulting \c vector_range can be manipulated like a normal vector.
     * If the specified range falls outside that of of the index range of the vector, then the resulting \c vector_range is not a well formed
     * Vector Expression and access to an element outside of index range of the vector is \b undefined.
     */
    template<class V>
    
    vector_range<V> project (V &data, typename vector_range<V>::range_type const &r) {
        return vector_range<V> (data, r);
    }

    /** \brief Return a \c vector_range on a specified vector and \c range
     * Return a \c vector_range on a specified vector and \c range. The resulting \c vector_range can be manipulated like a normal vector.
     * If the specified range falls outside that of of the index range of the vector, then the resulting \c vector_range is not a well formed
     * Vector Expression and access to an element outside of index range of the vector is \b undefined.
     */
    template<class V>
    
    const vector_range<const V> project (const V &data, typename vector_range<V>::range_type const &r) {
        // ISSUE was: return vector_range<V> (const_cast<V &> (data), r);
        return vector_range<const V> (data, r);
   }

    // Specialization of temporary_traits
    template <class V>
    struct vector_temporary_traits< vector_range<V> >
    : vector_temporary_traits< V > {} ;
    template <class V>
    struct vector_temporary_traits< const vector_range<V> >
    : vector_temporary_traits< V > {} ;

}}

#endif
