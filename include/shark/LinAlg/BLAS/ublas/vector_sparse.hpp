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

#ifndef _BOOST_UBLAS_VECTOR_SPARSE_
#define _BOOST_UBLAS_VECTOR_SPARSE_

#include <shark/LinAlg/BLAS/ublas/vector_expression.hpp>
#include <shark/LinAlg/BLAS/ublas/detail/vector_assign.hpp>

// Iterators based on ideas of Jeremy Siek

namespace shark{ namespace blas{
	
    namespace detail {

        template<class I, class T, class C>
        
        I lower_bound (const I &begin, const I &end, const T &t, C compare) {
            // t <= *begin <=> ! (*begin < t)
            if (begin == end || ! compare (*begin, t))
                return begin;
            if (compare (*(end - 1), t))
                return end;
            return std::lower_bound (begin, end, t, compare);
        }
        template<class I, class T, class C>
        
        I upper_bound (const I &begin, const I &end, const T &t, C compare) {
            if (begin == end || compare (t, *begin))
                return begin;
            // (*end - 1) <= t <=> ! (t < *end)
            if (! compare (t, *(end - 1)))
                return end;
            return std::upper_bound (begin, end, t, compare);
        }

        template<class P>
        struct less_pair {
            
            bool operator () (const P &p1, const P &p2) {
                return p1.first < p2.first;
            }
        };
        template<class T>
        struct less_triple {
            
            bool operator () (const T &t1, const T &t2) {
                return t1.first.first < t2.first.first ||
                       (t1.first.first == t2.first.first && t1.first.second < t2.first.second);
            }
        };

    }

    // Thanks to Kresimir Fresl for extending this to cover different index bases.
    
    /** \brief Compressed array based sparse vector
     *
     * a sparse vector of values of type T of variable size. The non zero values are stored as 
     * two seperate arrays: an index array and a value array. The index array is always sorted 
     * and there is at most one entry for each index. Inserting an element can be time consuming.
     * If the vector contains a few zero entries, then it is better to have a normal vector.
     * If the vector has a very high dimension with a few non-zero values, then this vector is
     * very memory efficient (at the cost of a few more computations).
     *
     * For a \f$n\f$-dimensional compressed vector and \f$0 \leq i < n\f$ the non-zero elements 
     * \f$v_i\f$ are mapped to consecutive elements of the index and value container, i.e. for 
     * elements \f$k = v_{i_1}\f$ and \f$k + 1 = v_{i_2}\f$ of these containers holds \f$i_1 < i_2\f$.
     *
     * Supported parameters for the adapted array (indices and values) are \c unbounded_array<> ,
     * \c bounded_array<> and \c std::vector<>.
     *
     * \tparam T the type of object stored in the vector (like double, float, complex, etc...)
     * \tparam IB the index base of the compressed vector. Default is 0. Other supported value is 1
     * \tparam IA the type of adapted array for indices. Default is \c unbounded_array<std::size_t>
     * \tparam TA the type of adapted array for values. Default is unbounded_array<T>
     */
    template<class T, std::size_t IB, class IA, class TA>
    class compressed_vector:
        public vector_container<compressed_vector<T, IB, IA, TA> > {

        typedef T &true_reference;
        typedef T *pointer;
        typedef const T *const_pointer;
        typedef compressed_vector<T, IB, IA, TA> self_type;
    public:

        // ISSUE require type consistency check
        // is_convertable (IA::size_type, TA::size_type)
        typedef typename IA::value_type size_type;
        typedef typename IA::difference_type difference_type;
        typedef T value_type;
        typedef const T &const_reference;
        typedef T &reference;

        typedef IA index_array_type;
        typedef TA value_array_type;
        typedef const vector_reference<const self_type> const_closure_type;
        typedef vector_reference<self_type> closure_type;
        typedef self_type vector_temporary_type;
        typedef sparse_tag storage_category;

        // Construction and destruction
        
        compressed_vector ():
            vector_container<self_type> (),
            size_ (0), capacity_ (restrict_capacity (0)), filled_ (0),
            index_data_ (capacity_), value_data_ (capacity_) {
            storage_invariants ();
        }
        explicit 
        compressed_vector (size_type size, size_type non_zeros = 0):
            vector_container<self_type> (),
            size_ (size), capacity_ (restrict_capacity (non_zeros)), filled_ (0),
            index_data_ (capacity_), value_data_ (capacity_) {
        storage_invariants ();
        }
        
        compressed_vector (const compressed_vector &v):
            vector_container<self_type> (),
            size_ (v.size_), capacity_ (v.capacity_), filled_ (v.filled_),
            index_data_ (v.index_data_), value_data_ (v.value_data_) {
            storage_invariants ();
        }
        template<class AE>
        
        compressed_vector (const vector_expression<AE> &ae, size_type non_zeros = 0):
            vector_container<self_type> (),
            size_ (ae ().size ()), capacity_ (restrict_capacity (non_zeros)), filled_ (0),
            index_data_ (capacity_), value_data_ (capacity_) {
            storage_invariants ();
            vector_assign<scalar_assign> (*this, ae);
        }

        // Accessors
        
        size_type size () const {
            return size_;
        }
        
        size_type nnz_capacity () const {
            return capacity_;
        }
        
        size_type nnz () const {
            return filled_;
        }

        // Storage accessors
        
        static size_type index_base () {
            return IB;
        }
        
        typename index_array_type::size_type filled () const {
            return filled_;
        }
        
        const index_array_type &index_data () const {
            return index_data_;
        }
        
        const value_array_type &value_data () const {
            return value_data_;
        }
        
        void set_filled (const typename index_array_type::size_type & filled) {
            filled_ = filled;
            storage_invariants ();
        }
        
        index_array_type &index_data () {
            return index_data_;
        }
        
        value_array_type &value_data () {
            return value_data_;
        }

        // Resizing
    private:
        
        size_type restrict_capacity (size_type non_zeros) const {
            non_zeros = (std::max) (non_zeros, size_type (1));
            non_zeros = (std::min) (non_zeros, size_);
            return non_zeros;
        }
    public:
        
        void resize (size_type size, bool preserve = true) {
            size_ = size;
            capacity_ = restrict_capacity (capacity_);
            if (preserve) {
                index_data_. resize (capacity_, size_type ());
                value_data_. resize (capacity_, value_type ());
                filled_ = (std::min) (capacity_, filled_);
                while ((filled_ > 0) && (zero_based(index_data_[filled_ - 1]) >= size)) {
                    --filled_;
                }
            }
            else {
                index_data_. resize (capacity_);
                value_data_. resize (capacity_);
                filled_ = 0;
            }
            storage_invariants ();
        }

        // Reserving
        
        void reserve (size_type non_zeros, bool preserve = true) {
            capacity_ = restrict_capacity (non_zeros);
            if (preserve) {
                index_data_. resize (capacity_, size_type ());
                value_data_. resize (capacity_, value_type ());
                filled_ = (std::min) (capacity_, filled_);
            }
            else {
                index_data_. resize (capacity_);
                value_data_. resize (capacity_);
                filled_ = 0;
            }
            storage_invariants ();
        }

        // Element support
        
        pointer find_element (size_type i) {
            return const_cast<pointer> (const_cast<const self_type&>(*this).find_element (i));
        }
        
        const_pointer find_element (size_type i) const {
            const_subiterator_type it (detail::lower_bound (index_data_.begin (), index_data_.begin () + filled_, k_based (i), std::less<size_type> ()));
            if (it == index_data_.begin () + filled_ || *it != k_based (i))
                return 0;
            return &value_data_ [it - index_data_.begin ()];
        }

        // Element access
        
        const_reference operator () (size_type i) const {
            BOOST_UBLAS_CHECK (i < size_, bad_index ());
            const_subiterator_type it (detail::lower_bound (index_data_.begin (), index_data_.begin () + filled_, k_based (i), std::less<size_type> ()));
            if (it == index_data_.begin () + filled_ || *it != k_based (i))
                return zero_;
            return value_data_ [it - index_data_.begin ()];
        }
        
        true_reference ref (size_type i) {
            BOOST_UBLAS_CHECK (i < size_, bad_index ());
            subiterator_type it (detail::lower_bound (index_data_.begin (), index_data_.begin () + filled_, k_based (i), std::less<size_type> ()));
            if (it == index_data_.begin () + filled_ || *it != k_based (i))
                return insert_element (i, value_type/*zero*/());
            else
                return value_data_ [it - index_data_.begin ()];
        }
        
        reference operator () (size_type i) {
            return ref (i) ;
        }

        
        const_reference operator [] (size_type i) const {
            return (*this) (i);
        }
        
        reference operator [] (size_type i) {
            return (*this) (i);
        }

        // Element assignment
        
        true_reference insert_element (size_type i, const_reference t) {
            BOOST_UBLAS_CHECK (!find_element (i), bad_index ());        // duplicate element
            if (filled_ >= capacity_)
                reserve (2 * capacity_, true);
            subiterator_type it (detail::lower_bound (index_data_.begin (), index_data_.begin () + filled_, k_based (i), std::less<size_type> ()));
            // ISSUE max_capacity limit due to difference_type
            typename std::iterator_traits<subiterator_type>::difference_type n = it - index_data_.begin ();
            BOOST_UBLAS_CHECK (filled_ == 0 || filled_ == typename index_array_type::size_type (n) || *it != k_based (i), internal_logic ());   // duplicate found by lower_bound
            ++ filled_;
            it = index_data_.begin () + n;
            std::copy_backward (it, index_data_.begin () + filled_ - 1, index_data_.begin () + filled_);
            *it = k_based (i);
            typename value_array_type::iterator itt (value_data_.begin () + n);
            std::copy_backward (itt, value_data_.begin () + filled_ - 1, value_data_.begin () + filled_);
            *itt = t;
            storage_invariants ();
            return *itt;
        }
        
        void erase_element (size_type i) {
            subiterator_type it (detail::lower_bound (index_data_.begin (), index_data_.begin () + filled_, k_based (i), std::less<size_type> ()));
            typename std::iterator_traits<subiterator_type>::difference_type  n = it - index_data_.begin ();
            if (filled_ > typename index_array_type::size_type (n) && *it == k_based (i)) {
                std::copy (it + 1, index_data_.begin () + filled_, it);
                typename value_array_type::iterator itt (value_data_.begin () + n);
                std::copy (itt + 1, value_data_.begin () + filled_, itt);
                -- filled_;
            }
            storage_invariants ();
        }

        // Zeroing
        
        void clear () {
            filled_ = 0;
            storage_invariants ();
        }

        // Assignment
        
        compressed_vector &operator = (const compressed_vector &v) {
            if (this != &v) {
                size_ = v.size_;
                capacity_ = v.capacity_;
                filled_ = v.filled_;
                index_data_ = v.index_data_;
                value_data_ = v.value_data_;
            }
            storage_invariants ();
            return *this;
        }
        template<class C>          // Container assignment without temporary
        
        compressed_vector &operator = (const vector_container<C> &v) {
            resize (v ().size (), false);
            assign (v);
            return *this;
        }
        
        compressed_vector &assign_temporary (compressed_vector &v) {
            swap (v);
            return *this;
        }
        template<class AE>
        
        compressed_vector &operator = (const vector_expression<AE> &ae) {
            self_type temporary (ae, capacity_);
            return assign_temporary (temporary);
        }
        template<class AE>
        
        compressed_vector &assign (const vector_expression<AE> &ae) {
            vector_assign<scalar_assign> (*this, ae);
            return *this;
        }

        // Computed assignment
        template<class AE>
        
        compressed_vector &operator += (const vector_expression<AE> &ae) {
            self_type temporary (*this + ae, capacity_);
            return assign_temporary (temporary);
        }
        template<class C>          // Container assignment without temporary
        
        compressed_vector &operator += (const vector_container<C> &v) {
            plus_assign (v);
            return *this;
        }
        template<class AE>
        
        compressed_vector &plus_assign (const vector_expression<AE> &ae) {
            vector_assign<scalar_plus_assign> (*this, ae);
            return *this;
        }
        template<class AE>
        
        compressed_vector &operator -= (const vector_expression<AE> &ae) {
            self_type temporary (*this - ae, capacity_);
            return assign_temporary (temporary);
        }
        template<class C>          // Container assignment without temporary
        
        compressed_vector &operator -= (const vector_container<C> &v) {
            minus_assign (v);
            return *this;
        }
        template<class AE>
        
        compressed_vector &minus_assign (const vector_expression<AE> &ae) {
            vector_assign<scalar_minus_assign> (*this, ae);
            return *this;
        }
        template<class AT>
        
        compressed_vector &operator *= (const AT &at) {
            vector_assign_scalar<scalar_multiplies_assign> (*this, at);
            return *this;
        }
        template<class AT>
        
        compressed_vector &operator /= (const AT &at) {
            vector_assign_scalar<scalar_divides_assign> (*this, at);
            return *this;
        }

        // Swapping
        
        void swap (compressed_vector &v) {
            if (this != &v) {
                std::swap (size_, v.size_);
                std::swap (capacity_, v.capacity_);
                std::swap (filled_, v.filled_);
                index_data_.swap (v.index_data_);
                value_data_.swap (v.value_data_);
            }
            storage_invariants ();
        }
        
        friend void swap (compressed_vector &v1, compressed_vector &v2) {
            v1.swap (v2);
        }

        // Back element insertion and erasure
        
        void push_back (size_type i, const_reference t) {
            BOOST_UBLAS_CHECK (filled_ == 0 || index_data_ [filled_ - 1] < k_based (i), external_logic ());
            if (filled_ >= capacity_)
                reserve (2 * capacity_, true);
            BOOST_UBLAS_CHECK (filled_ < capacity_, internal_logic ());
            index_data_ [filled_] = k_based (i);
            value_data_ [filled_] = t;
            ++ filled_;
            storage_invariants ();
        }
        
        void pop_back () {
            BOOST_UBLAS_CHECK (filled_ > 0, external_logic ());
            -- filled_;
            storage_invariants ();
        }

        // Iterator types
    private:
        // Use index array iterator
        typedef typename IA::const_iterator const_subiterator_type;
        typedef typename IA::iterator subiterator_type;

        
        true_reference at_element (size_type i) {
            BOOST_UBLAS_CHECK (i < size_, bad_index ());
            subiterator_type it (detail::lower_bound (index_data_.begin (), index_data_.begin () + filled_, k_based (i), std::less<size_type> ()));
            BOOST_UBLAS_CHECK (it != index_data_.begin () + filled_ && *it == k_based (i), bad_index ());
            return value_data_ [it - index_data_.begin ()];
        }

    public:
        class const_iterator;
        class iterator;

        // Element lookup
        //  This function seems to be big. So we do not let the compiler inline it.
        const_iterator find (size_type i) const {
            return const_iterator (*this, detail::lower_bound (index_data_.begin (), index_data_.begin () + filled_, k_based (i), std::less<size_type> ()));
        }
        //  This function seems to be big. So we do not let the compiler inline it.
        iterator find (size_type i) {
            return iterator (*this, detail::lower_bound (index_data_.begin (), index_data_.begin () + filled_, k_based (i), std::less<size_type> ()));
        }


        class const_iterator:
            public container_const_reference<compressed_vector>,
            public bidirectional_iterator_base<sparse_bidirectional_iterator_tag,
                                               const_iterator, value_type> {
        public:
            typedef typename compressed_vector::value_type value_type;
            typedef typename compressed_vector::difference_type difference_type;
            typedef typename compressed_vector::const_reference reference;
            typedef const typename compressed_vector::pointer pointer;

            // Construction and destruction
            
            const_iterator ():
                container_const_reference<self_type> (), it_ () {}
            
            const_iterator (const self_type &v, const const_subiterator_type &it):
                container_const_reference<self_type> (v), it_ (it) {}
            
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

            // Dereference
            
            const_reference operator * () const {
                BOOST_UBLAS_CHECK (index () < (*this) ().size (), bad_index ());
                return (*this) ().value_data_ [it_ - (*this) ().index_data_.begin ()];
            }

            // Index
            
            size_type index () const {
                BOOST_UBLAS_CHECK (*this != (*this) ().end (), bad_index ());
                BOOST_UBLAS_CHECK ((*this) ().zero_based (*it_) < (*this) ().size (), bad_index ());
                return (*this) ().zero_based (*it_);
            }

            // Assignment
            
            const_iterator &operator = (const const_iterator &it) {
                container_const_reference<self_type>::assign (&it ());
                it_ = it.it_;
                return *this;
            }

            // Comparison
            
            bool operator == (const const_iterator &it) const {
                BOOST_UBLAS_CHECK (&(*this) () == &it (), external_logic ());
                return it_ == it.it_;
            }

        private:
            const_subiterator_type it_;
        };

        
        const_iterator begin () const {
            return find (0);
        }
        
        const_iterator end () const {
            return find (size_);
        }

        class iterator:
            public container_reference<compressed_vector>,
            public bidirectional_iterator_base<sparse_bidirectional_iterator_tag,
                                               iterator, value_type> {
        public:
            typedef typename compressed_vector::value_type value_type;
            typedef typename compressed_vector::difference_type difference_type;
            typedef typename compressed_vector::true_reference reference;
            typedef typename compressed_vector::pointer pointer;

            // Construction and destruction
            
            iterator ():
                container_reference<self_type> (), it_ () {}
            
            iterator (self_type &v, const subiterator_type &it):
                container_reference<self_type> (v), it_ (it) {}

            // Arithmetic
            
            iterator &operator ++ () {
                ++ it_;
                return *this;
            }
            
            iterator &operator -- () {
                -- it_;
                return *this;
            }

            // Dereference
            
            reference operator * () const {
                BOOST_UBLAS_CHECK (index () < (*this) ().size (), bad_index ());
                return (*this) ().value_data_ [it_ - (*this) ().index_data_.begin ()];
            }

            // Index
            
            size_type index () const {
                BOOST_UBLAS_CHECK (*this != (*this) ().end (), bad_index ());
                BOOST_UBLAS_CHECK ((*this) ().zero_based (*it_) < (*this) ().size (), bad_index ());
                return (*this) ().zero_based (*it_);
            }

            // Assignment
            
            iterator &operator = (const iterator &it) {
                container_reference<self_type>::assign (&it ());
                it_ = it.it_;
                return *this;
            }

            // Comparison
            
            bool operator == (const iterator &it) const {
                BOOST_UBLAS_CHECK (&(*this) () == &it (), external_logic ());
                return it_ == it.it_;
            }

        private:
            subiterator_type it_;

            friend class const_iterator;
        };

        
        iterator begin () {
            return find (0);
        }
        
        iterator end () {
            return find (size_);
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

         // Serialization
        template<class Archive>
        void serialize(Archive & ar, const unsigned int /* file_version */){
            boost::serialization::collection_size_type s (size_);
            ar & boost::serialization::make_nvp("size",s);
            if (Archive::is_loading::value) {
                size_ = s;
            }
            // ISSUE: filled may be much less than capacity
            // ISSUE: index_data_ and value_data_ are undefined between filled and capacity (trouble with 'nan'-values)
            ar & boost::serialization::make_nvp("capacity", capacity_);
            ar & boost::serialization::make_nvp("filled", filled_);
            ar & boost::serialization::make_nvp("index_data", index_data_);
            ar & boost::serialization::make_nvp("value_data", value_data_);
            storage_invariants();
        }

    private:
        void storage_invariants () const
        {
            BOOST_UBLAS_CHECK (capacity_ == index_data_.size (), internal_logic ());
            BOOST_UBLAS_CHECK (capacity_ == value_data_.size (), internal_logic ());
            BOOST_UBLAS_CHECK (filled_ <= capacity_, internal_logic ());
            BOOST_UBLAS_CHECK ((0 == filled_) || (zero_based(index_data_[filled_ - 1]) < size_), internal_logic ());
        }

        size_type size_;
        typename index_array_type::size_type capacity_;
        typename index_array_type::size_type filled_;
        index_array_type index_data_;
        value_array_type value_data_;
        static const value_type zero_;

        
        static size_type zero_based (size_type k_based_index) {
            return k_based_index - IB;
        }
        
        static size_type k_based (size_type zero_based_index) {
            return zero_based_index + IB;
        }

        friend class iterator;
        friend class const_iterator;
    };

    template<class T, std::size_t IB, class IA, class TA>
    const typename compressed_vector<T, IB, IA, TA>::value_type compressed_vector<T, IB, IA, TA>::zero_ = value_type/*zero*/();
}}

#endif
