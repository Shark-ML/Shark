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

#ifndef BOOST_UBLAS_STORAGE_H
#define BOOST_UBLAS_STORAGE_H

#include <algorithm>

#include <boost/serialization/array.hpp>
#include <boost/serialization/collection_size_type.hpp>
#include <boost/serialization/nvp.hpp>

#include <shark/LinAlg/BLAS/ublas/exception.hpp>
#include <shark/LinAlg/BLAS/ublas/traits.hpp>
#include <shark/LinAlg/BLAS/ublas/detail/iterator.hpp>


namespace shark{ namespace blas{
    // Unbounded array - with allocator
    template<class T, class ALLOC>
    class unbounded_array{

        typedef unbounded_array<T, ALLOC> self_type;
    public:
        typedef ALLOC allocator_type;
        typedef typename ALLOC::size_type size_type;
        typedef typename ALLOC::difference_type difference_type;
        typedef T value_type;
        typedef const T &const_reference;
        typedef T &reference;
        typedef const T *const_pointer;
        typedef T *pointer;
        typedef const_pointer const_iterator;
        typedef pointer iterator;

        // Construction and destruction
        explicit 
        unbounded_array (const ALLOC &a = ALLOC()):
            alloc_ (a), size_ (0) {
            data_ = 0;
        }
        explicit 
        unbounded_array (size_type size, const ALLOC &a = ALLOC()):
            alloc_(a), size_ (size) {
          if (size_) {
              data_ = alloc_.allocate (size_);
                  for (pointer d = data_; d != data_ + size_; ++d)
                      alloc_.construct(d, value_type());
          }
          else
              data_ = 0;
        }
        // No value initialised, but still be default constructed
        
        unbounded_array (size_type size, const value_type &init, const ALLOC &a = ALLOC()):
            alloc_ (a), size_ (size) {
            if (size_) {
                data_ = alloc_.allocate (size_);
                std::uninitialized_fill (begin(), end(), init);
            }
            else
                data_ = 0;
        }
        
        unbounded_array (const unbounded_array &c):
            alloc_ (c.alloc_), size_ (c.size_) {
            if (size_) {
                data_ = alloc_.allocate (size_);
                std::uninitialized_copy (c.begin(), c.end(), begin());
            }
            else
                data_ = 0;
        }
        
        ~unbounded_array () {
            if (size_) {
                    // std::_Destroy (begin(), end(), alloc_);
                    const iterator i_end = end();
                    for (iterator i = begin (); i != i_end; ++i) {
                        iterator_destroy (i); 
                    }
                alloc_.deallocate (data_, size_);
            }
        }

        // Resizing
    private:
        
        void resize_internal (const size_type size, const value_type init, const bool preserve) {
            if (size != size_) {
                pointer p_data = data_;
                if (size) {
                    data_ = alloc_.allocate (size);
                    if (preserve) {
                        pointer si = p_data;
                        pointer di = data_;
                        if (size < size_) {
                            for (; di != data_ + size; ++di) {
                                alloc_.construct (di, *si);
                                ++si;
                            }
                        }
                        else {
                            for (pointer si = p_data; si != p_data + size_; ++si) {
                                alloc_.construct (di, *si);
                                ++di;
                            }
                            for (; di != data_ + size; ++di) {
                                alloc_.construct (di, init);
                            }
                        }
                    }
                    else {
                            for (pointer di = data_; di != data_ + size; ++di)
                                alloc_.construct (di, value_type());
                    }
                }

                if (size_) {
                        for (pointer si = p_data; si != p_data + size_; ++si)
                            alloc_.destroy (si);
                    alloc_.deallocate (p_data, size_);
                }

                if (!size)
                    data_ = 0;
                size_ = size;
            }
        }
    public:
        
        void resize (size_type size) {
            resize_internal (size, value_type (), false);
        }
        
        void resize (size_type size, value_type init) {
            resize_internal (size, init, true);
        }
                    
        // Random Access Container
        
        size_type max_size () const {
            return ALLOC ().max_size();
        }
        
        
        bool empty () const {
            return size_ == 0;
        }
            
        
        size_type size () const {
            return size_;
        }

        // Element access
        
        const_reference operator [] (size_type i) const {
            BOOST_UBLAS_CHECK (i < size_, bad_index ());
            return data_ [i];
        }
        
        reference operator [] (size_type i) {
            BOOST_UBLAS_CHECK (i < size_, bad_index ());
            return data_ [i];
        }

        // Assignment
        
        unbounded_array &operator = (const unbounded_array &a) {
            if (this != &a) {
                resize (a.size_);
                std::copy (a.data_, a.data_ + a.size_, data_);
            }
            return *this;
        }
        
        unbounded_array &assign_temporary (unbounded_array &a) {
            swap (a);
            return *this;
        }

        // Swapping
        
        void swap (unbounded_array &a) {
            if (this != &a) {
                std::swap (size_, a.size_);
                std::swap (data_, a.data_);
            }
        }
        
        friend void swap (unbounded_array &a1, unbounded_array &a2) {
            a1.swap (a2);
        }

        
        const_iterator begin () const {
            return data_;
        }
        
        const_iterator end () const {
            return data_ + size_;
        }

        
        iterator begin () {
            return data_;
        }
        
        iterator end () {
            return data_ + size_;
        }

        // Reverse iterators
        typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
        typedef std::reverse_iterator<iterator> reverse_iterator;

        
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

        // Allocator
        allocator_type get_allocator () {
            return alloc_;
        }

    private:
        friend class boost::serialization::access;

        // Serialization
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version)
        { 
            boost::serialization::collection_size_type s(size_);
            ar & boost::serialization::make_nvp("size",s);
            if ( Archive::is_loading::value ) {
                resize(s);
            }
            ar & boost::serialization::make_array(data_, s);
        }

    private:
        // Handle explict destroy on a (possibly indexed) iterator
        
        static void iterator_destroy (iterator &i) {
            (&(*i)) -> ~value_type ();
        }
        ALLOC alloc_;
        size_type size_;
        pointer data_;
    };


    // Range class
    template <class Z, class D>
    class basic_range {
        typedef basic_range<Z, D> self_type;
    public:
        typedef Z size_type;
        typedef D difference_type;
        typedef size_type value_type;
        typedef value_type const_reference;
        typedef const_reference reference;
        typedef const value_type *const_pointer;
        typedef value_type *pointer;

        // Construction and destruction
        
        basic_range ():
            start_ (0), size_ (0) {}
        
        basic_range (size_type start, size_type stop):
            start_ (start), size_ (stop - start) {
            BOOST_UBLAS_CHECK (start_ <= stop, bad_index ());
        }

        
        size_type start () const {
            return start_;
        }
        
        size_type size () const {
            return size_;
        }

        // Random Access Container
        
        size_type max_size () const {
            return size_;
        }
        
        
        bool empty () const {
            return size_ == 0;
        }
            
        // Element access
        
        const_reference operator () (size_type i) const {
            BOOST_UBLAS_CHECK (i < size_, bad_index ());
            return start_ + i;
        }

        // Composition
        
        basic_range compose (const basic_range &r) const {
            return basic_range (start_ + r.start_, start_ + r.start_ + r.size_);
        }

        // Comparison
        
        bool operator == (const basic_range &r) const {
            return start_ == r.start_ && size_ == r.size_;
        }
        
        bool operator != (const basic_range &r) const {
            return ! (*this == r);
        }

        // Iterator types
    private:
        // Use and index
        typedef size_type const_subiterator_type;

    public:
#ifdef BOOST_UBLAS_USE_INDEXED_ITERATOR
        typedef indexed_const_iterator<self_type, std::random_access_iterator_tag> const_iterator;
#else
        class const_iterator:
            public container_const_reference<basic_range>,
            public random_access_iterator_base<std::random_access_iterator_tag,
                                               const_iterator, value_type> {
        public:
            typedef typename basic_range::value_type value_type;
            typedef typename basic_range::difference_type difference_type;
            typedef typename basic_range::const_reference reference;
            typedef typename basic_range::const_pointer pointer;

            // Construction and destruction
            
            const_iterator ():
                container_const_reference<basic_range> (), it_ () {}
            
            const_iterator (const basic_range &r, const const_subiterator_type &it):
                container_const_reference<basic_range> (r), it_ (it) {}

            // Arithmetic
            
            const_iterator &operator ++ () {
                ++ it_;
                return *this;
            }
            
            const_iterator &operator -- () {
                BOOST_UBLAS_CHECK (it_ > 0, bad_index ());
                -- it_;
                return *this;
            }
            
            const_iterator &operator += (difference_type n) {
                BOOST_UBLAS_CHECK (n >= 0 || it_ >= size_type(-n), bad_index ());
                it_ += n;
                return *this;
            }
            
            const_iterator &operator -= (difference_type n) {
                BOOST_UBLAS_CHECK (n <= 0 || it_ >= size_type(n), bad_index ());
                it_ -= n;
                return *this;
            }
            
            difference_type operator - (const const_iterator &it) const {
                return it_ - it.it_;
            }

            // Dereference
            
            const_reference operator * () const {
                BOOST_UBLAS_CHECK ((*this) ().start () <= it_, bad_index ());
                BOOST_UBLAS_CHECK (it_ < (*this) ().start () + (*this) ().size (), bad_index ());
                return it_;
            }

            
            const_reference operator [] (difference_type n) const {
                return *(*this + n);
            }

            // Index
            
            size_type index () const {
                BOOST_UBLAS_CHECK ((*this) ().start () <= it_, bad_index ());
                BOOST_UBLAS_CHECK (it_ < (*this) ().start () + (*this) ().size (), bad_index ());
                return it_ - (*this) ().start ();
            }

            // Assignment
            
            const_iterator &operator = (const const_iterator &it) {
                // Comeau recommends...
                this->assign (&it ());
                it_ = it.it_;
                return *this;
            }

            // Comparison
            
            bool operator == (const const_iterator &it) const {
                BOOST_UBLAS_CHECK ((*this) () == it (), external_logic ());
                return it_ == it.it_;
            }
            
            bool operator < (const const_iterator &it) const {
                BOOST_UBLAS_CHECK ((*this) () == it (), external_logic ());
                return it_ < it.it_;
            }

        private:
            const_subiterator_type it_;
        };
#endif

        
        const_iterator begin () const {
            return const_iterator (*this, start_);
        }
        
        const_iterator end () const {
            return const_iterator (*this, start_ + size_);
        }

        // Reverse iterator
        typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

        
        const_reverse_iterator rbegin () const {
            return const_reverse_iterator (end ());
        }
        
        const_reverse_iterator rend () const {
            return const_reverse_iterator (begin ());
        }

        
        basic_range preprocess (size_type size) const {
            if (this != &all_)
                return *this;
            return basic_range (0, size);
        }
        static
        
        const basic_range &all () {
            return all_;
        }

    private:
        size_type start_;
        size_type size_;
        static const basic_range all_;
    };

    template <class Z, class D>
    const basic_range<Z,D> basic_range<Z,D>::all_  (0, size_type (-1));
}}

#endif
