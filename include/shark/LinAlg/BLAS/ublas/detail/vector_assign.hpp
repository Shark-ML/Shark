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

#ifndef _BOOST_UBLAS_VECTOR_ASSIGN_
#define _BOOST_UBLAS_VECTOR_ASSIGN_

#include <shark/LinAlg/BLAS/ublas/functional.hpp> // scalar_assign
// Required for make_conformant storage
#include <vector>

// Iterators based on ideas of Jeremy Siek

namespace shark{ namespace blas{
namespace detail {

    // Make sparse proxies conformant
    template<class V, class E>
    //  This function seems to be big. So we do not let the compiler inline it.
    void make_conformant (V &v, const vector_expression<E> &e) {
        BOOST_UBLAS_CHECK (v.size () == e ().size (), bad_size ());
        typedef typename V::size_type size_type;
        typedef typename V::difference_type difference_type;
        typedef typename V::value_type value_type;
        // FIXME unbounded_array with push_back maybe better
        std::vector<size_type> index;
        typename V::iterator it (v.begin ());
        typename V::iterator it_end (v.end ());
        typename E::const_iterator ite (e ().begin ());
        typename E::const_iterator ite_end (e ().end ());
        if (it != it_end && ite != ite_end) {
            size_type it_index = it.index (), ite_index = ite.index ();
            while (true) {
                difference_type compare = it_index - ite_index;
                if (compare == 0) {
                    ++ it, ++ ite;
                    if (it != it_end && ite != ite_end) {
                        it_index = it.index ();
                        ite_index = ite.index ();
                    } else
                        break;
                } else if (compare < 0) {
                    increment (it, it_end, - compare);
                    if (it != it_end)
                        it_index = it.index ();
                    else
                        break;
                } else if (compare > 0) {
                    if (*ite != value_type/*zero*/())
                        index.push_back (ite.index ());
                    ++ ite;
                    if (ite != ite_end)
                        ite_index = ite.index ();
                    else
                        break;
                }
            }
        }

        while (ite != ite_end) {
            if (*ite != value_type/*zero*/())
                index.push_back (ite.index ());
            ++ ite;
        }
        for (size_type k = 0; k < index.size (); ++ k)
            v (index [k]) = value_type/*zero*/();
    }

}//namespace detail

    // Dense (proxy) case
    template<template <class T1, class T2> class F, class V, class T>
    //  This function seems to be big. So we do not let the compiler inline it.
    void vector_assign_scalar (V &v, const T &t, dense_proxy_tag) {
        typedef F<typename V::iterator::reference, T> functor_type;
        typedef typename V::difference_type difference_type;
        difference_type size (v.size ());
        typename V::iterator it (v.begin ());
        BOOST_UBLAS_CHECK (v.end () - it == size, bad_size ());
        while (-- size >= 0)
            functor_type::apply (*it, t), ++ it;
    }
    // Packed (proxy) case
    template<template <class T1, class T2> class F, class V, class T>
    //  This function seems to be big. So we do not let the compiler inline it.
    void vector_assign_scalar (V &v, const T &t, packed_proxy_tag) {
        typedef F<typename V::iterator::reference, T> functor_type;
        typedef typename V::difference_type difference_type;
        typename V::iterator it (v.begin ());
        difference_type size (v.end () - it);
        while (-- size >= 0)
            functor_type::apply (*it, t), ++ it;
    }
    // Sparse (proxy) case
    template<template <class T1, class T2> class F, class V, class T>
    //  This function seems to be big. So we do not let the compiler inline it.
    void vector_assign_scalar (V &v, const T &t, sparse_proxy_tag) {
        typedef F<typename V::iterator::reference, T> functor_type;
        typename V::iterator it (v.begin ());
        typename V::iterator it_end (v.end ());
        while (it != it_end)
            functor_type::apply (*it, t), ++ it;
    }

    // Dispatcher
    template<template <class T1, class T2> class F, class V, class T>
    
    void vector_assign_scalar (V &v, const T &t) {
        typedef typename V::storage_category storage_category;
        vector_assign_scalar<F> (v, t, storage_category ());
    }

    template<class SC, bool COMPUTED, class RI>
    struct vector_assign_traits {
        typedef SC storage_category;
    };

    template<bool COMPUTED>
    struct vector_assign_traits<dense_tag, COMPUTED, packed_random_access_iterator_tag> {
        typedef packed_tag storage_category;
    };
    template<>
    struct vector_assign_traits<dense_tag, false, sparse_bidirectional_iterator_tag> {
        typedef sparse_tag storage_category;
    };
    template<>
    struct vector_assign_traits<dense_tag, true, sparse_bidirectional_iterator_tag> {
        typedef sparse_proxy_tag storage_category;
    };

    template<bool COMPUTED>
    struct vector_assign_traits<dense_proxy_tag, COMPUTED, packed_random_access_iterator_tag> {
        typedef packed_proxy_tag storage_category;
    };
    template<>
    struct vector_assign_traits<dense_proxy_tag, false, sparse_bidirectional_iterator_tag> {
        typedef sparse_proxy_tag storage_category;
    };
    template<>
    struct vector_assign_traits<dense_proxy_tag, true, sparse_bidirectional_iterator_tag> {
        typedef sparse_proxy_tag storage_category;
    };

    template<>
    struct vector_assign_traits<packed_tag, false, sparse_bidirectional_iterator_tag> {
        typedef sparse_tag storage_category;
    };
    template<>
    struct vector_assign_traits<packed_tag, true, sparse_bidirectional_iterator_tag> {
        typedef sparse_proxy_tag storage_category;
    };

    template<bool COMPUTED>
    struct vector_assign_traits<packed_proxy_tag, COMPUTED, sparse_bidirectional_iterator_tag> {
        typedef sparse_proxy_tag storage_category;
    };

    template<>
    struct vector_assign_traits<sparse_tag, true, dense_random_access_iterator_tag> {
        typedef sparse_proxy_tag storage_category;
    };
    template<>
    struct vector_assign_traits<sparse_tag, true, packed_random_access_iterator_tag> {
        typedef sparse_proxy_tag storage_category;
    };
    template<>
    struct vector_assign_traits<sparse_tag, true, sparse_bidirectional_iterator_tag> {
        typedef sparse_proxy_tag storage_category;
    };

    // Dense (proxy) case
    template<template <class T1, class T2> class F, class V, class E>
    //  This function seems to be big. So we do not let the compiler inline it.
    void vector_assign (V &v, const vector_expression<E> &e, dense_proxy_tag) {
        typedef F<typename V::iterator::reference, typename E::value_type> functor_type;
        typedef typename V::difference_type difference_type;
        difference_type size (BOOST_UBLAS_SAME (v.size (), e ().size ()));
        typename V::iterator it (v.begin ());
        BOOST_UBLAS_CHECK (v.end () - it == size, bad_size ());
        typename E::const_iterator ite (e ().begin ());
        BOOST_UBLAS_CHECK (e ().end () - ite == size, bad_size ());
        while (-- size >= 0)
            functor_type::apply (*it, *ite), ++ it, ++ ite;
    }
    // Packed (proxy) case
    template<template <class T1, class T2> class F, class V, class E>
    //  This function seems to be big. So we do not let the compiler inline it.
    void vector_assign (V &v, const vector_expression<E> &e, packed_proxy_tag) {
        BOOST_UBLAS_CHECK (v.size () == e ().size (), bad_size ());
        typedef F<typename V::iterator::reference, typename E::value_type> functor_type;
        typedef typename V::difference_type difference_type;
        typedef typename V::value_type value_type;
        typename V::iterator it (v.begin ());
        typename V::iterator it_end (v.end ());
        typename E::const_iterator ite (e ().begin ());
        typename E::const_iterator ite_end (e ().end ());
        difference_type it_size (it_end - it);
        difference_type ite_size (ite_end - ite);
        if (it_size > 0 && ite_size > 0) {
            difference_type size ((std::min) (difference_type (it.index () - ite.index ()), ite_size));
            if (size > 0) {
                ite += size;
                ite_size -= size;
            }
        }
        if (it_size > 0 && ite_size > 0) {
            difference_type size ((std::min) (difference_type (ite.index () - it.index ()), it_size));
            if (size > 0) {
                it_size -= size;
                if (!functor_type::computed) {
                    while (-- size >= 0)    // zeroing
                        functor_type::apply (*it, value_type/*zero*/()), ++ it;
                } else {
                    it += size;
                }
            }
        }
        difference_type size ((std::min) (it_size, ite_size));
        it_size -= size;
        ite_size -= size;
        while (-- size >= 0)
            functor_type::apply (*it, *ite), ++ it, ++ ite;
        size = it_size;
        if (!functor_type::computed) {
            while (-- size >= 0)    // zeroing
                functor_type::apply (*it, value_type/*zero*/()), ++ it;
        } else {
            it += size;
        }
    }
    // Sparse case
    template<template <class T1, class T2> class F, class V, class E>
    //  This function seems to be big. So we do not let the compiler inline it.
    void vector_assign (V &v, const vector_expression<E> &e, sparse_tag) {
        BOOST_UBLAS_CHECK (v.size () == e ().size (), bad_size ());
        typedef F<typename V::iterator::reference, typename E::value_type> functor_type;
        BOOST_STATIC_ASSERT ((!functor_type::computed));
        typedef typename V::value_type value_type;
        v.clear ();
        typename E::const_iterator ite (e ().begin ());
        typename E::const_iterator ite_end (e ().end ());
        while (ite != ite_end) {
            value_type t (*ite);
            if (t != value_type/*zero*/())
                v.insert_element (ite.index (), t);
            ++ ite;
        }
    }
    // Sparse proxy or functional case
    template<template <class T1, class T2> class F, class V, class E>
    //  This function seems to be big. So we do not let the compiler inline it.
    void vector_assign (V &v, const vector_expression<E> &e, sparse_proxy_tag) {
        BOOST_UBLAS_CHECK (v.size () == e ().size (), bad_size ());
        typedef F<typename V::iterator::reference, typename E::value_type> functor_type;
        typedef typename V::size_type size_type;
        typedef typename V::difference_type difference_type;
        typedef typename V::value_type value_type;
        detail::make_conformant (v, e);

        typename V::iterator it (v.begin ());
        typename V::iterator it_end (v.end ());
        typename E::const_iterator ite (e ().begin ());
        typename E::const_iterator ite_end (e ().end ());
        if (it != it_end && ite != ite_end) {
            size_type it_index = it.index (), ite_index = ite.index ();
            while (true) {
                difference_type compare = it_index - ite_index;
                if (compare == 0) {
                    functor_type::apply (*it, *ite);
                    ++ it, ++ ite;
                    if (it != it_end && ite != ite_end) {
                        it_index = it.index ();
                        ite_index = ite.index ();
                    } else
                        break;
                } else if (compare < 0) {
                    if (!functor_type::computed) {
                        functor_type::apply (*it, value_type/*zero*/());
                        ++ it;
                    } else
                        increment (it, it_end, - compare);
                    if (it != it_end)
                        it_index = it.index ();
                    else
                        break;
                } else if (compare > 0) {
                    increment (ite, ite_end, compare);
                    if (ite != ite_end)
                        ite_index = ite.index ();
                    else
                        break;
                }
            }
        }

        if (!functor_type::computed) {
            while (it != it_end) {  // zeroing
                functor_type::apply (*it, value_type/*zero*/());
                ++ it;
            }
        } else {
            it = it_end;
        }
    }

    // Dispatcher
    template<template <class T1, class T2> class F, class V, class E>
    
    void vector_assign (V &v, const vector_expression<E> &e) {
        typedef typename vector_assign_traits<typename V::storage_category,
                                              F<typename V::reference, typename E::value_type>::computed,
                                              typename E::const_iterator::iterator_category>::storage_category storage_category;
        vector_assign<F> (v, e, storage_category ());
    }

    template<class SC, class RI>
    struct vector_swap_traits {
        typedef SC storage_category;
    };

    template<>
    struct vector_swap_traits<dense_proxy_tag, sparse_bidirectional_iterator_tag> {
        typedef sparse_proxy_tag storage_category;
    };

    template<>
    struct vector_swap_traits<packed_proxy_tag, sparse_bidirectional_iterator_tag> {
        typedef sparse_proxy_tag storage_category;
    };

    // Dense (proxy) case
    template<template <class T1, class T2> class F, class V, class E>
    //  This function seems to be big. So we do not let the compiler inline it.
    void vector_swap (V &v, vector_expression<E> &e, dense_proxy_tag) {
        typedef F<typename V::iterator::reference, typename E::iterator::reference> functor_type;
        typedef typename V::difference_type difference_type;
        difference_type size (BOOST_UBLAS_SAME (v.size (), e ().size ()));
        typename V::iterator it (v.begin ());
        typename E::iterator ite (e ().begin ());
        while (-- size >= 0)
            functor_type::apply (*it, *ite), ++ it, ++ ite;
    }
    // Packed (proxy) case
    template<template <class T1, class T2> class F, class V, class E>
    //  This function seems to be big. So we do not let the compiler inline it.
    void vector_swap (V &v, vector_expression<E> &e, packed_proxy_tag) {
        typedef F<typename V::iterator::reference, typename E::iterator::reference> functor_type;
        typedef typename V::difference_type difference_type;
        typename V::iterator it (v.begin ());
        typename V::iterator it_end (v.end ());
        typename E::iterator ite (e ().begin ());
        typename E::iterator ite_end (e ().end ());
        difference_type it_size (it_end - it);
        difference_type ite_size (ite_end - ite);
        if (it_size > 0 && ite_size > 0) {
            difference_type size ((std::min) (difference_type (it.index () - ite.index ()), ite_size));
            if (size > 0) {
                ite += size;
                ite_size -= size;
            }
        }
        if (it_size > 0 && ite_size > 0) {
            difference_type size ((std::min) (difference_type (ite.index () - it.index ()), it_size));
            if (size > 0)
                it_size -= size;
        }
        difference_type size ((std::min) (it_size, ite_size));
        it_size -= size;
        ite_size -= size;
        while (-- size >= 0)
            functor_type::apply (*it, *ite), ++ it, ++ ite;
    }
    // Sparse proxy case
    template<template <class T1, class T2> class F, class V, class E>
    //  This function seems to be big. So we do not let the compiler inline it.
    void vector_swap (V &v, vector_expression<E> &e, sparse_proxy_tag) {
        BOOST_UBLAS_CHECK (v.size () == e ().size (), bad_size ());
        typedef F<typename V::iterator::reference, typename E::iterator::reference> functor_type;
        typedef typename V::size_type size_type;
        typedef typename V::difference_type difference_type;

        detail::make_conformant (v, e);
        // FIXME should be a seperate restriction for E
        detail::make_conformant (e (), v);

        typename V::iterator it (v.begin ());
        typename V::iterator it_end (v.end ());
        typename E::iterator ite (e ().begin ());
        typename E::iterator ite_end (e ().end ());
        if (it != it_end && ite != ite_end) {
            size_type it_index = it.index (), ite_index = ite.index ();
            while (true) {
                difference_type compare = it_index - ite_index;
                if (compare == 0) {
                    functor_type::apply (*it, *ite);
                    ++ it, ++ ite;
                    if (it != it_end && ite != ite_end) {
                        it_index = it.index ();
                        ite_index = ite.index ();
                    } else
                        break;
                } else if (compare < 0) {
                    increment (it, it_end, - compare);
                    if (it != it_end)
                        it_index = it.index ();
                    else
                        break;
                } else if (compare > 0) {
                    increment (ite, ite_end, compare);
                    if (ite != ite_end)
                        ite_index = ite.index ();
                    else
                        break;
                }
            }
        }
    }

    // Dispatcher
    template<template <class T1, class T2> class F, class V, class E>
    
    void vector_swap (V &v, vector_expression<E> &e) {
        typedef typename vector_swap_traits<typename V::storage_category,
                                            typename E::const_iterator::iterator_category>::storage_category storage_category;
        vector_swap<F> (v, e, storage_category ());
    }

}}

#endif
