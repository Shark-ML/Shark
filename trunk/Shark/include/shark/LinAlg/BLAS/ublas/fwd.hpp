//
//  Copyright (c) 2000-2010
//  Joerg Walter, Mathias Koch, David Bellot
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  GeNeSys mbH & Co. KG in producing this work.
//

/// \file fwd.hpp is essentially used to forward declare the main types

#ifndef BOOST_UBLAS_FWD_H
#define BOOST_UBLAS_FWD_H

#include <memory>

namespace shark {
namespace blas {

// Storage types
template<class T, class ALLOC = std::allocator<T> >
class unbounded_array;

template <class Z = std::size_t, class D = std::ptrdiff_t>
class basic_range;
typedef basic_range<> range;

// Expression types

template<class E>
struct vector_expression;
template<class C>
struct vector_container;
template<class E>
class vector_reference;

template<class E>
struct matrix_expression;
template<class C>
struct matrix_container;
template<class E>
class matrix_reference;

template<class V>
class vector_range;

template<class M>
class matrix_row;
template<class M>
class matrix_column;
template<class M>
class matrix_vector_range;
template<class M>
class matrix_range;

template<class T, class A = unbounded_array<T> >
class vector;


// Sparse vectors
template<class T, std::size_t IB = 0, class IA = unbounded_array<std::size_t>, class TA = unbounded_array<T> >
class compressed_vector;

// Matrix orientation type
struct unknown_orientation_tag {};
struct row_major_tag {};
struct column_major_tag {};

// Matrix storage layout parameterisation
template <class Z = std::size_t, class D = std::ptrdiff_t>
struct basic_row_major;
typedef basic_row_major<> row_major;

template <class Z = std::size_t, class D = std::ptrdiff_t>
struct basic_column_major;
typedef basic_column_major<> column_major;

template<class T, class L = row_major, class A = unbounded_array<T> >
class matrix;

// Triangular matrix type
struct lower_tag {};
struct upper_tag {};
struct unit_lower_tag : public lower_tag {};
struct unit_upper_tag : public upper_tag {};
struct strict_lower_tag : public lower_tag {};
struct strict_upper_tag : public upper_tag {};

// Triangular matrix parameterisation
template <class Z = std::size_t>
struct basic_full;
typedef basic_full<> full;

// Sparse matrices
template<class T, class L = row_major, std::size_t IB = 0, class IA = unbounded_array<std::size_t>, class TA = unbounded_array<T> >
class compressed_matrix;

}
}

#endif
