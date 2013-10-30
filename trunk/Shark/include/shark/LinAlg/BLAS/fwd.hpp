#ifndef SHARK_LINALG_BLAS_UBLAS_FWD_H
#define SHARK_LINALG_BLAS_UBLAS_FWD_H

namespace shark {
namespace blas {

// Storage types
class range;

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


// Sparse vectors
template<class T, class I = std::size_t>
class compressed_vector;

}
}

#endif
