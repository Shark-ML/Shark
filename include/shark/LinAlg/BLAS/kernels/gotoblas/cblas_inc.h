#ifndef SHARK_LINALG_BLAS_KERNELS_GOTOBLAS_CBLAS_INC_H
#define SHARK_LINALG_BLAS_KERNELS_GOTOBLAS_CBLAS_INC_H

extern "C" {
#include <common.h>
#include <cblas.h>
}
#include <complex>
#include <shark/LinAlg/BLAS/traits/matrix_raw.hpp>
#include <shark/LinAlg/BLAS/traits/vector_raw.hpp>
#include <shark/Core/Exception.h>

namespace shark{ namespace detail{ namespace bindings {

template <typename Ord> struct storage_order {};
template<> struct storage_order<blas::row_major> {
	enum ename { value = CblasRowMajor };
};
template<> struct storage_order<blas::column_major> {
	enum ename { value = CblasColMajor };
};


template <typename UpLo> struct uplo_triang {};
template<> struct uplo_triang<blas::upper_tag> {
	enum ename { value = CblasUpper };
};
template<> struct uplo_triang<blas::lower_tag> {
	enum ename { value = CblasLower };
};

}}}



#endif
