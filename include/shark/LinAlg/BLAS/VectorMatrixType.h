/**
*
*  \brief Convenience macros for defining vector and matrix types.
*
*  \author O.Krause, T.Glasmachers, T. Voss
*  \date 2010-2011
*
*  \par Copyright (c) 1998-2007:
*      Institut f&uuml;r Neuroinformatik<BR>
*      Ruhr-Universit&auml;t Bochum<BR>
*      D-44780 Bochum, Germany<BR>
*      Phone: +49-234-32-25558<BR>
*      Fax:   +49-234-32-14209<BR>
*      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
*      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
*      <BR>
*
*
*  <BR><HR>
*  This file is part of Shark. This library is free software;
*  you can redistribute it and/or modify it under the terms of the
*  GNU General Public License as published by the Free Software
*  Foundation; either version 3, or (at your option) any later version.
*
*  This library is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with this library; if not, see <http://www.gnu.org/licenses/>.
*
*/
#ifndef SHARK_LINALG_BLAS_MACROS_H
#define SHARK_LINALG_BLAS_MACROS_H

//All ublas files required by shark
#include <shark/LinAlg/BLAS/ublas.h>
#include <shark/LinAlg/BLAS/Proxy.h>
///for complex vectors (currently not supported very well)
#include <complex>

/**
* \brief Convenience macro for mass template specialization on the
*  supplied type with the given prefix.
*/
#define SHARK_VECTOR_MATRIX_TYPEDEFS(basetype, prefix) \
	typedef blas::vector< basetype > prefix##Vector; \
	typedef blas::vector< const basetype > Const##prefix##Vector; \
	typedef blas::matrix< basetype, blas::row_major > prefix##Matrix; \
	typedef blas::matrix< const basetype, blas::row_major > Const##prefix##Matrix; \
	typedef blas::zero_vector< basetype > prefix##ZeroVector; \
	typedef blas::zero_vector< const basetype > Const##prefix##ZeroVector; \
	typedef blas::unit_vector< basetype > prefix##UnitVector; \
	typedef blas::unit_vector< const basetype > Const##prefix##UnitVector; \
	typedef blas::scalar_vector< basetype > prefix##ScalarVector; \
	typedef blas::scalar_vector< const basetype > Const##prefix##ScalarVector; \
	typedef blas::zero_matrix< basetype > prefix##ZeroMatrix; \
	typedef blas::zero_matrix< const basetype > Const##prefix##ZeroMatrix; \
	typedef blas::identity_matrix< basetype > prefix##Identity; \
	typedef blas::identity_matrix< basetype > prefix##IdentityMatrix; \
	typedef blas::scalar_matrix< basetype > prefix##ScalarMatrix; \
	typedef blas::scalar_matrix< const basetype > Const##prefix##ScalarMatrix; \
	typedef blas::diagonal_matrix< basetype > prefix##DiagonalMatrix; \
	typedef blas::diagonal_matrix< const basetype > Const##prefix##DiagonalMatrix; \
	typedef blas::vector_range< prefix##Vector > prefix##VectorRange; \
	typedef blas::vector_range< const prefix##Vector > Const##prefix##VectorRange; \
	typedef blas::vector_range< prefix##Vector > prefix##SubVector; \
	typedef blas::vector_range< const prefix##Vector > Const##prefix##SubVector; \
	typedef blas::matrix_row< prefix##Matrix > prefix##MatrixRow; \
	typedef blas::matrix_row< const prefix##Matrix > Const##prefix##MatrixRow; \
	typedef blas::matrix_column< prefix##Matrix > prefix##MatrixColumn; \
	typedef blas::matrix_column< const prefix##Matrix > Const##prefix##MatrixColumn; \
	typedef blas::matrix_vector_range< prefix##Matrix > prefix##MatrixVectorRange; \
	typedef blas::matrix_vector_range< const prefix##Matrix > Const##prefix##MatrixVectorRange; \
	typedef blas::matrix_range< prefix##Matrix > prefix##SubMatrix; \
	typedef blas::matrix_range< const prefix##Matrix > Const##prefix##SubMatrix; \
	typedef blas::mapped_vector< basetype > Mapped##prefix##Vector; \
	typedef blas::vector_range< Mapped##prefix##Vector > Mapped##prefix##VectorRange; \
	typedef blas::vector_range< const Mapped##prefix##Vector > ConstMapped##prefix##VectorRange; \
	typedef blas::vector_range< Mapped##prefix##Vector > Mapped##prefix##SubVector; \
	typedef blas::vector_range< const Mapped##prefix##Vector > ConstMapped##prefix##SubVector; \
	typedef blas::compressed_vector< basetype > Compressed##prefix##Vector; \
	typedef blas::vector_range< Compressed##prefix##Vector > Compressed##prefix##VectorRange; \
	typedef blas::vector_range< const Compressed##prefix##Vector > ConstCompressed##prefix##VectorRange; \
	typedef blas::vector_range< Compressed##prefix##Vector > Compressed##prefix##SubVector; \
	typedef blas::vector_range< const Compressed##prefix##Vector > ConstCompressed##prefix##SubVector; \
	typedef blas::coordinate_vector< basetype > Coordinate##prefix##Vector; \
	typedef blas::vector_range< Coordinate##prefix##Vector > Coordinate##prefix##VectorRange; \
	typedef blas::vector_range< const Coordinate##prefix##Vector > ConstCoordinate##prefix##VectorRange; \
	typedef blas::vector_range< Coordinate##prefix##Vector > Coordinate##prefix##SubVector; \
	typedef blas::vector_range< const Coordinate##prefix##Vector > ConstCoordinate##prefix##SubVector; \
	typedef blas::mapped_matrix< basetype > Mapped##prefix##Matrix; \
	typedef blas::matrix_row< Mapped##prefix##Matrix > Mapped##prefix##MatrixRow; \
	typedef blas::matrix_row< const Mapped##prefix##Matrix > ConstMapped##prefix##MatrixRow; \
	typedef blas::matrix_column< Mapped##prefix##Matrix > Mapped##prefix##MatrixColumn; \
	typedef blas::matrix_column< const Mapped##prefix##Matrix > ConstMapped##prefix##MatrixColumn; \
	typedef blas::matrix_vector_range< Mapped##prefix##Matrix > Mapped##prefix##MatrixVectorRange; \
	typedef blas::matrix_vector_range< const Mapped##prefix##Matrix > ConstMapped##prefix##MatrixVectorRange; \
	typedef blas::matrix_range< Mapped##prefix##Matrix > Mapped##prefix##SubMatrix; \
	typedef blas::matrix_range< const Mapped##prefix##Matrix > ConstMapped##prefix##SubMatrix; \
	typedef blas::compressed_matrix< basetype > Compressed##prefix##Matrix; \
	typedef blas::matrix_row< Compressed##prefix##Matrix > Compressed##prefix##MatrixRow; \
	typedef blas::matrix_row< const Compressed##prefix##Matrix > ConstCompressed##prefix##MatrixRow; \
	typedef blas::matrix_column< Compressed##prefix##Matrix > Compressed##prefix##MatrixColumn; \
	typedef blas::matrix_column< const Compressed##prefix##Matrix > ConstCompressed##prefix##MatrixColumn; \
	typedef blas::matrix_vector_range< Compressed##prefix##Matrix > Compressed##prefix##MatrixVectorRange; \
	typedef blas::matrix_vector_range< const Compressed##prefix##Matrix > ConstCompressed##prefix##MatrixVectorRange; \
	typedef blas::matrix_range< Compressed##prefix##Matrix > Compressed##prefix##SubMatrix; \
	typedef blas::matrix_range< const Compressed##prefix##Matrix > ConstCompressed##prefix##SubMatrix;

/**
	* \brief Convenience macro for pulling device-specific linear-algebra
	* traits into arbitrary scopes/namespaces.
	*/
#define SELECT_DEFAULT_COMPUTING_DEVICE( scope, prefix) \
	typedef scope prefix##Vector prefix##Vector; \
	typedef scope Const##prefix##Vector Const##prefix##Vector; \
	typedef scope prefix##Matrix prefix##Matrix; \
	typedef scope Const##prefix##Matrix Const##prefix##Matrix; \
	typedef scope prefix##ZeroVector prefix##ZeroVector; \
	typedef scope Const##prefix##ZeroVector Const##prefix##ZeroVector; \
	typedef scope prefix##UnitVector prefix##UnitVector; \
	typedef scope Const##prefix##UnitVector Const##prefix##UnitVector; \
	typedef scope prefix##ScalarVector prefix##ScalarVector; \
	typedef scope Const##prefix##ScalarVector Const##prefix##ScalarVector; \
	typedef scope prefix##ZeroMatrix prefix##ZeroMatrix; \
	typedef scope Const##prefix##ZeroMatrix Const##prefix##ZeroMatrix; \
	typedef scope prefix##Identity prefix##Identity; \
	typedef scope prefix##IdentityMatrix prefix##IdentityMatrix; \
	typedef scope prefix##ScalarMatrix prefix##ScalarMatrix; \
	typedef scope Const##prefix##ScalarMatrix Const##prefix##ScalarMatrix; \
	typedef scope prefix##DiagonalMatrix prefix##DiagonalMatrix; \
	typedef scope Const##prefix##DiagonalMatrix Const##prefix##DiagonalMatrix; \
	typedef scope prefix##VectorRange prefix##VectorRange; \
	typedef scope Const##prefix##VectorRange Const##prefix##VectorRange; \
	typedef scope prefix##SubVector prefix##SubVector; \
	typedef scope Const##prefix##SubVector Const##prefix##SubVector; \
	typedef scope prefix##MatrixRow prefix##MatrixRow; \
	typedef scope Const##prefix##MatrixRow Const##prefix##MatrixRow; \
	typedef scope prefix##MatrixColumn prefix##MatrixColumn; \
	typedef scope Const##prefix##MatrixColumn Const##prefix##MatrixColumn; \
	typedef scope prefix##MatrixVectorRange prefix##MatrixVectorRange; \
	typedef scope Const##prefix##MatrixVectorRange Const##prefix##MatrixVectorRange; \
	typedef scope prefix##SubMatrix prefix##SubMatrix; \
	typedef scope Const##prefix##SubMatrix Const##prefix##SubMatrix; \
	typedef scope Mapped##prefix##Vector Mapped##prefix##Vector; \
	typedef scope Mapped##prefix##VectorRange Mapped##prefix##VectorRange; \
	typedef scope ConstMapped##prefix##VectorRange ConstMapped##prefix##VectorRange; \
	typedef scope Mapped##prefix##SubVector Mapped##prefix##SubVector; \
	typedef scope ConstMapped##prefix##SubVector ConstMapped##prefix##SubVector; \
	typedef scope Compressed##prefix##Vector Compressed##prefix##Vector; \
	typedef scope Compressed##prefix##VectorRange Compressed##prefix##VectorRange; \
	typedef scope ConstCompressed##prefix##VectorRange ConstCompressed##prefix##VectorRange; \
	typedef scope Compressed##prefix##SubVector Compressed##prefix##SubVector; \
	typedef scope ConstCompressed##prefix##SubVector ConstCompressed##prefix##SubVector; \
	typedef scope Coordinate##prefix##Vector Coordinate##prefix##Vector; \
	typedef scope Coordinate##prefix##VectorRange Coordinate##prefix##VectorRange; \
	typedef scope ConstCoordinate##prefix##VectorRange ConstCoordinate##prefix##VectorRange; \
	typedef scope Coordinate##prefix##SubVector Coordinate##prefix##SubVector; \
	typedef scope ConstCoordinate##prefix##SubVector ConstCoordinate##prefix##SubVector; \
	typedef scope Mapped##prefix##Matrix Mapped##prefix##Matrix; \
	typedef scope Mapped##prefix##MatrixRow Mapped##prefix##MatrixRow; \
	typedef scope ConstMapped##prefix##MatrixRow ConstMapped##prefix##MatrixRow; \
	typedef scope Mapped##prefix##MatrixColumn Mapped##prefix##MatrixColumn; \
	typedef scope ConstMapped##prefix##MatrixColumn ConstMapped##prefix##MatrixColumn; \
	typedef scope Mapped##prefix##MatrixVectorRange Mapped##prefix##MatrixVectorRange; \
	typedef scope ConstMapped##prefix##MatrixVectorRange ConstMapped##prefix##MatrixVectorRange; \
	typedef scope Mapped##prefix##SubMatrix Mapped##prefix##SubMatrix; \
	typedef scope ConstMapped##prefix##SubMatrix ConstMapped##prefix##SubMatrix; \
	typedef scope Compressed##prefix##Matrix Compressed##prefix##Matrix; \
	typedef scope Compressed##prefix##MatrixRow Compressed##prefix##MatrixRow; \
	typedef scope ConstCompressed##prefix##MatrixRow ConstCompressed##prefix##MatrixRow; \
	typedef scope Compressed##prefix##MatrixColumn Compressed##prefix##MatrixColumn; \
	typedef scope ConstCompressed##prefix##MatrixColumn ConstCompressed##prefix##MatrixColumn; \
	typedef scope Compressed##prefix##MatrixVectorRange Compressed##prefix##MatrixVectorRange; \
	typedef scope ConstCompressed##prefix##MatrixVectorRange ConstCompressed##prefix##MatrixVectorRange; \
	typedef scope Compressed##prefix##SubMatrix Compressed##prefix##SubMatrix; \
	typedef scope ConstCompressed##prefix##SubMatrix ConstCompressed##prefix##SubMatrix;

#define SHARK_VECTOR_MATRIX_ASSIGNMENT(prefix) \
	template<> struct VectorMatrixTraits< prefix##Vector >{ \
		typedef prefix##Matrix MatrixType;\
		typedef prefix##Matrix DenseMatrixType;\
		typedef prefix##Vector VectorType;\
		typedef prefix##Vector DenseVectorType;\
		typedef prefix##VectorRange SubType;\
		typedef prefix##VectorRange DenseSubType;\
		typedef Const##prefix##VectorRange ConstSubType;\
		typedef Const##prefix##VectorRange ConstDenseSubType;\
	};\
	template<> struct VectorMatrixTraits< Mapped##prefix##Vector >{ \
		typedef Mapped##prefix##Matrix MatrixType;\
		typedef prefix##Matrix DenseMatrixType;\
		typedef Mapped##prefix##Vector VectorType;\
		typedef prefix##Vector DenseVectorType;\
		typedef Mapped##prefix##VectorRange SubType;\
		typedef prefix##VectorRange DenseSubType;\
		typedef ConstMapped##prefix##VectorRange ConstSubType;\
		typedef Const##prefix##VectorRange ConstDenseSubType;\
	};\
	template<> struct VectorMatrixTraits< Compressed##prefix##Vector >{ \
		typedef Compressed##prefix##Matrix MatrixType;\
		typedef prefix##Matrix DenseMatrixType;\
		typedef Compressed##prefix##Vector VectorType;\
		typedef prefix##Vector DenseVectorType;\
		typedef Compressed##prefix##VectorRange SubType;\
		typedef prefix##VectorRange DenseSubType;\
		typedef ConstCompressed##prefix##VectorRange ConstSubType;\
		typedef Const##prefix##VectorRange ConstDenseSubType;\
	};\
	template<> struct VectorMatrixTraits< prefix##VectorRange > { \
		typedef prefix##Matrix MatrixType;\
		typedef prefix##Matrix DenseMatrixType;\
		typedef prefix##Vector VectorType;\
		typedef prefix##Vector DenseVectorType;\
		typedef prefix##VectorRange SubType;\
		typedef prefix##VectorRange DenseSubType;\
		typedef Const##prefix##VectorRange ConstSubType;\
		typedef Const##prefix##VectorRange ConstDenseSubType;\
	};\
	template<> struct VectorMatrixTraits< Mapped##prefix##VectorRange > { \
		typedef Mapped##prefix##Matrix MatrixType;\
		typedef prefix##Matrix DenseMatrixType;\
		typedef Mapped##prefix##Vector VectorType;\
		typedef prefix##Vector DenseVectorType;\
		typedef Mapped##prefix##VectorRange SubType;\
		typedef prefix##VectorRange DenseSubType;\
		typedef ConstMapped##prefix##VectorRange ConstSubType;\
		typedef Const##prefix##VectorRange ConstDenseSubType;\
	};\
	template<> struct VectorMatrixTraits< Compressed##prefix##VectorRange > { \
		typedef Compressed##prefix##Matrix MatrixType;\
		typedef prefix##Matrix DenseMatrixType;\
		typedef Compressed##prefix##Vector VectorType;\
		typedef prefix##Vector DenseVectorType;\
		typedef Compressed##prefix##VectorRange SubType;\
		typedef prefix##VectorRange DenseSubType;\
		typedef ConstCompressed##prefix##VectorRange ConstSubType;\
		typedef Const##prefix##VectorRange ConstDenseSubType;\
	};
	
namespace shark {

namespace detail {

	/** 
	* \brief Models a central processing unit. 
	*/
	struct Cpu {};

	/** 
	* \brief Models special purpose processing units, e.g., GPU or 
	*  the DSP of a sound card (see OpenCL). 
	*/
	struct Spu {};

	/**
	* \brief Models linear algebra traits with respect to a computing device, e.g.,
	* the CPU or the GPU.
	*/
	template<typename ProcessorTag>
	struct LinearAlgebraTraits {
	};

	/**
	* \brief Template specialization for handling CPU traits.
	*/
	template<>
	struct LinearAlgebraTraits< Cpu > {
		/** \brief quadruple precision floating point. */
		SHARK_VECTOR_MATRIX_TYPEDEFS(long double, BigReal);
		
		/** \brief Double precision floating point. */
		SHARK_VECTOR_MATRIX_TYPEDEFS(double, Real)

		/** \brief Single precision floating point. */
		SHARK_VECTOR_MATRIX_TYPEDEFS(float, Float)

		/** \brief Double precision complex numbers. */
		SHARK_VECTOR_MATRIX_TYPEDEFS(std::complex<double>, Complex)

		/** \brief Signed integer type. */
		SHARK_VECTOR_MATRIX_TYPEDEFS(int, Int)

		/** \brief Unsigned integer type. */
		SHARK_VECTOR_MATRIX_TYPEDEFS(unsigned int, UInt)
	};

}

typedef blas::range Range;
typedef blas::permutation_matrix<std::size_t> PermutationMatrix;

/** \brief Injects RealVector, RealMatrix etc. into the shark namespace. */
SELECT_DEFAULT_COMPUTING_DEVICE( detail::LinearAlgebraTraits< shark::detail::Cpu >::, BigReal );

/** \brief Injects RealVector, RealMatrix etc. into the shark namespace. */
SELECT_DEFAULT_COMPUTING_DEVICE( detail::LinearAlgebraTraits< shark::detail::Cpu >::, Real )

/** \brief Injects FloatVector, FloatMatrix etc. into the shark namespace. */
SELECT_DEFAULT_COMPUTING_DEVICE( detail::LinearAlgebraTraits< shark::detail::Cpu >::, Float )

/** \brief Injects ComplexVector, ComplexMatrix etc. into the shark namespace. */
SELECT_DEFAULT_COMPUTING_DEVICE( detail::LinearAlgebraTraits< shark::detail::Cpu >::, Complex )

/** \brief Injects IntVector, IntMatrix etc. into the shark namespace. */
SELECT_DEFAULT_COMPUTING_DEVICE( detail::LinearAlgebraTraits< shark::detail::Cpu >::, Int )

/** \brief Injects UIntVector, UIntMatrix etc. into the shark namespace. */
SELECT_DEFAULT_COMPUTING_DEVICE( detail::LinearAlgebraTraits< shark::detail::Cpu >::, UInt )


#undef SHARK_VECTOR_MATRIX_TYPEDEFS

///\brief Template which finds for every Vector type the best fitting Matrix.
///
///As default a RealMatrix is used
template<class VectorType>
struct VectorMatrixTraits {
	typedef RealMatrix MatrixType;
	typedef RealMatrix DenseMatrixType;
	typedef RealVector SuperType;
	typedef RealVector DenseSuperType;
	typedef RealVectorRange SubType;
	typedef RealVectorRange DenseSubType;
	typedef ConstRealVectorRange ConstSubType;
	typedef ConstRealVectorRange ConstDenseSubType;
};

SHARK_VECTOR_MATRIX_ASSIGNMENT(BigReal);
SHARK_VECTOR_MATRIX_ASSIGNMENT(Real)
SHARK_VECTOR_MATRIX_ASSIGNMENT(Float)
SHARK_VECTOR_MATRIX_ASSIGNMENT(Complex)
SHARK_VECTOR_MATRIX_ASSIGNMENT(Int)
SHARK_VECTOR_MATRIX_ASSIGNMENT(UInt)
}
	
#endif 
