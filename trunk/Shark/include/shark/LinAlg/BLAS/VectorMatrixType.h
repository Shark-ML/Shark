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

namespace shark {
/**
* \brief Convenience macro for mass template specialization on the
*  supplied type with the given prefix.
*/
#define SHARK_VECTOR_MATRIX_TYPEDEFS(basetype, prefix) \
	typedef blas::vector< basetype > prefix##Vector; \
	typedef blas::vector< const basetype > Const##prefix##Vector; \
	typedef blas::matrix< basetype, blas::row_major > prefix##Matrix; \
	typedef blas::zero_matrix< basetype > prefix##ZeroMatrix; \
	typedef blas::identity_matrix< basetype > prefix##Identity; \
	typedef blas::identity_matrix< basetype > prefix##IdentityMatrix; \
	typedef blas::scalar_matrix< basetype > prefix##ScalarMatrix; \
	typedef blas::vector_range< prefix##Vector > prefix##VectorRange; \
	typedef blas::vector_range< const prefix##Vector > Const##prefix##VectorRange; \
	typedef blas::matrix_row< prefix##Matrix > prefix##MatrixRow; \
	typedef blas::matrix_row< const prefix##Matrix > Const##prefix##MatrixRow; \
	typedef blas::matrix_column< prefix##Matrix > prefix##MatrixColumn; \
	typedef blas::matrix_column< const prefix##Matrix > Const##prefix##MatrixColumn; \
	typedef blas::matrix_range< prefix##Matrix > prefix##SubMatrix; \
	typedef blas::matrix_range< const prefix##Matrix > Const##prefix##SubMatrix; \
	typedef blas::compressed_vector< basetype > Compressed##prefix##Vector; \
	typedef blas::vector_range< Compressed##prefix##Vector > Compressed##prefix##VectorRange; \
	typedef blas::vector_range< const Compressed##prefix##Vector > ConstCompressed##prefix##VectorRange; \
	typedef blas::compressed_matrix< basetype > Compressed##prefix##Matrix; \
	typedef blas::matrix_row< Compressed##prefix##Matrix > Compressed##prefix##MatrixRow; \
	typedef blas::matrix_row< const Compressed##prefix##Matrix > ConstCompressed##prefix##MatrixRow; \
	typedef blas::matrix_column< Compressed##prefix##Matrix > Compressed##prefix##MatrixColumn; \
	typedef blas::matrix_column< const Compressed##prefix##Matrix > ConstCompressed##prefix##MatrixColumn; \
	typedef blas::matrix_range< Compressed##prefix##Matrix > Compressed##prefix##SubMatrix; \
	typedef blas::matrix_range< const Compressed##prefix##Matrix > ConstCompressed##prefix##SubMatrix;\
	typedef blas::diagonal_matrix<blas::vector< basetype > > prefix##DiagonalMatrix;

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

	SHARK_VECTOR_MATRIX_TYPEDEFS(long double, BigReal);
	SHARK_VECTOR_MATRIX_TYPEDEFS(double, Real)
	SHARK_VECTOR_MATRIX_TYPEDEFS(float, Float)
	SHARK_VECTOR_MATRIX_TYPEDEFS(std::complex<double>, Complex)
	SHARK_VECTOR_MATRIX_TYPEDEFS(int, Int)
	SHARK_VECTOR_MATRIX_TYPEDEFS(unsigned int, UInt)
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

typedef blas::range Range;
typedef blas::permutation_matrix<std::size_t> PermutationMatrix;

SHARK_VECTOR_MATRIX_ASSIGNMENT(BigReal);
SHARK_VECTOR_MATRIX_ASSIGNMENT(Real)
SHARK_VECTOR_MATRIX_ASSIGNMENT(Float)
SHARK_VECTOR_MATRIX_ASSIGNMENT(Complex)
SHARK_VECTOR_MATRIX_ASSIGNMENT(Int)
SHARK_VECTOR_MATRIX_ASSIGNMENT(UInt)
}
	
#endif 
