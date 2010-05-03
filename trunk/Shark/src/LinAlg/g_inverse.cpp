//===========================================================================
/*!
 *  \file g_inverse.cpp
 *
 *  \brief Determines the generalized inverse matrix of an input matrix
 *         by using singular value decomposition.
 *
 *  \author  M. Kreutz
 *  \date    1998
 *
 *  \par Copyright (c) 1998-2000:
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
 *  This file is part of Shark. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 2, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, write to the Free Software
 *  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 *
 *
 */
//===========================================================================

#include <cmath>
#include <SharkDefs.h>
#include <LinAlg/LinAlg.h>

//===========================================================================
/*!
 *  \brief Calculates the generalized inverse matrix of input matrix "amatA".
 *
 *  Given an input matrix \f$ X \f$ this function uses singular value
 *  decomposition to determine the generalized inverse matrix \f$ X' \f$,
 *  so that
 *
 *  \f$
 *      XX'X = X
 *  \f$
 *
 *  If \f$ X \f$ is singular, i.e. \f$ det(X) = 0 \f$ or \f$ X \f$ is
 *  non-square then \f$ X' \f$ is not unique.
 *
 *      \param  amatA \f$ m \times n \f$ input matrix.
 *      \param	bmatA \f$ n \times m \f$ generalized inverse matrix.
 *      \param  maxIterations Number of iterations after which the SVD calculation
 *					  algorithm gives up, if the solution has still not converged.
 *					  Default ist 200 Iterations.
 *		  \param	tolerance singular values less than this value will be considered zero.
 *					  Default is 1e-10.
 *      \param  ignoreThreshold If set to false, the method throws an exception if 
 *              the threshold maxIterations is exceeded. Otherwise it uses the 
 *              approximate intermediate results in the further calculations. 
 *              The default is true.
 *      \return       none.
 *
 *
 *  Please follow the link to view the source code of the example.
 *  The example can be executed in the example directory
 *  of package LinAlg.
 *
 *  \author  M. Kreutz
 *  \date    1998
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 *  \sa svd.cpp
 *
 */
unsigned g_inverse
(
	const Array2D< double >& amatA,
	Array2D< double >& bmatA,
	unsigned maxIterations,
	double tolerance,
	bool ignoreThreshold
)
{
	unsigned i, j, k, m, n, r;

	m = amatA.rows();
	n = amatA.cols();

	Array2D< double > umatL(m, n);
	Array2D< double > vmatL(n, n);
	Array  < double > wvecL(n);

	bmatA.resize(n, m, false);

	if (m == 0 || n == 0) {
		return 0;
	}
	//calculate singular value decomposition
	//throw exception if the can't be calculated
	//within maxIterations iterations
	try {
		svd(amatA, umatL, vmatL, wvecL, maxIterations, ignoreThreshold);
	} catch (SharkException e) { 
		throw(e);
	}
	svdsort(umatL, vmatL, wvecL);
	//determine rank
	r = svdrank(amatA, umatL, vmatL, wvecL);
	//normalize singular values
	for (i = 0; i < r; i++) {
		if (wvecL(i) < tolerance)
			wvecL(i) = 0.;
		else
			wvecL(i) = 1. / wvecL(i);
	}
	for (; i < n; i++) {
		wvecL(i) = 0.;
	}

	for (i = 0; i < n; i++) {
		for (j = 0; j < m; j++) {
			double  t  = 0.;
			//calculate (pseudo-)inverse
			for (k = 0; k < n; k++) {
				t += vmatL(i, k) * wvecL(k) * umatL(j, k);
			}

			bmatA(i, j) = t;
		}
	}

	return r;
}

//===========================================================================
/*!
 *  \file geninv.h
 *
 *  \brief Determines the generalized inverse matrix of an input matrix
 *         by using ...
 *
 *  \author  Thorsten Suttorp
 *  \date    2008
 *
 *  \par Copyright (c) 1998-2008:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-27978<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
 *
 *  \par Project:
 *      LinAlg
 *
 *
 *  <BR>
 *
 *
 *  <BR><HR>
 *  This file is part of LinAlg. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 2, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, write to the Free Software
 *  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */
//===========================================================================

#include <cmath>
#include <SharkDefs.h>
#include <LinAlg/LinAlg.h>

void invertSymmPositiveDefinite(Array2D<double> &I, const Array2D< double >& ArrSymm) {
	Array2D<double> CholArraySymm, CholArraySymmInv, CholArraySymmInvT;

	CholeskyDecomposition(ArrSymm, CholArraySymm);

	CholArraySymmInv.resize(CholArraySymm, false);
	CholArraySymmInv = 0;

	unsigned m = CholArraySymm.dim(0);
	double s;
	unsigned i, j, k;

	for(j = 0; j < m; j++) 
		CholArraySymmInv(j ,j) = 1/CholArraySymm(j,j);

	for(j = 0; j < m; j++) {
		for(i = j+1; i < m; i++) {
			s = 0;
			for(k = 0; k < i; k++) {
				s += CholArraySymm(i, k) * CholArraySymmInv(k, j);
			}			
			CholArraySymmInv(i ,j) = -1/CholArraySymm(i, i)*s;
		}
	}

	CholArraySymmInvT = CholArraySymmInv;
	CholArraySymmInvT.transpose();
	
	matMat(I, CholArraySymmInvT, CholArraySymmInv);
}


//===========================================================================
/*!
 *  \brief Returns the generalized inverse matrix of input matrix "A"
 *         according to the paper: Fast computation of Moore-Penrose
 *         inverse matrices. Neural Information Processing-Letters and
 *         Reviews 8(2), pp. 25-29, 2005
 *
 *  \param  A The input matrix.
 *  \return   The generalized inverse matrix.
 *  \throw check_exception the type of the exception will be
 *         "size mismatch" and indicates that \em A is not a
 *         square matrix
 *
 *  \author  T. Suttorp
 *  \date    2008
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 *
 */
void g_inverseCholesky(const Array2D< double >& A, Array2D< double >& outA, double thresholdFactor) {
	SIZE_CHECK(A.ndim() == 2)

	bool bTranspose = false;
	unsigned dimX = A.dim(0);
	unsigned dimY = A.dim(1);

	Array2D< double > AT = Array2D<double>(transpose(A));
	Array2D< double > AMat;

	if (dimX < dimY) {
		bTranspose = true;

		matMat( AMat, A, AT );
		dimY = dimX;
	}
	else 
		matMat( AMat, AT, A );
	

	double minDiagElem = AMat(0, 0);

	for (unsigned i = 1; i<dimY; ++i) 
		if (AMat(i,i) < minDiagElem)
			minDiagElem = AMat(i,i);

	minDiagElem	= minDiagElem * thresholdFactor;

	unsigned m = dimY;
	unsigned n = dimY;
	unsigned r = 0;
	unsigned i, k, l;
	double s;


	Array2D< double > C;
	C.resize(m, m, false);
	C=0;

	for(k = 0; k < n; k++) {
		r++;
		for(i = k; i < n; i++) {
			s = 0;
			for(l = 0; l < r-1; l++) 
				s += C(i, l) * C(k, l);
	
			C(i, r-1) = AMat(i, k)  - s;
		}
		if (C(k, r-1)> minDiagElem) {
			C(k, r-1)  = sqrt(C(k, r-1));
			if (k<n)
				for(l = k+1; l < n; l++) 
					C(l, r-1) =  C(l, r-1) / C(k, r-1);
		}
		else
			r--;
	}

	// delete m-r last cols
	Array2D< double > C2(m, r);

	for (unsigned i = 0; i < m; ++i)
		for (unsigned j = 0; j < r; ++j)
			C2(i,j)  =  C(i,j); 

		
	Array2D< double > ArrSymm, ArrSymmInv;
	Array2D<double> tmpMatrix, tmpMatrix2;

	Array2D< double > C2T = transpose(C2);
	matMat( ArrSymm, C2T, C2 );

#if 0
	invertSymm(ArrSymmInv, ArrSymm);
#else
	invertSymmPositiveDefinite(ArrSymmInv, ArrSymm);
#endif

	matMat( tmpMatrix, ArrSymmInv, ArrSymmInv );
	matMat( tmpMatrix2, tmpMatrix, C2T );
	matMat( tmpMatrix, C2, tmpMatrix2 );

	if (bTranspose) 
		matMat(outA, AT, tmpMatrix);
	else 
		matMat(outA, tmpMatrix, AT);
}

//===========================================================================
/*!
 *  \brief Returns the generalized inverse matrix of input matrix
 *         "A" using Cholesky decomposition assuming that \f$ A^T A \f$ has full rank.
 *
 *  \param  A The input matrix.
 *  \return   The generalized inverse matrix.
 *  \throw check_exception the type of the exception will be
 *         "size mismatch" and indicates that \em A is not a
 *         square matrix
 *
 *  \author  T. Suttorp
 *  \date    2008
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 *
 */
void g_inverseMoorePenrose(const Array2D< double >& A, Array2D< double >& outA) {
	SIZE_CHECK(A.ndim() == 2)

	bool bTranspose = false;
	unsigned dimX = A.dim(0);
	unsigned dimY = A.dim(1);

	Array2D< double > AT = Array2D<double>(transpose(A));
	Array2D< double > AMat;

	if (dimX < dimY) {
		bTranspose = true;
		matMat( AMat, A, AT );
		dimY = dimX;
	}
	else 
		matMat( AMat, AT, A );

	Array2D<double> AMatInv;
	invertSymmPositiveDefinite(AMatInv, AMat);

	if (bTranspose) 
		matMat(outA, AT, AMatInv);
	else
		matMat(outA, AMatInv, AT);

}
