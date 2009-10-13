//===========================================================================
/*!
 *  \file MixtureLinearRegression.cpp
 *
 *  \brief This file offers a class for creating a regression model that
 *         will approximate data pairs (x,y) by finding a linear mapping for
 *         them.
 *
 *  \author  M. Kreutz
 *  \date    1995-01-01
 *
 *  \par Copyright (c) 1995,1999:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *
 *  \par Project:
 *      LinAlg
 *
 *
 *
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
 *
 *
 */
//===========================================================================

#include <Array/ArrayOp.h>
#include <LinAlg/LinAlg.h>
#include <Mixture/MixtureLinearRegression.h>


//===========================================================================
/*!
 *  \brief Creates a new Linear Regression object and resets
 *         internal values.
 *
 *  \author  M. Kreutz
 *  \date    1995
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
MixtureLinearRegression::MixtureLinearRegression()
{
	//
	// the following initialization is not really necessary since
	// all comparision use sxM.nelem( ) instead of sxM.dim( 0 )
	//
	syM.resize(0U, false);
	sxM.resize(0U, false);

	reset();
}


//===========================================================================
/*!
 *  \brief Destructs a Linear Regression object.
 *
 *  \author  M. Kreutz
 *  \date    1995
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
MixtureLinearRegression::~MixtureLinearRegression()
{}


//===========================================================================
/*!
 *  \brief Resets internal variables of the class to initial values.
 *
 *  \return none
 *
 *  \author  M. Kreutz
 *  \date    1995
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
void MixtureLinearRegression::reset()
{
	modifiedM = true;
	countM    = 0;
	syM       = 0.;
	sxM       = 0.;
	rxxM      = 0.;
	rxyM      = 0.;
}


//===========================================================================
/*!
 *  \brief Calculates the current regression model.
 *
 *  Based on internal variables the current regression model
 *  is calculated if these internal variables have changed before
 *  due to adding/removing vector pair(s). This method is called
 *  automatically when you are calling the methods for returning
 *  the regression coefficients \f$A\f$ or \f$b\f$ or when
 *  you are calling the mehod for approximating an \f$y\f$ for a
 *  given \f$x\f$ (methods #A, #b, #recall), so you can be sure
 *  that always the most current model is used.
 *
 *  \return none
 *
 *  \author  M. Kreutz
 *  \date    1995
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 *  \sa #train, #remove, #A, #b, #recall
 *
 */
void MixtureLinearRegression::update()
{
	if (modifiedM && sxM.nelem() > 0 && countM > 0)
	{
		unsigned i, j;
		double   s, t;

		s = countM;

		Array  < double > dvecL(sxM.dim(0));
		Array  < double > mxL(sxM / s);
		Array  < double > myL(syM / s);
		Array2D< double > cxxL(sxM.dim(0), sxM.dim(0));
		Array2D< double > cxyL(sxM.dim(0), syM.dim(0));

		for (i = 0; i < sxM.dim(0); ++i)
		{
			t = mxL(i);

			for (j = 0; j <= i; j++)
			{
				cxxL(i, j) = rxxM(i, j) / s - t * mxL(j);
			}

			for (j = 0; j < syM.dim(0); ++j)
			{
				cxyL(i, j) = rxyM(i, j) / s - t * myL(j);
			}
		}

		linearRegress(cxxL, cxyL, mxL, myL, amatM, bvecM, dvecL);

		modifiedM = false;
	}
}


//===========================================================================
/*!
 *  \brief Adds the data vector pair(s) "x"/"y" to the regression model.
 *
 *  Given the data vector \f$x\f$ and its corresponding \f$y\f$,
 *  the internal variables from which the model is calculated
 *  are changed, i.e. the information given by \f$x\f$ and \f$y\f$
 *  is added to these variables. <br>
 *  You can also add several x/y-vector-pairs at once. <br>
 *  The model itself is not changed
 *  here, it will changed after calling method #update. <br>
 *  This separation is more efficient, e.g. when you are adding
 *  several data vectors at different time, because the model itself
 *  is then calculated
 *  only once after all vectors are added instead of each time
 *  after a single vector pair is added.
 *
 *  \param  x Vector(s) containing the first component of the data.
 *  \param  y Vector(s) containing the second component of the data.
 *  \return none
 *  \throw check_exception the type of the exception will be
 *         "size mismatch" and indicates that \em x and \em y
 *         are not one- and 2-dimensional or that one of both
 *         is one-dimensional and the other one 2-dimensional
 *         or that the number of vectors in \em x is different
 *         to that in \em y
 *
 *  \author  M. Kreutz
 *  \date    1995
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 *  \sa #update
 *
 */
void MixtureLinearRegression::train
(
	const Array< double >& x,
	const Array< double >& y
)
{
	SIZE_CHECK
	(
		(x.ndim() == 1 && y.ndim() == 1) ||
		(x.ndim() == 2 && y.ndim() == 2)
	)

	if (x.ndim() == 2)
	{
		SIZE_CHECK(x.dim(0) == y.dim(0))

		for (unsigned i = 0; i < x.dim(0); ++i)
		{
			train(x[ i ], y[ i ]);
		}
	}
	else
	{
		if (sxM.nelem() != x.dim(0) || syM.nelem() != y.dim(0))
		{
			if (countM > 0)
			{
				throw 1;
			}
			else
			{

				sxM  .resize(x.dim(0), false);
				syM  .resize(y.dim(0), false);
				rxxM .resize(x.dim(0), x.dim(0), false);
				rxyM .resize(x.dim(0), y.dim(0), false);
				amatM.resize(y.dim(0), x.dim(0), false);
				bvecM.resize(y.dim(0), false);

				sxM  = 0.;
				syM  = 0.;
				rxxM = 0.;
				rxyM = 0.;
			}
		}

		//
		// train linear regression model
		//
		++countM;

		syM += y;

		for (unsigned i = 0, j; i < sxM.dim(0); ++i)
		{
			double t  = x(i);
			sxM(i) += t;

			for (j = 0; j <= i; ++j)
			{
				rxxM(i, j) += t * x(j);
			}

			for (j = 0; j < syM.dim(0); ++j)
			{
				rxyM(i, j) += t * y(j);
			}
		}

		modifiedM = true;
	}
}



//===========================================================================
/*!
 *  \brief Removes the data vector pair(s) "x"/"y" from the regression model.
 *
 *  Given the data vector \f$x\f$ and its corresponding \f$y\f$,
 *  the internal variables from which the model is calculated
 *  are changed, i.e. the information given by \f$x\f$ and \f$y\f$
 *  is removed from these variables. <br>
 *  You can also remove several x/y-vector-pairs at once. <br>
 *  The model itself is not changed
 *  here, it will changed after calling method #update. <br>
 *  This separation is more efficient, e.g. when you are removing
 *  several data vectors at different time, because the model itself
 *  is then calculated
 *  only once after all vectors are removed instead of each time
 *  after a single vector pair is removed.
 *
 *  \param  x Vector(s) containing the first component of the data.
 *  \param  y Vector(s) containing the second component of the data.
 *  \return none
 *  \throw check_exception the type of the exception will be
 *         "size mismatch" and indicates that \em x and \em y
 *         are not one- and 2-dimensional or that one of both
 *         is one-dimensional and the other one 2-dimensional
 *         or that the number of vectors in \em x is different
 *         to that in \em y or that the model was not
 *         calculated yet or that it was not calculated for
 *         as many vectors as there are vectors in \em x and \em y
 *
 *  \author  M. Kreutz
 *  \date    1995
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 *  \sa #update
 *
 */
void MixtureLinearRegression::remove
	(
		const Array< double >& x,
		const Array< double >& y
	)
{
	SIZE_CHECK
	(
		(x.ndim() == 1 && y.ndim() == 1) ||
		(x.ndim() == 2 && y.ndim() == 2)
	)

	if (x.ndim() == 2)
	{
		SIZE_CHECK(x.dim(0) == y.dim(0))

		for (unsigned i = 0; i < x.dim(0); ++i)
		{
			remove(x[ i ], y[ i ]);
		}
	}
	else
	{
		SIZE_CHECK
		(
			countM > 0 &&
			sxM.dim(0) == x.dim(0) &&
			syM.dim(0) == y.dim(0)
		)

		//
		// remove vector from linear regression model
		//
		--countM;

		syM -= y;

		for (unsigned i = 0, j; i < sxM.dim(0); ++i)
		{
			double t  = x(i);
			sxM(i) -= t;

			for (j = 0; j <= i; ++j)
			{
				rxxM(i, j) -= t * x(j);
			}

			for (j = 0; j < syM.dim(0); ++j)
			{
				rxyM(i, j) -= t * y(j);
			}
		}

		modifiedM = true;
	}
}


//===========================================================================
/*!
 *  \brief Given one/several data vector(s) "x" the corresponding
 *         vector(s) "y" is/are calculated by using the current regression
 *         model.
 *
 *  The current regression model \f$y = A \cdot x + b\f$ is used to
 *  approximate \f$y\f$ for one/several given data vector(s) \f$x\f$.
 *
 *  \param  x Input vector(s) for which the corresponding y-vector(s)
 *            will be calculated.
 *  \param  y The corresponding (approximated) y-vector(s) for the
 *            input vector(s).
 *  \return none
 *  \throw check_exception the type of the exception will be
 *         "size mismatch" and indicates that \em x is not
 *         one- or 2-dimensional or that the model was not calculated
 *         for any training vector yet
 *
 *  \author  M. Kreutz
 *  \date    1995
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 *  \sa #update
 *
 */
void MixtureLinearRegression::recall
(
	const Array< double >& x,
	Array< double >& y
)
{
	SIZE_CHECK(x.ndim() == 1 || x.ndim() == 2)
	SIZE_CHECK(sxM.nelem() > 0 && countM > 0)

	if (x.ndim() == 2)
	{
		y.resize(x.dim(0), syM.dim(0), false);

		for (unsigned i = 0; i < x.dim(0); ++i)
		{
			ArrayReference< double > yi(y[i]);
			recall(x[ i ], static_cast< Array< double >& >(yi));
		}
	}
	else
	{
		update();

		y.resize(syM.dim(0), false);

		for (unsigned i = 0; i < syM.dim(0); ++i)
		{
			double t = bvecM(i);

			for (unsigned j = 0; j < sxM.dim(0); ++j)
			{
				t += amatM(i, j) * x(j);
			}

			y(i) = t;
		}
	}
}


//===========================================================================
/*!
 *  \brief Temporarily removes the vector pair(s) "x"/"y" from the
 *         regression model and uses this modified model to approximate
 *         the vector(s) "y", when the vector(s) "x" is/are given.
 *
 *  This method can be used to check if some vector pairs \f$(x, y)\f$
 *  are necessary as information for the training of the regression model
 *  or not. <br>
 *  After the call of this method, the regression model is reset to the
 *  further state, where the \f$(x, y)\f$ pair(s) were included in the
 *  model.
 *
 *  \param x    The x-vector(s) that will be temporarily removed from the
 *              model.
 *  \param y    The corresponding y-vector(s) that will be temporarily
 *              removed from the model.
 *  \param yout The y-vector(s) for the given vector(s) \em x, approximated
 *              by the modified model.
 *  \return none
 *  \throw check_exception the type of the exception will be
 *         "size mismatch" and indicates that \em x and \em y
 *         are not one- and 2-dimensional or that one of both
 *         is one-dimensional and the other one 2-dimensional
 *         or that the number of vectors in \em x is different
 *         to that in \em y
 *
 *  \author  M. Kreutz
 *  \date    1995
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
void MixtureLinearRegression::leaveOneOut
(
	const Array< double >& x,
	const Array< double >& y,
	Array< double >&       yout
)
{
	SIZE_CHECK
	(
		(x.ndim() == 1 && y.ndim() == 1) ||
		(x.ndim() == 2 && y.ndim() == 2)
	)

	SIZE_CHECK(sxM.nelem() > 0 && countM > 0)

	if (x.ndim() == 2)
	{
		SIZE_CHECK(x.dim(0) == y.dim(0))

		yout.resize(y, false);

		for (unsigned i = 0; i < x.dim(0); ++i)
		{
			ArrayReference< double > youti(yout[ i ]);
			leaveOneOut(x[ i ], y[ i ], static_cast< Array< double >& >(youti));
		}
	}
	else
	{
		remove(x, y);
		recall(x, yout);
		train(x, y);
	}
}


//===========================================================================
/*!
 *  \brief Returns the regression coefficient "A" of the regression model.
 *
 *  Given the current regression model \f$y = A \cdot x + b\f$,
 *  this method returns \f$A\f$.
 *
 *  \return The regression coefficient \f$A\f$.
 *
 *  \author  M. Kreutz
 *  \date    1995
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
Array< double > MixtureLinearRegression::A()
{
	update();
	return amatM;
}


//===========================================================================
/*!
 *  \brief Returns the regression coefficient "b" of the regression model.
 *
 *  Given the current regression model \f$y = A \cdot x + b\f$,
 *  this method returns \f$b\f$.
 *
 *  \return The regression coefficient \f$b\f$.
 *
 *  \author  M. Kreutz
 *  \date    1995
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
Array< double > MixtureLinearRegression::b()
{
	update();
	return bvecM;
}

/*!
 *  \brief Given the correlations of the n-dimensional data vector "x"
 *         and the m-dimensional data vector "y" and also given
 *         their mean values, this function summarizes the data by
 *         finding a linear mapping that will approximate the data.
 *
 *  This function is used as subroutine for the class
 *  MixtureLinearRegression, where you can directly work with the
 *  data vectors \f$x\f$ and \f$y\f$ itself and the correlation
 *  and mean values are calculated for you. <br>
 *  Please refer to the reference of this class for more
 *  detailed information about linear regression.
 *
 *      \param  cxxMatA \f$n \times n\f$ matrix, that contains
 *                      the covariances between the single dimensions
 *                      of the vector \f$x\f$.
 *                      Only the lower triangle matrix must contain values.
 *      \param  cxyMatA \f$n \times m\f$ matrix, that contains
 *                      the covariances between the single values of
 *                      \f$x\f$ and \f$y\f$.
 *      \param	mxVecA  \f$n\f$-dimensional vector that contains the mean
 *                      values of vector \f$x\f$.
 *      \param	myVecA  \f$m\f$-dimensional vector that contains the mean
 *                      values of vector \f$y\f$.
 *      \param  amatA   \f$m \times n\f$ matrix, that will contain the
 *                      transposed transformation matrix \f$A^T\f$,
 *                      that is used for the linear mapping
 *                      \f$y = A \cdot x + b\f$ of the data.
 *      \param	bvecA   The \f$m\f$-dimensional vector \f$b\f$,
 *                      that is used for the linear mapping
 *                      \f$y = A \cdot x + b\f$ of the data.
 *      \param	dvecA   \f$n\f$-dimensional temporary vector, that
 *                      will contain the eigenvalues of matrix
 *                      \em cxxMatA in descending order.
 *      \return none
 *      \throw check_exception the type of the exception will be
 *             "size mismatch" and indicates that \em mxVecA or
 *             \em myVecA are not one-dimensional or that the dimensions
 *             of the matrices \em cxxMatA or \em cxyMatA don't
 *             correspond to the sizes of \em mxVecA and \em myVecA
 *             (the size of the first dimension of \em cxxMatA and
 *              \em cxyMatA and the size of the second dimension
 *              of \em cxxMatA must be the same than the size of
 *              \em mxVecA and the size of the second dimension
 *              of \em cxyMatA must be the same than the size
 *              of \em myVecA)
 *
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
 */
void MixtureLinearRegression::linearRegress
(
	Array2D< double >& cxxMatA,
	Array2D< double >& cxyMatA,
	Array  < double >& mxVecA,
	Array  < double >& myVecA,
	Array2D< double >& amatA,
	Array  < double >& bvecA,
	Array  < double >& dvecA
)
{
	SIZE_CHECK
	(
		mxVecA.ndim() == 1 &&
		myVecA.ndim() == 1 &&
		mxVecA.dim(0) == cxxMatA.dim(0) &&
		mxVecA.dim(0) == cxxMatA.dim(1) &&
		mxVecA.dim(0) == cxyMatA.dim(0) &&
		myVecA.dim(0) == cxyMatA.dim(1)
	)

	unsigned i, j, k;

	Array2D< double > vmatL(cxxMatA.dim(0), cxxMatA.dim(0));
	Array2D< double > hmatL(cxxMatA.dim(0), cxxMatA.dim(0));
	rankDecomp(cxxMatA, vmatL, hmatL, dvecA);

	amatA.resize(myVecA.dim(0), mxVecA.dim(0), false);
	bvecA.resize(myVecA, false);

	//eigenvalue decompistion of cxxMatA
	for (i = 0; i < cxxMatA.dim(0); ++i) {
		for (j = 0; j < cxxMatA.dim(1); ++j) {
			double t = 0.;

			for (k = 0; k < vmatL.dim(1); ++k) {
				t += vmatL(i, k) * vmatL(j, k);
			}

			cxxMatA(i, j) = t;
		}
	}

	//calculate amatA
	for (i = 0; i < amatA.dim(0); ++i) {
		for (j = 0; j < amatA.dim(1); ++j) {
			double t = 0.;

			for (k = 0; k < cxxMatA.dim(0); ++k) {
				t += cxyMatA(k, i) * cxxMatA(k, j);
			}

			amatA(i, j) = t;
		}
	}

       	//calculate b
	for (i = 0; i < bvecA.dim(0); ++i) {
		double t = 0.;

		for (j = 0; j < mxVecA.dim(0); ++j) {
			t += amatA(i, j) * mxVecA(j);
		}

		bvecA(i) = myVecA(i) - t;
	}
}

