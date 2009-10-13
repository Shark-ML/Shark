//===========================================================================
/*!
 *  \file MixtureLinearRegression.h
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
 *      <BR>
 *
 *  \par Project:
 *      Mixture
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


#ifndef MixtureLinearRegression_H
#define MixtureLinearRegression_H


#include <Array/Array2D.h>


//! This class should be replaced by the ReClaM version
//! in the future.
class MixtureLinearRegression
{
public:

	//! Creates a new empty Linear Regression object and resets
	//! internal values.
	MixtureLinearRegression();

	//! Destructs a Linear Regression object.
	~MixtureLinearRegression();

	//! Resets internal variables of the class to initial values.
	void reset();

	//! Calculates the current regression model.
	void update();

	//! Adds the data vector pair(s) "x"/"y" to the regression model.
	void train
	(
		const Array< double >&,
		const Array< double >&
	);

	//! Removes the data vector pair(s) "x"/"y" from the regression model.
	void remove
		(
			const Array< double >&,
			const Array< double >&
		);

	//! Given one/several data vector(s) "x" the corresponding
	//! vector(s) "y" is/are calculated by using the current regression model.
	void recall
	(
		const Array< double >&,
		Array< double >&
	);

	//! Temporarily removes the vector pair(s) "x"/"y" from the
	//! regression model and uses this modified model to approximate
	//! the vector(s) "y", when the vector(s) "x" is/are given.
	void leaveOneOut
	(
		const Array< double >&,
		const Array< double >&,
		Array< double >&
	);

	//! Returns the regression coefficient "A" of the regression model.
	Array< double > A();

	//! Returns the regression coefficient "b" of the regression model.
	Array< double > b();


#ifdef _WIN32
	// dummy !!!
	bool operator == (const MixtureLinearRegression&) const
	{
		return false;
	}

	// dummy !!!
	bool operator < (const MixtureLinearRegression&) const
	{
		return false;
	}
#endif


protected:

	// Set to "true" if the internal variables sxM, syM, rxxM and rxyM
	// are changed due to adding/removing vector pair(s) x/y.
	bool modifiedM;

	// The no. of data vector pairs x/y used for training of the model.
	unsigned countM;

	// Given k data vectors x with dimension n and the corresponding
	// k vectors y with dimension m, this n-dimensional vector
	// contains the sum of all x_l for l = 1,...,k.
	Array  < double > sxM;

	// Given k data vectors x with dimension n and the corresponding
	// k vectors y with dimension m, this n-dimensional vector
	// contains the sum of all y_l for l = 1,...,k.
	Array  < double > syM;

	// Given k data vectors x with dimension n and the corresponding
	// k vectors y with dimension m, this n x n matrix contains the
	// sum of all s_l for l = 1,...,k with s_l = x_i * x_j
	// for i,j = 1,...,n.
	Array2D< double > rxxM;

	// Given k data vectors x with dimension n and the corresponding
	// k vectors y with dimension m, this n x m matrix contains the
	// sum of all s_l for l = 1,...,k with s_l = x_i * y_j
	// for i = 1,...,n and j = 1,...,m.
	Array2D< double > rxyM;

	// The transformation matrix "A" of the model y = Ax + b.
	Array2D< double > amatM;

	// The vector "b" of the model y = Ax + b.
	Array  < double > bvecM;

	// core method
	void linearRegress
	(
		Array2D< double >& cxxMatA,
		Array2D< double >& cxyMatA,
		Array  < double >& mxVecA,
		Array  < double >& myVecA,
		Array2D< double >& amatA,
		Array  < double >& bvecA,
		Array  < double >& dvecA
	);

	friend class LocalLinearRegression;
};


#endif

