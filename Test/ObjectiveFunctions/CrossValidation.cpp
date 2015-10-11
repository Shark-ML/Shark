//===========================================================================
/*!
 * 
 *
 * \brief       Cross validation test case
 * 
 * 
 * 
 *
 * \author      T. Glasmachers
 * \date        2011
 *
 *
 * \par Copyright 1995-2015 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://image.diku.dk/shark/>
 * 
 * Shark is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Shark is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with Shark.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
//===========================================================================

#include <shark/Models/LinearModel.h>
#include <shark/Algorithms/Trainers/LinearRegression.h>
#include <shark/ObjectiveFunctions/CrossValidationError.h>
#include <shark/ObjectiveFunctions/Loss/AbsoluteLoss.h>
#include <shark/Data/Dataset.h>

#define BOOST_TEST_MODULE ObjectiveFunctions_CrossValidation
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
using namespace shark;


// This test case judges a linear regression of data
// *not* on a line by means of cross-validation. In
// this very simple case the error can be computed
// analytically, and thus can serve validation.
BOOST_AUTO_TEST_SUITE (ObjectiveFunctions_CrossValidation)

BOOST_AUTO_TEST_CASE( ObjectiveFunctions_CrossValidation )
{
	// create a simple dataset with three point,
	// which also form three folds
	std::vector<RealVector> data(3);
	std::vector<RealVector> target(3);
	size_t i;
	for (i=0; i<3; i++)
	{
		data[i].resize(1);
		target[i].resize(1);
	}
	data[0](0) = 0.0; target[0](0) = 0.0;	// regression of this point from the others gives 2 --> loss = 2
	data[1](0) = 1.0; target[1](0) = 1.0;	// regression of this point from the others gives 0 --> loss = 1
	data[2](0) = 2.0; target[2](0) = 0.0;	// regression of this point from the others gives 2 --> loss = 2
	RegressionDataset dataset = createLabeledDataFromRange(data, target);
	CVFolds<RegressionDataset> folds = createCVSameSize(dataset, 3);	// there is only one way to form three folds out of three points

	// setup (regularized) linear regression for cross-validation
	LinearModel<> lin(1, 1, true);		// affine linear function
	LinearRegression trainer;
	AbsoluteLoss<> loss;
	CrossValidationError<LinearModel<> > cvError(
		folds,
		&trainer,
		&lin,
		&trainer,
		&loss
	);

	// set the regularization to zero and evaluation this setting
	RealVector param(1); param(0) = 0.0;
	double cve = cvError.eval(param);
	std::cout << "***  cross-validation test case  ***" << std::endl;
	std::cout << "CV-error: " << cve << "    (should be 5/3 = 1.66666...)" << std::endl;

	BOOST_CHECK_LT(std::abs(cve - 5.0 / 3.0), 1e-12);
}

BOOST_AUTO_TEST_SUITE_END()
