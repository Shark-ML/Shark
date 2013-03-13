//===========================================================================
/*!
 *  \brief LASSO Regression
 *
 *  This program is the LASSO counter part of the linear regression
 *  tutorial example program.
 *
 *  \author T. Glasmachers
 *  \date 2013
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
//===========================================================================

#include <shark/Data/DataDistribution.h>
#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h>
#include <shark/Algorithms/Trainers/LassoRegression.h>

#include <iostream>

using namespace shark;
using namespace std;

int main() {
	Wave prob;
	RegressionDataset data = prob.generateDataset(200);

	// regularization parameter
	double lambda = 1.0;

	// trainer and model
	LassoRegression<> trainer(lambda);
	LinearModel<> model;

	// train model
	trainer.train(model, data);

	// find non-zero coefficients
	RealMatrix m = model.matrix();
	// TODO...
}
