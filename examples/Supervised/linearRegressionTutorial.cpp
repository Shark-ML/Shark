//===========================================================================
/*!
 *  \brief Linear Regression Tutorial Sample Code
 *
 *  This file is part of the "Linear Regression" tutorial.
 *  It requires some toy sample data that comes with the library.
 *
 *  \author C. Igel
 *  \date 2011
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

#include <shark/Data/Csv.h>
#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h>
#include <shark/Algorithms/Trainers/LinearRegression.h>

#include <iostream>

using namespace shark;
using namespace std;

int main() {
	Data<RealVector> inputs;
	Data<RealVector> labels;
	import_csv(inputs, "data/regressionInputs.csv", " ");
	import_csv(labels, "data/regressionLabels.csv", " ");
	RegressionDataset data(inputs, labels);

	// trainer and model
	LinearRegression trainer;
	LinearModel<> model;

	// train model
	trainer.train(model, data);

	// show model parameters
	cout << "intercept: " << model.offset() << endl;
	cout << "matrix: " << model.matrix() << endl;

	Data<RealVector> prediction = model(data.inputs()); 
	SquaredLoss<> loss;
	cout << "squared loss: " << loss(data.labels(), prediction) << endl;
}
