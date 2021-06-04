//===========================================================================
/*!
 * 
 *
 * \brief       Linear Regression Tutorial Sample Code
 * 
 * This file is part of the "Linear Regression" tutorial.
 * It requires some toy sample data that comes with the library.
 * 
 * 
 *
 * \author      C. Igel
 * \date        2011
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://shark-ml.org/>
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

#include <shark/Data/Csv.h>
#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h>
#include <shark/Algorithms/Trainers/LinearRegression.h>

#include <iostream>

using namespace shark;
using namespace std;

int main(int argc, char **argv) {
	if(argc < 3) {
		cerr << "usage: " << argv[0] << " (file with inputs/independent variables) (file with outputs/dependent variables)" << endl;
		exit(EXIT_FAILURE);
	}
	Data<RealVector> inputs;
	Data<RealVector> labels;
	try {
		importCSV(inputs, argv[1], ' ');
	} 
	catch (...) {
		cerr << "unable to read input data from file " <<  argv[1] << endl;
		exit(EXIT_FAILURE);
	}

	try {
		importCSV(labels, argv[2]);
	}
	catch (...) {
		cerr << "unable to read labels from file " <<  argv[2] << endl;
		exit(EXIT_FAILURE);
	}

	RegressionDataset data(inputs, labels);



	// trainer and model
	LinearRegression trainer;
	LinearModel<> model;

	// train model
	trainer.train(model, data);

	// show model parameters
	cout << "intercept: " << model.offset() << endl;
	cout << "matrix: " << model.matrix() << endl;

	SquaredLoss<> loss;
	Data<RealVector> prediction = model(data.inputs()); 
	cout << "squared loss: " << loss(data.labels(), prediction) << endl;
}
