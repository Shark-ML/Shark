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

//###begin<csv_include>
#include <shark/Data/Csv.h>
//###end<csv_include>
//###begin<loss_include>
#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h>
//###end<loss_include>
//###begin<regression_include>
#include <shark/Algorithms/Trainers/LinearRegression.h>
//###end<regression_include>

#include <iostream>

//###begin<namespaces>
using namespace shark;
using namespace std;
//###end<namespaces>

//###begin<load_data>
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
//###end<load_data>



	// trainer and model
//###begin<trainer>
	LinearRegression trainer;
//###end<trainer>
//###begin<model>
	LinearModel<> model;
//###end<model>

	// train model
//###begin<train>
	trainer.train(model, data);
//###end<train>

	// show model parameters
//###begin<inspect>	
	cout << "intercept: " << model.offset() << endl;
	cout << "matrix: " << model.matrix() << endl;
//###end<inspect>	

//###begin<loss>
	SquaredLoss<> loss;
//###end<loss>
//###begin<apply>
	Data<RealVector> prediction = model(data.inputs()); 
//###end<apply>	
	cout << "squared loss: " << loss(data.labels(), prediction) << endl;
}
