//===========================================================================
/*!
 * 
 *
 * \brief       Linear Discriminant Analysis Tutorial Sample Code
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
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>

#include <shark/Algorithms/Trainers/LDA.h>

#include <iostream>

using namespace shark;
using namespace std;

int main(int argc, char **argv) {
	if(argc < 2) {
		cerr << "usage: " << argv[0] << " (filename)" << endl;
		exit(EXIT_FAILURE);
	}
	// read data
	ClassificationDataset data;
	try {
		importCSV(data, argv[1], LAST_COLUMN, ' ');
	} 
	catch (...) {
		cerr << "unable to read data from file " <<  argv[1] << endl;
		exit(EXIT_FAILURE);
	}

	cout << "overall number of data points: " << data.numberOfElements() << " "
	     << "number of classes: " << numberOfClasses(data) << " "
	     << "input dimension: " << inputDimension(data) << endl;

	// split data into training and test set
	ClassificationDataset dataTest = splitAtElement(data, .5 * data.numberOfElements() );
	cout << "training data points: " << data.numberOfElements() << endl;
	cout << "test data points: " << dataTest.numberOfElements() << endl;

	// define learning algorithm
	LDA ldaTrainer;

	// define linear model
	LinearClassifier<> lda;

	// train model
	ldaTrainer.train(lda, data);

	// evaluate classifier
	Data<unsigned int> prediction;
	ZeroOneLoss<unsigned int> loss;

	prediction = lda(data.inputs());
	cout << "LDA on training set accuracy: " << 1. - loss(data.labels(), prediction) << endl;
	prediction = lda(dataTest.inputs());
	cout << "LDA on test set accuracy:     " << 1. - loss(dataTest.labels(), prediction) << endl;
}
