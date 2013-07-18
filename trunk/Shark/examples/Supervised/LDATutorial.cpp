//===========================================================================
/*!
 *  \brief Linear Discriminant Analysis Tutorial Sample Code
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
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
#include <shark/Algorithms/Trainers/LDA.h>

#include <iostream>

using namespace shark;
using namespace std;

//###begin<load_data>
int main(int argc, char **argv) {
	if(argc < 2) {
		cerr << "usage: " << argv[0] << " (filename)" << endl;
		exit(EXIT_FAILURE);
	}
	// read data
	ClassificationDataset data;
	try {
		import_csv(data, argv[1], LAST_COLUMN, " ", "#");
	} 
	catch (...) {
		cerr << "unable to read data from file " <<  argv[1] << endl;
		exit(EXIT_FAILURE);
	}

	cout << "number of data points: " << data.numberOfElements() << " "
	     << "number of classes: " << numberOfClasses(data) << " "
	     << "input dimension: " << inputDimension(data) << endl;

	// split data into training and test set
	ClassificationDataset dataTest = splitAtElement(data,311);
///###end<load_data>

	// define learning algorithm
	LDA ldaTrainer;

	// define linear model
	LinearClassifier lda;

	// train model
	ldaTrainer.train(lda, data);

	// evaluate classifier
	Data<unsigned int> prediction;
	ZeroOneLoss<unsigned int> loss;
	prediction = lda(data.inputs());
	cout << "LDA on training set accuracy: " << 1. - loss(data.labels(), prediction) << endl;
	prediction = lda(dataTest.inputs());
	cout << "LDA on test set accuracy:     " << 1. - loss(data.labels(), prediction) << endl;
}
