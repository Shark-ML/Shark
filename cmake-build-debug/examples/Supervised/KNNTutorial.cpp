//===========================================================================
/*!
 * 
 *
 * \brief       Nearest Neighbor Tutorial Sample Code
 * 
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
#include <shark/Models/NearestNeighborModel.h>
#include <shark/Algorithms/NearestNeighbors/TreeNearestNeighbors.h>
#include <shark/Models/Trees/KDTree.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
#include <shark/Data/DataView.h>
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

	cout << "number of data points: " << data.numberOfElements()
	     << " number of classes: " << numberOfClasses(data)
	     << " input dimension: " << inputDimension(data) << endl;

	// split data into training and test set
	ClassificationDataset dataTest = splitAtElement(data, static_cast<std::size_t>(.5 * data.numberOfElements()));
	cout << "training data points: " << data.numberOfElements() << endl;
	cout << "test data points: " << dataTest.numberOfElements() << endl;

	//create a binary search tree and initialize the search algorithm - a fast tree search
	KDTree<RealVector> tree(data.inputs());
	TreeNearestNeighbors<RealVector,unsigned int> algorithm(data,&tree);
	//instantiate the classifier
	const unsigned int K = 1; // number of neighbors for kNN
	NearestNeighborModel<RealVector, unsigned int> KNN(&algorithm,K);

	// evaluate classifier
	ZeroOneLoss<unsigned int> loss;
	Data<unsigned int> prediction = KNN(data.inputs());
	cout << K << "-KNN on training set accuracy: " << 1. - loss.eval(data.labels(), prediction) << endl;
	prediction = KNN(dataTest.inputs());
	cout << K << "-KNN on test set accuracy:     " << 1. - loss.eval(dataTest.labels(), prediction) << endl;
}
