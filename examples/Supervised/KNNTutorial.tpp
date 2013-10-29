//===========================================================================
/*!
 *  \brief Nearest Neighbor Tutorial Sample Code
 *
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

//###begin<dataheader>
#include <shark/Data/Csv.h>
//###end<dataheader>
//###begin<nnheader>
#include <shark/Models/NearestNeighborClassifier.h>
#include <shark/Algorithms/NearestNeighbors/TreeNearestNeighbors.h>
#include <shark/Models/Trees/KDTree.h>
//###end<nnheader>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
#include <shark/Data/DataView.h>
#include <iostream>

//###begin<dataheader>
using namespace shark;
using namespace std;
//###end<dataheader>

//###begin<dataimport>
int main(int argc, char **argv) {
	if(argc < 2) {
		cerr << "usage: " << argv[0] << " (filename)" << endl;
		exit(EXIT_FAILURE);
	}
	// read data
	ClassificationDataset data;
	try {
		import_csv(data, argv[1], LAST_COLUMN, ' ');
	} 
	catch (...) {
		cerr << "unable to read data from file " <<  argv[1] << endl;
		exit(EXIT_FAILURE);
	}
//###end<dataimport>

//###begin<inspectdata>
	cout << "number of data points: " << data.numberOfElements()
	     << " number of classes: " << numberOfClasses(data)
	     << " input dimension: " << inputDimension(data) << endl;
//###end<inspectdata>

	// split data into training and test set
//###begin<splitdata>
	ClassificationDataset dataTest = splitAtElement(data, static_cast<std::size_t>(.5 * data.numberOfElements()));
	cout << "training data points: " << data.numberOfElements() << endl;
	cout << "test data points: " << dataTest.numberOfElements() << endl;
//###end<splitdata>

	//create a binary search tree and initialize the search algorithm - a fast tree search
//###begin<kdtree>
	KDTree<RealVector> tree(data.inputs());
//###end<kdtree>
//###begin<treeNN>
	TreeNearestNeighbors<RealVector,unsigned int> algorithm(data,&tree);
//###end<treeNN>
	//instantiate the classifier
//###begin<NNC>
	const unsigned int K = 1; // number of neighbors for kNN
	NearestNeighborClassifier<RealVector> KNN(&algorithm,K);
//###end<NNC>

	// evaluate classifier
//###begin<eval>
	ZeroOneLoss<unsigned int> loss;
	Data<unsigned int> prediction = KNN(data.inputs());
	cout << K << "-KNN on training set accuracy: " << 1. - loss.eval(data.labels(), prediction) << endl;
	prediction = KNN(dataTest.inputs());
	cout << K << "-KNN on test set accuracy:     " << 1. - loss.eval(dataTest.labels(), prediction) << endl;
//###end<eval>
}
