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

int main() {
	const unsigned int K = 1; // number of neighbors for kNN

	// read data
//###begin<dataimport>
	ClassificationDataset data;
	import_csv(data, "data/C.csv", LAST_COLUMN, " ", "#");
//###end<dataimport>

//###begin<inspectdata>
	cout << "number of data points: " << data.numberOfElements()
	     << " number of classes: " << numberOfClasses(data)
	     << " input dimension: " << inputDimension(data) << endl;
//###end<inspectdata>

	// split data into training and test set
//###begin<splitdata>
	ClassificationDataset dataTest = splitAtElement(data,311);
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
	NearestNeighborClassifier<RealVector> KNN(&algorithm,K);
//###end<NNC>

	// evaluate classifier
//###begin<eval>
	ZeroOneLoss<unsigned int> loss;
	Data<unsigned int> prediction = KNN(data.inputs());
	cout << K << "-KNN on training set accuracy: " << 1. - loss.eval(data.labels(), prediction) << endl;
	prediction = KNN(dataTest.inputs());
	cout << K << "-KNN on test set accuracy:     " << 1. - loss.eval(data.labels(), prediction) << endl;
//###end<eval>
}
