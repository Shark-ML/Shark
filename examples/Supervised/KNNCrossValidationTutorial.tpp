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
 * \par Copyright 1995-2014 Shark Development Team
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

#include <shark/Data/Csv.h>
#include <shark/Models/NearestNeighborClassifier.h>
#include <shark/Algorithms/NearestNeighbors/SimpleNearestNeighbors.h>
#include <shark/Models/Kernels/LinearKernel.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
#include <shark/Data/CVDatasetTools.h>
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
	ClassificationDataset dataTest = splitAtElement(data, .5 * data.numberOfElements());
	cout << "training data points: " << data.numberOfElements() << endl;
	cout << "test data points: " << dataTest.numberOfElements() << endl;

	//create 10 CV-Folds
	const unsigned int NFolds= 10;
	CVFolds<ClassificationDataset> folds = createCVSameSizeBalanced(data, NFolds);

	//we have 5 different values of k to test
	unsigned int k[]={1,3,5,7,9};
	unsigned int numParameters = 5;

	ZeroOneLoss<unsigned int> loss;//loss for evaluation
	LinearKernel<> metric;//linear distance measure

	//find best #-neighbors using CV
	unsigned int best_k = 0;
	double best_error = 2;//maximum 0-1loss is 1
	//for every parameter....
	for(std::size_t p = 0; p != numParameters; ++p){
		double error = 0;
		//calculate CV-error
		for(std::size_t i = 0; i != NFolds; ++i){
			SimpleNearestNeighbors<RealVector, unsigned int> algorithm(folds.training(i), &metric);
			NearestNeighborClassifier<RealVector> KNN(&algorithm, k[p]);
			error += loss(folds.validation(i).labels(),KNN(folds.validation(i).inputs()));
		}
		error /=NFolds;
		//print cv-error for current parameter
		std::cout<<k[p]<<" "<<error<<std::endl;
		//if the error is better, we keep it.
		if(error < best_error){
			best_k = k[p];
			best_error = error;
		}
	}
	//evaluate the best paramter found on test set using the full training set
	SimpleNearestNeighbors<RealVector, unsigned int> algorithm(data, &metric);
	NearestNeighborClassifier<RealVector> KNN(&algorithm, best_k);
	std::cout<<"NearestNeighbors: " << loss(dataTest.labels(),KNN(dataTest.inputs()))<<'\n';
	std::cout<<"K: "<<best_k<<std::endl; 
}
