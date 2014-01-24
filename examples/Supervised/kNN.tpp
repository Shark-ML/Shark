//===========================================================================
/*!
 * 
 * \file        kNN.tpp
 *
 * \brief       K-nearest-neighbor (k-NN) example program.
 * 
 * 
 *
 * \author      T. Glasmachers
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

#include <shark/Data/DataDistribution.h>
#include <shark/Data/DataView.h>
#include <shark/Models/NearestNeighborClassifier.h>
#include <shark/Algorithms/NearestNeighbors/TreeNearestNeighbors.h>
#include <shark/Algorithms/NearestNeighbors/SimpleNearestNeighbors.h>
#include <shark/Models/Trees/KDTree.h>
#include <shark/Models/Trees/KHCTree.h>
#include <shark/Models/Kernels/MonomialKernel.h>
#include <shark/Models/Kernels/LinearKernel.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>


using namespace shark;


// data set sizes
#define TRAINING 10000
#define TEST 10


///
/// \brief Example program for nearest neighbor classification.
///
/// \par
/// This program demonstrates how to use
/// nearest neighbor classifiers with
/// Euclidean distance or kernel distance
/// for arbitrary number of neighbors and different algorithms
///
int main(int argc, char** argv)
{
	ZeroOneLoss<unsigned int> loss;
	double testerror;

	// read number of neighbors from the command line
	unsigned int k = 1;
	if (argc > 1) k = atoi(argv[1]);
	std::cout << k << "-nearest-neighbor classifier" << std::endl;

	// generate a data set for binary classification as well as a view of the dataset suitable
	// for tree generation
	std::cout << "generating problem ..." << std::flush;
	Chessboard chess;
	ClassificationDataset training = chess.generateDataset(TRAINING);
	ClassificationDataset test = chess.generateDataset(TEST);
	std::cout << " done." << std::endl;

	// setup a k-nearest-neighbor classifier based on Euclidean distance
	std::cout << "preparing classifier based on Euclidean distance..." << std::flush;
	KDTree<RealVector> kdtree(training.inputs());
	TreeNearestNeighbors<RealVector,unsigned int> algorithmKD(training,&kdtree);
	NearestNeighborClassifier<RealVector> knn1(&algorithmKD, k);
	std::cout << " done." << std::endl;

	// evaluate the classifier on the test set
	std::cout << "computing predictions ..." << std::flush;
	testerror = loss.eval(test.labels(), knn1(test.inputs()));
	std::cout << " done." << std::endl;

	// output the error rate
	std::cout << "test error rate: " << 100.0 * testerror << " %" << std::endl;

	
	
	//now the same setup with euclidean distance
	std::cout << "preparing classifier based on polynomial kernel distance ..." << std::flush;
	MonomialKernel<RealVector> kernel(3);
	DataView<Data<RealVector> > trainingInputs(training.inputs());
	KHCTree<DataView<Data<RealVector> > > khctree(trainingInputs, &kernel);
	TreeNearestNeighbors<RealVector,unsigned int> algorithmKHC(training,&khctree);
	NearestNeighborClassifier<RealVector> knn2(&algorithmKHC, k);
	std::cout << " done." << std::endl;

	// evaluate the classifier on the test set
	std::cout << "computing predictions ..." << std::flush;
	testerror = loss.eval(test.labels(), knn2(test.inputs()));
	std::cout << " done." << std::endl;

	// output the error rate
	std::cout << "test error rate: " << 100.0 * testerror << " %" << std::endl;
	
	
	//both setups can also be computed using a Brute force nearest neighbor.
	//this is likely to be faster on big datasets with high dimensionality.
	std::cout << "preparing brute force classifier based on Euclidean distance..." << std::flush;
	LinearKernel<RealVector> euclideanKernel;
	SimpleNearestNeighbors<RealVector,unsigned int> simpleAlgorithm(training,&euclideanKernel);
	NearestNeighborClassifier<RealVector> knn3(&simpleAlgorithm, k);
	std::cout << " done." << std::endl;

	// evaluate the classifier on the test set
	std::cout << "computing predictions ..." << std::flush;
	testerror = loss.eval(test.labels(), knn3(test.inputs()));
	std::cout << " done." << std::endl;

	// output the error rate
	std::cout << "test error rate: " << 100.0 * testerror << " %" << std::endl;
}
