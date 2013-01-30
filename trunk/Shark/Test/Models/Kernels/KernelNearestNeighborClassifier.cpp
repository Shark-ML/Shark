//===========================================================================
/*!
 *  \file KernelNearestNeighborClassifier.cpp
 *
 *  \brief unit test for kernel nearest neighbor classifier
 *
 *
 *  \author  T. Glasmachers
 *  \date    2010-2011
 *
 *  \par Copyright (c) 2010-2011:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *
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


#include <iostream>
#include <boost/numeric/ublas/io.hpp>
#define BOOST_TEST_MODULE MODELS_KERNEL_NEAREST_NEIGHBOR_CLASSIFIER
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Models/NearestNeighborClassifier.h>
#include <shark/Algorithms/NearestNeighbors/TreeNearestNeighbors.h>
#include <shark/Algorithms/NearestNeighbors/SimpleNearestNeighbors.h>

#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Models/Kernels/LinearKernel.h>

#include <shark/Models/Trees/KHCTree.h>
#include <shark/Models/Trees/KDTree.h>

#include <shark/Rng/GlobalRng.h>
#include <queue>

using namespace shark;

BOOST_AUTO_TEST_CASE( KERNEL_NEAREST_NEIGHBOR_CLASSIFIER ) {
	std::vector<RealVector> input(6, RealVector(2));
	input[0](0)=1;
	input[0](1)=3;
	input[1](0)=-1;
	input[1](1)=3;
	input[2](0)=1;
	input[2](1)=0;
	input[3](0)=-1;
	input[3](1)=0;
	input[4](0)=1;
	input[4](1)=-3;
	input[5](0)=-1;
	input[5](1)=-3;
	std::vector<unsigned int> target(6);
	target[0]=0;
	target[1]=1;
	target[2]=0;
	target[3]=1;
	target[4]=0;
	target[5]=1;

	ClassificationDataset dataset(input, target);
	DataView<Data<RealVector> > view(dataset.inputs());
	DenseRbfKernel kernel(0.5);

	KHCTree<DataView<Data<RealVector> > > tree(view,&kernel);
	TreeNearestNeighbors<RealVector,unsigned int> algorithm(dataset, &tree);
	NearestNeighborClassifier<RealVector> model(&algorithm, 3);

	for (size_t i = 0; i<6; ++i)
	{
		unsigned int label = model(input[i]);
		BOOST_CHECK_EQUAL(target[i], label);
	}

	//~ RealVector param;
//~ #ifdef DEBUG
	//~ BOOST_CHECK_THROW(model.setParameterVector(param), Exception);
//~ #endif
	//~ param.resize(1);
	//~ param(0) = 3.0;
	//~ BOOST_CHECK_NO_THROW(model.setParameterVector(param));
//~ #ifdef DEBUG
	//~ param(0) = 4.5;
	//~ BOOST_CHECK_THROW(model.setParameterVector(param), Exception);
	//~ param(0) = -1.0;
	//~ BOOST_CHECK_THROW(model.setParameterVector(param), Exception);
//~ #endif
}

BOOST_AUTO_TEST_CASE( SIMPLE_NEAREST_NEIGHBOR_CLASSIFIER ) {
	std::vector<RealVector> input(6, RealVector(2));
	input[0](0)=1;
	input[0](1)=3;
	input[1](0)=-1;
	input[1](1)=3;
	input[2](0)=1;
	input[2](1)=0;
	input[3](0)=-1;
	input[3](1)=0;
	input[4](0)=1;
	input[4](1)=-3;
	input[5](0)=-1;
	input[5](1)=-3;
	std::vector<unsigned int> target(6);
	target[0]=0;
	target[1]=1;
	target[2]=0;
	target[3]=1;
	target[4]=0;
	target[5]=1;

	ClassificationDataset dataset(input, target,2);

	DenseRbfKernel kernel(0.5);
	SimpleNearestNeighbors<RealVector,unsigned int> algorithm(dataset, &kernel);
	NearestNeighborClassifier<RealVector> model(&algorithm, 3);

	Data<unsigned int> labels = model(input);
	for (size_t i = 0; i<6; ++i)
	{
		unsigned int label = model(input[i]);
		BOOST_CHECK_EQUAL(target[i], label);
		BOOST_CHECK_EQUAL(target[i], labels(i));
	}

	//~ RealVector param;
//~ #ifdef DEBUG
	//~ BOOST_CHECK_THROW(model.setParameterVector(param), Exception);
//~ #endif
	//~ param.resize(1);
	//~ param(0) = 3.0;
	//~ BOOST_CHECK_NO_THROW(model.setParameterVector(param));
//~ #ifdef DEBUG
	//~ param(0) = 4.5;
	//~ BOOST_CHECK_THROW(model.setParameterVector(param), Exception);
	//~ param(0) = -1.0;
	//~ BOOST_CHECK_THROW(model.setParameterVector(param), Exception);
//~ #endif
}

BOOST_AUTO_TEST_CASE( SIMPLE_NEAREST_NEIGHBOR_CLASSIFIER_BRUTE_FORCE ) {
	std::size_t Dimension = 5;
	std::size_t Points = 100;
	std::size_t TestPoints = 100;
	std::vector<RealVector> input(Points, RealVector(Dimension));
	std::vector<unsigned int> target(Points);
	for(std::size_t i = 0; i != Points; ++i){
		target[i] = i%2;
		for(std::size_t d = 0; d != Dimension; ++d){
			input[i][d]=Rng::gauss(3*target[i],1);
		}
	}
	std::vector<RealVector> testInput(Points, RealVector(Dimension));
	std::vector<unsigned int> testTarget(Points);
	for(std::size_t i = 0; i != TestPoints; ++i){
		testTarget[i] = i%2;
		for(std::size_t d = 0; d != Dimension; ++d){
			testInput[i][d]=Rng::gauss(3*testTarget[i],1);
		}
	}

	ClassificationDataset dataset(input, target,10);
	ClassificationDataset testDataset(testInput, testTarget,10);
	
	//test using the brute force algorithm, whether the test points
	//are classified correctly in 1-NN
	{
		DenseRbfKernel kernel(0.5);
		SimpleNearestNeighbors<RealVector,unsigned int> algorithm(dataset, &kernel);
		NearestNeighborClassifier<RealVector> model(&algorithm, 1);

		Data<unsigned int> labels = model(dataset.inputs());
		for (size_t i = 0; i<Points; ++i)
		{
			unsigned int label = model(input[i]);
			BOOST_REQUIRE_EQUAL(target[i], label);
			BOOST_REQUIRE_EQUAL(target[i], labels(i));
		}
		
		Data<unsigned int> testLabels = model(testDataset.inputs());
		for (size_t i = 0; i<TestPoints; ++i)
		{
			unsigned int bruteforceLabel = 0;
			double minDistance = std::numeric_limits<double>::max();
			for(std::size_t j = 0; j != Points; ++j){
				double distance=distanceSqr(testInput[i],input[j]);
				if(distance < minDistance){
					minDistance = distance;
					bruteforceLabel=target[j];
				}
			}
			
			unsigned int label = model(testInput[i]);
			BOOST_CHECK_EQUAL(bruteforceLabel, label);
			BOOST_CHECK_EQUAL(bruteforceLabel, testLabels(i));
		}
	}
	
	//test using the brute force algorithm, whether the test points
	//are classified correctly in 3-NN
	{
		DenseRbfKernel kernel(0.5);
		SimpleNearestNeighbors<RealVector,unsigned int> algorithm(dataset, &kernel);
		NearestNeighborClassifier<RealVector> model(&algorithm, 3);
		
		Data<unsigned int> testLabels = model(testDataset.inputs());
		for (size_t i = 0; i<TestPoints; ++i)
		{
			typedef KeyValuePair<double,unsigned int> Pair;
			std::priority_queue<Pair,std::vector<Pair>,std::greater<Pair> > queue;
			for(std::size_t j = 0; j != Points; ++j){
				double distance=norm_2(testInput[i]-input[j]);
				queue.push(Pair(distance,target[j]));
			}
			
			unsigned int res = 0;
			for(std::size_t j = 0; j != 3; ++j){
				res+=queue.top().value;
				queue.pop();
			}
			unsigned int bruteforceLabel = res > 1;
			unsigned int label = model(testInput[i]);
			BOOST_CHECK_EQUAL(bruteforceLabel, label);
			BOOST_CHECK_EQUAL(bruteforceLabel, testLabels(i));
		}
	
	}
}

BOOST_AUTO_TEST_CASE( NEAREST_NEIGHBOR_CLASSIFIER_KDTREE_BRUTE_FORCE ) {
	std::size_t Dimension = 5;
	std::size_t Points = 100;
	std::size_t TestPoints = 100;
	std::vector<RealVector> input(Points, RealVector(Dimension));
	std::vector<unsigned int> target(Points);
	for(std::size_t i = 0; i != Points; ++i){
		target[i] = i%2;
		for(std::size_t d = 0; d != Dimension; ++d){
			input[i][d]=Rng::gauss(3*target[i],1);
		}
	}
	std::vector<RealVector> testInput(Points, RealVector(Dimension));
	std::vector<unsigned int> testTarget(Points);
	for(std::size_t i = 0; i != TestPoints; ++i){
		testTarget[i] = i%2;
		for(std::size_t d = 0; d != Dimension; ++d){
			testInput[i][d]=Rng::gauss(3*testTarget[i],1);
		}
	}

	ClassificationDataset dataset(input, target,10);
	ClassificationDataset testDataset(testInput, testTarget,10);
	
	{
		KDTree<RealVector> tree(dataset.inputs());
		TreeNearestNeighbors<RealVector,unsigned int> algorithm(dataset, &tree);
		NearestNeighborClassifier<RealVector> model(&algorithm, 1);

		//test whether 1-NN classifies it's training set correctly.
		Data<unsigned int> labels = model(dataset.inputs());
		for (size_t i = 0; i<Points; ++i)
		{
			unsigned int label = model(input[i]);
			BOOST_REQUIRE_EQUAL(target[i], label);
			BOOST_REQUIRE_EQUAL(target[i], labels(i));
		}
		
		//test using the brute force algorithm, whether the test points
		//are classified correctly in 1-NN
		Data<unsigned int> testLabels = model(testDataset.inputs());
		for (size_t i = 0; i<TestPoints; ++i)
		{
			unsigned int bruteforceLabel = 0;
			double minDistance = std::numeric_limits<double>::max();
			for(std::size_t j = 0; j != Points; ++j){
				double distance=distanceSqr(testInput[i],input[j]);
				if(distance < minDistance){
					minDistance = distance;
					bruteforceLabel=target[j];
				}
			}
			
			unsigned int label = model(testInput[i]);
			BOOST_CHECK_EQUAL(bruteforceLabel, label);
			BOOST_CHECK_EQUAL(bruteforceLabel, testLabels(i));
		}
	}
	
	//test using the brute force algorithm, whether the test points
	//are classified correctly in 3-NN
	{
		KDTree<RealVector> tree(dataset.inputs());
		TreeNearestNeighbors<RealVector,unsigned int> algorithm(dataset, &tree);
		NearestNeighborClassifier<RealVector> model(&algorithm, 3);
		
		Data<unsigned int> testLabels = model(testDataset.inputs());
		for (size_t i = 0; i<TestPoints; ++i)
		{
			typedef KeyValuePair<double,unsigned int> Pair;
			std::priority_queue<Pair,std::vector<Pair>,std::greater<Pair> > queue;
			for(std::size_t j = 0; j != Points; ++j){
				double distance=norm_2(testInput[i]-input[j]);
				queue.push(Pair(distance,target[j]));
			}
			
			unsigned int res = 0;
			for(std::size_t j = 0; j != 3; ++j){
				res+=queue.top().value;
				queue.pop();
			}
			unsigned int bruteforceLabel = res > 1;
			unsigned int label = model(testInput[i]);
			BOOST_CHECK_EQUAL(bruteforceLabel, label);
			BOOST_CHECK_EQUAL(bruteforceLabel, testLabels(i));
		}
	}
}

