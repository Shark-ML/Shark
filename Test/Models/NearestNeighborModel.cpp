//===========================================================================
/*!
 * 
 *
 * \brief       unit test for soft nearest neighbor classifier
 * 
 * 
 * 
 *
 * \author      T. Glasmachers
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

#define BOOST_TEST_MODULE MODELS_NEAREST_NEIGHBOR
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Models/NearestNeighborModel.h>
#include <shark/Algorithms/NearestNeighbors/TreeNearestNeighbors.h>
#include <shark/Algorithms/NearestNeighbors/SimpleNearestNeighbors.h>

#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Models/Kernels/LinearKernel.h>

#include <shark/Models/Trees/KHCTree.h>
#include <shark/Models/Trees/KDTree.h>

#include <shark/Core/Random.h>
#include <queue>
using namespace shark;

BOOST_AUTO_TEST_SUITE (Models_NearestNeighbor)

BOOST_AUTO_TEST_CASE( Models_NearestNeighbor_Regression ) {

	// simple data set with paired points
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
	std::vector<RealVector> target(6, RealVector(1));
	target[0](0)=-5.0;
	target[1](0)=-3.0;
	target[2](0)=-1.0;
	target[3](0)=+1.0;
	target[4](0)=+3.0;
	target[5](0)=+5.0;
	RegressionDataset dataset = createLabeledDataFromRange(input, target);
	dataset.inputShape()={1,2};
	// model
	DenseLinearKernel kernel;
	SimpleNearestNeighbors<RealVector,RealVector> algorithm(dataset, &kernel);
	NearestNeighborModel<RealVector, RealVector> model(&algorithm, 2);
	BOOST_CHECK_EQUAL(dataset.inputShape(),model.inputShape());
	BOOST_CHECK_EQUAL(dataset.labelShape(),model.outputShape());

	// predictions must be pair averages
	Data<RealVector> prediction = model(dataset.inputs());
	for (int i = 0; i<6; ++i)
	{
		BOOST_CHECK_SMALL(prediction.elements()[i](0) - 4.0 * (i/2 - 1), 1e-14);
	}
}

BOOST_AUTO_TEST_CASE( Models_NearestNeighbor_Classification_Simple ) {
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
	target[1]=0;
	target[2]=1;
	target[3]=1;
	target[4]=2;
	target[5]=2;

	ClassificationDataset dataset = createLabeledDataFromRange(input, target);
	dataset.inputShape()={1,2};
	
	DenseRbfKernel kernel(0.5);
	SimpleNearestNeighbors<RealVector,unsigned int> algorithm(dataset, &kernel);
	NearestNeighborModel<RealVector, unsigned int> model(&algorithm, 3);
	BOOST_CHECK_EQUAL(dataset.inputShape(),model.inputShape());
	BOOST_CHECK_EQUAL(Shape({3}),model.outputShape());

	Data<unsigned int> prediction=model(dataset.inputs());
	Data<RealVector> soft_prediction=model.decisionFunction()(dataset.inputs());
	for (size_t i = 0; i<6; ++i)
	{
		BOOST_CHECK_EQUAL(prediction.elements()[i],target[i]);
		BOOST_CHECK_CLOSE(soft_prediction.elements()[i](target[i]), 2.0/3.0, 1e-12);
	}
}

BOOST_AUTO_TEST_CASE( Models_NearestNeighbor_Classification_KHCTree ) {
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

	ClassificationDataset dataset = createLabeledDataFromRange(input, target);
	dataset.inputShape()={1,2};
	
	DataView<Data<RealVector> > view(dataset.inputs());
	DenseRbfKernel kernel(0.5);

	KHCTree<DataView<Data<RealVector> > > tree(view,&kernel);
	TreeNearestNeighbors<RealVector,unsigned int> algorithm(dataset, &tree);
	NearestNeighborModel<RealVector, unsigned int> model(&algorithm, 3);
	BOOST_CHECK_EQUAL(dataset.inputShape(),model.inputShape());
	BOOST_CHECK_EQUAL(Shape({2}),model.outputShape());

	for (size_t i = 0; i<6; ++i)
	{
		unsigned int label = model(input[i]);
		BOOST_CHECK_EQUAL(target[i], label);
	}
}


BOOST_AUTO_TEST_CASE( Models_NearestNeighbor_Classification_Simple_Brute_Force ) {
	std::size_t Dimension = 5;
	std::size_t Points = 100;
	std::size_t TestPoints = 100;
	std::vector<RealVector> input(Points, RealVector(Dimension));
	std::vector<unsigned int> target(Points);
	for(std::size_t i = 0; i != Points; ++i){
		target[i] = i%2;
		for(std::size_t d = 0; d != Dimension; ++d){
			input[i][d]=random::gauss(random::globalRng,3*target[i],1);
		}
	}
	std::vector<RealVector> testInput(Points, RealVector(Dimension));
	std::vector<unsigned int> testTarget(Points);
	for(std::size_t i = 0; i != TestPoints; ++i){
		testTarget[i] = i%2;
		for(std::size_t d = 0; d != Dimension; ++d){
			testInput[i][d]=random::gauss(random::globalRng,3*testTarget[i],1);
		}
	}

	ClassificationDataset dataset = createLabeledDataFromRange(input, target,10);
	ClassificationDataset testDataset = createLabeledDataFromRange(testInput, testTarget,10);
	
	//test using the brute force algorithm, whether the test points
	//are classified correctly in 1-NN
	{
		DenseRbfKernel kernel(0.5);
		SimpleNearestNeighbors<RealVector,unsigned int> algorithm(dataset, &kernel);
		NearestNeighborModel<RealVector, unsigned int> model(&algorithm, 1);

		Data<unsigned int> labels = model(dataset.inputs());
		for (size_t i = 0; i<Points; ++i)
		{
			unsigned int label = model(input[i]);
			BOOST_REQUIRE_EQUAL(target[i], label);
			BOOST_REQUIRE_EQUAL(target[i], labels.elements()[i]);
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
			BOOST_CHECK_EQUAL(bruteforceLabel, testLabels.elements()[i]);
		}
	}
	
	//test using the brute force algorithm, whether the test points
	//are classified correctly in 3-NN
	{
		DenseRbfKernel kernel(0.5);
		SimpleNearestNeighbors<RealVector,unsigned int> algorithm(dataset, &kernel);
		NearestNeighborModel<RealVector, unsigned int> model(&algorithm, 3);
		
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
			RealVector prob = model.decisionFunction()(testInput[i]);
			BOOST_CHECK_EQUAL(bruteforceLabel, label);
			BOOST_CHECK_CLOSE(prob(bruteforceLabel), std::max(res/3.0, 1.0 - res/3.0), 1.e-10);
			BOOST_CHECK_EQUAL(bruteforceLabel, testLabels.elements()[i]);
		}
	
	}
}

BOOST_AUTO_TEST_CASE( Models_NearestNeighbor_Classification_KDTree_Brute_Force) {
	std::size_t Dimension = 5;
	std::size_t Points = 100;
	std::size_t TestPoints = 100;
	std::vector<RealVector> input(Points, RealVector(Dimension));
	std::vector<unsigned int> target(Points);
	for(std::size_t i = 0; i != Points; ++i){
		target[i] = i%2;
		for(std::size_t d = 0; d != Dimension; ++d){
			input[i][d]=random::gauss(random::globalRng,3*target[i],1);
		}
	}
	std::vector<RealVector> testInput(Points, RealVector(Dimension));
	std::vector<unsigned int> testTarget(Points);
	for(std::size_t i = 0; i != TestPoints; ++i){
		testTarget[i] = i%2;
		for(std::size_t d = 0; d != Dimension; ++d){
			testInput[i][d]=random::gauss(random::globalRng,3*testTarget[i],1);
		}
	}

	ClassificationDataset dataset = createLabeledDataFromRange(input, target,10);
	ClassificationDataset testDataset = createLabeledDataFromRange(testInput, testTarget,10);
	
	{
		KDTree<RealVector> tree(dataset.inputs());
		TreeNearestNeighbors<RealVector,unsigned int> algorithm(dataset, &tree);
		NearestNeighborModel<RealVector, unsigned int> model(&algorithm, 1);

		//test whether 1-NN classifies it's training set correctly.
		Data<unsigned int> labels = model(dataset.inputs());
		for (size_t i = 0; i<Points; ++i)
		{
			unsigned int label = model(input[i]);
			BOOST_REQUIRE_EQUAL(target[i], label);
			BOOST_REQUIRE_EQUAL(target[i], labels.elements()[i]);
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
			BOOST_CHECK_EQUAL(bruteforceLabel, testLabels.elements()[i]);
		}
	}
	
	//test using the brute force algorithm, whether the test points
	//are classified correctly in 3-NN
	{
		KDTree<RealVector> tree(dataset.inputs());
		TreeNearestNeighbors<RealVector,unsigned int> algorithm(dataset, &tree);
		NearestNeighborModel<RealVector, unsigned int> model(&algorithm, 3);
		
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
			BOOST_CHECK_EQUAL(bruteforceLabel, testLabels.elements()[i]);
		}
	}
}

BOOST_AUTO_TEST_SUITE_END()
