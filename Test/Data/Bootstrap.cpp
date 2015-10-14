#define BOOST_TEST_MODULE DATA_BOOTSTRAP
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Data/WeightedDataset.h>

using namespace shark;

BOOST_AUTO_TEST_SUITE (Data_Bootstrap)

BOOST_AUTO_TEST_CASE( Bootstrap_LabeledData ){
	//create a toy dataset
	std::vector<unsigned int> inputs;
	std::vector<unsigned int> labels;

	for(size_t i=0;i != 20;++i){
		inputs.push_back(i);
		labels.push_back(20+i);
	}
	LabeledData<unsigned int, unsigned int> set=createLabeledDataFromRange(inputs,labels,8);
	
	//create Bootstrap subsets and create a running sum of the weights to check
	// that every point is chosen equally.
	RealVector weightSums(20,0.0);
	std::size_t iterations = 10000;
	for(std::size_t iteration = 0; iteration != iterations; ++iteration){
		WeightedLabeledData<unsigned int,unsigned int> bootstrapSet = bootstrap(set);
		BOOST_REQUIRE_EQUAL(bootstrapSet.numberOfElements(),20);
		double setWeightSum = 0.0;
		for(std::size_t i = 0; i != 20; ++i){
			BOOST_CHECK_EQUAL(bootstrapSet.element(i).data.input,set.element(i).input);
			BOOST_CHECK_EQUAL(bootstrapSet.element(i).data.label,set.element(i).label);
			weightSums[i] += bootstrapSet.element(i).weight;
			setWeightSum += bootstrapSet.element(i).weight;
		}
		BOOST_CHECK_CLOSE(setWeightSum,20,1.e-7);
	}
	weightSums/=iterations;
	for(std::size_t i = 0; i != 20; ++i){
		BOOST_CHECK_CLOSE(weightSums[i],1.0,5);
	}
}

BOOST_AUTO_TEST_CASE( Bootstrap_UnlabeledData ){
	//create a toy dataset
	std::vector<unsigned int> inputs;

	for(size_t i=0;i != 20;++i){
		inputs.push_back(i);
	}
	UnlabeledData<unsigned int> set=createDataFromRange(inputs,8);
	
	//create Bootstrap subsets and create a running sum of the weights to check
	// that every point is chosen equally.
	RealVector weightSums(20,0.0);
	std::size_t iterations = 10000;
	for(std::size_t iteration = 0; iteration != iterations; ++iteration){
		WeightedUnlabeledData<unsigned int> bootstrapSet = bootstrap(set);
		BOOST_REQUIRE_EQUAL(bootstrapSet.numberOfElements(),20);
		double setWeightSum = 0.0;
		for(std::size_t i = 0; i != 20; ++i){
			BOOST_CHECK_EQUAL(bootstrapSet.element(i).data,set.element(i));
			weightSums[i] += bootstrapSet.element(i).weight;
			setWeightSum += bootstrapSet.element(i).weight;
		}
		BOOST_CHECK_CLOSE(setWeightSum,20,1.e-7);
	}
	weightSums/=iterations;
	for(std::size_t i = 0; i != 20; ++i){
		BOOST_CHECK_CLOSE(weightSums[i],1.0,5);
	}
}
BOOST_AUTO_TEST_SUITE_END()
