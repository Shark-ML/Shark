#define BOOST_TEST_MODULE ML_CVDatasetTools
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Data/CVDatasetTools.h>

#include <algorithm>
#include <iterator>
using namespace shark;

const size_t numPartitions=4;

template<class T,class U>
void testEqualCollections(const T& set,const U& vec){
	BOOST_REQUIRE_EQUAL(set.numberOfElements(),vec.size());
	for(size_t i=0;i!=set.numberOfElements();++i){
		bool found = false;
		for(std::size_t j = 0; j != set.numberOfElements(); ++j)
			if(set.element(j)==vec[i])
				found = true;
		BOOST_CHECK_EQUAL( found,true );
	}
}

BOOST_AUTO_TEST_SUITE (Data_CVDatasetTools)

BOOST_AUTO_TEST_CASE( CVDatasetTools_CreateIndexed ){
	//input for createIndexed
	std::vector<double> inputs;
	std::vector<double> labels;
	std::vector<size_t> indizes;

	//the testresults
	std::vector<double> testInputPartitions[numPartitions];
	//fill the vectors. inputs are the number [0,19], labels go from [20...39]
	//the indizes are assigned cyclically 0,1,2,3,0,1,2,3 to the numbers.
	for(size_t i=0;i!=20;++i){
		inputs.push_back(i);
		labels.push_back(20+i);
		indizes.push_back(i%numPartitions);
		for(size_t j=0;j!=numPartitions;++j){
			if(j!=i%numPartitions){
				testInputPartitions[j].push_back(i);
			}
		}
	}
	LabeledData<double,double> set=createLabeledDataFromRange(inputs,labels,8);
	CVFolds<LabeledData<double,double> > folds= createCVIndexed(set, numPartitions, indizes);

	BOOST_REQUIRE_EQUAL(folds.size(), numPartitions);
	//now test the partitions
	for(size_t i=0;i!=numPartitions;++i){
		LabeledData<double,double> partition = folds.training(i);
		testEqualCollections( partition.inputs(), testInputPartitions[i] );
	}

}


//you can't really test this, because its based on randomness...
//there are a few things like the partitionsSize which should be roughly equal
//the rest is tested through CVDatasetTools_CreateIndexed
//~ BOOST_AUTO_TEST_CASE( CVDatasetTools_CreateIID )
//~ {
	//~ const size_t numExamples=50000;
	//~ const double numValidationSize=numExamples/numPartitions;
	//~ const double numTrainingSize=numExamples-numValidationSize;

	//~ //input for createIID
	//~ std::vector<double> inputs;
	//~ std::vector<double> labels;

	//~ for(size_t i=0;i!=numExamples;++i){
		//~ inputs.push_back(i);
		//~ labels.push_back(numExamples+i);
	//~ }
	//~ LabeledData<double,double> set(inputs,labels,8);

	//~ CVFolds<LabeledData<double,double> > folds= createCVIID(set,numPartitions);
	//~ BOOST_REQUIRE_EQUAL(folds.size(), numPartitions);
	//~ //now test the partitions
	//~ for(size_t i=0;i!=numPartitions;++i){
		//~ LabeledData<double,double> partition = folds.training(i);
		//~ double partitionSizeError=1-partition.numberOfElements()/(double)numTrainingSize;
		//~ BOOST_CHECK_SMALL(partitionSizeError,0.01);
	//~ }
//~ }

BOOST_AUTO_TEST_CASE( CVDatasetTools_CreateSameSize )
{
	const size_t numExamples=102;

	size_t trainSize[]={76,76,77,77};

	//input for createSameSize
	std::vector<double> inputs;
	std::vector<double> labels;

	for(size_t i=0;i!=numExamples;++i){
		inputs.push_back(i);
		labels.push_back(numExamples+i);
	}
	LabeledData<double,double> set = createLabeledDataFromRange(inputs,labels,8);

	CVFolds<LabeledData<double,double> > folds =  createCVSameSize(set,numPartitions);
	BOOST_REQUIRE_EQUAL(folds.size(), numPartitions);
	//now comes the complex part: to ensure that everything is ok
	//we check that all validation vectors together form the input and label array.
	//the test works as following:
	//1. all vectors are appended
	//2. the resulting vector is sorted
	//3.after that it holds vec[i]==i for inputs and vec[i]==i+numExamples for labels
	std::vector<double> validationInputs;
	std::vector<double> validationLabels;
	for(size_t i=0;i!=numPartitions;++i){
		LabeledData<double,double> partition = folds.training(i);
		LabeledData<double,double> validation = folds.validation(i);
		BOOST_REQUIRE_EQUAL(partition.numberOfElements(),trainSize[i]);
		BOOST_REQUIRE_EQUAL(validation.numberOfElements(),numExamples-trainSize[i]);
		for(size_t j=0;j!=validation.numberOfElements();++j){
			validationInputs.push_back(validation.element(j).input);
			validationLabels.push_back(validation.element(j).label);
		}
	}
	std::sort(validationInputs.begin(),validationInputs.end());
	std::sort(validationLabels.begin(),validationLabels.end());
	for(size_t i=0;i!=numExamples;++i){
		BOOST_CHECK_EQUAL(validationInputs[i],i);
		BOOST_CHECK_EQUAL(validationLabels[i],i+numExamples);
	}
}
BOOST_AUTO_TEST_CASE( CVDatasetTools_CreateSameSizeBalancedUnsigned )
{
	const size_t numExamples=102;

	size_t trainSize[]={51,51};
	//number of.numberOfElements with label 0 per partition
	size_t zeroSize[]={26,25};


	//input for createSameSize
	std::vector<RealVector> inputs;
	std::vector<unsigned int> labels;

	for(size_t i=0;i!=numExamples;++i){
		RealVector vec(1);
		vec(0)=i;
		inputs.push_back(vec);
		labels.push_back(i%2);
	}
	ClassificationDataset set = createLabeledDataFromRange(inputs,labels,8);

	CVFolds<ClassificationDataset > folds =  createCVSameSizeBalanced(set,2);
	BOOST_REQUIRE_EQUAL(folds.size(), 2);

	//first check equals previous test
	std::vector<double> validationInputs;
	for(size_t i=0;i!=2;++i){
		ClassificationDataset partition = folds.training(i);
		ClassificationDataset validation = folds.validation(i);
		BOOST_REQUIRE_EQUAL(partition.numberOfElements(),trainSize[i]);
		size_t zeroCount=0;
		for(size_t j=0;j!=validation.numberOfElements();++j){
			validationInputs.push_back(validation.element(j).input(0));
			zeroCount+= !validation.element(j).label;
		}
		BOOST_CHECK_EQUAL(zeroCount,zeroSize[i]);
	}
	std::sort(validationInputs.begin(),validationInputs.end());
	for(size_t i=0;i!=numExamples;++i){
		BOOST_CHECK_EQUAL(validationInputs[i],i);
	}


}
BOOST_AUTO_TEST_SUITE_END()
