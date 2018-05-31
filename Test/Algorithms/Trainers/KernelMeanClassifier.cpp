#define BOOST_TEST_MODULE ML_PERCEPTRON
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/Trainers/KernelMeanClassifier.h>
#include <shark/Models/Kernels/LinearKernel.h>
#include <shark/Statistics/Distributions/MultiVariateNormalDistribution.h>

using namespace shark;
using namespace std;

BOOST_AUTO_TEST_SUITE (Algorithms_Trainers_KernelMeanClassifier)

BOOST_AUTO_TEST_CASE( KERNEL_MEAN_CLASSIFIER ) {
	
	DenseLinearKernel kernel;
	KernelMeanClassifier<RealVector> trainer(&kernel);
	KernelClassifier<RealVector> model;

	std::vector<RealVector> input(6,RealVector(2));
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

	ClassificationDataset dataset = createLabeledDataFromRange(input,target);

	trainer.train(model, dataset);

	for(size_t i = 0; i != 6; ++i){
		RealVector result = model.decisionFunction()(input[i]);
		BOOST_CHECK_EQUAL(result.size(),1u);
		unsigned int label = result(0)>0;
		BOOST_CHECK_EQUAL(target[i],label);
	}

}

BOOST_AUTO_TEST_CASE( KMC_TEST_MULTICLASS){
	const size_t TrainExamples = 100;
	const unsigned int classes = 10;
	

	//create datatsets - overlapping normal distributions
	MultiVariateNormalDistribution dist(blas::identity_matrix<double>(2));


	std::vector<RealVector> mean(classes,RealVector(2,0.0));
	for(unsigned int c = 0; c != classes; ++c){
		mean[c](0) = random::gauss(random::globalRng,0,30);
		mean[c](1) = random::gauss(random::globalRng,0,30);
	}

	std::vector<RealVector> input(TrainExamples,RealVector(2));
	std::vector<unsigned int> target(TrainExamples);

	RealMatrix empiricalMean(classes,2,0.0);
	for(size_t i=0;i!=TrainExamples;++i){
		//create sample
		target[i]=i%classes;
		input[i]=dist(random::globalRng).first+mean[target[i]];
		noalias(row(empiricalMean,target[i])) += input[i]/(TrainExamples/classes);
	}
	std::vector<unsigned int> expectedResult(TrainExamples);
	for(size_t i=0;i!=TrainExamples;++i){
		RealMatrix m = sqr(empiricalMean - blas::repeat(input[i],classes));
		expectedResult[i] = arg_min(sum(as_rows(m)));
	}
	
	ClassificationDataset dataset = createLabeledDataFromRange(input,target);
	
	DenseLinearKernel kernel;
	KernelMeanClassifier<RealVector> trainer(&kernel);
	KernelClassifier<RealVector> model;
	trainer.train(model,dataset);
	
	auto resultLabel = model(dataset.inputs());
	for(size_t i=0;i!=TrainExamples;++i){
		BOOST_CHECK_EQUAL(expectedResult[i], resultLabel.element(i));
	}
}
BOOST_AUTO_TEST_CASE( KMC_TEST_MULTICLASS_WEIGHTING ){
	const size_t TrainExamples = 100;
	const size_t Trials = 10;
	const size_t DatasetSize = 500;//size of the dataset after resampling
	const unsigned int classes = 10;
	

	//create datatsets - overlapping normal distributions
	MultiVariateNormalDistribution dist(blas::identity_matrix<double>(2));


	std::vector<RealVector> mean(classes,RealVector(2));
	for(unsigned int c = 0; c != classes; ++c){
		mean[c](0) = random::gauss(random::globalRng,0,30);
		mean[c](1) = random::gauss(random::globalRng,0,30);
	}

	std::vector<RealVector> input(TrainExamples,RealVector(2));
	std::vector<unsigned int> target(TrainExamples);

	for(size_t i=0;i!=TrainExamples;++i){
		//create sample
		target[i]=i%classes;
		input[i]=dist(random::globalRng).first+mean[target[i]];
	}
	ClassificationDataset dataset = createLabeledDataFromRange(input,target);
	
	//resample the dataset by creating duplications. This must be the same as the normal
	//dataset initialized with the correct multiplicities
	for(std::size_t trial = 0; trial != Trials; ++trial){
		//generate weighted and unweighted dataset
		WeightedLabeledData<RealVector,unsigned int> weightedDataset(dataset,0.0);
		ClassificationDataset unweightedDataset(1);
		unweightedDataset.batch(0).input.resize(DatasetSize,inputDimension(dataset));
		unweightedDataset.batch(0).label.resize(DatasetSize);
		RealVector classWeight(classes,0);
		for(std::size_t i = 0; i != DatasetSize; ++i){
			std::size_t index = random::discrete(random::globalRng,std::size_t(0),TrainExamples-1);
			weightedDataset.element(index).weight +=1.0;
			unweightedDataset.element(i) = dataset.element(index);
			classWeight(weightedDataset.element(index).data.label) += 1.0/DatasetSize;
		}
		DenseLinearKernel kernel;
		KernelMeanClassifier<RealVector> trainer(&kernel);
		KernelClassifier<RealVector> modelWeighted;
		KernelClassifier<RealVector> modelUnweighted;
		trainer.train(modelUnweighted, unweightedDataset);
		trainer.train(modelWeighted, weightedDataset);
		
		auto resultUnweighted = modelUnweighted.decisionFunction()(dataset.inputs());
		auto resultWeighted = modelWeighted.decisionFunction()(dataset.inputs());
		
		for(std::size_t i = 0; i != TrainExamples; ++i){
			BOOST_CHECK_SMALL(norm_inf(resultUnweighted.element(i) - resultWeighted.element(i)), 1.e-8);
		}
	}
}


BOOST_AUTO_TEST_SUITE_END()
