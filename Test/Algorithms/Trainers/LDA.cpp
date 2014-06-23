#define BOOST_TEST_MODULE TRAINERS_LDA
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/Trainers/LDA.h>
#include <shark/Statistics/Distributions/MultiVariateNormalDistribution.h>
#include <shark/LinAlg/solveSystem.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
using namespace shark;

BOOST_AUTO_TEST_CASE( LDA_TEST_TWOCLASS ){
	const size_t TrainExamples = 12000;
	LDA trainer;
	LinearClassifier<> model;


	//create datatsets - two overlapping normal distributions
	RealMatrix covariance(2,2);
	covariance(0,0)=16;
	covariance(0,1)=8;
	covariance(1,0)=8;
	covariance(1,1)=16;
	RealMatrix inverse(2,2,0.0);
	inverse(0,0) = inverse(1,1) = 1.0;
	blas::solveSymmSystemInPlace<blas::SolveAXB>(covariance,inverse);


	RealVector mean[]={RealVector(2),RealVector(2)};
	mean[0](0)=0;
	mean[0](1)=0;
	mean[1](0)=6;
	mean[1](1)=6;
	
	double prior[2]={std::log(1.0/3.0),std::log(2.0/3.0)};
	MultiVariateNormalDistribution dist(covariance);


	std::vector<RealVector> input(TrainExamples,RealVector(2));
	std::vector<unsigned int> target(TrainExamples);

	double statisticalBayesRisk=0;
	for(size_t i=0;i!=TrainExamples;++i){
		//create samples. class 1 has double as many elements as class 0
		target[i]=(i%3 == 0);
		input[i]=dist().first+mean[target[i]];
		//calculate bayes Target - the best fit to the distributions
		RealVector diff=input[i]-mean[0];
		double dist0=inner_prod(diff,prod(inverse,diff))+prior[0];
		diff=input[i]-mean[1];
		double dist1=inner_prod(diff,prod(inverse,diff))+prior[1];
		unsigned int bayesTarget = dist0>dist1;
		statisticalBayesRisk+= bayesTarget != target[i];
	}
	statisticalBayesRisk/=TrainExamples;

	ClassificationDataset dataset = createLabeledDataFromRange(input,target);

	trainer.train(model, dataset);

	
	Data<unsigned int> results = model(dataset.inputs());
	
	ZeroOneLoss<> loss;
	double classificatorRisk = loss.eval(dataset.labels(),results);

	std::cout<<"statistical bayes Risk: "<<statisticalBayesRisk<<std::endl;
	std::cout<<"classificator Risk: "<<classificatorRisk<<std::endl;
	BOOST_CHECK_SMALL(classificatorRisk-statisticalBayesRisk,0.01);

}

BOOST_AUTO_TEST_CASE( LDA_TEST_TWOCLASS_SINGULAR ){
	const size_t TrainExamples = 12000;
	LDA trainer;
	LinearClassifier<> model;


	//create datatsets - two overlapping normal distributions
	//same as in the previous test aside from that we add a third
	//variable which is allways 0

	RealMatrix covariance(2,2);
	covariance(0,0)=16;
	covariance(0,1)=8;
	covariance(1,0)=8;
	covariance(1,1)=16;
	RealMatrix inverse(2,2,0.0);
	inverse(0,0) = inverse(1,1) = 1.0;
	blas::solveSymmSystemInPlace<blas::SolveAXB>(covariance,inverse);


	RealVector mean[]={RealVector(2),RealVector(2)};
	mean[0](0)=0;
	mean[0](1)=0;
	mean[1](0)=6;
	mean[1](1)=6;
	MultiVariateNormalDistribution dist(covariance);
	
	double prior[2]={std::log(2.0/3.0),std::log(1.0/3.0)};


	std::vector<RealVector> input(TrainExamples,RealVector(3));
	std::vector<unsigned int> target(TrainExamples);

	double statisticalBayesRisk=0;
	for(size_t i=0;i!=TrainExamples;++i){
		//create sample
		target[i]= (i%3 != 0);
		RealVector vec = dist().first+mean[target[i]];
		//calculate bayes Target - the best fit to the distributions
		RealVector diff=vec-mean[0];
		double dist0=inner_prod(diff,prod(inverse,diff)) + prior[0];
		diff=vec-mean[1];
		double dist1=inner_prod(diff,prod(inverse,diff)) + prior[1];
		unsigned int bayesTarget = dist0>dist1;
		statisticalBayesRisk+= bayesTarget != target[i];
		init(input[i])<<vec,0;//add third zero
	}
	statisticalBayesRisk/=TrainExamples;

	ClassificationDataset dataset = createLabeledDataFromRange(input,target);

	trainer.train(model, dataset);

	
	Data<unsigned int> results = model(dataset.inputs());
	
	ZeroOneLoss<> loss;
	double classificatorRisk = loss.eval(dataset.labels(),results);

	std::cout<<"statistical bayes Risk: "<<statisticalBayesRisk<<std::endl;
	std::cout<<"classificator Risk: "<<classificatorRisk<<std::endl;
	BOOST_CHECK_SMALL(classificatorRisk-statisticalBayesRisk,0.01);

}

//~ BOOST_AUTO_TEST_CASE( LDA_TEST_MULTICLASS ){
	//~ const size_t TrainExamples = 200000;
	//~ const unsigned int classes = 10;
	//~ LDA trainer;
	//~ LinearClassifier<> model;
	//~ Rng::seed(44);


	//~ //create datatsets - overlapping normal distributions

	//~ RealMatrix covariance(2,2);
	//~ covariance(0,0)=16;
	//~ covariance(0,1)=8;
	//~ covariance(1,0)=8;
	//~ covariance(1,1)=16;
	//~ RealMatrix inverse(2,2,0.0);
	//~ inverse(0,0) = inverse(1,1) = 1.0;
	//~ blas::solveSymmSystemInPlace<blas::SolveAXB>(covariance,inverse);
	
	//~ std::vector<RealVector> mean(classes,RealVector(2));
	//~ for(unsigned int c = 0; c != classes; ++c){
		//~ for(std::size_t j = 0; j != 2; ++j){
			//~ mean[c](j) = Rng::gauss(0,30);
		//~ }
	//~ }
	//~ MultiVariateNormalDistribution dist(covariance);


	//~ std::vector<RealVector> input(TrainExamples,RealVector(2));
	//~ std::vector<unsigned int> target(TrainExamples);

	//~ double statisticalBayesRisk=0;
	//~ for(size_t i=0;i!=TrainExamples;++i){
		//~ //create sample
		//~ target[i]=i%classes;
		//~ input[i]=dist().first+mean[target[i]];
		//~ //calculate bayes Target - the best fit to the distributions
		//~ unsigned int bayesTarget = 0;
		//~ double minDist = 1.e30;
		//~ for(unsigned int c = 0; c != classes; ++c){
			//~ RealVector diff=input[i]-mean[c];
			//~ double dist=inner_prod(diff,prod(inverse,diff));
			//~ if(dist<minDist){
				//~ minDist=dist;
				//~ bayesTarget=c;
			//~ }
		//~ }
		//~ statisticalBayesRisk+= bayesTarget != target[i];
	//~ }
	//~ statisticalBayesRisk/=TrainExamples;

	//~ ClassificationDataset dataset = createLabeledDataFromRange(input,target);

	//~ trainer.train(model, dataset);

	//~ //double classificatorRisk=0;
	//~ Data<unsigned int> results = model(dataset.inputs());
	
	//~ ZeroOneLoss<> loss;
	//~ double classificatorRisk = loss.eval(dataset.labels(),results);
	
	//~ std::cout<<"statistical bayes Risk: "<<statisticalBayesRisk<<std::endl;
	//~ std::cout<<"classificator Risk: "<<classificatorRisk<<std::endl;
	//~ BOOST_CHECK_SMALL(classificatorRisk-statisticalBayesRisk,10e-2);

//~ }



BOOST_AUTO_TEST_CASE( LDA_TEST_MULTICLASS_WEIGHTING ){
	const size_t TrainExamples = 100;
	const size_t Trials = 10;
	const size_t DatasetSize = 500;//size of the dataset after resampling
	const unsigned int classes = 10;
	

	//create datatsets - overlapping normal distributions
	RealMatrix covariance(2,2);
	covariance(0,0)=16;
	covariance(0,1)=8;
	covariance(1,0)=8;
	covariance(1,1)=16;
	RealMatrix inverse(2,2,0.0);
	inverse(0,0) = inverse(1,1) = 1.0;
	blas::solveSymmSystemInPlace<blas::SolveAXB>(covariance,inverse);
	
	std::vector<RealVector> mean(classes,RealVector(2));
	for(unsigned int c = 0; c != classes; ++c){
		mean[c](0) = Rng::gauss(0,30);
		mean[c](1) = Rng::gauss(0,30);
	}
	MultiVariateNormalDistribution dist(covariance);

	std::vector<RealVector> input(TrainExamples,RealVector(2));
	std::vector<unsigned int> target(TrainExamples);

	for(size_t i=0;i!=TrainExamples;++i){
		//create sample
		target[i]=i%classes;
		input[i]=dist().first+mean[target[i]];
	}
	ClassificationDataset dataset = createLabeledDataFromRange(input,target);
	
	//first check that the weighted and unweighted version give the same results when all weights are equal
	LDA trainer;
	LinearClassifier<> modelUnweighted;
	LinearClassifier<> modelWeighted;
	
	trainer.train(modelUnweighted, dataset);
	trainer.train(modelWeighted, WeightedLabeledData<RealVector,unsigned int>(dataset,2.0));//two as 1 might hide errors as sqrt(1)=1
	
	//we need to correct for the fact that the unweighted versions normalizes with inputs-classes
	//and the unnormalized only with sumOfWeights=inputs.
	BOOST_CHECK_SMALL(
		norm_frobenius(modelUnweighted.decisionFunction().matrix()/0.9-modelWeighted.decisionFunction().matrix()),
		0.0001
	);
	RealVector normalizedOffset = (modelUnweighted.decisionFunction().offset()-blas::repeat(std::log(0.1),10))/0.9;
	BOOST_CHECK_SMALL(
		norm_2(normalizedOffset-modelWeighted.decisionFunction().offset()+blas::repeat(std::log(0.1),10)),
		0.0001
	);
	
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
			std::size_t index = Rng::discrete(0,TrainExamples-1);
			weightedDataset.element(index).weight +=1.0;
			unweightedDataset.element(i) = dataset.element(index);
			classWeight(weightedDataset.element(index).data.label) += 1.0/DatasetSize;
		}
		trainer.train(modelUnweighted, unweightedDataset);
		trainer.train(modelWeighted, weightedDataset);
		double covarianceCorrection = double(DatasetSize-classes)/DatasetSize;
		BOOST_CHECK_SMALL(
			norm_frobenius(modelUnweighted.decisionFunction().matrix()/covarianceCorrection-modelWeighted.decisionFunction().matrix()),
			0.0001
		);
		RealVector normalizedOffset = (modelUnweighted.decisionFunction().offset()-log(classWeight))/covarianceCorrection;
		BOOST_CHECK_SMALL(
			norm_2(normalizedOffset-modelWeighted.decisionFunction().offset()+log(classWeight)),
			0.0001
		);
		
	}
}