#include <shark/Unsupervised/RBM/ConvolutionalBinaryRBM.h>
#include <shark/LinAlg/Base.h>
#include <shark/Unsupervised/RBM/Sampling/GibbsOperator.h>
#include <shark/Rng/GlobalRng.h>

#define BOOST_TEST_MODULE RBM_ConvolutionalEnergyGradient
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include "../ObjectiveFunctions/TestObjectiveFunction.h"

using namespace shark;

//test, that the weighted gradient produces correct results for binary units when using addVH
//test1 is with weight 1
BOOST_AUTO_TEST_SUITE (RBM_ConvolutionalEnergyGradient)

BOOST_AUTO_TEST_CASE( ConvolutionalEnergyGradient_Weighted_One_Visible )
{
	ConvolutionalBinaryRBM rbm(Rng::globalRng);
	rbm.setStructure(4,4,1,3);
	initRandomNormal(rbm,2);
	
	RealMatrix batch(10,16);
	for(std::size_t j = 0; j != 10; ++j){
		for(std::size_t k = 0; k != 16; ++k){
			batch(j,k)=Rng::coinToss(0.5);
		}
	}
	
	ConvolutionalBinaryGibbsOperator::HiddenSampleBatch hiddenBatch(10,4);
	ConvolutionalBinaryGibbsOperator::VisibleSampleBatch visibleBatch(10,16);
	
	ConvolutionalBinaryGibbsOperator gibbs(&rbm);
	detail::ConvolutionalEnergyGradient<ConvolutionalBinaryRBM> grad(&rbm);
	detail::ConvolutionalEnergyGradient<ConvolutionalBinaryRBM> gradTest(&rbm);
	
	gibbs.createSample(hiddenBatch,visibleBatch,batch);
	
	grad.addVH(hiddenBatch,visibleBatch);
	gradTest.addVH(hiddenBatch,visibleBatch,blas::repeat(0.0,10));
	
	RealVector diff = grad.result()-gradTest.result();
	
	BOOST_CHECK_SMALL(norm_1(diff),1.e-5);
}
//test, that the weighted gradient produces correct results for binary units when using addHV
//test1 is with weight 1
BOOST_AUTO_TEST_CASE( ConvolutionalEnergyGradient_Weighted_One_Hidden )
{
	ConvolutionalBinaryRBM rbm(Rng::globalRng);
	rbm.setStructure(4,4,1,3);
	initRandomNormal(rbm,2);
	
	RealMatrix batch(10,4);
	for(std::size_t j = 0; j != 10; ++j){
		for(std::size_t k = 0; k != 4; ++k){
			batch(j,k)=Rng::coinToss(0.5);
		}
	}
	
	ConvolutionalBinaryGibbsOperator::HiddenSampleBatch hiddenBatch(10,4);
	ConvolutionalBinaryGibbsOperator::VisibleSampleBatch visibleBatch(10,16);
	
	ConvolutionalBinaryGibbsOperator gibbs(&rbm);
	detail::ConvolutionalEnergyGradient<ConvolutionalBinaryRBM> grad(&rbm);
	detail::ConvolutionalEnergyGradient<ConvolutionalBinaryRBM> gradTest(&rbm);
	
	hiddenBatch.state = batch;
	gibbs.precomputeVisible(hiddenBatch,visibleBatch,blas::repeat(1.0,10));
	
	grad.addHV(hiddenBatch,visibleBatch);
	gradTest.addHV(hiddenBatch,visibleBatch,blas::repeat(0.0,10));
	
	RealVector diff = grad.result()-gradTest.result();
	
	BOOST_CHECK_SMALL(norm_1(diff),1.e-5);
}

//~ //test, that the weighted gradient produces correct results for binary units when using addVH
//~ //test2 is with different weights
//~ BOOST_AUTO_TEST_CASE( ConvolutionalEnergyGradient_Weighted_Visible )
//~ {
	//~ ConvolutionalBinaryRBM rbm(Rng::globalRng);
	//~ rbm.setStructure(4,4);
	//~ initRandomNormal(rbm,2);
	
	//~ RealMatrix batch(10,4);
	//~ RealVector weights(10);
	//~ for(std::size_t j = 0; j != 10; ++j){
		//~ for(std::size_t k = 0; k != 4; ++k){
			//~ batch(j,k)=Rng::coinToss(0.5);
		//~ }
		//~ weights(j)=j;
	//~ }
	//~ double logWeightSum=std::log(sum(weights));
	
	//~ ConvolutionalBinaryGibbsOperator::HiddenSampleBatch hiddenBatch(10,4);
	//~ ConvolutionalBinaryGibbsOperator::VisibleSampleBatch visibleBatch(10,4);
	
	//~ ConvolutionalBinaryGibbsOperator gibbs(&rbm);
	//~ detail::ConvolutionalEnergyGradient<ConvolutionalBinaryRBM> grad(&rbm);
	//~ detail::ConvolutionalEnergyGradient<ConvolutionalBinaryRBM> gradTest(&rbm);
	
	//~ gibbs.createSample(hiddenBatch,visibleBatch,batch);
	
	//~ grad.addVH(hiddenBatch,visibleBatch,log(weights));
	//~ BOOST_CHECK_CLOSE(logWeightSum,grad.logWeightSum(),1.e-5);
	
	//~ //calculate ground truth data by incorporating the weights into 
	//~ //the state. we have to correct the probability part of the gradient afterwards
	//~ //as this is not changed.
	//~ RealVector newWeights=weights/std::exp(logWeightSum)*10;
	//~ for(std::size_t i = 0; i != 10; ++i){
		//~ row(visibleBatch.state,i)*=newWeights(i);
	//~ }
	//~ gradTest.addVH(hiddenBatch,visibleBatch);
	//~ RealVector gradTestResult= gradTest.result();
	//~ noalias(subrange(gradTestResult,16,20))= 
		//~ prod(weights,hiddenBatch.statistics)/std::exp(logWeightSum);
	
	//~ RealVector diff = grad.result()-gradTestResult;
	
	//~ BOOST_CHECK_SMALL(norm_1(diff),1.e-5);
//~ }

//~ //test, that the weighted gradient produces correct results for binary units when using addHV
//~ //test2 is with different weights
//~ BOOST_AUTO_TEST_CASE( ConvolutionalEnergyGradient_Weighted_Hidden )
//~ {
	//~ ConvolutionalBinaryRBM rbm(Rng::globalRng);
	//~ rbm.setStructure(4,4);
	//~ initRandomNormal(rbm,2);
	
	//~ RealMatrix batch(10,4);
	//~ RealVector weights(10);
	//~ for(std::size_t j = 0; j != 10; ++j){
		//~ for(std::size_t k = 0; k != 4; ++k){
			//~ batch(j,k)=Rng::coinToss(0.5);
		//~ }
		//~ weights(j)=j;
	//~ }
	//~ double logWeightSum=std::log(sum(weights));
	
	//~ ConvolutionalBinaryGibbsOperator::HiddenSampleBatch hiddenBatch(10,4);
	//~ ConvolutionalBinaryGibbsOperator::VisibleSampleBatch visibleBatch(10,4);
	
	//~ ConvolutionalBinaryGibbsOperator gibbs(&rbm);
	//~ detail::ConvolutionalEnergyGradient<ConvolutionalBinaryRBM> grad(&rbm);
	//~ detail::ConvolutionalEnergyGradient<ConvolutionalBinaryRBM> gradTest(&rbm);
	
	//~ hiddenBatch.state = batch;
	//~ gibbs.precomputeVisible(hiddenBatch,visibleBatch,blas::repeat(1.0,10));
	
	//~ grad.addHV(hiddenBatch,visibleBatch,log(weights));
	//~ BOOST_CHECK_CLOSE(logWeightSum,grad.logWeightSum(),1.e-5);
	
	//~ //calculate ground truth data
	//~ RealVector newWeights=weights/std::exp(logWeightSum)*10;
	//~ for(std::size_t i = 0; i != 10; ++i){
		//~ row(hiddenBatch.state,i)*=newWeights(i);
	//~ }
	//~ gradTest.addHV(hiddenBatch,visibleBatch);
	//~ RealVector gradTestResult= gradTest.result();
	//~ noalias(subrange(gradTestResult,20,24))= 
		//~ prod(weights,visibleBatch.statistics)/std::exp(logWeightSum);
	
	//~ RealVector diff = grad.result()-gradTestResult;
	
	//~ BOOST_CHECK_SMALL(norm_1(diff),1.e-5);
//~ }
//now, that we now, that the unweighted version does the same as the weighted, we check
//that the unweighted version works, by calculating the numerical gradient
class TestGradientVH : public SingleObjectiveFunction{
public:
	TestGradientVH(ConvolutionalBinaryRBM* rbm): mpe_rbm(rbm){
		m_features |= HAS_FIRST_DERIVATIVE;
		m_features |= CAN_PROPOSE_STARTING_POINT;
	};

	std::string name() const
	{ return "TestGradientVH"; }

	void setData(UnlabeledData<RealVector> const& data){
		m_data = data;	
	}

	void proposeStartingPoint(SearchPointType& startingPoint) const{
		startingPoint = mpe_rbm->parameterVector();
	}
	std::size_t numberOfVariables()const{
		return mpe_rbm->numberOfParameters();
	}
	
	double eval( SearchPointType const & parameter) const {
		mpe_rbm->setParameterVector(parameter);
		double result=0;
		for(std::size_t i =0; i != m_data.numberOfBatches();++i){
			result+=sum(mpe_rbm->energy().logUnnormalizedProbabilityVisible(
				m_data.batch(i),blas::repeat(1.0,m_data.batch(i).size1())
			));
		}
		result/=m_data.numberOfElements();
		return result;
	}

	double evalDerivative( SearchPointType const & parameter, FirstOrderDerivative & derivative ) const {
		mpe_rbm->setParameterVector(parameter);
		detail::ConvolutionalEnergyGradient<ConvolutionalBinaryRBM> gradient(mpe_rbm);
		GibbsOperator<ConvolutionalBinaryRBM> sampler(mpe_rbm);
		
		//create Energy from RBM
		
		//calculate the expectation of the energy gradient with respect to the data
		for(std::size_t i=0; i != m_data.numberOfBatches(); i++){
			std::size_t currentBatchSize=m_data.batch(i).size1();
			GibbsOperator<ConvolutionalBinaryRBM>::HiddenSampleBatch hiddenSamples(currentBatchSize,mpe_rbm->numberOfHN());
			GibbsOperator<ConvolutionalBinaryRBM>::VisibleSampleBatch visibleSamples(currentBatchSize,mpe_rbm->numberOfVN());
		
			sampler.createSample(hiddenSamples,visibleSamples,m_data.batch(i));
			gradient.addVH(hiddenSamples, visibleSamples);
		}
		derivative = gradient.result();
		return eval(parameter);
	}
private:
	ConvolutionalBinaryRBM* mpe_rbm;
	UnlabeledData<RealVector> m_data;
};

BOOST_AUTO_TEST_CASE( ConvolutionalEnergyGradient_DerivativeVH )
{
	ConvolutionalBinaryRBM rbm(Rng::globalRng);
	rbm.setStructure(4,4,2,3);
	
	std::vector<RealVector> dataVec(50,RealVector(16));
	for(std::size_t j = 0; j != 16; ++j){
		for(std::size_t k = 0; k != 10; ++k){
			dataVec[j](k)=Rng::coinToss(0.5);
		}
	}
	UnlabeledData<RealVector> data = createDataFromRange(dataVec,25);
	
	TestGradientVH gradient(&rbm);
	gradient.setData(data);
	
	for(std::size_t i = 0; i != 100; ++i){
		initRandomNormal(rbm,1);
		RealVector parameters = rbm.parameterVector();
		testDerivative(gradient,parameters,1.e-2,1.e-10,0.025);
	}
}

class TestGradientHV : public SingleObjectiveFunction{
public:
	TestGradientHV(ConvolutionalBinaryRBM* rbm): mpe_rbm(rbm){
		m_features |= HAS_FIRST_DERIVATIVE;
		m_features |= CAN_PROPOSE_STARTING_POINT;
	};

	std::string name() const
	{ return "TestGradientHV"; }

	void setData(UnlabeledData<RealVector> const& data){
		m_data = data;	
	}

	void proposeStartingPoint(SearchPointType& startingPoint) const{
		startingPoint = mpe_rbm->parameterVector();
	}
	
	std::size_t numberOfVariables()const{
		return mpe_rbm->numberOfParameters();
	}
	
	double eval( SearchPointType const & parameter) const {
		mpe_rbm->setParameterVector(parameter);
		
		double result=0;
		for(std::size_t i =0; i != m_data.numberOfBatches();++i){
			result+=sum(mpe_rbm->energy().logUnnormalizedProbabilityHidden(
				m_data.batch(i),blas::repeat(1,m_data.batch(i).size1())
			));
		}
		result/=m_data.numberOfElements();
		return result;
	}

	double evalDerivative( SearchPointType const & parameter, FirstOrderDerivative & derivative ) const {
		mpe_rbm->setParameterVector(parameter);
		detail::ConvolutionalEnergyGradient<ConvolutionalBinaryRBM> gradient(mpe_rbm);
		GibbsOperator<ConvolutionalBinaryRBM> sampler(mpe_rbm);
		
		//create Energy from RBM

		//calculate the expectation of the energy gradient with respect to the data
		for(std::size_t i=0; i != m_data.numberOfBatches(); i++){
			std::size_t currentBatchSize=m_data.batch(i).size1();
			GibbsOperator<ConvolutionalBinaryRBM>::HiddenSampleBatch hiddenSamples(currentBatchSize,mpe_rbm->numberOfHN());
			GibbsOperator<ConvolutionalBinaryRBM>::VisibleSampleBatch visibleSamples(currentBatchSize,mpe_rbm->numberOfVN());
		
			hiddenSamples.state = m_data.batch(i);
			sampler.precomputeVisible(hiddenSamples,visibleSamples,blas::repeat(1.0,10));
			gradient.addHV(hiddenSamples, visibleSamples);
		}
		derivative = gradient.result();
		return eval(parameter);
	}
private:
	ConvolutionalBinaryRBM* mpe_rbm;
	UnlabeledData<RealVector> m_data;
};

BOOST_AUTO_TEST_CASE( ConvolutionalEnergyGradient_DerivativeHV )
{
	ConvolutionalBinaryRBM rbm(Rng::globalRng);
	rbm.setStructure(4,4,2,3);
	
	std::vector<RealVector> dataVec(10,RealVector(8));
	for(std::size_t j = 0; j != 10; ++j){
		for(std::size_t k = 0; k != 8; ++k){
			dataVec[j](k)=Rng::coinToss(0.5);
		}
	}
	UnlabeledData<RealVector> data = createDataFromRange(dataVec,25);
	
	TestGradientHV gradient(&rbm);
	gradient.setData(data);
	
	for(std::size_t i = 0; i != 100; ++i){
		initRandomNormal(rbm,1);
		RealVector parameters = rbm.parameterVector();
		testDerivative(gradient,parameters,2.e-2,1.e-10,0.01);
	}
}


BOOST_AUTO_TEST_SUITE_END()
