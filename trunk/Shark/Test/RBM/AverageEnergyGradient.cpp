#include <shark/Unsupervised/RBM/BinaryRBM.h>
#include <shark/LinAlg/Base.h>
#include <shark/Unsupervised/RBM/Sampling/GibbsOperator.h>
#include <shark/Rng/GlobalRng.h>

#define BOOST_TEST_MODULE RBM_AverageEnergyGradient
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include "../ObjectiveFunctions/TestObjectiveFunction.h"

using namespace shark;

//test, that the weighted gradient produces correct results for binary units when using addVH
//test1 is with weight 1
BOOST_AUTO_TEST_CASE( AverageEnergyGradient_Weighted_One_Visible )
{
	BinaryRBM rbm(Rng::globalRng);
	rbm.structure().setStructure(4,4);
	initRandomNormal(rbm,2);
	
	RealMatrix batch(10,4);
	for(std::size_t j = 0; j != 10; ++j){
		for(std::size_t k = 0; k != 4; ++k){
			batch(j,k)=Rng::coinToss(0.5);
		}
	}
	
	BinaryGibbsOperator::HiddenSampleBatch hiddenBatch(10,4);
	BinaryGibbsOperator::VisibleSampleBatch visibleBatch(10,4);
	
	BinaryGibbsOperator gibbs(&rbm);
	BinaryRBM::Energy::AverageEnergyGradient grad(&rbm.structure());
	BinaryRBM::Energy::AverageEnergyGradient gradTest(&rbm.structure());
	gibbs.flags() = grad.flagsVH();
	
	RealScalarVector zero(10,0);
	gibbs.createSample(hiddenBatch,visibleBatch,batch);
	
	grad.addVH(hiddenBatch,visibleBatch);
	gradTest.addVH(hiddenBatch,visibleBatch,zero);
	
	RealVector diff = grad.result()-gradTest.result();
	
	BOOST_CHECK_SMALL(norm_1(diff),1.e-5);
}
//test, that the weighted gradient produces correct results for binary units when using addHV
//test1 is with weight 1
BOOST_AUTO_TEST_CASE( AverageEnergyGradient_Weighted_One_Hidden )
{
	BinaryRBM rbm(Rng::globalRng);
	rbm.structure().setStructure(4,4);
	initRandomNormal(rbm,2);
	
	RealMatrix batch(10,4);
	for(std::size_t j = 0; j != 10; ++j){
		for(std::size_t k = 0; k != 4; ++k){
			batch(j,k)=Rng::coinToss(0.5);
		}
	}
	
	BinaryGibbsOperator::HiddenSampleBatch hiddenBatch(10,4);
	BinaryGibbsOperator::VisibleSampleBatch visibleBatch(10,4);
	
	BinaryGibbsOperator gibbs(&rbm);
	BinaryRBM::Energy::AverageEnergyGradient grad(&rbm.structure());
	BinaryRBM::Energy::AverageEnergyGradient gradTest(&rbm.structure());
	gibbs.flags() = grad.flagsHV();
	
	RealScalarVector zero(10,0);
	hiddenBatch.state = batch;
	gibbs.precomputeVisible(hiddenBatch,visibleBatch);
	
	grad.addHV(hiddenBatch,visibleBatch);
	gradTest.addHV(hiddenBatch,visibleBatch,zero);
	
	RealVector diff = grad.result()-gradTest.result();
	
	BOOST_CHECK_SMALL(norm_1(diff),1.e-5);
}

//test, that the weighted gradient produces correct results for binary units when using addVH
//test2 is with different weights
BOOST_AUTO_TEST_CASE( AverageEnergyGradient_Weighted_Visible )
{
	BinaryRBM rbm(Rng::globalRng);
	rbm.structure().setStructure(4,4);
	initRandomNormal(rbm,2);
	
	RealMatrix batch(10,4);
	RealVector weights(10);
	for(std::size_t j = 0; j != 10; ++j){
		for(std::size_t k = 0; k != 4; ++k){
			batch(j,k)=Rng::coinToss(0.5);
		}
		weights(j)=j;
	}
	double logWeightSum=std::log(sum(weights));
	
	BinaryGibbsOperator::HiddenSampleBatch hiddenBatch(10,4);
	BinaryGibbsOperator::VisibleSampleBatch visibleBatch(10,4);
	
	BinaryGibbsOperator gibbs(&rbm);
	BinaryRBM::Energy::AverageEnergyGradient grad(&rbm.structure());
	BinaryRBM::Energy::AverageEnergyGradient gradTest(&rbm.structure());
	gibbs.flags() = grad.flagsVH();
	
	gibbs.createSample(hiddenBatch,visibleBatch,batch);
	
	gradTest.addVH(hiddenBatch,visibleBatch,log(weights));
	BOOST_CHECK_CLOSE(logWeightSum,gradTest.logWeightSum(),1.e-5);
	
	//calculate ground truth data
	RealVector newWeights=weights/std::exp(logWeightSum)*10;
	for(std::size_t i = 0; i != 10; ++i){
		row(visibleBatch.state,i)*=newWeights(i);
		row(hiddenBatch.statistics.probability,i)*=newWeights(i);
	}
	grad.addVH(hiddenBatch,visibleBatch);
	
	
	RealVector diff = grad.result()-gradTest.result();
	
	BOOST_CHECK_SMALL(norm_1(diff),1.e-5);
}

//test, that the weighted gradient produces correct results for binary units when using addHV
//test2 is with different weights
BOOST_AUTO_TEST_CASE( AverageEnergyGradient_Weighted_Hidden )
{
	BinaryRBM rbm(Rng::globalRng);
	rbm.structure().setStructure(4,4);
	initRandomNormal(rbm,2);
	
	RealMatrix batch(10,4);
	RealVector weights(10);
	for(std::size_t j = 0; j != 10; ++j){
		for(std::size_t k = 0; k != 4; ++k){
			batch(j,k)=Rng::coinToss(0.5);
		}
		weights(j)=j;
	}
	double logWeightSum=std::log(sum(weights));
	
	BinaryGibbsOperator::HiddenSampleBatch hiddenBatch(10,4);
	BinaryGibbsOperator::VisibleSampleBatch visibleBatch(10,4);
	
	BinaryGibbsOperator gibbs(&rbm);
	BinaryRBM::Energy::AverageEnergyGradient grad(&rbm.structure());
	BinaryRBM::Energy::AverageEnergyGradient gradTest(&rbm.structure());
	gibbs.flags() = grad.flagsHV();
	
	hiddenBatch.state = batch;
	gibbs.precomputeVisible(hiddenBatch,visibleBatch);
	
	gradTest.addHV(hiddenBatch,visibleBatch,log(weights));
	BOOST_CHECK_CLOSE(logWeightSum,gradTest.logWeightSum(),1.e-5);
	
	//calculate ground truth data
	RealVector newWeights=weights/std::exp(logWeightSum)*10;
	for(std::size_t i = 0; i != 10; ++i){
		row(hiddenBatch.state,i)*=newWeights(i);
		row(visibleBatch.statistics.probability,i)*=newWeights(i);
	}
	grad.addHV(hiddenBatch,visibleBatch);
	
	
	RealVector diff = grad.result()-gradTest.result();
	
	BOOST_CHECK_SMALL(norm_1(diff),1.e-5);
}
//now, that we now, that the unweighted version does the same as the weighted, we check
//that the unweighted version works, by calculating the numerical gradient
class TestGradientVH : public UnsupervisedObjectiveFunction<RealVector>{
public:
	typedef BinaryRBM::Energy Energy;

	TestGradientVH(BinaryRBM* rbm): mpe_rbm(rbm){
		m_features |= HAS_FIRST_DERIVATIVE;
		m_features |= CAN_PROPOSE_STARTING_POINT;
	};

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
		Energy energy(&mpe_rbm->structure());
		
		double result=0;
		for(std::size_t i =0; i != m_data.numberOfBatches();++i){
			RealScalarVector beta(m_data.batch(i).size1(),1);
			result+=sum(energy.logUnnormalizedPropabilityVisible(m_data.batch(i),beta));
		}
		result/=m_data.numberOfElements();
		return result;
	}

	double evalDerivative( SearchPointType const & parameter, FirstOrderDerivative & derivative ) const {
		mpe_rbm->setParameterVector(parameter);
		Energy::AverageEnergyGradient gradient(&mpe_rbm->structure());
		GibbsOperator<BinaryRBM> sampler(mpe_rbm);
		sampler.flags() = gradient.flagsVH();
		
		//create Energy from RBM
		Energy energy(&mpe_rbm->structure());
		typedef Energy::VisibleType::StateSpace VisibleStateSpace;
		
		//calculate the expectation of the energy gradient with respect to the data
		for(std::size_t i=0; i != m_data.numberOfBatches(); i++){
			std::size_t currentBatchSize=m_data.batch(i).size1();
			GibbsOperator<BinaryRBM>::HiddenSampleBatch hiddenSamples(currentBatchSize,mpe_rbm->numberOfHN());
			GibbsOperator<BinaryRBM>::VisibleSampleBatch visibleSamples(currentBatchSize,mpe_rbm->numberOfVN());
		
			sampler.createSample(hiddenSamples,visibleSamples,m_data.batch(i));
			gradient.addVH(hiddenSamples, visibleSamples);
		}
		derivative = gradient.result();
		return eval(parameter);
	}
private:
	BinaryRBM* mpe_rbm;
	UnlabeledData<RealVector> m_data;
};

BOOST_AUTO_TEST_CASE( AverageEnergyGradient_DerivativeVH )
{
	BinaryRBM rbm(Rng::globalRng);
	rbm.structure().setStructure(10,16);
	initRandomNormal(rbm,2);
	RealVector parameters = rbm.parameterVector();
	initRandomNormal(rbm,2);
	
	std::vector<RealVector> dataVec(50,RealVector(10));
	for(std::size_t j = 0; j != 50; ++j){
		for(std::size_t k = 0; k != 10; ++k){
			dataVec[j](k)=Rng::coinToss(0.5);
		}
	}
	UnlabeledData<RealVector> data = createDataFromRange(dataVec,25);
	
	TestGradientVH gradient(&rbm);
	gradient.setData(data);
	
	testDerivative(gradient,parameters,1.e-5);
}

class TestGradientHV : public UnsupervisedObjectiveFunction<RealVector>{
public:
	typedef BinaryRBM::Energy Energy;

	TestGradientHV(BinaryRBM* rbm): mpe_rbm(rbm){
		m_features |= HAS_FIRST_DERIVATIVE;
		m_features |= CAN_PROPOSE_STARTING_POINT;
	};

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
		Energy energy(&mpe_rbm->structure());
		
		double result=0;
		for(std::size_t i =0; i != m_data.numberOfBatches();++i){
			RealScalarVector beta(m_data.batch(i).size1(),1);
			result+=sum(energy.logUnnormalizedPropabilityHidden(m_data.batch(i),beta));
		}
		result/=m_data.numberOfElements();
		return result;
	}

	double evalDerivative( SearchPointType const & parameter, FirstOrderDerivative & derivative ) const {
		mpe_rbm->setParameterVector(parameter);
		Energy::AverageEnergyGradient gradient(&mpe_rbm->structure());
		GibbsOperator<BinaryRBM> sampler(mpe_rbm);
		sampler.flags() = gradient.flagsHV();
		
		//create Energy from RBM
		Energy energy(&mpe_rbm->structure());
		typedef Energy::VisibleType::StateSpace VisibleStateSpace;
		
		//calculate the expectation of the energy gradient with respect to the data
		for(std::size_t i=0; i != m_data.numberOfBatches(); i++){
			std::size_t currentBatchSize=m_data.batch(i).size1();
			GibbsOperator<BinaryRBM>::HiddenSampleBatch hiddenSamples(currentBatchSize,mpe_rbm->numberOfHN());
			GibbsOperator<BinaryRBM>::VisibleSampleBatch visibleSamples(currentBatchSize,mpe_rbm->numberOfVN());
		
			hiddenSamples.state = m_data.batch(i);
			sampler.precomputeVisible(hiddenSamples,visibleSamples);
			gradient.addHV(hiddenSamples, visibleSamples);
		}
		derivative = gradient.result();
		return eval(parameter);
	}
private:
	BinaryRBM* mpe_rbm;
	UnlabeledData<RealVector> m_data;
};

BOOST_AUTO_TEST_CASE( AverageEnergyGradient_DerivativeHV )
{
	BinaryRBM rbm(Rng::globalRng);
	rbm.structure().setStructure(16,10);
	initRandomNormal(rbm,2);
	RealVector parameters = rbm.parameterVector();
	initRandomNormal(rbm,2);
	
	std::vector<RealVector> dataVec(1,RealVector(10));
	for(std::size_t j = 0; j != 1; ++j){
		for(std::size_t k = 0; k != 10; ++k){
			dataVec[j](k)=Rng::coinToss(0.5);
		}
	}
	UnlabeledData<RealVector> data = createDataFromRange(dataVec,25);
	
	TestGradientHV gradient(&rbm);
	gradient.setData(data);
	
	testDerivative(gradient,parameters,1.e-5);
}

