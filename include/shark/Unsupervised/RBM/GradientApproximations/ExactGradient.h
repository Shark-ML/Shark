/*!
 * 
 *
 * \brief       -
 *
 * \author      -
 * \date        -
 *
 *
 * \par Copyright 1995-2015 Shark Development Team
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
#ifndef SHARK_UNSUPERVISED_RBM_GRADIENTAPPROXIMATIONS_EXACTGRADIENT_H
#define SHARK_UNSUPERVISED_RBM_GRADIENTAPPROXIMATIONS_EXACTGRADIENT_H

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/Unsupervised/RBM/Sampling/GibbsOperator.h>
#include <shark/Unsupervised/RBM/analytics.h>
#include <boost/type_traits/is_same.hpp>

namespace shark{

template<class RBMType>
class ExactGradient: public SingleObjectiveFunction{
private:
	typedef GibbsOperator<RBMType> Gibbs;
public:
	typedef RBMType RBM;

	ExactGradient(RBM* rbm): mpe_rbm(rbm),m_regularizer(0){
		SHARK_ASSERT(rbm != NULL);

		m_features |= HAS_FIRST_DERIVATIVE;
		m_features |= CAN_PROPOSE_STARTING_POINT;
	};

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "ExactGradient"; }

	void setData(UnlabeledData<RealVector> const& data){
		m_data = data;
	}
	
	void configure(PropertyTree const& node){
		PropertyTree::const_assoc_iterator it = node.find("rbm");
		if(it!=node.not_found())
		{
			mpe_rbm->configure(it->second);
		}
	}

	void proposeStartingPoint(SearchPointType& startingPoint) const{
		startingPoint = mpe_rbm->parameterVector();
	}
	
	std::size_t numberOfVariables()const{
		return mpe_rbm->numberOfParameters();
	}
	
	void setRegularizer(double factor, SingleObjectiveFunction* regularizer){
		m_regularizer = regularizer;
		m_regularizationStrength = factor;
	}
	
	double eval( SearchPointType const & parameter) const {
		mpe_rbm->setParameterVector(parameter);
		
		double negLogLikelihood = negativeLogLikelihood(*mpe_rbm,m_data)/m_data.numberOfElements();
		if(m_regularizer){
			negLogLikelihood += m_regularizationStrength * m_regularizer->eval(parameter);
		}
		return negLogLikelihood;
	}

	double evalDerivative( SearchPointType const & parameter, FirstOrderDerivative & derivative ) const {
		mpe_rbm->setParameterVector(parameter);
		
		//the gradient approximation for the energy terms of the RBM		
		AverageEnergyGradient<RBM> empiricalExpectation(mpe_rbm);
		AverageEnergyGradient<RBM> modelExpectation(mpe_rbm);

		Gibbs gibbsSampler(mpe_rbm);
		
		//calculate the expectation of the energy gradient with respect to the data
		double negLogLikelihood = 0;
		BOOST_FOREACH(RealMatrix const& batch,m_data.batches()) {
			std::size_t currentBatchSize = batch.size1();
			typename Gibbs::HiddenSampleBatch hiddenSamples(currentBatchSize,mpe_rbm->numberOfHN());
			typename Gibbs::VisibleSampleBatch visibleSamples(currentBatchSize,mpe_rbm->numberOfVN());
		
			gibbsSampler.createSample(hiddenSamples,visibleSamples,batch);
			empiricalExpectation.addVH(hiddenSamples, visibleSamples);
			negLogLikelihood -= sum(mpe_rbm->energy().logUnnormalizedProbabilityVisible(
				batch,hiddenSamples.input,blas::repeat(1,currentBatchSize)
			));
		}
		
		//calculate the expectation of the energy gradient with respect to the model distribution
		if(mpe_rbm->numberOfVN() < mpe_rbm->numberOfHN()){
			integrateOverVisible(modelExpectation);
		}
		else{
			integrateOverHidden(modelExpectation);
		}
		
		derivative.resize(mpe_rbm->numberOfParameters());
		noalias(derivative) = modelExpectation.result() - empiricalExpectation.result();
	
		m_logPartition = modelExpectation.logWeightSum();
		negLogLikelihood/=m_data.numberOfElements();
		negLogLikelihood += m_logPartition;
		
		if(m_regularizer){
			FirstOrderDerivative regularizerDerivative;
			negLogLikelihood += m_regularizationStrength * m_regularizer->evalDerivative(parameter,regularizerDerivative);
			noalias(derivative) += m_regularizationStrength * regularizerDerivative;
		}
		
		return negLogLikelihood;
	}

	long double getLogPartition(){
		return m_logPartition;
	}

private:
	RBM* mpe_rbm;

	SingleObjectiveFunction* m_regularizer;
	double m_regularizationStrength;
	
	//batchwise loops over all hidden units to calculate the gradient as well as partition
	template<class GradientApproximator>//mostly dummy right now
	void integrateOverVisible(GradientApproximator & modelExpectation) const{
		
		Gibbs sampler(mpe_rbm);
		
		typedef typename RBM::VisibleType::StateSpace VisibleStateSpace;
		std::size_t values = VisibleStateSpace::numberOfStates(mpe_rbm->numberOfVN());
		std::size_t batchSize = std::min(values, std::size_t(256));
		
		for (std::size_t x = 0; x < values; x+=batchSize) {
			//create batch of states
			std::size_t currentBatchSize=std::min(batchSize,values-x);
			typename Batch<RealVector>::type stateBatch(currentBatchSize,mpe_rbm->numberOfVN());
			for(std::size_t elem = 0; elem != currentBatchSize;++elem){
				//generation of the x+elem-th state vector
				VisibleStateSpace::state(row(stateBatch,elem),x+elem);
			}
			
			//create sample from state batch
			typename Gibbs::HiddenSampleBatch hiddenBatch(currentBatchSize,mpe_rbm->numberOfHN());
			typename Gibbs::VisibleSampleBatch visibleBatch(currentBatchSize,mpe_rbm->numberOfVN());
			sampler.createSample(hiddenBatch,visibleBatch,stateBatch);
			
			//calculate probabilities and update 
			RealVector logP = mpe_rbm->energy().logUnnormalizedProbabilityVisible(
				stateBatch,hiddenBatch.input,blas::repeat(1,currentBatchSize)
			);
			modelExpectation.addVH(hiddenBatch, visibleBatch, logP);
		}
	}
	
	//batchwise loops over all hidden units to calculate the gradient as well as partition
	template<class GradientApproximator>//mostly dummy right now
	void integrateOverHidden(GradientApproximator & modelExpectation) const{
		
		Gibbs sampler(mpe_rbm);
		
		typedef typename RBM::HiddenType::StateSpace HiddenStateSpace;
		std::size_t values = HiddenStateSpace::numberOfStates(mpe_rbm->numberOfHN());
		std::size_t batchSize = std::min(values, std::size_t(256) );
		
		for (std::size_t x = 0; x < values; x+=batchSize) {
			//create batch of states
			std::size_t currentBatchSize=std::min(batchSize,values-x);
			typename Batch<RealVector>::type stateBatch(currentBatchSize,mpe_rbm->numberOfHN());
			for(std::size_t elem = 0; elem != currentBatchSize;++elem){
				//generation of the x+elem-th state vector
				HiddenStateSpace::state(row(stateBatch,elem),x+elem);
			}
			
			//create sample from state batch
			typename Gibbs::HiddenSampleBatch hiddenBatch(currentBatchSize,mpe_rbm->numberOfHN());
			typename Gibbs::VisibleSampleBatch visibleBatch(currentBatchSize,mpe_rbm->numberOfVN());
			hiddenBatch.state=stateBatch;
			sampler.precomputeVisible(hiddenBatch,visibleBatch, blas::repeat(1,currentBatchSize));
			
			//calculate probabilities and update 
			RealVector logP = mpe_rbm->energy().logUnnormalizedProbabilityHidden(
				stateBatch,visibleBatch.input,blas::repeat(1,currentBatchSize)
			);
			modelExpectation.addHV(hiddenBatch, visibleBatch, logP);
		}
	}

	UnlabeledData<RealVector> m_data;

	mutable double m_logPartition; //the partition function of the model distribution
};	
	
}

#endif
