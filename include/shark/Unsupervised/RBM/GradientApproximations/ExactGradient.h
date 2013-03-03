/*
*  \par Copyright (c) 1998-2007:
*      Institut f&uuml;r Neuroinformatik<BR>
*      Ruhr-Universit&auml;t Bochum<BR>
*      D-44780 Bochum, Germany<BR>
*      Phone: +49-234-32-25558<BR>
*      Fax:   +49-234-32-14209<BR>
*      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
*      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
*      <BR>
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
*/
#ifndef SHARK_UNSUPERVISED_RBM_GRADIENTAPPROXIMATIONS_EXACTGRADIENT_H
#define SHARK_UNSUPERVISED_RBM_GRADIENTAPPROXIMATIONS_EXACTGRADIENT_H

#include <shark/ObjectiveFunctions/DataObjectiveFunction.h>
#include <shark/Unsupervised/RBM/Sampling/GibbsOperator.h>
#include <shark/Unsupervised/RBM/analytics.h>
#include <boost/type_traits/is_same.hpp>

namespace shark{

template<class RBMType>
class ExactGradient: public UnsupervisedObjectiveFunction<typename RBMType::VectorType>{
public:

	typedef RBMType RBM;
	typedef UnsupervisedObjectiveFunction<typename RBM::VectorType> base_type;
	typedef typename RBM::Energy Energy;
	typedef GibbsOperator<RBM> Gibbs;
	typedef typename base_type::SearchPointType SearchPointType;
	typedef typename base_type::FirstOrderDerivative FirstOrderDerivative;

	ExactGradient(RBM* rbm): mpe_rbm(rbm){
		SHARK_ASSERT(rbm != NULL);

		base_type::m_features |= base_type::HAS_FIRST_DERIVATIVE;
		base_type::m_features |= base_type::CAN_PROPOSE_STARTING_POINT;
	};


	void setData(UnlabeledData<typename RBM::VectorType> const& data){
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
	
	double eval( SearchPointType const & parameter) const {
		mpe_rbm->setParameterVector(parameter);
		return negativeLogLikelihood(*mpe_rbm,m_data)/m_data.numberOfElements();
	}

	double evalDerivative( SearchPointType const & parameter, FirstOrderDerivative & derivative ) const {
		mpe_rbm->setParameterVector(parameter);
		
		//the gradient approximation for the energy terms of the RBM		
		typename Energy::AverageEnergyGradient empiricalExpectation(&mpe_rbm->structure());
		typename Energy::AverageEnergyGradient modelExpectation(&mpe_rbm->structure());

		Gibbs gibbsSampler(mpe_rbm);
		gibbsSampler.flags() = empiricalExpectation.flagsVH();
		
		//calculate the expectation of the energy gradient with respect to the data
		Energy energy(&mpe_rbm->structure());
		double negLogLikelihood = 0;
		BOOST_FOREACH(RealMatrix const& batch,m_data.batches()) {
			std::size_t currentBatchSize = batch.size1();
			typename Gibbs::HiddenSampleBatch hiddenSamples(currentBatchSize,mpe_rbm->numberOfHN());
			typename Gibbs::VisibleSampleBatch visibleSamples(currentBatchSize,mpe_rbm->numberOfVN());
			RealScalarVector betaBatch(currentBatchSize,1);
		
			gibbsSampler.createSample(hiddenSamples,visibleSamples,batch);
			empiricalExpectation.addVH(hiddenSamples, visibleSamples);
			negLogLikelihood -= sum(energy.logUnnormalizedPropabilityVisible(batch,hiddenSamples.input,betaBatch));
		}
		
		//calculate the expectation of the energy gradient with respect to the model distribution
		if(mpe_rbm->numberOfVN() < mpe_rbm->numberOfHN()){
			integrateOverVisible(modelExpectation);
		}
		else{
			integrateOverHidden(modelExpectation);
		}
		
		derivative.m_gradient.resize(mpe_rbm->numberOfParameters());
		noalias(derivative.m_gradient) = modelExpectation.result() - empiricalExpectation.result();
	
		m_logPartition = modelExpectation.logWeightSum();
		negLogLikelihood/=m_data.numberOfElements();
		negLogLikelihood += m_logPartition;
		return negLogLikelihood;
	}

	long double getLogPartition(){
		return m_logPartition;
	}

private:
	RBM* mpe_rbm;
	
	//batchwise loops over all hidden units to calculate the gradient as well as partition
	template<class GradientApproximator>//mostly dummy right now
	void integrateOverVisible(GradientApproximator & modelExpectation) const{
		
		Gibbs sampler(mpe_rbm);
		sampler.flags() = modelExpectation.flagsVH();
		
		typedef typename Energy::VisibleType::StateSpace VisibleStateSpace;
		std::size_t values = VisibleStateSpace::numberOfStates(mpe_rbm->numberOfVN());
		std::size_t batchSize = std::min(values, std::size_t(256));
		
		Energy energy(&mpe_rbm->structure());
		for (std::size_t x = 0; x < values; x+=batchSize) {
			//create batch of states
			std::size_t currentBatchSize=std::min(batchSize,values-x);
			typename Batch<typename RBMType::VectorType>::type stateBatch(currentBatchSize,mpe_rbm->numberOfVN());
			for(std::size_t elem = 0; elem != currentBatchSize;++elem){
				//generation of the x+elem-th state vector
				VisibleStateSpace::state(row(stateBatch,elem),x+elem);
			}
			
			RealScalarVector beta(currentBatchSize,1);
			//create sample from state batch
			typename Gibbs::HiddenSampleBatch hiddenBatch(currentBatchSize,mpe_rbm->numberOfHN());
			typename Gibbs::VisibleSampleBatch visibleBatch(currentBatchSize,mpe_rbm->numberOfVN());
			sampler.createSample(hiddenBatch,visibleBatch,stateBatch);
			
			//calculate probabilities and update 
			RealVector logP = energy.logUnnormalizedPropabilityVisible(stateBatch,hiddenBatch.input,beta);
			modelExpectation.addVH(hiddenBatch, visibleBatch, logP);
			//std::cout<<exp(logP)<<" "<<std::exp(modelExpectation.logWeightSum())<<std::endl;;
			
		}
	}
	
	//batchwise loops over all hidden units to calculate the gradient as well as partition
	template<class GradientApproximator>//mostly dummy right now
	void integrateOverHidden(GradientApproximator & modelExpectation) const{
		
		Gibbs sampler(mpe_rbm);
		sampler.flags() = modelExpectation.flagsHV();
		
		typedef typename Energy::HiddenType::StateSpace HiddenStateSpace;
		std::size_t values = HiddenStateSpace::numberOfStates(mpe_rbm->numberOfHN());
		std::size_t batchSize = std::min(values, std::size_t(256) );
		
		Energy energy(&mpe_rbm->structure());
		for (std::size_t x = 0; x < values; x+=batchSize) {
			//create batch of states
			std::size_t currentBatchSize=std::min(batchSize,values-x);
			typename Batch<typename RBMType::VectorType>::type stateBatch(currentBatchSize,mpe_rbm->numberOfHN());
			for(std::size_t elem = 0; elem != currentBatchSize;++elem){
				//generation of the x+elem-th state vector
				HiddenStateSpace::state(row(stateBatch,elem),x+elem);
			}
			
			RealScalarVector beta(currentBatchSize,1);
			//create sample from state batch
			typename Gibbs::HiddenSampleBatch hiddenBatch(currentBatchSize,mpe_rbm->numberOfHN());
			typename Gibbs::VisibleSampleBatch visibleBatch(currentBatchSize,mpe_rbm->numberOfVN());
			hiddenBatch.state=stateBatch;
			sampler.precomputeVisible(hiddenBatch,visibleBatch, beta);
			
			//calculate probabilities and update 
			RealVector logP = energy.logUnnormalizedPropabilityHidden(stateBatch,visibleBatch.input,beta);
			modelExpectation.addHV(hiddenBatch, visibleBatch, logP);
		}
	}

	UnlabeledData<typename RBM::VectorType> m_data;

	mutable double m_logPartition; //the partition function of the model distribution
};	
	
}

#endif
