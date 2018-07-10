/*!
 * 
 *
 * \brief       -
 *
 * \author      -
 * \date        -
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
#ifndef SHARK_UNSUPERVISED_RBM_GRADIENTAPPROXIMATIONS_CONTRASTIVEDIVERGENCE_H
#define SHARK_UNSUPERVISED_RBM_GRADIENTAPPROXIMATIONS_CONTRASTIVEDIVERGENCE_H

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/Unsupervised/RBM/Energy.h>
#include <shark/Core/Threading/Algorithms.h>

namespace shark{

/// \brief Implements k-step Contrastive Divergence described by Hinton et al. (2006).
///
/// k-step Contrastive Divergence approximates the gradient by initializing a Gibbs
/// chain with a training example and run it for k steps. 
/// The sample gained after k steps than samples is than used to approximate the mean of the RBM distribution in the gradient.
template<class Operator>	
class ContrastiveDivergence: public SingleObjectiveFunction{
public:
	typedef typename Operator::RBM RBM;
	
	/// \brief The constructor 
	///
	///@param rbm pointer to the RBM which shell be trained 
	ContrastiveDivergence(RBM* rbm)
	: mpe_rbm(rbm),m_operator(rbm)
	, m_k(1), m_numBatches(0),m_regularizer(0){
		SHARK_ASSERT(rbm != NULL);

		m_features.reset(HAS_VALUE);
		m_features |= HAS_FIRST_DERIVATIVE;
		m_features |= CAN_PROPOSE_STARTING_POINT;
	};

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "ContrastiveDivergence"; }

	/// \brief Sets the training batch.
	///
	/// @param data the batch of training data
	void setData(Data<RealVector> const& data){
		m_data = data;
	}
	
	/// \brief Sets the value of k- the number of steps of the Gibbs Chain 
	///
	/// @param k  the number of steps
	void setK(unsigned int k){
		m_k = k;
	}

	SearchPointType proposeStartingPoint() const{
		return  mpe_rbm->parameterVector();
	}
	
	/// \brief Returns the number of variables of the RBM.
	///
	/// @return the number of variables of the RBM
	std::size_t numberOfVariables()const{
		return mpe_rbm->numberOfParameters();
	}
	
	/// \brief Returns the number of batches of the dataset that are used in every iteration.
	///
	/// If it is less than all batches, the batches are chosen at random. if it is 0, all batches are used
	std::size_t numBatches()const{
		return m_numBatches;
	}
	
	/// \brief Returns a reference to the number of batches of the dataset that are used in every iteration.
	///
	/// If it is less than all batches, the batches are chosen at random.if it is 0, all batches are used.
	std::size_t& numBatches(){
		return m_numBatches;
	}
	
	void setRegularizer(double factor, SingleObjectiveFunction* regularizer){
		m_regularizer = regularizer;
		m_regularizationStrength = factor;
	}
	
	/// \brief Gives the CD-k approximation of the log-likelihood gradient.
	///
	/// @param parameter the actual parameters of the RBM
	/// @param derivative holds later the CD-k approximation of the log-likelihood gradient
	double evalDerivative( SearchPointType const & parameter, FirstOrderDerivative & derivative ) const{
		mpe_rbm->setParameterVector(parameter);
		derivative.resize(mpe_rbm->numberOfParameters());
		derivative.clear();
		
		std::size_t batchesForTraining = m_numBatches > 0? m_numBatches: m_data.size();
		std::size_t elements = 0;
		//get the batches for this iteration
		std::vector<std::size_t> batchIds(m_data.size());
		{
			for(std::size_t i = 0; i != m_data.size(); ++i){
				batchIds[i] = i;
			}
			std::shuffle(batchIds.begin(),batchIds.end(), random::globalRng());
			batchIds.erase(batchIds.begin() + batchesForTraining, batchIds.end());
			for(std::size_t i = 0; i != batchesForTraining; ++i){
				elements += m_data[batchIds[i]].size1();
			}
		}
		
		auto map = [&](std::size_t i){//maps the ith batch to its gradient
			typename RBM::GradientType empiricalAverage(mpe_rbm);
			typename RBM::GradientType modelAverage(mpe_rbm);
		
			RealMatrix const& batch = m_data[i];
			
			//create the batches for evaluation
			typename Operator::HiddenSample hiddenBatch(batch.size1(),mpe_rbm->numberOfHN());
			typename Operator::VisibleSample visibleBatch(batch.size1(),mpe_rbm->numberOfVN());
			
			visibleBatch.state = batch;
			m_operator.precomputeHidden(hiddenBatch,visibleBatch,blas::repeat(1.0,batch.size1()));
			m_operator.sampleHidden(hiddenBatch);
			empiricalAverage.addVH(hiddenBatch,visibleBatch);
			
			for(std::size_t step = 0; step != m_k; ++step){
				m_operator.precomputeVisible(hiddenBatch, visibleBatch,blas::repeat(1.0,batch.size1()));
				m_operator.sampleVisible(visibleBatch);
				m_operator.precomputeHidden(hiddenBatch, visibleBatch,blas::repeat(1.0,batch.size1()));
				if( step != m_k-1){
					m_operator.sampleHidden(hiddenBatch);
				}
			}
			modelAverage.addVH(hiddenBatch,visibleBatch);
			double weight = batch.size1()/double(elements);
			return RealVector(weight*(modelAverage.result() - empiricalAverage.result()));
		};
		derivative = threading::mapAccumulate( batchIds, std::move(derivative), map, threading::globalThreadPool());
		
		if(m_regularizer){
			FirstOrderDerivative regularizerDerivative;
			m_regularizer->evalDerivative(parameter,regularizerDerivative);
			noalias(derivative) += m_regularizationStrength*regularizerDerivative;
		}
		
		return std::numeric_limits<double>::quiet_NaN();
	}

private:	
	Data<RealVector> m_data;
	RBM* mpe_rbm;
	Operator m_operator;
	unsigned int m_k;
	std::size_t m_numBatches;///< number of batches used in every iteration. 0 means all.

	SingleObjectiveFunction* m_regularizer;
	double m_regularizationStrength;
};	
	
}

#endif
