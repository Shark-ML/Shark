/*!
 * 
 *
 * \brief       -
 *
 * \author      -
 * \date        -
 *
 *
 * \par Copyright 1995-2014 Shark Development Team
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
#ifndef SHARK_UNSUPERVISED_RBM_GRADIENTAPPROXIMATIONS_CONTRASTIVEDIVERGENCE_H
#define SHARK_UNSUPERVISED_RBM_GRADIENTAPPROXIMATIONS_CONTRASTIVEDIVERGENCE_H

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/Unsupervised/RBM/Energy.h>

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
	void setData(UnlabeledData<RealVector> const& data){
		m_data = data;
	}

	void configure(PropertyTree const& node){
		PropertyTree::const_assoc_iterator it = node.find("rbm");
		if(it!=node.not_found())
		{
			mpe_rbm->configure(it->second);
		}
		it = node.find("sampling");
		if(it!=node.not_found())
		{
			m_operator.configure(it->second);
		}
		setK(node.get<unsigned int>("k",1));
	}
	
	/// \brief Sets the value of k- the number of steps of the Gibbs Chain 
	///
	/// @param k  the number of steps
	void setK(unsigned int k){
		m_k = k;
	}

	void proposeStartingPoint(SearchPointType& startingPoint) const{
		startingPoint = mpe_rbm->parameterVector();
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
		
		std::size_t batchesForTraining = m_numBatches > 0? m_numBatches: m_data.numberOfBatches();
		std::size_t elements = 0;
		//get the batches for this iteration
		std::vector<std::size_t> batchIds(m_data.numberOfBatches());
		{
			for(std::size_t i = 0; i != m_data.numberOfBatches(); ++i){
				batchIds[i] = i;
			}
			DiscreteUniform<typename RBM::RngType> uni(mpe_rbm->rng(),0,1);
			std::random_shuffle(batchIds.begin(),batchIds.end(),uni);
			for(std::size_t i = 0; i != batchesForTraining; ++i){
				elements += m_data.batch(batchIds[i]).size1();
			}
		}
		
		std::size_t threads = std::min<std::size_t>(batchesForTraining,SHARK_NUM_THREADS);
		std::size_t numBatches = batchesForTraining/threads;
		
		
		SHARK_PARALLEL_FOR(int t = 0; t < (int)threads; ++t){
			AverageEnergyGradient<RBM> empiricalAverage(mpe_rbm);
			AverageEnergyGradient<RBM> modelAverage(mpe_rbm);
			
			std::size_t threadElements = 0;
			
			std::size_t batchStart = t*numBatches;
			std::size_t batchEnd = (t== (int)threads-1)? batchesForTraining : batchStart+numBatches;
			for(std::size_t i = batchStart; i != batchEnd; ++i){
				RealMatrix const& batch = m_data.batch(batchIds[i]);
				threadElements += batch.size1();
				
				//create the batches for evaluation
				typename Operator::HiddenSampleBatch hiddenBatch(batch.size1(),mpe_rbm->numberOfHN());
				typename Operator::VisibleSampleBatch visibleBatch(batch.size1(),mpe_rbm->numberOfVN());
				
				visibleBatch.state = batch;
				m_operator.precomputeHidden(hiddenBatch,visibleBatch,blas::repeat(1.0,batch.size1()));
				SHARK_CRITICAL_REGION{
					m_operator.sampleHidden(hiddenBatch);
				}
				empiricalAverage.addVH(hiddenBatch,visibleBatch);
				
				for(std::size_t step = 0; step != m_k; ++step){
					m_operator.precomputeVisible(hiddenBatch, visibleBatch,blas::repeat(1.0,batch.size1()));
					SHARK_CRITICAL_REGION{
						m_operator.sampleVisible(visibleBatch);
					}
					m_operator.precomputeHidden(hiddenBatch, visibleBatch,blas::repeat(1.0,batch.size1()));
					if( step != m_k-1){
						SHARK_CRITICAL_REGION{
							m_operator.sampleHidden(hiddenBatch);
						}
					}
				}
				modelAverage.addVH(hiddenBatch,visibleBatch);
			}
			SHARK_CRITICAL_REGION{
				double weight = threadElements/double(elements);
				noalias(derivative) += weight*(modelAverage.result() - empiricalAverage.result());
			}
			
		}
		
		if(m_regularizer){
			FirstOrderDerivative regularizerDerivative;
			m_regularizer->evalDerivative(parameter,regularizerDerivative);
			noalias(derivative) += m_regularizationStrength*regularizerDerivative;
		}
		
		return std::numeric_limits<double>::quiet_NaN();
	}

private:	
	UnlabeledData<RealVector> m_data;
	RBM* mpe_rbm;
	Operator m_operator;
	unsigned int m_k;
	std::size_t m_numBatches;///< number of batches used in every iteration. 0 means all.

	SingleObjectiveFunction* m_regularizer;
	double m_regularizationStrength;
};	
	
}

#endif
