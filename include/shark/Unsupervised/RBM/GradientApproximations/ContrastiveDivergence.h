/*
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
#ifndef SHARK_UNSUPERVISED_RBM_GRADIENTAPPROXIMATIONS_CONTRASTIVEDIVERGENCE_H
#define SHARK_UNSUPERVISED_RBM_GRADIENTAPPROXIMATIONS_CONTRASTIVEDIVERGENCE_H

#include <shark/ObjectiveFunctions/DataObjectiveFunction.h>
#include <shark/Unsupervised/RBM/Energy.h>

namespace shark{

/// \brief Implements k-step Contrastive Divergence described by Hinton et al. (2006).
///
/// k-step Contrastive Divergence approximates the gradient by initializing a Gibbs
/// chain with a training example and run it for k steps. 
/// The sample gained after k steps than samples is than used to approximate the mean of the RBM distribution in the gradient.
template<class Operator>	
class ContrastiveDivergence: public UnsupervisedObjectiveFunction<RealVector>{
private:
	typedef UnsupervisedObjectiveFunction<RealVector> base_type;
public:
	typedef typename Operator::RBM RBM;
	typedef typename base_type::SearchPointType SearchPointType;
	typedef typename base_type::FirstOrderDerivative FirstOrderDerivative;
	
	
	/// \brief The constructor 
	///
	///@param rbm pointer to the RBM which shell be trained 
	ContrastiveDivergence(RBM* rbm):mpe_rbm(rbm),m_operator(rbm),m_k(1){
		SHARK_ASSERT(rbm != NULL);

		base_type::m_features.reset(base_type::HAS_VALUE);
		base_type::m_features |= base_type::HAS_FIRST_DERIVATIVE;
		base_type::m_features |= base_type::CAN_PROPOSE_STARTING_POINT;
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
	
	/// \brief Gives the CD-k approximation of the log-likelihood gradient.
    ///
	/// @param parameter the actual parameters of the RBM
    /// @param derivative holds later the CD-k approximation of the log-likelihood gradient
	double evalDerivative( SearchPointType const & parameter, FirstOrderDerivative & derivative ) const{
		mpe_rbm->setParameterVector(parameter);
		
		AverageEnergyGradient<RBM> empiricalAverage(mpe_rbm);
		AverageEnergyGradient<RBM> modelAverage(mpe_rbm);
	
		BOOST_FOREACH(RealMatrix const& batch,m_data.batches()) {
			//create the batches for evaluation
			typename Operator::HiddenSampleBatch hiddenBatch(batch.size1(),mpe_rbm->numberOfHN());
			typename Operator::VisibleSampleBatch visibleBatch(batch.size1(),mpe_rbm->numberOfVN());
			
			m_operator.createSample(hiddenBatch,visibleBatch,batch);
			empiricalAverage.addVH(hiddenBatch,visibleBatch);
			
			for(std::size_t step = 0; step != m_k; ++step){
				m_operator.precomputeVisible(hiddenBatch, visibleBatch);
				m_operator.sampleVisible(visibleBatch);
				m_operator.precomputeHidden(hiddenBatch, visibleBatch);
				if( step != m_k-1){
					m_operator.sampleHidden(hiddenBatch);
				}
			}
			modelAverage.addVH(hiddenBatch,visibleBatch);
		}
		
		derivative.resize(mpe_rbm->numberOfParameters());
		noalias(derivative) = modelAverage.result() - empiricalAverage.result();
	
		return std::numeric_limits<double>::quiet_NaN();
	}

private:	
	UnlabeledData<RealVector> m_data;
	RBM* mpe_rbm;
	Operator m_operator;
	unsigned int m_k;
};	
	
}

#endif
