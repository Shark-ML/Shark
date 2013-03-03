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
#ifndef SHARK_UNSUPERVISED_RBM_GRADIENTAPPROXIMATIONS_CONTRASTIVEDIVERGENCE_H
#define SHARK_UNSUPERVISED_RBM_GRADIENTAPPROXIMATIONS_CONTRASTIVEDIVERGENCE_H

#include <shark/ObjectiveFunctions/DataObjectiveFunction.h>
#include <shark/Unsupervised/RBM/Tags.h>
#include <shark/Unsupervised/RBM/Tags.h>

namespace shark{

/// \brief Implements k-step Contrastive Divergence described by Hinton et all. (2006).
///
/// k-step Contrastive Divergence approximates the gradient by initializing a Markov
/// chain with a training example and than samples one new sample based on this for the mean of the distribution
template<class Operator>	
class ContrastiveDivergence: public UnsupervisedObjectiveFunction<typename Operator::RBM::VectorType>{
public:
	typedef typename Operator::RBM RBM;
	typedef UnsupervisedObjectiveFunction<typename RBM::VectorType> base_type;
	typedef typename base_type::SearchPointType SearchPointType;
	typedef typename base_type::FirstOrderDerivative FirstOrderDerivative;
	typedef typename RBM::Energy Energy;
	
	
	ContrastiveDivergence(RBM* rbm):mpe_rbm(rbm),m_operator(rbm),m_k(1){
		SHARK_ASSERT(rbm != NULL);

		base_type::m_features.reset(base_type::HAS_VALUE);
		base_type::m_features |= base_type::HAS_FIRST_DERIVATIVE;
		base_type::m_features |= base_type::CAN_PROPOSE_STARTING_POINT;
	};

        void setData(UnlabeledData<typename RBM::VectorType> const& data){
		m_data = data;
		typename Energy::AverageEnergyGradient flagHelper(&mpe_rbm->structure());
		m_operator.flags() = flagHelper.flagsVH();
	}

	void configure(PropertyTree const& node){
		PropertyTree::const_assoc_iterator it = node.find("rbm");
		if(it!=node.not_found())
		{
			mpe_rbm->structure().configure(it->second);
		}
		it = node.find("sampling");
		if(it!=node.not_found())
		{
			m_operator.configure(it->second);
		}
		setK(node.get<unsigned int>("k",1));
	}
	
	void setK(unsigned int k){
		m_k = k;
	}

	void proposeStartingPoint(SearchPointType& startingPoint) const{
		startingPoint = mpe_rbm->structure().parameterVector();
	}
	
	std::size_t numberOfVariables()const{
		return mpe_rbm->numberOfParameters();
	}
	
	double evalDerivative( SearchPointType const & parameter, FirstOrderDerivative & derivative ) const{
		mpe_rbm->structure().setParameterVector(parameter);
		
		typename Energy::AverageEnergyGradient empiricalAverage(&mpe_rbm->structure());
		typename Energy::AverageEnergyGradient modelAverage(&mpe_rbm->structure());
	
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
				if( step != m_k-1 || m_operator.flags() & StoreHiddenState){
					m_operator.sampleHidden(hiddenBatch);
				}
			}
			modelAverage.addVH(hiddenBatch,visibleBatch);
		}
		
		derivative.m_gradient.resize(mpe_rbm->structure().numberOfParameters());
		noalias(derivative.m_gradient) = modelAverage.result() - empiricalAverage.result();
	
		return std::numeric_limits<double>::quiet_NaN();
	}

private:	
	UnlabeledData<typename RBM::VectorType> m_data;
	RBM* mpe_rbm;
	Operator m_operator;
	unsigned int m_k;
};	
	
}

#endif
