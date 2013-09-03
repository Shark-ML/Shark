//===========================================================================
/*!
 *  \brief Offers the functions to create and to work with a
 *         recurrent neural network.
 *
 *  \author  O. Krause
 *  \date    2010
 *
 *  \par Copyright (c) 1999-2001:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-27974<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *
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
 *  
 */
#ifndef SHARK_MODELS_RNNET_H
#define SHARK_MODELS_RNNET_H

#include <shark/Models/AbstractModel.h>
#include <shark/Models/RecurrentStructure.h>

namespace shark{

//!  \brief A recurrent neural network regression model that learns
//!  with Back Propagation Through Time
//!
//!  This class defines a recurrent neural network regression
//!  model. Its inputs and output types are Matrices which represet
//!  sequences of inputs. The gradient is calculated via
//!  BackPropagationTroughTime (BPTT).
//!
//! The inputs of this Network are not sigmoidal, but the hidden and output
//! neurons are.
//!
//!  This class is optimized for batch learning. See OnlineRNNet for an online
//!  version.
class RNNet:public AbstractModel<Sequence,Sequence >
{
private:
	struct InternalState: public State{
		//! Activation of the neurons after processing the time series.
		//! m_timeActivation(b,t,i) is a 3-dimensional array, the first dimension
		//! returns the i-th element of the batch, the second dimension returns
		//! the activation for timestep t, the third dimension the activation
		//! of the neuron at the timestep of the batch element.
		std::vector<Sequence> timeActivation;
	};
public:

	//! creates a neural network with a potentially shared structure
	//! \param structure the structure of this neural network. It can be shared between multiple instances or with then
	//!                  online version of this net.
	RNNet(RecurrentStructure* structure):mpe_structure(structure){
		SHARK_CHECK(mpe_structure,"[RNNet] structure is not allowed to be empty");
		m_features|=HAS_FIRST_PARAMETER_DERIVATIVE;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "RNNet"; }

	//!  \brief Sets the warm up sequence
	//!
	//!  Usually, when processing a new data series all the
	//!  `states' of the network are reset to zero. By `states' I mean the
	//!  buffered activations to which time-delayed synapses refer
	//!  to. Effectively, this means one assumes a zero activation history.
	//!
	//!  The advantage of this is, that it makes the model behavior well
	//!  defined. The disadvantage is that you can't predict a time series
	//!  well with a zero history. Thus, one should use a data series to
	//!  initialize the network, i.e., to let it converge into a `normal'
	//!  dynamic state from which prediction of new data is possible.
	//!  This phase is called the warmup phase.
	//!
	//! With this method, the warm up sequence can be set, which is then used
	//! during the warm up phase.
	//!
	//!  \param warmUpSequence the warm up sequence used before each batch of data. The
	//!                        default is an empty sequence
	void setWarmUpSequence(Sequence const& warmUpSequence = Sequence()){
		m_warmUpSequence = warmUpSequence;
	}
	
	boost::shared_ptr<State> createState()const{
		return boost::shared_ptr<State>(new InternalState());
	}

	//!  \brief Feed a data series to the model. The output (i.e., the time
	//!  series of activations of the output neurons) it copied into the
	//!  output buffer.
	//!
	//!  \param  pattern  batch of timeseries for the network.
	//!  \param  output Used to store the outputs of the network.
	//!  \param  state stores additional information which can be reused for the computation of the derivative
	void eval(BatchInputType const& pattern, BatchOutputType& output, State& state)const;
	using AbstractModel<Sequence,Sequence>::eval;
	
	/// obtain the input dimension
	std::size_t inputSize() const{
		return mpe_structure->inputs();
	}

	/// obtain the output dimension
	std::size_t outputSize() const{
		return mpe_structure->outputs();
	}
	
	//!\brief calculates the weighted sum of gradients w.r.t the parameters
	//!
	//!The RNNet uses internally BPTT to calculate the gradient.
	//! Stores the BPTT error values for the calculation
	//! of the gradient.
	//!
	//! Given the gradient of the loss function \f$ \frac{\delta L(t)}{\delta y_i(t)}\f$,
	//! the BPTT error is calculated as
	//!\f[ \frac{\delta E}{\delta y_i(t)}= \mu_i \frac{\delta L(t)}{\delta y_i(t)}
	//! +\sum_{j=1}^N \frac{\delta E}{\delta y_i(t+1)} y_i'(t+1) w^R_{ij} \f]
	//! Where \f$ L \f$ is the loss, \f$ y_i \f$ the ith neuron and
	//! \f$ w^R_ij\f$ is the recurrent weight of the connection from neuron i to j.
	//! The factor \f$ \mu_i \f$ is one of the neuron is an output neuron, else zero.
	//!
	//! \todo expand documentation
	//!
	//! \param patterns the batch of patterns to evaluate
	//! \param coefficients the coefficients which are used to calculate the weighted sum
	//! \param state the last state stord during eval
	//! \param gradient the calculated gradient
	void weightedParameterDerivative(
		BatchInputType const& patterns, BatchInputType const& coefficients,  State const& state, 
		RealVector& gradient
	)const;
	
	//! get internal parameters of the model
	RealVector parameterVector() const{
		return mpe_structure->parameterVector();
	}
	
	//! set internal parameters of the model
	//! \param newParameters the new parameters of the model. this changes the internal referenced RecurrentStructure
	void setParameterVector(RealVector const& newParameters){
		mpe_structure->setParameterVector(newParameters);
	}

	//!number of parameters of the network
	std::size_t numberOfParameters() const{
		return mpe_structure->parameters();
	}
protected:
	//! the warm up sequence of the network
	Sequence m_warmUpSequence;

	//! the topology of the network.
	RecurrentStructure* mpe_structure;

	RealMatrix m_errorDerivative;
};
}

#endif //RNNET_H









