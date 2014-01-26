//===========================================================================
/*!
 * 
 *
 * \brief       Offers the functions to create and to work with a
 * recurrent neural network.
 * 
 * 
 *
 * \author      O. Krause
 * \date        2010
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
#ifndef SHARK_MODELS_ONLINERNNET_H
#define SHARK_MODELS_ONLINERNNET_H

#include <shark/Models/AbstractModel.h>
#include <shark/Models/RecurrentStructure.h>
namespace shark{

//!  \brief A recurrent neural network regression model optimized
//!         for online learning. 
//!
//! The OnlineRNNet can only process a single input at a time. Internally
//! it stores the last activation as well as the derivatives which get updated 
//! over the course of the sequence. Instead of feeding in the whole sequence,
//! the inputs must be given on after another. However if the whole sequence is
//! available in advance, this implementation is not advisable, since it is a lot slower
//! than RNNet which is targeted to whole sequences. 
//! 
class OnlineRNNet:public AbstractModel<RealVector,RealVector>
{
public:
	//! creates a configured neural network
	OnlineRNNet(RecurrentStructure* structure);

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "OnlineRNNet"; }

	//!  \brief Feeds a timestep of a time series to the model and
	//!         calculates it's output.
	//!
	//!  \param  pattern  Input patterns for the network.
	//!  \param  output Used to store the outputs of the network.
	void eval(RealMatrix const& pattern,RealMatrix& output);
	using AbstractModel<RealVector,RealVector>::eval;

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
	//!Uses an iterative update scheme to calculate the gradient at timestep t from the gradient
	//!at timestep t-1 using forward propagation. This Methods requires O(n^3) Memory and O(n^4) computations,
	//!where n is the number of neurons. So if the network is very large, RNNet should be used!
	//!
	//! \param pattern the pattern to evaluate
	//! \param coefficients the oefficients which are used to calculate the weighted sum
	//! \param gradient the calculated gradient
	void weightedParameterDerivative(RealMatrix const& pattern, RealMatrix const& coefficients,  RealVector& gradient);

	//! get internal parameters of the model
	RealVector parameterVector() const{
		return mpe_structure->parameterVector();
	}
	//! set internal parameters of the model
	void setParameterVector(RealVector const& newParameters){
		mpe_structure->setParameterVector(newParameters);
	}

	//!number of parameters of the network
	std::size_t numberOfParameters() const{
		return mpe_structure->parameters();
	}

	//!resets the internal state of the network.
	//!it resets the network to 0 activation and clears the derivative
	//!this method needs to be called, when a sequence ends and a new sequence is to be started
	void resetInternalState(){
		m_lastActivation.clear();
		m_activation.clear();
		m_unitGradient.clear();
	}

	//!  \brief This Method sets the activation of the output neurons
	//!
	//!  This is usefull when teacher forcing is used. When the network
	//!  is trained to predict a timeseries and diverges from the sequence
	//!  at an early stage, the resulting gradient might not be very helpfull.
	//!  In this case, teacher forcing can be applied to prevent diverging.
	//!  However, the network might become unstable, when teacher-forcing is turned off
	//!  because there is no force which prevents it from diverging anymore.
	//!
	//!  \param  activation  Input patterns for the network.
	void setOutputActivation(RealVector const& activation){
		m_activation.resize(mpe_structure->numberOfUnits());
		subrange(m_activation,mpe_structure->numberOfUnits()-outputSize(),mpe_structure->numberOfUnits()) = activation;
	}
protected:
	
	//! the topology of the network.
	RecurrentStructure* mpe_structure;

	//!the activation of the network at time t (after evaluation)
	RealVector m_activation;
	//!the activation of the network at time t-1 (before evaluation)
	RealVector m_lastActivation;

	//!\brief the gradient of the hidden units with respect to every weight
	//!
	//!The gradient \f$ \frac{\delta y_k(t)}{\delta w_{ij}} \f$ is stored in this
	//!structure. Using this gradient, the derivative of the Network can be calculated as
	//!\f[ \frac{\delta E(y(t))}{\delta w_{ij}}=\sum_{k=1}^n\frac{\delta E(y(t))}{\delta y_k} \frac{\delta y_k(t)}{\delta w_{ij}} \f]
	//!where \f$ y_k(t) \f$ is the activation of neuron \f$ k \f$ at timestep \f$ t \f$
	//!the gradient needs to be updated after every timestep using the formula
	//!\f[ \frac{\delta y_k(t+1)}{\delta w_{ij}}= y'_k(t)= \left[\sum_{l=1}^n w_{il}\frac{\delta y_l(t)}{\delta w_{ij}} +\delta_{kl}y_l(t-1)\right]\f]
	//!so if the gradient is needed, don't forget to call weightedParameterDerivative at every timestep!
	RealMatrix m_unitGradient;
};
}

#endif //RNNET_H









