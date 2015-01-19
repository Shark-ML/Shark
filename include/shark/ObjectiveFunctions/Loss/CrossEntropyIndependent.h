//===========================================================================
/*!
 * 
 *
 * \brief       Error measure for classification tasks of non exclusive attributes
 * that can be used for model training.
 * 
 * 
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

#ifndef SHARK_OBJECTIVEFUNCTIONS_LOSS_CROSS_ENTROPY_INDEPENDENT_H
#define SHARK_OBJECTIVEFUNCTIONS_LOSS_CROSS_ENTROPY_INDEPENDENT_H

#include <shark/ObjectiveFunctions/Loss/AbstractLoss.h>

namespace shark{
/*!
  *  \brief Error measure for classification tasks of non exclusive attributes
 *         that can be used for model training.
 *
 *  If your model should return a vector whose components are connected to
 *  multiple mutually independent attributes and the \em k-th component of the
 *  output vector is representing the probability of the presence of the \em k-th
 *  attribute given any input vector, 'CrossEntropyIndependent' is the adequate
 *  error measure for model-training. For \em C>1, dimension of model's output
 *  and every output dimension represents a single attribute or class respectively,
 *  it follows the formular
 *  \f[
 *      E = - \sum_{i=1}^N \sum_{k=1}^{C} \{tar^i_k \ln model_k(in^i) + (1-tar^i_k) \ln
 *          (1-model_k(in^i))\}
 *  \f]
 *  where \em i runs over all input patterns.
 *  This error functional can be derivated and so used for training. In case
 *  of only one single output dimension 'CrossEntropyIndependent' returns actually the
 *  true cross entropy for two classes, using the formalism
 *  \f[
 *      E = - \sum_{i=1}^N \{tar^i \ln model(in^i) + (1-tar^i) \ln
 *          (1-model(in^i))\}
 *  \f]
 *  For theoretical reasons it is suggested to use for neural networks
 *  the logistic sigmoid activation function at the output neurons.
 *  However, actually every sigmoid activation could be applied to the output
 *  neurons as far as the image of this function is identical to the intervall
 *  \em [0,1].
 *  In this implementation every target value to be chosen from {0,1}
 *  (binary encoding). For detailed information refer to
 *  (C.M. Bishop, Neural Networks for Pattern Recognition, Clarendon Press 1996, Chapter 6.8.)
 */
class CrossEntropyIndependent : public AbstractLoss<unsigned int,RealVector>
{
	typedef AbstractLoss<unsigned int, RealVector> base_type;

	// This code uses a different formula to compute the binary case for 1 output.
	// It should be numerically more stable.
	// formula: ln(1+exp(-yx)) with y = -1/1 
	double evalError(unsigned int target,double exponential,double value) const {
		double label = 2 * static_cast<double>(target) - 1;   // converts labels from 0/1 to -1/+1
		if(value*label < -100 ){
			//below this, we might get numeric instabilities
			//but we know, that ln(1+exp(x)) converges to x for big arguments
			return - value * label;
		}
		if(target == 0)
			exponential = 1/exponential;

		return std::log(1+exponential);
	}

public:
	CrossEntropyIndependent()
	{
		m_features |= HAS_FIRST_DERIVATIVE;
		m_features |= HAS_SECOND_DERIVATIVE;
	}


	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "CrossEntropyIndependent"; }

	// annoyingness of C++ templates
	using base_type::eval;

	double eval(unsigned int const& target, RealVector const& prediction) const
	{
		double error = 0;
		for (std::size_t c = 0; c != prediction.size(); c++){
			double exponential =  exp ( -prediction ( c ) );
			error += evalError(target,exponential,prediction ( c ));
		}

		return error;
	}

	double evalDerivative(unsigned int const& target, RealVector const& prediction, RealVector& gradient) const
	{
		gradient.resize(target.size());
		
		double error = 0;
		for (std::size_t c = 0; c < output.nelem(); c++){
			double exponential = exp ( -prediction ( c ) );
			double sigmoid = 1/ ( 1+ exponential);
			gradient ( c ) = ( sigmoid - target );
			error += evalError(target, exponential, prediction ( c ));
		}
		return error;
	}

	double evalDerivative (
			unsigned int const& target, 
			RealVector const& prediction,
			RealVector& gradient, 
			typename base_type::MatrixType& hessian) const
	{
		gradient.resize(target.size());
		hessian.resize(0,0);
		hessian.clear();

		double error = 0;
		for (std::size_t c = 0; c < output.nelem(); c++){
			double exponential = exp ( -prediction ( c ) );
			double sigmoid = 1/ ( 1+ exponential);
			gradient ( c ) = ( sigmoid - target );
			hessian ( c,c ) = std::max(0.0, sigmoid * ( 1-sigmoid ));
			error += evalError(target, exponential, prediction ( c ));
		}
		return error;
	}
};


}
#endif
