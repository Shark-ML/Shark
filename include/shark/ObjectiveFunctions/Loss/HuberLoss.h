/*!
 * 
 * \brief       Implements the Huber loss function for robust regression
 * 
 *
 * \author    Oswin Krause
 * \date        2014
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

#ifndef SHARK_OBJECTIVEFUNCTIONS_LOSS_HUBERLOSS_H
#define SHARK_OBJECTIVEFUNCTIONS_LOSS_HUBERLOSS_H

#include <shark/ObjectiveFunctions/Loss/AbstractLoss.h>

namespace shark {

/// \brief Huber-loss for for robust regression
///
/// The Huber loss is a function that is quadratic if\f$ ||f(x)-y||_2 \leq \delta \f$. 
/// Outside this region, whn the error is larger, it is defined as a linear continuation. The function is once
/// but not twice differentiable. This loss is important for regression as it weights outliers lower than
/// ordinary least squares regression while still preserving a convex shape of the loss function.
///
/// Please not that, due to its nature, the error function is not scale invariant. thus rescaling the dataset
/// changes the behaviour. This function has the hyper parameter delta which marks thee region where
/// the function changes from quadratic to linear.
class HuberLoss : public AbstractLoss<RealVector, RealVector>
{
public:
	/// constructor
	HuberLoss(double delta = 1.0):m_delta(delta){ 
		m_features |= base_type::HAS_FIRST_DERIVATIVE;
	}
	
	/// \brief Returns class name "HuberLoss"
	std::string name() const
	{ return "HuberLoss"; }


	///\brief calculates the sum of all 
	double eval(BatchLabelType const& labels, BatchOutputType const& predictions) const{
		SIZE_CHECK(labels.size1() == predictions.size1());
		SIZE_CHECK(labels.size2() == predictions.size2());
		std::size_t numInputs = labels.size1();
		
		double error = 0;
		for(std::size_t i = 0; i != numInputs;++i){
			double norm2 = norm_sqr(row(predictions,i)-row(labels,i));
			
			//check whether we are in the quadratic area
			if(norm2 <= sqr(m_delta)){
				error += 0.5*norm2;
			}
			else{
				error += m_delta*std::sqrt(norm2)-0.5*sqr(m_delta);
			}
		}
		return error;
	}

	double evalDerivative(BatchLabelType const& labels, BatchOutputType const& predictions, BatchOutputType& gradient)const{
		std::size_t numInputs = size(labels);
		std::size_t outputDim = predictions.size2();
		SIZE_CHECK(numInputs == size(predictions));
		
		gradient.resize(numInputs,outputDim);
		double error = 0;
		for(std::size_t i = 0; i != numInputs;++i){
			double norm2 = norm_sqr(row(predictions,i)-row(labels,i));
			
			//check whether we are in the quadratic area
			if(norm2 <= sqr(m_delta)){
				error += 0.5*norm2;
				noalias(row(gradient,i)) = row(predictions,i)-row(labels,i);
			}
			else{
				double norm = std::sqrt(norm2);
				error += m_delta*norm-0.5*sqr(m_delta);
				noalias(row(gradient,i)) = m_delta/norm*(row(predictions,i)-row(labels,i));
			}
		}
		return error;
	}
	
private: 
	double m_delta;
};

}
#endif
