/*!
 * 
 * \brief       Implements Tukey's Biweight-loss function for robust regression
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

/// \brief Tukey's Biweight-loss for robust regression
///
/// Tukey's Biweight-loss is a robust regression function. For predictions close to the correct classification,
/// it is convex, but close to a value k, it approaches a constant value. for differences greater than k,
/// the function is constant and has gradient 0. This effectively ignores large outliers at the cost
/// of loosing the convexity of the loss function. 
/// The 1-dimensional loss is defined as
///\f[ f(x)= \frac {x^6}{6k^4} - \frac {x^4} {2k^2}+\frac {x^2} {2}  \f]
/// for \f$ x \in [-k,k]\f$. outside it is the constant function \f$\frac {k^2}{6}\f$.
///
/// For multidimensional problems we define it with x being the two-norm of the difference
/// between the label and predicted values.
class TukeyBiweightLoss : public AbstractLoss<RealVector, RealVector>
{
public:
	/// constructor
	TukeyBiweightLoss(double k = 1.0):m_k(k){ 
		m_features |= base_type::HAS_FIRST_DERIVATIVE;
	}
	
	/// \brief Returns class name "HuberLoss"
	std::string name() const
	{ return "TukeyBiweightLoss"; }


	///\brief calculates the sum of all 
	double eval(BatchLabelType const& labels, BatchOutputType const& predictions) const{
		SIZE_CHECK(labels.size1() == predictions.size1());
		SIZE_CHECK(labels.size2() == predictions.size2());
		std::size_t numInputs = labels.size1();
		
		double error = 0;
		double k2 = sqr(m_k);
		double k4 = sqr(k2);
		double maxErr = k2/6;
		for(std::size_t i = 0; i != numInputs;++i){
			double norm2 = norm_sqr(row(predictions,i)-row(labels,i));
			
			//check whether we are in the quadratic area
			if(norm2 <= sqr(m_k)){
				error = norm2/2+sqr(norm2)/6*(norm2/k4-3/k2);
			}
			else{
				error += maxErr;
			}
		}
		return error;
	}

	double evalDerivative(BatchLabelType const& labels, BatchOutputType const& predictions, BatchOutputType& gradient)const{
		std::size_t numInputs = size(labels);
		std::size_t outputDim = predictions.size2();
		SIZE_CHECK(numInputs == size(predictions));
		
		gradient.resize(numInputs,outputDim);
		gradient.clear();
		double error = 0;
		double k2 = sqr(m_k);
		double k4 = sqr(k2);
		double maxErr = k2/6;
		for(std::size_t i = 0; i != numInputs;++i){
			double norm2 = norm_sqr(row(predictions,i)-row(labels,i));
			
			//check whether we are in the quadratic area
			if(norm2 <= sqr(m_k)){
				error = norm2/2+sqr(norm2)/6*(norm2/k4-3/k2);
				noalias(row(gradient,i)) = (1+sqr(norm2)/k4-2*norm2/k2)*(row(predictions,i)-row(labels,i));
			}
			else{
				error += maxErr;
				//gradient is initialized to 0!
			}
		}
		return error;
	}
	
private: 
	double m_k;
};

}
#endif
