/*!
 * 
 * \brief       Implements the Hinge Loss function for maximum margin regression
 * 
 *
 * \author    Oswin Krause
 * \date        2014
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

#ifndef SHARK_OBJECTIVEFUNCTIONS_LOSS_EPSILONHINGELOSS_H
#define SHARK_OBJECTIVEFUNCTIONS_LOSS_EPSILONHINGELOSS_H

#include <shark/ObjectiveFunctions/Loss/AbstractLoss.h>

namespace shark {

///
/// \brief Hinge-loss for large margin regression
///
/// The loss is defined as \f$L_i = \sum_j^N \max\{0.0, |f_j(x_i)-y_{i,j}|-\epsilon\} \f$  
/// where \f$ y_i =(y_{i,1},\dots,y_{i_N} \f$ is the label of dimension N
/// and \f$ f_j(x_i) \f$ is the j-th output of the prediction of the model for the ith input. 
/// The loss introduces the concept of a margin to regression, that is, points are not punished
/// when they are sufficiently close to the function. Points which are outside of the
/// margin are linearly punished, that is the loss is outlier resistant. 
///
/// Epsilon describes the size of the margin.
///
/// The hinge-loss is not differentiable at the points y_{i,j}+epsilon and y_{i,j}-epsilon.
class EpsilonHingeLoss : public AbstractLoss<RealVector, RealVector>
{
public:
	/// constructor
	EpsilonHingeLoss(double epsilon):m_epsilon(epsilon){ 
		m_features |= base_type::HAS_FIRST_DERIVATIVE;
	}
	
	/// \brief Returns class name "HingeLoss"
	std::string name() const
	{ return "EpsilonHingeLoss"; }


	///\brief calculates the sum of all 
	double eval(BatchLabelType const& labels, BatchOutputType const& predictions) const{
		SIZE_CHECK(predictions.size1() == labels.size1());
		SIZE_CHECK(predictions.size2() == labels.size2());
		std::size_t numInputs = predictions.size1();
		std::size_t outputDim = predictions.size2();
		
		return sum(max(0.0,abs(labels-predictions)-blas::repeat(m_epsilon,numInputs,outputDim)));
	}

	double evalDerivative(BatchLabelType const& labels, BatchOutputType const& predictions, BatchOutputType& gradient)const{
		SIZE_CHECK(predictions.size1() == labels.size1());
		SIZE_CHECK(predictions.size2() == labels.size2());
		std::size_t numInputs = predictions.size1();
		std::size_t outputDim = predictions.size2();
		
		gradient.resize(numInputs,outputDim);
		double error = 0;
		for(std::size_t i = 0; i != numInputs;++i){
			for(std::size_t o = 0; o != outputDim;++o){
				double sampleLoss = std::max(0.0,std::abs(predictions(i,o)-labels(i,o))-m_epsilon);
				error+=sampleLoss;
				gradient(i,o) = 0;
				if(sampleLoss > 0){
					if(predictions(i,o) > labels(i,o))
						gradient(i,o) = 1;
					else
						gradient(i,o) = -1;
				}
				
			}
		}
		return error;
	}
private:
	double m_epsilon;
};

}
#endif
