/*!
 * 
 * \brief       Implements the squard Hinge Loss function for maximum margin regression.
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

#ifndef SHARK_OBJECTIVEFUNCTIONS_LOSS_SQUAREDEPSILONHINGELOSS_H
#define SHARK_OBJECTIVEFUNCTIONS_LOSS_SQUAREDEPSILONHINGELOSS_H

#include <shark/ObjectiveFunctions/Loss/AbstractLoss.h>

namespace shark {

///
/// \brief Hinge-loss for large margin regression using th squared two-norm
///
/// The loss is defined as \f$L_i = 1/2 \max\{0.0, ||f(x_i)-y{i,j}||^2- \epsilon^2\} \f$  
/// where \f$ y_i =(y_{i,1},\dots,y_{i_N} \f$ is the label of dimension N
/// and \f$ f_j(x_i) \f$ is the j-th output of the prediction of the model for the ith input. 
/// The loss introduces the concept of a margin to regression, that is, points are not punished
/// when they are sufficiently close to the function.
///
/// epsilon describes the distance from the label to the margin that is allowed until the point leaves
/// the margin.
///
/// Contrary to th EpsilonHingeLoss, this loss is differentiable.
class SquaredEpsilonHingeLoss : public AbstractLoss<RealVector, RealVector>
{
public:
	/// constructor
	SquaredEpsilonHingeLoss(double epsilon):m_sqrEpsilon(sqr(epsilon)){ 
		m_features |= base_type::HAS_FIRST_DERIVATIVE;
	}
	
	/// \brief Returns class name "HingeLoss"
	std::string name() const
	{ return "SquaredEpsilonHingeLoss"; }


	///\brief calculates the sum of all 
	double eval(BatchLabelType const& labels, BatchOutputType const& predictions) const{
		SIZE_CHECK(predictions.size1() == labels.size1());
		SIZE_CHECK(predictions.size2() == labels.size2());
		std::size_t numInputs = predictions.size1();
		
		return 0.5*sum(max(0.0,sum_columns(sqr(labels-predictions))-blas::repeat(m_sqrEpsilon,numInputs)));
	}

	double evalDerivative(BatchLabelType const& labels, BatchOutputType const& predictions, BatchOutputType& gradient)const{
		SIZE_CHECK(predictions.size1() == labels.size1());
		SIZE_CHECK(predictions.size2() == labels.size2());
		std::size_t numInputs = predictions.size1();
		
		gradient.resize(numInputs,predictions.size2());
		double error = 0;
		for(std::size_t i = 0; i != numInputs;++i){
			double sampleLoss = 0.5*std::max(0.0,norm_sqr(row(predictions,i)-row(labels,i))-m_sqrEpsilon);
			error+=sampleLoss;
			if(sampleLoss > 0){
				noalias(row(gradient,i)) = row(predictions,i)-row(labels,i);
			}
			else{
				row(gradient,i).clear();
			}
		}
		return error;
	}
private:
	double m_sqrEpsilon;
};

}
#endif
