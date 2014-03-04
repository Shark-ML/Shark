/*!
 * 
 * \brief       Implements the Squared Hinge Loss function for maximum margin classification.
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

#ifndef SHARK_OBJECTIVEFUNCTIONS_LOSS_SQUAREDHINGELOSS_H
#define SHARK_OBJECTIVEFUNCTIONS_LOSS_SQUAREDHINGELOSS_H

#include <shark/ObjectiveFunctions/Loss/AbstractLoss.h>

namespace shark {

///
/// \brief Squared Hinge-loss for large margin classification
///
/// The squared hinge loss for two class problems is defined as \f$ L_i = 1/2\max \{ 0 , (1- y_i f(x_i))^2 \} \f$ where \f$ y_i \in \{-1,1} \f$ is the label
/// and \f$ f(x_i) \f$ is the prediction of the model for the ith input. The loss introduces the concept of
/// a margin, that is, the point should not only be correctly classified but also not too close to the
/// decision boundary. Therefore even correctly classified points are getting punished. 
///
/// for multi class problems the concept of sums of the relative margin is used:
/// \f$ L_i = 1/2 \sum_{c \neq y_i} \max \{ 0 , 1- 1/2 (f_{y_i}(x_i)- f_c(x_i) \} \f$. This loss requires that there is a margin
/// between the different class outputs and the functions needs as many outputs as classes. the pre-factor
/// 1/2 ensures that in the 2 class 2 output case with a linear function the value of loss is the same as in the single
/// output version.
///
/// The loss is implemented for class labels 0,1,...,n, even in the binary cases.
/// 
/// The difference to the normal hinge loss is, that the squared hinge-loss is always differentiable.
/// However compared to the hinge loss, small margin violations are not as much punished - but big deviations
/// are punished much stronger.
class SquaredHingeLoss : public AbstractLoss<unsigned int, RealVector>
{
public:
	/// constructor
	SquaredHingeLoss(){ 
		m_features |= base_type::HAS_FIRST_DERIVATIVE;
	}
	
	/// \brief Returns class name "HingeLoss"
	std::string name() const
	{ return "SquaredHingeLoss"; }


	///\brief calculates the sum of all 
	double eval(BatchLabelType const& labels, BatchOutputType const& predictions) const{
		std::size_t numInputs = size(labels);
		SIZE_CHECK(numInputs == size(predictions));
		
		double error = 0;
		//binary case for models with single output
		if(predictions.size2() == 1){
			for(std::size_t i = 0; i != numInputs;++i){
				SIZE_CHECK(labels(i) < 2);
				double y = 2.0*labels(i)-1.0;
				error += sqr(std::max(0.0,1.0-y*predictions(i,0)));
			}
		}
		else
		{//multi-class or multiple output case
			for(std::size_t i = 0; i != numInputs;++i){
				SIZE_CHECK(labels(i) < predictions.size2());
				for(std::size_t o = 0; o != predictions.size2();++o){
					if( o == labels(i)) continue;
					error += sqr(std::max(0.0,2.0 - predictions(i,labels(i))+predictions(i,o)));
				}
			}
			error/=4;
		}
		
		return error/2;
	}

	double evalDerivative(BatchLabelType const& labels, BatchOutputType const& predictions, BatchOutputType& gradient)const{
		std::size_t numInputs = size(labels);
		std::size_t outputDim = predictions.size2();
		SIZE_CHECK(numInputs == size(predictions));
		
		gradient.resize(numInputs,outputDim);
		gradient.clear();
		double error = 0;
		//binary case for models with single output
		if(outputDim == 1){
			for(std::size_t i = 0; i != numInputs; ++i){
				double y = 2.0*labels(i)-1.0;
				double sampleLoss = std::max(0.0,1.0-y*predictions(i,0));
				if(sampleLoss > 0)
					gradient(i,0) = -y*sampleLoss;
				error += sqr(sampleLoss);
			}
		}
		else
		{//multi-class or multiple output case
			for(std::size_t i = 0; i != numInputs;++i){
				SIZE_CHECK(labels(i) < predictions.size2());
				for(std::size_t o = 0; o != predictions.size2();++o){
					if( o == labels(i)) continue;
					double sampleLoss = std::max(0.0,2.0 - predictions(i,labels(i))+predictions(i,o));
					if(sampleLoss > 0){
						gradient(i,o) = sampleLoss*0.25;
						gradient(i,labels(i)) -= sampleLoss*0.25;
					}
					error+= sqr(sampleLoss);
				}
			}
			error/=4;
		}
		
		return error/2;
	}
	
};

}
#endif
