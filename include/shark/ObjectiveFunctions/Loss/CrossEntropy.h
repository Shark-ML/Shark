//===========================================================================
/*!
 *  \brief Error measure for classification tasks that can be used
 *         as the objective function for training.
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

#ifndef SHARK_OBJECTIVEFUNCTIONS_LOSS_CROSS_ENTROPY_H
#define SHARK_OBJECTIVEFUNCTIONS_LOSS_CROSS_ENTROPY_H

#include <shark/ObjectiveFunctions/Loss/AbstractLoss.h>

namespace shark{

/*!
 *  \brief Error measure for classication tasks that can be used
 *         as the objective function for training.
 *
 *  If your model should return a vector whose components reflect the
 *  conditonal probabilities of class membership given any input vector
 *  'CrossEntropy' is the adequate error measure for model-training.
 *  For \em C>1 classes the loss function is defined as
 *  \f[
 *      E = - \ln \frac{\exp{x_c}} {\sum_{c^{\prime}=1}^C \exp{x_c^{\prime}}} = - x_c + \ln \sum_{c^{\prime}=1}^C \exp{x_c^{\prime}} 
 *  \f]
 *  where \em x is the prediction vector of the model and \em c is the class label. In the case of only one
 *  model output and binary classification, another more numerically stable formulation is used:
 *  \f[
 *     E = \ln(1+ e^{-yx})
 *  \f]
 *  here, \em y are class labels between -1 and 1 and y = -2 c+1. The reason why this is numerically more stable is,
 *  that when \f$ e^{-yx} \f$ is big, the error function is well approximated by the linear function \em x. Also if
 *  the exponential is very small, the case \f$ \ln(0) \f$ is avoided.
 *
 * The class labels must be integers starting from 0. Also for theoretical reasons, the output neurons of a neural
 *  Network must be linear.
 */
class CrossEntropy : public AbstractLoss<unsigned int,RealVector>
{
private:
	typedef AbstractLoss<unsigned int,RealVector> base_type;

	//uses different formula to compute the binary case for 1 output.
	//should be numerically more stable
	//formula: ln(1+exp(-yx)) with y = -1/1
	double evalError(double label,double exponential,double value) const {

		if(value*label < -200 ){
			//below this, we might get numeric instabilities
			//but we know, that ln(1+exp(x)) converges to x for big arguments
			return - value * label;
		}
		return std::log(1+exponential);
	}
	template<class Vec>
	double expNorm(Vec const& vec) const {
		double result = 0;
		for( std::size_t i = 0; i != vec.size(); ++ i)
		{
			double term = std::exp(vec(i));
			if(term > 1.e200)
				term = 1.e200;
			result += term;
		}
		return result;
	}
	template<class Vec1,class Vec2>
	double expNorm(Vec1 const& vec, Vec2& expVec) const {
		double result = 0;
		for( std::size_t i = 0; i != vec.size(); ++ i)
		{
			double term = std::exp(vec(i));
			if(term > 1.e200)
				term = 1.e200;
			result += term;
			expVec(i) = term;
		}
		return result;
	}
public:
	CrossEntropy()
	{
		m_features |= HAS_FIRST_DERIVATIVE;
		//~ m_features |= HAS_SECOND_DERIVATIVE;
	}


	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "CrossEntropy"; }

	// annoyingness of C++ templates
	using base_type::eval;

	double eval(UIntVector const& target, RealMatrix const& prediction) const {
		if ( prediction.size2() == 1 )
		{
			double error = 0;
			for(std::size_t i = 0; i != prediction.size1(); ++i){
				RANGE_CHECK ( target(i) < 2 );
				double label = 2 * static_cast<double>(target(i)) - 1;   //converts labels from 0/1 to -1/1
				double exponential =  std::exp ( -label * prediction ( i,0 ) );
				error+= evalError(label,exponential,prediction (i, 0 ));
			}
			return error;
		}
		else
		{
			double error = 0;
			for(std::size_t i = 0; i != prediction.size1(); ++i){
				RANGE_CHECK ( target(i) < prediction.size2() );
				double norm = expNorm(row(prediction,i));
				error+= std::log(norm) - prediction(i,target(i));
			}
			return error;
		}
	}

	double evalDerivative(UIntVector const& target, RealMatrix const& prediction, RealMatrix& gradient) const {
		gradient.resize(prediction.size1(),prediction.size2());
		if ( prediction.size2() == 1 )
		{
			double error = 0;
			for(std::size_t i = 0; i != prediction.size1(); ++i){
				RANGE_CHECK ( target(i) < 2 );
				double label = 2 * static_cast<double>(target(i)) - 1;   //converts labels from 0/1 to -1/1
				double exponential =  std::exp ( -label * prediction (i, 0 ) );
				double sigmoid = 1.0/(1.0+exponential);
				gradient ( i,0 ) = -label * (1.0 - sigmoid);
				error+=evalError(label,exponential,prediction (i, 0 ));
			}
			return error;
		}
		else
		{
			double error = 0;
			for(std::size_t i = 0; i != prediction.size1(); ++i){
				RANGE_CHECK ( target(i) < prediction.size2() );
				RealMatrixRow gradRow=row(gradient,i);
				double norm = expNorm(row(prediction,i),gradRow);
				row(gradient,i)/=norm;
				gradient(i,target(i)) -= 1;
				error+=std::log(norm) - prediction(i,target(i));
			}
			return error;
		}
	}

	double evalDerivative (
		unsigned int const& target,
		RealVector const& prediction,
		RealVector& gradient,
		RealMatrix& hessian
	) const{
		gradient.resize(prediction.size());
		hessian.resize(prediction.size(),prediction.size());
		if ( prediction.size() == 1 )
		{
			RANGE_CHECK ( target < 2 );
			double label = 2 * static_cast<double>(target) - 1;   //converts labels from 0/1 to -1/1
			double exponential =  std::exp ( -label * prediction ( 0 ) );
			double sigmoid = 1.0/(1.0+exponential);
			gradient ( 0 ) = -label * (1.0-sigmoid);
			hessian ( 0,0 ) = sigmoid * ( 1-sigmoid );
			return evalError(label,exponential,prediction ( 0 ));
		}
		else
		{
			RANGE_CHECK ( target < prediction.size() );
			double norm = expNorm(prediction,gradient);
			gradient/=norm;

			noalias(hessian)=-outer_prod(gradient,gradient);
			noalias(diag(hessian)) += gradient;
			gradient(target) -= 1;

			return std::log(norm) - prediction(target);
		}
	}
};


}
#endif
