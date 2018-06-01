//===========================================================================
/*!
 * 
 *
 * \brief       Error measure for classification tasks that can be used
 * as the objective function for training.
 * 
 * 
 * 
 *
 * \author      -
 * \date        -
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 * 3
 * <BR><HR>
 * This file is part of Shark.
 * <http://shark-ml.org/>
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

#ifndef SHARK_OBJECTIVEFUNCTIONS_LOSS_CROSS_ENTROPY_H
#define SHARK_OBJECTIVEFUNCTIONS_LOSS_CROSS_ENTROPY_H

#include <shark/ObjectiveFunctions/Loss/AbstractLoss.h>

namespace shark{

///  \brief Error measure for classification tasks that can be used
///         as the objective function for training.
///
///  If your model should return a vector whose components reflect the
///  logarithmic conditional probabilities of class membership given any input vector
///  'CrossEntropy' is the adequate error measure for model-training.
///  For \em C>1 classes the loss function is defined as
///  \f[
///      E = - \ln \frac{\exp{x_c}} {\sum_{c^{\prime}=1}^C \exp{x_c^{\prime}}} = - x_c + \ln \sum_{c^{\prime}=1}^C \exp{x_c^{\prime}} 
///  \f]
///  where \em x is the prediction vector of the model and \em c is the class label. In the case of only one
///  model output and binary classification, another more numerically stable formulation is used:
///  \f[
///     E = \ln(1+ e^{-yx})
///  \f]
///  here, \em y are class labels between -1 and 1 and y = -2 c+1. The reason why this is numerically more stable is,
///  that when \f$ e^{-yx} \f$ is big, the error function is well approximated by the linear function \em x. Also if
///  the exponential is very small, the case \f$ \ln(0) \f$ is avoided.
///
/// If the class labels are integers, they must be starting from 0. If class labels are vectors, there must be a proper
/// probability vector. i.e. values must be bigger or equal to zero and sum to one. This incldues one-hot-encoding of labels.
/// Also for theoretical reasons, the output neurons of a neural Network that is trained with this loss should be linear.
/// \ingroup lossfunctions

template<class LabelType, class OutputType>
class CrossEntropy;


template<class OutputType>
class CrossEntropy<unsigned int, OutputType> : public AbstractLoss<unsigned int,OutputType>
{
private:
	typedef AbstractLoss<unsigned int,OutputType> base_type;
	typedef typename base_type::ConstLabelReference ConstLabelReference;
	typedef typename base_type::ConstOutputReference ConstOutputReference;
	typedef typename base_type::BatchOutputType BatchOutputType;
	typedef typename base_type::MatrixType MatrixType;

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
public:
	CrossEntropy()
	{ this->m_features |= base_type::HAS_FIRST_DERIVATIVE;}


	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "CrossEntropy"; }

	// annoyingness of C++ templates
	using base_type::eval;

	double eval(UIntVector const& target, BatchOutputType const& prediction) const {
		double error = 0;
		for(std::size_t i = 0; i != prediction.size1(); ++i){
			error += eval(target(i), row(prediction,i));
		}
		return error;
	}
	
	double eval( ConstLabelReference target, ConstOutputReference prediction)const{
		if ( prediction.size() == 1 )
		{
			RANGE_CHECK ( target < 2 );
			double label = 2.0 * target - 1;   //converts labels from 0/1 to -1/1
			double exponential =  std::exp( -label * prediction(0) );
			return evalError(label,exponential,prediction(0 ));
		}else{
			RANGE_CHECK ( target < prediction.size() );
			
			//calculate the log norm in a numerically stable way
			//we subtract the maximum prior to exponentiation to 
			//ensure that the exponentiation result will still fit in double
			double maximum = max(prediction);
			double logNorm = sum(exp(prediction-maximum));
			logNorm = std::log(logNorm) + maximum;
			return logNorm - prediction(target);
		}
	}

	double evalDerivative(UIntVector const& target, BatchOutputType const& prediction, BatchOutputType& gradient) const {
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
		}else{
			double error = 0;
			for(std::size_t i = 0; i != prediction.size1(); ++i){
				RANGE_CHECK ( target(i) < prediction.size2() );
				auto gradRow=row(gradient,i);
				
				//calculate the log norm in a numerically stable way
				//we subtract the maximum prior to exponentiation to 
				//ensure that the exponentiation result will still fit in double
				//this does not change the result as the values get normalized by
				//their sum and thus the correction term cancels out.
				double maximum = max(row(prediction,i));
				noalias(gradRow) = exp(row(prediction,i) - maximum);
				double norm = sum(gradRow);
				gradRow/=norm;
				gradient(i,target(i)) -= 1;
				error+=std::log(norm) - prediction(i,target(i))+maximum;
			}
			return error;
		}
	}
	double evalDerivative(ConstLabelReference target, ConstOutputReference prediction, OutputType& gradient) const {
		gradient.resize(prediction.size());
		if ( prediction.size() == 1 ){
			RANGE_CHECK ( target < 2 );
			double label = 2.0 * target - 1;   //converts labels from 0/1 to -1/1
			double exponential =  std::exp ( - label * prediction(0));
			double sigmoid = 1.0/(1.0+exponential);
			gradient(0) = -label * (1.0 - sigmoid);
			return evalError(label,exponential,prediction(0));
		}else{
			RANGE_CHECK ( target < prediction.size() );
			
			//calculate the log norm in a numerically stable way
			//we subtract the maximum prior to exponentiation to 
			//ensure that the exponentiation result will still fit in double
			//this does not change the result as the values get normalized by
			//their sum and thus the correction term cancels out.
			double maximum = max(prediction);
			noalias(gradient) = exp(prediction - maximum);
			double norm = sum(gradient);
			gradient /= norm;
			gradient(target) -= 1;
			return std::log(norm) - prediction(target) + maximum;
		}
	}

	double evalDerivative(
		ConstLabelReference target, ConstOutputReference prediction,
		BatchOutputType& gradient,MatrixType & hessian
	) const {
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
			//calculate the log norm in a numerically stable way
			//we subtract the maximum prior to exponentiation to 
			//ensure that the exponentiation result will still fit in double
			//this does not change the result as the values get normalized by
			//their sum and thus the correction term cancels out.
			double maximum = max(prediction);
			noalias(gradient) = exp(prediction-maximum);
			double norm = sum(gradient);
			gradient/=norm;

			noalias(hessian)=-outer_prod(gradient,gradient);
			noalias(diag(hessian)) += gradient;
			gradient(target) -= 1;

			return std::log(norm) - prediction(target) - maximum;
		}
	}
};


template<class T, class Device>
class CrossEntropy<blas::vector<T, Device>, blas::vector<T, Device> >
: public AbstractLoss<blas::vector<T, Device>, blas::vector<T, Device>>
{
private:
	typedef blas::vector<T, Device> OutputType;
	typedef AbstractLoss<OutputType,OutputType> base_type;
	typedef typename base_type::ConstLabelReference ConstLabelReference;
	typedef typename base_type::ConstOutputReference ConstOutputReference;
	typedef typename base_type::BatchOutputType BatchOutputType;
	typedef typename base_type::MatrixType MatrixType;
public:
	CrossEntropy()
	{ this->m_features |= base_type::HAS_FIRST_DERIVATIVE;}


	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "CrossEntropy"; }

	// annoyingness of C++ templates
	using base_type::eval;

	double eval(BatchOutputType const& target, BatchOutputType const& prediction) const {
		SIZE_CHECK(target.size1() == prediction.size1());
		SIZE_CHECK(target.size2() == prediction.size2());
		std::size_t m = target.size2();
		
		OutputType maximum = max(as_rows(prediction));
		auto safeExp = exp(prediction - trans(blas::repeat(maximum, m)));
		OutputType norm = sum(as_rows(safeExp));
		double error = sum(log(norm)) - sum(target * prediction) + sum(maximum);
		return error;
	}

	double evalDerivative(BatchOutputType const& target, BatchOutputType const& prediction, BatchOutputType& gradient) const {
		gradient.resize(prediction.size1(),prediction.size2());
		std::size_t m = target.size2();
		
		OutputType maximum = max(as_rows(prediction));
		noalias(gradient) = exp(prediction - trans(blas::repeat(maximum, m)));
		OutputType norm = sum(as_rows(gradient));
		noalias(gradient) /= trans(blas::repeat(norm, m));
		noalias(gradient) -= target;
		double error = sum(log(norm)) - sum(target * prediction) + sum(maximum);
		return error;
	}
};


}
#endif
