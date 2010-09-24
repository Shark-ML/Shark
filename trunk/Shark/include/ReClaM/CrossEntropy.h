//===========================================================================
/*!
 *  \file CrossEntropy.h
 *
 *  \brief Error measure for classication tasks that can be used
 *         as the objective function for training.
 *
 *  If your model should return a vector whose components are reflecting the
 *  class conditonal probabilities of class membership given any input vector
 *  'CrossEntropy' is the adequate error measure for model-training.
 *  For \em C>1, dimension of model's output and every output dimension
 *  represents the probability for class membership of the given input vector,
 *  the error measure applied is defined as
 *  \f[
 *      E = - \sum_{i=1}^N \sum_{k=1}^C  \left\{tar^i_k \cdot\ln \frac{\exp{(model_k(in^i))}}
 *                          {\sum_{k^{\prime}=1}^C \exp{(model_{k^{\prime}}(in^i))}} \right\}
 *  \f]
 *  where \em i runs over all input patterns and every term in the sum equals zero
 *  if the coefficient equals zero, since \em x \em ln(x) is zero in limes of x running
 *  to zero. The argument of the logarithm calculates the so called softmax-activation
 *  to guarantee for unity at the outputs, i.e.
 *  \f[
 *      \sum_{k=1}^C
 *          \frac{\exp{(model_k(in^i))}}{\sum_{k^{\prime}=1}^C \exp{(model_{k^{\prime}}(in^i))}} = 1
 *  \f]
 *  This is neccessary in order to interprete the output values as probabilities.
 *  This error functional can be derivated and so used for training. In case
 *  of only one single output dimension 'CrossEntropy' returns
 *  the corresponding cross entropy for two classes, using the formalism
 *  \f[
 *      E = - \sum_{i=1}^N \left\{tar^i\cdot \ln model(in^i) + (1-tar^i) \cdot\ln
 *          (1-model(in^i))\right\}
 *  \f]
 *
 *  In this implementation every target value has to be chosen from {0,1} (binary encoding).
 *  For theoretical reasons it is suggested to use for neural networks with one output neuron
 *  the logistic sigmoid activation function at the output. For multiple outputs it's required
 *  to use linear activiation, since this error implementation transforms the linear output
 *  to the softmax activiation as described above. For detailed information refer to
 *  (C.M. Bishop, Neural Networks for Pattern Recognition, Clarendon Press 1996, Chapter 6.9.)
 *
 *  This implementation of the cross entropy performs less efficient than the implemetation given
 *  in 'DF_CrossEntropy.h' for more than one output dimensions, if standard model classes as 'FFNet'
 *  are simultaneously linked, because redundant calculations of the outer derivatives are not circumvented.
 *
 *
 *  \author  M. Huesken
 *  \date    1999
 *
 *  \par Copyright (c) 1999-2001:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *
 *  \par Project:
 *      ReClaM
 *
 *
 *  <BR>
 *
 *
 *  <BR><HR>
 *  This file is part of ReClaM. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 2, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, write to the Free Software
 *  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */
//===========================================================================


#ifndef CROSS_ENTROPY_H
#define CROSS_ENTROPY_H

#include <climits>
#include <cmath>
#include <SharkDefs.h>
#include <ReClaM/ErrorFunction.h>


//===========================================================================
/*!
 *  \brief Error measure for classication tasks that can be used
 *         as the objective function for training.
 *
 *  If your model should return a vector whose components are reflecting the
 *  class conditonal probabilities of class membership given any input vector
 *  'CrossEntropy' is the adequate error measure for model-training.
 *  For \em C>1, dimension of model's output and every output dimension
 *  represents the probability for class membership of the given input vector,
 *  the error measure applied is defined as
 *  \f[
 *      E = - \sum_{i=1}^N \sum_{k=1}^C  \left\{tar^i_k \cdot\ln \frac{\exp{(model_k(in^i))}}
 *                          {\sum_{k^{\prime}=1}^C \exp{(model_{k^{\prime}}(in^i))}} \right\}
 *  \f]
 *  where \em i runs over all input patterns and every term in the sum equals zero
 *  if the coefficient equals zero, since \em x \em ln(x) is zero in limes of x running
 *  to zero. The argument of the logarithm calculates the so called softmax-activation
 *  to guarantee for unity at the outputs, i.e.
 *  \f[
 *      \sum_{k=1}^C
 *          \frac{\exp{(model_k(in^i))}}{\sum_{k^{\prime}=1}^C \exp{(model_{k^{\prime}}(in^i))}} = 1
 *  \f]
 *  This is neccessary in order to interprete the output values as probabilities.
 *  This error functional can be derivated and so used for training. In case
 *  of only one single output dimension 'CrossEntropy' returns
 *  the corresponding cross entropy for two classes, using the formalism
 *  \f[
 *      E = - \sum_{i=1}^N \left\{tar^i\cdot \ln model(in^i) + (1-tar^i) \cdot\ln
 *          (1-model(in^i))\right\}
 *  \f]
 *
 *  In this implementation every target value has to be chosen from {0,1} (binary encoding).
 *  For theoretical reasons it is suggested to use for neural networks with one output neuron
 *  the logistic sigmoid activation function at the output. For multiple outputs it's required
 *  to use linear activiation, since this error implementation transforms the linear output
 *  to the softmax activiation as described above. For detailed information refer to
 *  (C.M. Bishop, Neural Networks for Pattern Recognition, Clarendon Press 1996, Chapter 6.9.)
 *
 *  This implementation of the cross entropy performs less efficient than the implemetation given
 *  in 'DF_CrossEntropy.h' for more than one output dimensions, if standard model classes as 'FFNet'
 *  are simultaneously linked, because redundant calculations of the outer derivatives are not circumvented.
 *
 *  \author  M. Huesken
 *  \date    1999
 *
 *  \par Changes:
 *      Revision 2003/06/03 (S. Wiegand):
 *      softmax activation introduced, bugs in calculation of derivatives corrected
 *
 *
 *  \par Status:
 *      stable
 */
class CrossEntropy : public ErrorFunction
{
	// This value is used instead of log of a small number to avoid numerical instabilities.
	// This value is chosen, as minLog = log(0) on a unix system.
	const static long minLog = -730;
	const static unsigned long maxDeriv = ULONG_MAX;
	const static unsigned long maxexp = ULONG_MAX;
public:

//===========================================================================
	/*!
	 *  \brief Calculates the cross entropy error.
	 *
	 *  The cross entropy function for \em N patterns and
	 *  \em C>1 class-dimensions within the output vector is calculated via
	 *  \f[
	 *      E = - \sum_{i=1}^N \sum_{k=1}^C \left\{tar^i_k\cdot \ln \frac{\exp{(model_k(in^i))}}
	 *                          {\sum_{k^{\prime}=1}^C \exp{(model_{k^{\prime}}(in^i))}}\right\}
	 *  \f]
	 *  respectively for only one single output dimension and two classes via
	 *  \f[
	 *      E = - \sum_{i=1}^N \left\{tar^i\cdot \ln model(in^i) + (1-tar^i) \ln
	 *          (1-model(in^i))\right\}
	 *  \f]
	 *
	 *      \param  in Input-Matrix for the model.
	 *      \param  target Target vector.
	 *      \return The error \em E.
	 *
	 *  \author  M. Huesken, S. Wiegand
	 *  \date    1999, 2003
	 *
	 */
	double error(Model& model, const Array<double> &in, const Array<double> &target)
	{
		double ce = 0;
		double smaxsum;

		if (model.getOutputDimension() > 1)
			// more then one output neuron
			if (in.ndim() == 1)
			{
				Array<double> output(target.dim(0));
				model.model(in, output);
				smaxsum = smaxnorm(output);
				output /= smaxsum;
				for (unsigned c = 0; c < target.dim(0); c++)
					if (target(c) != 0)
						if (output(c) != 0.)
							ce -= target(c) * log(output(c)) ;
						else
							ce -= minLog;
			}
			else
			{
				Array<double> output(target.dim(1));
				for (unsigned pattern = 0; pattern < in.dim(0); ++pattern)
				{
					model.model(in[pattern], output);
					smaxsum = smaxnorm(output);
					output /= smaxsum;
					for (unsigned c = 0; c < target.dim(1); c++)
						if (target(pattern, c) != 0)
							if (output(c) != 0.)
								ce -= target(pattern, c) * log(output(c));
							else
								ce -= minLog;
				}
			}
		else
		{
			// only one output neuron
			if (in.ndim() == 1)
			{
				// one pattern
				Array<double> output(1);
				model.model(in, output);
				if ((output(0) != 0) && (1. - output(0) != 0))
					ce -= (target(0) * log(output(0)) + (1. - target(0)) * log(1. - output(0)));
				else if (output(0) != target(0))
					ce -= minLog;
			}
			else
			{
				// more then one pattern
				Array<double> output(1);
				for (unsigned pattern = 0; pattern < in.dim(0); ++pattern)
				{
					model.model(in[pattern], output);
					if ((output(0) != 0) && (1. - output(0) != 0))
						ce -= target(pattern, 0) * log(output(0)) + (1. - target(pattern, 0)) * log(1. - output(0));
					else if (output(0) != target(pattern, 0))
						ce -= minLog;
				}
			}
		}
		return ce;
	}

//===========================================================================
	/*!
	 *  \brief Calculates the derivatives of the cross entropy error
	 *         (see #error) with respect to the parameters ModelInterface::w.
	 *
	 *  The derivatives of the cross entropy for \em N patterns and \em C>1
	 *  class-dimensions within the output vector with respect to model parameters
	 *  \em w are calculated via
	 *  \f[
	 *      \frac{\partial E}{\partial w} = - \sum_{i=1}^N \sum_{k=1}^C
	 *                                        \left\{
	 *                                        \left(  tar^i_k -
	 *      \frac{\exp{(model_k(in^i))}}{\sum_{k^{\prime}=1}^C \exp{(model_{k^{\prime}}(in^i))}}
	 *                                        \right)
	 *					  \cdot\frac{\partial model_k(in^i)}{\partial w}\right\}
	 *  \f]
	 *  respectively for only one single output dimension via
	 *  \f[
	 *  \frac{\partial E}{\partial w} = - \sum_{i=1}^N
	 *                                    \left\{ \frac{tar^i - model(in^i)}{model(in^i)\cdot(1-model(in^i))}
	 *				      \cdot\frac{\partial model(in^i)}{\partial w}\right\}
	 *  \f]
	 *  Usually, as a byproduct of the calculation of the derivative one gets the
	 *  cross entropy error itself very efficiently. Therefore, the method #error
	 *  gives back this value. This additional effect can be switched of by means
	 *  of the third parameter (returnError = false).
	 *
	 *      \param  in          Input vector for the model.
	 *      \param  target      Target vector.
	 *      \param  returnError Determines whether or not to calculate the cross
	 *                          entropy error itself. By default the
	 *                          error is calculated.
	 *      \return The cross entropy error if \em returnError is set to "true",
	 *              "-1" otherwise.
	 *
	 *  \author  M. Huesken
	 *  \date    1999
	 *
	 *  \par Changes
	 *      Revision 2003/06/03 (S. Wiegand):
	 *      softmax activation introduced, bugs in calculation of derivatives corrected
	 *
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	double errorDerivative(Model& model, const Array<double> &in, const Array<double> &target, Array<double>& derivative)
	{
		derivative.resize(model.getParameterDimension(), false);
		derivative = 0;
		double ce = 0;
		double smaxsum;
		double isconsistent = 0;
		Array<double> dmdw;

		if (model.getOutputDimension() > 1)
		{
			// more then one output neuron
			if (in.ndim() == 1)
			{
				// one input pattern
				Array<double> output(target.dim(0));
				coeffs.resize(output.nelem(), false);
				model.modelDerivative(in, output, dmdw);
				smaxsum = smaxnorm(output);
				output /= smaxsum;
				// calculate cross entropy
				for (unsigned c = 0; c < target.dim(0); c++)
					if (target(c) != 0)
						if (output(c) != 0.)
							ce -= target(c) * log(output(c)) ;
						else
							ce -= minLog;
				// calculate derivative
				for (unsigned c = 0; c < output.nelem(); c++)
				{
					isconsistent += target(c);
					for (unsigned i = 0; i < derivative.nelem(); i++)
						derivative(i) += (output(c) - target(c)) * dmdw(c, i);
				}
				if (isconsistent != 1)
				{
					throw SHARKEXCEPTION("[CrossEntropy] teacher labels do not sum to unity!");
				}
			}
			else
			{
				// more than one input pattern
				//std::cout << "derror: Hallo x-Out x-pattern! CE" << std::endl;
				Array<double> output(target.dim(1));
				coeffs.resize(output.nelem(), false);
				for (unsigned pattern = 0; pattern < in.dim(0); ++pattern)
				{
					isconsistent = 0;
					model.modelDerivative(in[pattern], output, dmdw);
					smaxsum = smaxnorm(output);
					output /= smaxsum;
					// calculate cross-entropy
					for (unsigned c = 0; c < target.dim(1); c++)
						if (target(pattern, c) != 0)
							if (output(c) != 0.)
								ce -= target(pattern, c) * log(output(c));
							else
								ce -= minLog;
					// calculate derivative
					for (unsigned c = 0; c < output.nelem(); c++)
					{
						isconsistent += target(pattern, c);
						for (unsigned i = 0; i < derivative.nelem(); i++)
							derivative(i) += (output(c) - target(pattern, c)) * dmdw(c, i);
					}
					if (isconsistent != 1)
					{
						throw SHARKEXCEPTION("[CrossEntropy] teacher labels do not sum to unity!");
					}
				}
			}
		}
		else
		{
			// one output neuron
			if (in.ndim() == 1)
			{
				// only one pattern
				Array<double> output(1);
				coeffs.resize(output.nelem(), false);
					model.modelDerivative(in, output, dmdw);
				// calculate ce
				if ((output(0) != 0) && (1. - output(0) != 0))
					ce -= (target(0) * log(output(0)) + (1. - target(0)) * log(1. - output(0)));
				else if (output(0) != target(0))
					ce -= minLog;
				// calculate derivative
				for (unsigned i = 0; i < derivative.nelem(); i++)
					if ((output(0) != 0) && (1. - output(0) != 0))
					{
						derivative(i) -= (target(0) - output(0)) / (output(0) * (1. - output(0))) * dmdw(0, i);
					}
					else
					{
						if (output(0) != target(0))
						{
							// the sign changes due to the value of the target,
							derivative(i) -= (2. * target(0) - 1.) * maxDeriv * dmdw(0, i);
						}
					}
			}
			else
			{
				// a number of patterns
				Array<double> output(1);
				coeffs.resize(output.nelem(), false);
				for (unsigned pattern = 0; pattern < in.dim(0); ++pattern)
				{
					model.modelDerivative(in[pattern], output, dmdw);
	
					// calculate ce
						if ((output(0) != 0) && (1. - output(0) != 0))
							ce -= target(pattern, 0) * log(output(0)) + (1. - target(pattern, 0)) * log(1. - output(0));
						else  if (output(0) != target(pattern, 0))
							ce -= minLog;

					// calculate derivative
					for (unsigned i = 0; i < derivative.nelem(); i++)
						if ((output(0) != 0) && (1. - output(0) != 0))
						{
							derivative(i) -= (target(pattern, 0) - output(0)) / (output(0) * (1. - output(0))) * dmdw(0, i);
						}
						else
						{
							if (output(0) != target(pattern, 0))
							{
								// the sign changes due to the value of the target, maxDeriv is devided by reason of computational bounds
								derivative(i) -= (2. * target(pattern, 0) - 1.) * maxDeriv / in.dim(0) * dmdw(0, i);
							}
						}
				}
			}
		}

		return ce;
	}

private:
	double norm(Array<double> &output)
	{
		double sum = 0;
		if (output.ndim() == 1)
		{
			for (unsigned i = 0; i < output.nelem(); i++) sum += output(i);
		}
		else
		{
			throw SHARKEXCEPTION("[CrossEntropy] internal error ?!?");
		}
		return sum;
	}

	double smaxnorm(Array<double> &output)
	{
		double sum = 0;
		if (output.ndim() == 1)
		{
			for (unsigned i = 0; i < output.nelem(); i++)
			{
				if (output(i) < 301)
					output(i) = exp(output(i));
				else output(i) = maxexp / output.nelem();
				sum += output(i);
			}
		}
		else
		{
			throw SHARKEXCEPTION("[CrossEntropy] internal error ?!?");
		}
		return sum;
	}
	Array<double> coeffs;
};


#endif

