//===========================================================================
/*!
 *  \file CrossEntropyIndependent.h
 *
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
 *
 *  For theoretical reasons it is suggested to use for neural networks
 *  the logistic sigmoid activation function at the output neurons.
 *  However, actually every sigmoid activation could be applied to the output
 *  neurons as far as the image of this function is identical to the intervall
 *  \em [0,1].
 *  In this implementation every target value to be chosen from {0,1}
 *  (binary encoding). For detailed information refer to
 *  (C.M. Bishop, Neural Networks for Pattern Recognition, Clarendon Press 1996, Chapter 6.8.)
 *
 *  This implementation of the cross entropy performs less efficient than the implemetation given
 *  in 'DF_CrossEntropyIndependent.h' for more than one output dimensions, if standard model classes as 'FFNet'
 *  are simultaneously linked, because redundant calculations of the outer derivatives are not circumvented.
 *
 *  \author  C. Igel, M. Huesken
 *  \date    2002
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


#ifndef CROSS_ENTROPY_INDEPENDENT_H
#define CROSS_ENTROPY_INDEPENDENT_H

#include <cmath>
#include <ReClaM/ErrorFunction.h>

//===========================================================================
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
 *
 *  This implementation of the cross entropy performs less efficient than the implemetation given
 *  in 'DF_CrossEntropyIndependent.h' for more than one output dimensions, if standard model classes as 'FFNet'
 *  are simultaneously linked, because redundant calculations of the outer derivatives are not circumvented.
 *
 *  \author  M. Huesken
 *  \date    1999
 *
 *  \par Changes:
 *      Revision 2003/06/03 (S. Wiegand):
 *      bugs in formulars corrected
 *
 *  \par Status:
 *      stable
 */
class CrossEntropyIndependent : public ErrorFunction
{
	// This value is used instead of log of a small number to avoid numerical instabilities.
	// This value is chosen, as minLog = log(0) on a unix system.
	const static double minLog = -730;
	const static double maxDeriv = 1e300;
public:

//===========================================================================
	/*!
	 *  \brief Calculates the CrossEntropyIndependent error.
	 *
	 *  Calculates the CrossEntropyIndependent error function for \em N patterns and
	 *  \em C>1 independent attributes within the output vector via
	 *  \f[
	 *      E = - \sum_{i=1}^N \sum_{k=1}^{C} \{tar^i_k \ln model_k(in^i) + (1-tar^i_k) \ln
	 *          (1-model_k(in^i))\}
	 *  \f]
	 *  or the true cross entropy for only one single output dimension via
	 *  \f[
	 *      E = - \sum_{i=1}^N \{tar^i \ln model(in^i) + (1-tar^i) \ln
	 *          (1-model(in^i))\}
	 *  \f]
	 *      \param  in Input vector for the model.
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
		if (model.getOutputDimension() > 1)
			// more then one output neuron
			if (in.ndim() == 1)
			{
				Array<double> output(target.dim(0));
				model(in, output);
				for (unsigned c = 0; c < target.dim(0); c++)
					if ((output(c) != 0) && (1.0 - output(c) != 0))
						ce -= target(c) * log(output(c)) + (1. - target(c)) * log(1. - output(c));
					else if (output(c) != target(c))
						ce -= minLog;
			}
			else
			{
				Array<double> output(target.dim(1));
				for (unsigned pattern = 0; pattern < in.dim(0); ++pattern)
				{
					model(in[pattern], output);
					for (unsigned c = 0; c < target.dim(1); c++)
						if ((output(c) != 0) && (1.0 - output(c) != 0))
							ce -= target(pattern, c) * log(output(c)) + (1. - target(pattern, c)) * log(1. - output(c));
						else if (output(c) != target(pattern, c))
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
				model(in, output);
				if ((output(0) != 0) && (1.0 - output(0) != 0))
					ce -= (target(0) * log(output(0)) + (1.0 - target(0)) * log(1.0 - output(0)));
				else if (output(0) != target(0))
					ce -= minLog;
			}
			else
			{
				// more then one pattern
				Array<double> output(1);
				for (unsigned pattern = 0; pattern < in.dim(0); ++pattern)
				{
					model(in[pattern], output);
					if ((output(0) != 0) && (1.0 - output(0) != 0))
						ce -= target(pattern, 0) * log(output(0)) + (1.0 - target(pattern, 0)) * log(1.0 - output(0));
					else if (output(0) != target(pattern, 0))
						ce -= minLog;
				}
			}
		}
		return ce;
	}

//===========================================================================
	/*!
	 *  \brief Calculates the derivatives of the CrossEntropyIndependent error
	 *         (see #error) with respect to the parameters ModelInterface::w.
	 *
	 *  Calculates the CrossEntropyIndependent error derivative for \em N patterns and
	 *  \em C>1 independent attributes within the output vector corresponding to model parameter
	 *  \em w via
	 *  \f[
	 *      \frac{\partial E}{\partial w} = - \sum_{i=1}^N \sum_{k=1}^{C}
	 *                                        \left\{ \frac{tar^i_k - model_k(in^i)}{model_k(in^i)\cdot(1-model_k(in^i))}
	 *					  \cdot\frac{\partial model_k(in^i)}{\partial w}\right\}
	 *  \f]
	 *  or the derivative of the true cross entropy for only one single output dimension via
	 *  \f[
	 *  \frac{\partial E}{\partial w} = - \sum_{i=1}^N
	 *                                    \left\{ \frac{tar^i - model(in^i)}{model(in^i)\cdot(1-model(in^i))}
	 *				      \cdot\frac{\partial model(in^i)}{\partial w}\right\}
	 *  \f]
	 *  Usually, as a byproduct of the calculation of the derivative one gets the
	 *  CrossEntropyIndependent error itself very efficiently. Therefore, the method #error
	 *  gives back this value. This additional effect can be switched of by means
	 *  of the third parameter (returnError = false).
	 *
	 *      \param  in          Input vector for the model.
	 *      \param  target      Target vector.
	 *      \param  returnError Determines whether or not to calculate the cross
	 *                          entropy error itself. By default the
	 *                          error is calculated.
	 *      \return The CrossEntropyIndependent error if \em returnError is set to "true",
	 *              "-1" otherwise.
	 *
	 *  \author  M. Huesken
	 *  \date    1999
	 *
	 *  \par Changes
	 *      Revision 2003/06/03 (S. Wiegand):
	 *      bugs in formulars corrected
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	double errorDerivative(Model& model, const Array<double> &in, const Array<double> &target, Array<double>& derivative)
	{
		derivative.resize(model.getParameterDimension(), false);
		derivative = 0;
		Array<double> dmdw;
		double ce = 0;
		if (model.getOutputDimension() > 1)
			// more then one output neuron
			if (in.ndim() == 1)
			{
				// one input pattern
				Array<double> output(target.dim(0));
				model.modelDerivative(in, output, dmdw);
				if (returnError)
				{
					// calculate CrossEntropyIndependent
					for (unsigned c = 0; c < target.dim(0); c++)
						if ((output(c) != 0) && (1.0 - output(c) != 0))
							ce -= target(c) * log(output(c)) + (1. - target(c)) * log(1. - output(c));
						else if (output(c) != target(c))
							ce -= minLog;
				}
				// calculate derivative
				for (unsigned i = 0; i < derivative.nelem(); i++)
					for (unsigned c = 0; c < output.nelem(); c++)
						if ((output(c) != 0) && (1.0 - output(c) != 0))
							derivative(i) -= ((target(c) - output(c)) / ((1. - output(c)) * output(c))) * dmdw(c, i);
						else if (output(c) != target(c))
							derivative(i) -= (2.0 * target(c) - 1.0) * maxDeriv * dmdw(c, i);
			}
			else
			{
				// more than one input pattern
				Array<double> output(target.dim(1));
				for (unsigned pattern = 0; pattern < in.dim(0); ++pattern)
				{
					model.modelDerivative(in[pattern], output, dmdw);
					if (returnError)
					{
						// calculate cross-entropy
						for (unsigned c = 0; c < target.dim(1); c++)
							if ((output(c) != 0) && (1.0 - output(c) != 0))
								ce -= target(pattern, c) * log(output(c)) + (1. - target(pattern, c)) * log(1. - output(c));
							else if (output(c) != target(pattern, c))
								ce -= minLog;
					}
					// calculate dericative
					for (unsigned i = 0; i < derivative.nelem(); i++)
						for (unsigned c = 0; c < output.nelem(); c++)
							if ((output(c) != 0) && (1.0 - output(c) != 0))
								derivative(i) -= ((target(pattern, c) - output(c)) / ((1. - output(c)) * output(c))) * dmdw(c, i);
							else if (output(c) != target(pattern, c))
								// the sign changes due to the value of the target, maxDeriv is devided by reason of computational bounds
								derivative(i) -= (2.0 * target(pattern, c) - 1.0) * maxDeriv / in.dim(0) * dmdw(c, i);
				}
			}
		else
			// one output neuron
			if (in.ndim() == 1)
			{
				// only one pattern
				Array<double> output(1);
				model.modelDerivative(in, output, dmdw);
				if (returnError)
				{
					// calculate ce
					if ((output(0) != 0) && (1.0 - output(0) != 0))
						ce -= (target(0) * log(output(0)) + (1.0 - target(0)) * log(1.0 - output(0)));
					else if (output(0) != target(0))
						ce -= minLog;
				}
				// calculate derivative
				for (unsigned i = 0; i < derivative.nelem(); i++)
					if ((output(0) != 0) && (1.0 - output(0) != 0))
						derivative(i) -= (target(0) - output(0)) / (output(0) * (1.0 - output(0))) * dmdw(0, i);
					else if (output(0) != target(0))
						derivative(i) -= (2.0 * target(0) - 1.0) * maxDeriv * dmdw(0, i);
			}
			else
			{
				// a number of patterns
				Array<double> output(1);
				for (unsigned pattern = 0; pattern < in.dim(0); ++pattern)
				{
					model.modelDerivative(in[pattern], output, dmdw);
					if (returnError)
					{
						// calculate ce
						if ((output(0) != 0) && (1.0 - output(0) != 0))
							ce -= target(pattern, 0) * log(output(0)) + (1.0 - target(pattern, 0)) * log(1.0 - output(0));
						else if (output(0) != target(pattern, 0))
							ce -= minLog;
					}
					// calculate derivative
					for (unsigned i = 0; i < derivative.nelem(); i++)
						if ((output(0) != 0) && (1.0 - output(0) != 0))
							derivative(i) -= (target(pattern, 0) - output(0)) / (output(0) * (1.0 - output(0))) * dmdw(0, i);
						else if (output(0) != target(pattern, 0))
							// the sign changes due to the value of the target, maxDeriv is devided by reason of computational bounds
							derivative(i) -= (2.0 * target(pattern, 0) - 1.0) * maxDeriv / in.dim(0) * dmdw(0, i);
				}
			}
		if (!returnError)
			ce = -1;
		return ce;
	}

};
#endif









