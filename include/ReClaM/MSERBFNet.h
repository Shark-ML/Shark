//===========================================================================
/*!
 *  \file MSERBFNet.h
 *
 *  \brief Offers the functions to create and to work with
 *         radial basis function networks and to train it with the
 *         mean squared error. This combination provides more
 *         computational efficiency compared to using the class
 *         #MeanSquaredError
 *
 *  \author  C. Igel
 *  \date    2001
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

#ifndef MSERBFNET_H
#define MSERBFNET_H

#include "ReClaM/RBFNet.h"
#include "Array/Array.h"

//===========================================================================
/*!
 *  \brief Offers the functions to create and to work with
 *         radial basis function networks and to train it with the
 *         mean squared error. This combination provides more
 *         computational efficiency compared to using the class
 *         MeanSquaredError
 *
 *  Together with an error measure and an optimization algorithm class
 *  you can use this radial basis function network class to produce
 *  your own optimization tool.
 *  Radial Basis Function networks are especially used for
 *  classification tasks.
 *  For this purpose the class of models contains, besides
 *  the number of hidden neurons \f$nHidden\f$, 4 different kinds of
 *  parameters: the centers of the hidden neurons
 *  \f$\vec{m}_j=(m_{j1},\ldots,m_{jn})\f$, the standard deviations of
 *  the hidden neurons
 *  \f$\vec{\sigma}_j=(\sigma_{j1},\ldots,\sigma_{jn})\f$, the weights
 *  of the connections between the hidden neuron with index \f$j\f$
 *  and output neuron with index \f$k\f$ and the bias \f$b_k\f$ of the
 *  output neuron with index \f$k\f$. The output neuron with index
 *  \f$k\f$ yields the following output, assuming a
 *  \f$nInput\f$-dimensional input vector \f$\vec{x}\f$:
 *
 * \f[
 *   y_k = b_k + \sum_{j=1}^{nHidden} \exp\left[ -\frac{1}{2}\sum_{i=1}^{nInput}
 *          \frac{\left(x_i-m_{ji}\right)2}{\sigma_{ji}}\right]
 *
 * \f]
 *
 *  The mean squared error is used as the error measure.
 *
 * \f[
 *   E = \frac{1}{P\cdot nOutput}\sum_{p=1}^{P}\sum_{k=1}^{nOutput}
 *       \left(y_k^{(p)}-t_k^{(p)}\right)2
 * \f]
 *
 * Here, \f$P\f$ denotes the number of different patterns,
 * \f$y_k^{(p)}\f$ and \f$t_k^{(p)}\f$ the output and target value
 * belonging to the \f$p^{th}\f$ input pattern and \f$nOutput\f$ the
 * number of output neurons.
 *
 *  Please follow the link to view the source code of this example,
 *  that shows you, how you can construct your own radial basis function
 *  neural network, completely with one error measure for training
 *  and another for monitoring and an optimization
 *  algorithm.
 *  The example itself can be executed in the example directory
 *  of package ReClaM.
 *
 *
 *  \author  M. H&uuml;sken
 *  \date    2001
 *
 *  \par Changes:
 *      none
 *
 *  \par Status:
 *      stable
 */
class MSERBFNet : public RBFNet
{

public:

//===========================================================================
	/*!
	 *  \brief Creates a new Radial Basis Function Network (RBFN) with
	 *         mean squared error measuring.
	 *
	 *  This method creates a Radial Basis Function Network with
	 *  \em numInput input neurons, \em numOutput output neurons and \em numHidden
	 *  hidden neurons.
	 *
	 *
	 *      \param  numInput  Number of input neurons, equal to dimensionality of
	 *                        input space.
	 *      \param  numOutput Number of output neurons, equal to dimensionality
	 *                        of output space.
	 *      \param  numHidden Number of hidden neurons.
	 *      \return None.
	 *
	 *  \author  M. H&uuml;sken
	 *  \date    2001
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      none
	 *
	 */
	MSERBFNet(unsigned numInput, unsigned numOutput, unsigned numHidden)
			: RBFNet(numInput, numOutput, numHidden)
	{ }


//===========================================================================
	/*!
	 *  Constructor 2
	 *
	 *  \brief Creates a Radial Basis Function Network by reading information from
	 *         file "filename".
	 *
	 *  This method creates a Radial Basis Function Network (RBFN) which
	 *  is definded in the file \em filename.
	 *
	 *
	 *      \param  filename Name of the file the RBFN definition
	 *                       is read from.
	 *      \return None.
	 *
	 *  \author  L. Arnold
	 *  \date    2002
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	MSERBFNet(const std::string &filename) : RBFNet(filename)
	{};




//===========================================================================
	/*!
	 *  Constructor 3
	 *
	 *  This method creates a Radial Basis Function Network (RBFN) with
	 *  \em numInput input neurons, \em numOutput output neurons and \em numHidden
	 *  hidden neurons and initializes the parameters.
	 *
	 *
	 *  \param  numInput  Number of input neurons, equal to dimensionality of
	 *                    input space.
	 *  \param  numOutput Number of output neurons, equal to dimensionality of
	 *                    output space.
	 *  \param  numHidden Number of hidden neurons.
	 *
	 *  \param  _m        centers
	 *
	 *  \param  _A        output weights
	 *
	 *  \param  _b        biases
	 *
	 *  \param  _v        variances
	 *
	 *  \return None.
	 *
	 *  \author  C. Igel
	 *  \date    2003
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	MSERBFNet(unsigned numInput, unsigned numOutput, unsigned numHidden,
			  const Array<double> &_m,
			  const Array<double> &_A,
			  const Array<double> &_b,
			  const Array<double> &_v)
			: RBFNet(numInput, numOutput, numHidden, _m, _A, _b, _v)
	{};




//===========================================================================
	/*!
	 *  \brief Returns the mean squared error of the Radial Basis Function
	 *         Network.
	 *
	 *  This method calculates the error of the RBFN.
	 *  The output of the model for the input \f$input\f$, given as
	 *  \f$model(input)\f$\em is compared with the \f$target\f$ values.
	 *  Given a Radial Basis Function network with \f$N\f$ output neurons
	 *  and \f$P\f$ input patterns as stimulus, the error is calculated
	 *  as:
	 *
	 * \f$
	 *   E = \frac{1}{P\cdot N}\sum_{p=1}^{P}\sum_{i=1}^{N}
	 *       \left(model(input)^{(p)}_i-target_i^{(p)}\right)2
	 * \f$
	 *
	 *      \param  input Vector of input patterns.
	 *      \param  target Vector of target values.
	 *      \return None.
	 *
	 *  \author  C. Igel
	 *  \date    1999
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	double error(Model& model, const Array< double >& input, const Array< double >& target)
	{
		Array<double> myParameter(model.getParameterDimension());
		for (int i = 0; i < model.getParameterDimension(); i++)
		{
			myParameter[i] = model.getParameter(i);
		}
		setParams(myParameter);
		return mse(input, target) / (double)outputDimension;
	}

//===========================================================================
	/*!
	 *  \brief Calculates the derivative of the error
	 *         \f$E\f$ with respect to the parameters of the network.
	 *         Additionally, the error itself is returned.
	 *
	 *  Calculates the derivates of the mean squared error \f$E\f$
	 *  (see #error) with respect
	 *  to the parameters ModelInterface::w and stores the result in
	 *  ModelInterface::dedw.
	 *
	 *  As a byproduct of the calculation of the derivative one gets the
	 *  the mean squared error \f$E\f$ itself very efficiently. Therefore,
	 *  the method #error returns this value. This additional effect can be
	 *  switched of by means of the third parameter (\em returnError = false).
	 *
	 *      \param  input Vector of input patterns.
	 *      \param  target Vector of target values.
	 *      \param  returnError Determines whether or not to calculate the error
	 *                          itself. By default the error is calculated.
	 *      \return The error \f$E\f$ if \em returnError is set to "true",
	 *              otherwise "-1" is returned.
	 *
	 *  \author  M. Kreutz
	 *  \date    1999
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	double errorDerivative(Model& model, const Array< double >& input, const Array< double >& target, Array<double> &derivative)
	{
		Array <double> myParameter(model.getParameterDimension());
		for (int i = 0;i < model.getParameterDimension();i++)
		{
			myParameter[i] = model.getParameter(i);
		}

		setParams(myParameter);

		double mse = gradientMSE(input, target, derivative) / (double)outputDimension;
		derivative /= (double)outputDimension;
		return mse;
	}
};


#endif

