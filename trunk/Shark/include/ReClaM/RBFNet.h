//===========================================================================
/*!
 *  \file RBFNet.h
 *
 *  \brief Offers the functions to create and to work with
 *         radial basis function networks.
 *
 *  \author  C. Igel
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

#ifndef RBFNET_H
#define RBFNET_H

#include <fstream>
#include <iostream>
#include <sstream>
#include <SharkDefs.h>
#include <Array/ArrayOp.h>
#include <Array/Array.h>
#include <Array/ArrayIo.h>
#include <ReClaM/Model.h>
#include <Mixture/RBFN.h>


//===========================================================================
/*!
 *  \brief Offers the functions to create and to work with
 *         radial basis function network.
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
 *
 *  \author  C. Igel
 *  \date    1999
 *
 *  \par Changes:
 *      none
 *
 *  \par Status:
 *      stable
 */
class RBFNet : protected RBFN, public Model
{
public:
	void read(std::istream& is);
	void write(std::ostream& os) const;

//===========================================================================
	/*!
	 *  Constructor 1
	 *
	 *  \brief Creates a Radial Basis Function Network.
	 *
	 *  This method creates a Radial Basis Function Network (RBFN) with
	 *  \em numInput input neurons, \em numOutput output neurons and \em numHidden
	 *  hidden neurons.
	 *
	 *
	 *  \param  numInput  Number of input neurons, equal to dimensionality of
	 *                    input space.
	 *  \param  numOutput Number of output neurons, equal to dimensionality of
	 *                    output space.
	 *  \param  numHidden Number of hidden neurons.
	 *  \return None.
	 *
	 *  \author  C. Igel
	 *  \date    1999
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      none
	 *
	 */
	RBFNet(unsigned numInput, unsigned numOutput, unsigned numHidden)
			: RBFN(numInput, numOutput, numHidden)
	{
		Model::inputDimension = numInput;
		Model::outputDimension = numOutput;
		Model::parameter.resize(b.nelem() + A.nelem() + m.nelem() + v.nelem());
		getParams(Model::parameter);
	}


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
	RBFNet(const std::string &filename)
	{
		std::ifstream input(filename.c_str());
		if (!input)
		{
			std::stringstream s;
			s << "cannot open net file " << filename << std::endl;
			throw SHARKEXCEPTION(s.str().c_str());
		}
		input >> *this;
		input.close();
	}



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
	RBFNet(unsigned numInput, unsigned numOutput, unsigned numHidden,
		   const Array<double> &_m,
		   const Array<double> &_A,
		   const Array<double> &_b,
		   const Array<double> &_v)
			: RBFN(numInput, numOutput, numHidden)
	{
		outputDimension = numOutput;
		inputDimension  = numInput;

		m = _m; v = _v; A = _A; b = _b;

		Model::parameter.resize(b.nelem() +    // bias
								A.nelem() +   // weights
								m.nelem() +   // centers
								v.nelem());   // variances

		getParams(Model::parameter);
	}


//===========================================================================
	/*!
	 *  \brief Initializes the Radial Basis Function Network.
	 *
	 *  This method initializes the parameters of the RBFN-model. First,
	 *  it performs a K-means-Clustering to determine the centers and
	 *  variances of the hidden neurons. Thereafter, the connection
	 *  weights and bias values are determined by means of a linear
	 *  regression.
	 *
	 *      \param  input Vector of input values.
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
	 *      none
	 *
	 */
	void initRBFNet(const Array< double >& input, const Array< double >& target)
	{
		RBFN::initialize(input, target);
		getParams(Model::parameter);
	}

//===========================================================================
	/*!
	 *  \brief Evaluates the RBFN-Model and calculates the output values.
	 *
	 *  This method evaluates the RBFN and determines the output values
	 *  based on the following formula:
	 * \f[
	 *   y_k = b_k + \sum_{j=1}^{nHidden} \exp\left[ -\frac{1}{2}\sum_{i=1}^{nInput}
	 *          \frac{\left(x_i-m_{ji}\right)2}{\sigma_{ji}}\right]
	 *
	 * \f]
	 *
	 *      \param  input Vector of input values.
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
	 *      none
	 *
	 */
	void model(const Array< double >& input, Array< double >& target)
	{
		setParams(Model::parameter);
		recall(input, target);
	}

	//! Modifies a specific model parameter.
	void setParameter(unsigned int index, double value) {
		Model::setParameter(index,  value);
		setParams(Model::parameter);
	}


//===========================================================================
	/*!
	 *  \brief Calculates the derivatives of the model with respect to the
	 *         weights and stores the results in ModelInterface::dmdw.
	 *
	 *      \param  input Vector of input values.
	 *      \return None.
	 *
	 *  \author  C. Igel
	 *  \date    1999
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      none
	 *
	 */
	void modelDerivative(const Array<double>& input, Array<double>& derivative)
	{
		setParams(Model::parameter);
		gradientOut(input, derivative);
	}

//===========================================================================
	/*!
	 *  \brief Calculates the derivatives of the model with respect to the
	 *         weights and stores the results in ModelInterface::dmdw.
	 *         Furthermode, the output of the model is evaluated.
	 *
	 *      \param  input Vector of input values.
	 *      \param  output Vector of evaluated output values.
	 *      \return None.
	 *
	 *  \author  C. Igel
	 *  \date    1999
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      none
	 *
	 */
	void modelDerivative(const Array<double>& input, Array<double>& output, Array<double>& derivative)
	{
		setParams(Model::parameter);
		recall(input, output);
		gradientOut(input, derivative);
	}

//===========================================================================
	/*!
	 *  \brief Returns the weights of the connections between hidden and
	 *         output neurons.
	 *
	 *  Returns an array containing the weights of the connections between
	 *  hidden and output neurons. The array is a two-dimensional object,
	 *  the element \f$w_{ji}\f$ is the weight of the connection from the
	 *  \f$i^{th}\f$ hidden neuron to the \f$j^{th}\f$ output neuron.
	 *
	 *      \return The weight matrix.
	 *
	 *  \author  M. H&uuml;sken
	 *  \date    2001
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	const Array<double>& getWeights() const
	{
		//setParams(Model::parameter);
		return A;
	}

//===========================================================================
	/*!
	 *  \brief Returns the bias of all output neurons.
	 *
	 *  The bias values are contained in an 1-dimensional array.
	 *  The element
	 *  \f$b_{k}\f$ is the bias of the \f$k^{th}\f$ output neuron.
	 *
	 *      \return The bias values of the output neurons.
	 *
	 *  \author  M. H&uuml;sken
	 *  \date    2001
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	const Array<double>& getBias() const
	{
		//setParams(Model::parameter);
		return b;
	}

//===========================================================================
	/*!
	 *  \brief Returns the centers of all hidden neurons.
	 *
	 *  The center values are contained in an 2-dimensional array.
	 *  The element \f$m_{ji}\f$ is the \f$i^{th}\f$ coordinate
	 *  of the center of the \f$j^{th}\f$ hidden neuron.
	 *
	 *      \return The center values of the hidden neurons.
	 *
	 *  \author  M. H&uuml;sken
	 *  \date    2001
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	const Array<double>& getCenter() const
	{
		//setParams(Model::parameter);
		return m;
	}

//===========================================================================
	/*!
	 *  \brief Returns the variances of all hidden neurons.
	 *
	 *  The variances are contained in an 2-dimensional array.
	 *  The element \f$\sigma_{ji}\f$ is the \f$i^{th}\f$ coordinate
	 *  of the variance of the \f$j^{th}\f$ hidden neuron.
	 *
	 *      \return The variances of the hidden neurons.
	 *
	 *  \author  M. H&uuml;sken
	 *  \date    2001
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	const Array<double>& getVariance() const
	{
		//setParams(Model::parameter);
		return v;
	}

//===========================================================================
	/*!
	 *  \brief Returns the number of hidden neurons.
	 *
	 *      \return The number of hidden neurons.
	 *
	 *  \author  M. H&uuml;sken
	 *  \date    2001
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	const unsigned getNHidden() const
	{
		//setParams(Model::parameter);
		return v.dim(0);
	}

//===========================================================================
	/*!
	 *  \brief Sets the weights of connections between hidden and output neurons.
	 *
	 *  Sets the weights of the connections between hidden and output
	 *  neurons. The weights are described by means of a two-dimensional
	 *  array, the element \f$w_{ji}\f$ is the weight of the connection
	 *  from the \f$i^{th}\f$ hidden neuron to the \f$j^{th}\f$ output
	 *  neuron.
	 *
	 *      \param _A The new weight values.
	 *      \return None.
	 *
	 *  \author  M. H&uuml;sken
	 *  \date    2001
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	void setWeights(const Array<double>& _A)
	{
		if ((A.ndim() == _A.ndim()) && (A.dim(0) == _A.dim(0)) && (A.dim(1) == _A.dim(1)))
			A = _A;
		getParams(Model::parameter);
	}

//===========================================================================
	/*!
	 *  \brief Sets the bias of the output neurons.
	 *
	 *  This method sets the bias of the output neurons. The bias is
	 *  described by means of an one-dimensional array, the element
	 *  \f$b_{k}\f$ is the bias of the \f$k^{th}\f$ output neuron.
	 *
	 *      \param _b The new bias values.
	 *      \return None.
	 *
	 *  \author  M. H&uuml;sken
	 *  \date    2001
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	void setBias(const Array<double>& _b)
	{
		if ((b.ndim() == _b.ndim()) && (b.dim(0) == _b.dim(0)))
			b = _b;
		getParams(Model::parameter);
	}

//===========================================================================
	/*!
	 *  \brief Sets the centers of the hidden neurons.
	 *
	 *  This method sets the centers of the hidden neurons. The centers
	 *  are described by means of a two-dimensional array, the element
	 *  \f$m_{ji}\f$ is the \f$i^{th}\f$ coordinate of the center of the
	 *  \f$j^{th}\f$ hidden neuron.
	 *
	 *      \param _m The new center values.
	 *      \return None.
	 *
	 *  \author  M. H&uuml;sken
	 *  \date    2001
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	void setCenter(const Array<double>& _m)
	{
		if ((m.ndim() == _m.ndim()) && (m.dim(0) == _m.dim(0)) && (m.dim(1) == _m.dim(1)))
			m = _m;
		getParams(Model::parameter);
	}

//===========================================================================
	/*!
	 *  \brief Sets the variances of the hidden neurons.
	 *
	 *  This method sets the variances of the hidden neurons. The variances
	 *  are described by means of a two-dimensional array, the element
	 *  \f$\sigma_{ji}\f$ is the \f$i^{th}\f$ coordinate of the variance of the
	 *  \f$j^{th}\f$ hidden neuron.
	 *
	 *      \param _v The new variance values.
	 *      \return None.
	 *
	 *  \author  M. H&uuml;sken
	 *  \date    2001
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	void setVariance(const Array<double>& _v)
	{
		if ((v.ndim() == _v.ndim()) && (v.dim(0) == _v.dim(0)) && (v.dim(1) == _v.dim(1)))
			v = _v;
		getParams(Model::parameter);
	}



}
; // End of class RBFNet


//===========================================================================
/*!
 *  \brief Writes a network to the output stream "os".
 *
 *  The number of input, output and hidden neurons and also the
 *  center, variance, weights and bias matrices are written to the
 *  output stream \em os.
 *
 *      \param os The output stream.
 *      \param net The net that will be written to the output stream.
 *      \return Reference to the output stream.
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
void RBFNet::write(std::ostream& os) const
{
	Array<double> tmp;
	tmp = getCenter();
	os << tmp.dim(1) << " " ;
	tmp = getBias();
	os << tmp.dim(0) << " " ;
	os << getNHidden() << "\n\n";
	writeArray(getWeights(), os, "", "\n", ' ');
	writeArray(getBias(), os, "", "\n", ' ');
	os << "\n";
	writeArray(getCenter(), os, "", "\n", ' ');
	writeArray(getVariance(), os, "", "\n", ' ');
}


//===========================================================================
/*!
 *  \brief Reads a network from the input stream "is".
 *
 *  The number of input, output and hidden neurons, and also the
 *  center, variance, weight and bias matrices
 *  are read from the input stream \em is,
 *  memory for the internal net data structures is reserved
 *  and the weight and bias values are written to the
 *  parameter vector Model::parameter.
 *
 *      \param is The input stream.
 *      \return Reference to the input stream.
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
void RBFNet::read(std::istream& is)
{
	unsigned i, j;
	unsigned NumInput, NumOutput, NumHidden;
	is >> NumInput >> NumOutput >> NumHidden;

	a.resize(NumHidden);
	for (i = 0; i < a.dim(0); i++)
		a(i) = 1.0 / NumHidden;

	Array<double> Weights(NumOutput, NumHidden);
	Array<double> Bias(NumOutput);
	Array<double> Center(NumHidden, NumInput);
	Array<double> Variance(NumHidden, NumInput);


	for (i = 0; i < Weights.dim(0); i++)
		for (j = 0; j < Weights.dim(1); j++)
			is >> Weights(i, j);

	for (i = 0; i < Bias.dim(0); i++)
		is >> Bias(i);

	for (i = 0; i < Center.dim(0); i++)
		for (j = 0; j < Center.dim(1); j++)
			is >> Center(i, j);

	for (i = 0; i < Variance.dim(0); i++)
		for (j = 0; j < Variance.dim(1); j++)
			is >> Variance(i, j);




	A.resize(NumOutput, NumHidden);
	b.resize(NumOutput);
	m.resize(NumHidden, NumInput);
	v.resize(NumHidden, NumInput);


	setWeights(Weights);
	setBias(Bias);
	setCenter(Center);
	setVariance(Variance);

	Model::inputDimension = NumInput;
	Model::outputDimension = NumOutput;
	Model::parameter.resize(b.nelem() + A.nelem() + m.nelem() + v.nelem());

	getParams(Model::parameter);
}

#endif

