//===========================================================================
/*!
 *  \file FFNet.h
 *
 *  \brief Offers the functions to create and to work with
 *         a feed-forward network.
 *
 *  \author  C. Igel
 *  \date    1999
 *
 *  \par Copyright (c) 1999-2002:
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

#ifndef FFNET_H
#define FFNET_H

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <list>
#include <string>
#ifdef __GNUC__
#include <sys/param.h>
#endif

#include <Rng/GlobalRng.h>
#include <Array/Array.h>
#include <Array/ArrayOp.h>
#include <Array/ArrayIo.h>
#include <ReClaM/Model.h>
#include <FileUtil/FileUtil.h>


//===========================================================================
/*!
 *  \brief Offers the functions to create and to work with
 *         a feed-forward network.
 *
 *  Together with an error measure and an optimization algorithm
 *  class you can use this feed-forward network class to produce
 *  your own optimization tool.
 *
 *  \author  C. Igel
 *  \date    1999
 *
 *
 *  \par Changes:
 *      none
 *
 *  \par Status:
 *      stable
 */
class FFNet : public Model
{

public:

	//! Reads the data for the creation of a network from an input stream.
	void read(std::istream& is);

	//! Writes the data of a network to an output stream.
	void write(std::ostream& os) const;

	//! Creates a feed-forward network by reading the necessary
	//! information from a file named "filename".
	FFNet(const std::string &filename);

	//! Creates an empty feed-forward network with "in" input neurons and
	//! "out" output neurons.
	FFNet(const unsigned in = 0, const unsigned out = 0);

	//! Creates a feed-forward network with "in" input neurons and
	//! "out" output neurons. Additionally, the array "cmat" determines
	//! the topology (i.e., number of neurons and their connections).
	FFNet(const unsigned in, const unsigned out,
		  const Array<int>& cmat);

	//! Creates a feed-forward network with "in" input neurons and
	//! "out" output neurons. Additionally, the arrays "cmat" and
	//! "wmat" determine the topology (i.e., number of neurons and their
	//! connections) as well as the connection weights.
	FFNet(const unsigned in, const unsigned out,
		  const Array<int>& cmat, const Array<double>& wmat);

	//! Destructs an feed forward network object.
	virtual ~FFNet();

//===========================================================================
	/*!
	 *  \brief Assigns the data of network "net" to the current network.
	 *
	 *  This operator is defined in order to prevent the implicit definition
	 *  by the compiler, because a feed forward network includes
	 *  dynamic structures, that must be properly assigned.
	 *
	 *      \param  net Second network, whose data is assigned to the
	 *                  current network.
	 *      \return A reference to the current network (after assignment).
	 *
	 *  \author  C. Igel
	 *  \date    1999
	 *
	 *  \par Changes
	 *      2002-01-08, ci: <br>
	 *      Added some commands to remove memory leackage
	 *      that lead crashes, too.
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	FFNet &operator=(const FFNet &net)
	{
		numberOfNeurons   = net.numberOfNeurons;
		bias              = net.bias;
		firstOutputNeuron = net.firstOutputNeuron;
		connectionMatrix  = net.connectionMatrix;
		weightMatrix      = net.weightMatrix;
		delta             = net.delta;
		d                 = net.d;
		parameter         = net.parameter;
		if (z) delete [] z;
		if (numberOfNeurons)
			z = new double[numberOfNeurons];
		else
			z = NULL;
		inputDimension    = net.getInputDimension();
		outputDimension   = net.getOutputDimension();

		return *this;
	}


//===========================================================================
	/*!
	 *  \brief Copies the values of object "net" to the current object.
	 *
	 *  This operator is defined in order to prevent the implicit definition
	 *  by the compiler (which would result in unwanted behavior, e.g. if the
	 *  object contains pointer members).
	 *
	 *  \param  net Object, which values are copied.
	 *  \return A reference to the current object (after the copy process).
	 *
	 *  \author  C. Igel
	 *  \date    2001
	 *
	 *  \par Changes
	 *      2002-01-08: ci <br>
	 *      Constructor totally changed to remove memory leackage
	 *      that lead to crashes, too.
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	FFNet(const FFNet &net)
	{
		z = NULL;
		*this = net;
	}


//===========================================================================
	/*!
	 *  \brief Returns the weight value at position \f$(i,j)\f$ in the
	 *         weight matrix.
	 *
	 *  The weight matrix corresponds to the connection matrix. If a connection
	 *  exists from neuron no. \f$j\f$ to neuron no. \f$i\f$ with \f$j < i\f$, then
	 *  the function will return a weight value greater than zero for this
	 *  connection. If there is no connection or if \f$j \geq i\f$ then the
	 *  function will return zero. <br>
	 *  If you have a \f$N \times N + 1\f$ matrix \f$w\f$, then \f$w_{ij}\f$ for
	 *  \f$j = N\f$ will give the weight of the connection between the
	 *  neuron no. \f$i\f$ and the bias value.
	 *
	 *      \param  i Connection end point.
	 *      \param  j Connection start point.
	 *      \return The weight of the connection \f$c_{ij}\f$.
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
	inline double& weight(unsigned i, unsigned j)
	{
		return weightMatrix(i, j);
	}

//===========================================================================
	/*!
	 *  \brief Returns the value at position \f$(i,j)\f$ of the connection matrix.
	 *
	 *  If a connection exists from neuron no. \f$j\f$ to neuron no. \f$i\f$
	 *  with \f$j < i\f$, then the function will return "1". If there is no
	 *  connection or if \f$j \geq i\f$ then the function will return zero. <br>
	 *  If you have a \f$N \times N + 1\f$ matrix \f$c\f$, then \f$c_{ij}\f$ for
	 *  \f$j = N\f$ will indicate the status of the connection between the
	 *  neuron no. \f$i\f$ and the bias value. For all input neurons, this status
	 *  value is always zero.
	 *
	 *      \param  i Connection end point.
	 *      \param  j Connection start point.
	 *      \return Status of connection \f$c_{ij}\f$.
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
	inline int& connection(unsigned i, unsigned j)
	{
		return connectionMatrix(i, j);
	}

	//! Initializes the random number generator and the weights of the net.
	virtual void initWeights(long seed = 42,
							 double l = -.5,
							 double h = .5);

	//! Initializes the weights of the net.
	virtual void initWeights(double l = -.5,
							 double h = .5);

//===========================================================================
	/*!
	 *  \brief Initializes the random number generator.
	 *
	 *      \param  s The init (seed) value for the random number generator.
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
	void seed(long s)
	{
		Rng::seed(s);
	};


	//! Returns the current #weightMatrix.
	const Array< double >& getWeights();


	//! Returns the current #connectionMatrix.
	const Array< int >& getConnections();


//===========================================================================
	/*!
	 *  \brief Returns the output value of neuron no. "i".
	 *
	 *  Given an index \em i, the function returns the value, that
	 *  was calculated by the activation function for the neuron no.
	 *  \em i for the last given input pattern.
	 *
	 *      \param  i Index of the neuron.
	 *      \return Output value of the neuron.
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
	double outputValue(unsigned i)
	{
		return z[i];
	}

	//! Uses the input pattern(s) \em input for the Feed Forward Net
	//! model to produce the output vector(s) \em output.
	void model(const Array<double>& input, Array<double>& output);

	//! Reads in one input pattern for the Feed Forward Net
	//! model and calculates the derivatives of the resulting network
	//! output with respect to the weights.
	void modelDerivative(const Array<double>& input, Array<double>& derivative);

	//! Reads in one input pattern for the Feed Forward Net
	//! model and calculates the derivatives of the resulting network
	//! output with respect to the weights. Furthermore, the
	//! network output is given back.
	void modelDerivative(const Array<double>& input, Array<double> &output, Array<double>& derivative);

	void generalDerivative(const Array<double>& input, const Array<double>& coeffs, Array<double>& derivative);

	//inline
	//! Reads the parameters (weight and bias values) from the parameter
	//! vector ModelInterface::w and stores them in the #weightMatrix.
	void readParameters();

	//inline
	//! Reads the parameters (weight and bias values) from the
	//! #weightMatrix and writes them to the parameter vector
	//! ModelInterface::w.
	void writeParameters();

	//! Based on a given #connectionMatrix the structure of the
	//! current network is modified.
	void setStructure(const Array<int>& cmat);

	//! Based on a given #weightMatrix the structure of the current
	//! network is modified.
	void setStructure(const Array<double>& wmat, bool preserve = false);

	//! Based on a given #connectionMatrix and a #weightMatrix the
	//! structure of the current network is modified.
	void setStructure(const Array<int>& cmat, const Array<double>& wmat);

	//! Sets connection structure of a network with two hidden layers.
	void setStructure(unsigned in, unsigned hidden1, unsigned hidden2, unsigned out,
										bool ff_layer  = true,  
										bool ff_in_out = true,  
										bool ff_all    = true,  
										bool bias      = true);

	//! Sets connection structure of the network.
	void setStructure(Array<unsigned> &layers,
										bool ff_layer  = true, 
										bool ff_in_out = true, 
										bool ff_all    = true, 
										bool bias      = true);

	//! Sets connection structure of a network with a single hidden layer.
	void setStructure(unsigned in, unsigned hidden, unsigned out,
										bool ff_layer  = true, 
										bool ff_in_out = true, 
										bool ff_all    = true, 
										bool bias      = true);

	//! Writes the network to an outout stream as plain C code. 
	void writeSource(std::ostream &os, const char *g, const char *gOut, unsigned p = 4);

	protected:

	//! Activation function \f$g_{hidden}(x)\f$ of the hidden neurons.
	virtual double  g(double a);

	//! Computes the derivative of the activation function
	//! \f$g_{hidden}(x)\f$ for the hidden neurons.
	virtual double  dg(double a);

	//! Activation function \f$g_{output}(x)\f$ of the output neurons.
	virtual double  gOutput(double a);

	//! Computes the derivative of the activation function
	//! \f$g_{output}(x)\f$ for the output neurons.
	virtual double  dgOutput(double a);

	//! Reserves memory for all internal net data structures.
	virtual void    resize();


	//inline
//===========================================================================
	/*!
	 *  \brief The input pattern "in" is used to activate the
	 *         neurons of the net and produce some result
	 *         at the output neurons.
	 *
	 *  The input vector \em in contains a pattern given to the
	 *  input neurons. The input will be propagated through the network and
	 *  produce some "reaction" at the output neurons, depending
	 *  one the weights of the single connections.
	 *  The activation of the \f$c\f$ input neurons is done by
	 *
	 *  \f$
	 *      z_i = in_i \mbox{,\ } 0 \leq i < c
	 *  \f$
	 *
	 *  so the input neurons only propagate the input values.
	 *  For the further propagation of the input values through the
	 *  hidden to the output neurons, sigmoid activation functions
	 *  given by the methods #g and #gOutput are used.
	 *
	 *  \param  in The input pattern.
	 *  \return None.
	 *
	 *  \author  C. Igel
	 *  \date    2002
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	void activate(const Array<double> &in)
	{
		unsigned i, j;
		for (j = 0; j < inputDimension; j++)
		{
			z[j] = in(j);
		}
		for (i = inputDimension; i < firstOutputNeuron; i++)
		{
			z[i] = 0;
			for (j = 0; j < i; j++)
				if (connectionMatrix(i, j)) z[i] += z[j] * weight(i, j);
			if (connectionMatrix(i, bias)) z[i] += weight(i, bias); // bias
			z[i] =  g(z[i]);
		}
		for (i = firstOutputNeuron; i < numberOfNeurons; i++)
		{
			z[i] = 0;
			for (j = 0; j < i; j++)
				if (connectionMatrix(i, j)) z[i] += z[j] * weight(i, j);
			if (connectionMatrix(i, bias)) z[i] += weight(i, bias); // bias
			z[i] = gOutput(z[i]);
		}
	}

	//! Performs a backpropagation to calculate the derivatives of the
	//! outputs with respect to the parameters of the network.
	void backprop(Array<double>& derivative);

	//! Auxiliary function for writeSource
	void replaceInString(const std::string &str, std::string &newString, const std::string &token, const std::string &newToken1, int newToken2, const std::string &newToken3);


	/*!
	 *  \brief Number of all network neurons.
	 *
	 *  This is the total number of neurons in the network, i.e. 
	 *  input, hidden and output neurons.
	 *
	 */
	unsigned numberOfNeurons;


	/*!
	 *  \brief Index of bias weight in the weight matrix.
	 *
	 *  The bias term can be considered as an extra input 
	 *  neuron \f$x_0\f$, whose activation is permanently set to \f$+1\f$.
	 *  The index is used for direct access of the weight for the bias 
	 *  connection in the weight matrix.
	 *
	 *  \sa weightMatrix
	 *
	 */
	unsigned bias;


	/*!
	 *  \brief Index of the first output neuron. 
	 *
	 *  This value is used for direct access of the output neurons,
	 *  especially when working with the list #z of activation
	 *  results for all neurons.
	 *
	 */
	unsigned firstOutputNeuron;


	/*! \brief Matrix that identifies the connection status between
	 *         all neurons.
	 * 
	 * If the network consists of \f$M\f$ neurons, this is a
	 * \f$M \times M + 1\f$ matrix where a value of "1" at position \f$(i,j)\f$
	 * denotes a connection between neuron \f$j\f$ to
	 * neuron \f$i\f$ with \f$j < i\f$, a value of zero means
	 * no connection. <br>
	 * The last column of the matrix indicates the connection of the
	 * current neuron (given by the row number) to the #bias value. <br>
	 * For all input neurons, this value is always zero.
	 *
	 * \f$ 
	 * C = \left( \begin{array}{cccccc} 
	 *         0          & 0          & 0          & \cdots & 0 & c_{0b}\\ 
	 *         c_{10}     & 0          & 0          & \cdots & 0 & c_{1b}\\ 
	 *         c_{20}     & c_{21}     & 0          & \cdots & 0 & c_{2b}\\ 
	 *         \vdots     & \vdots     & \vdots     & \ddots & \vdots & \vdots\\ 
	 *         c_{(M-1)0} & c_{(M-1)1} & c_{(M-1)2} & \cdots & 0 & c_{(M-1)b}\\ 
	 * \end{array} \right)\f$
	 *
	 * \sa weightMatrix
	 *
	 */
	Array< int > connectionMatrix;


	/*!
	 *  \brief Corresponding to the #connectionMatrix, this 
	 *         matrix defines the weight of the considered
	 *         connection.
	 *
	 * If the network consists of \f$M\f$ neurons, this is 
	 * a \f$M \times M + 1\f$ matrix where a
	 * value of \f$w\f$ at position \f$(i,j)\f$
	 * denotes, that the connection between neuron \f$j\f$ to
	 * neuron \f$i\f$ with \f$j < i\f$ has a weight of \f$w\f$.
	 * If \f$c_{ij} = 0\f$ in the #connectionMatrix, then
	 * \f$w_{ij} = 0\f$.
	 * For each neuron there is also a so-called #bias term
	 * \f$\Theta\f$ at the last position of each matrix row:
	 *
	 * \f$ 
	 * W = \left( \begin{array}{cccccc} 
	 *    0          & 0          & 0          & \cdots & 0    & \Theta_0\\ 
	 *    w_{10}     & 0          & 0          & \cdots & 0    & \Theta_1\\ 
	 *    w_{20}     & w_{21}     & 0          & \cdots & 0    & \Theta_2\\ 
	 *    \vdots     & \vdots     & \vdots     & \ddots & \vdots & \vdots\\ 
	 *    w_{(M-1)0} & w_{(M-1)1} & w_{(M-1)2} & \cdots & 0  & \Theta_{(M-1)}\\ 
	 * \end{array} \right)\f$
	 *
	 * \sa connectionMatrix
	 *
	 */
	Array< double > weightMatrix;


	/*!
	 *  \brief Used to store the current results of the activation
	 *         function for all neurons for the input pattern \f$x\f$.
	 *
	 * Given a network with \f$M\f$ neurons, including
	 * \f$c\f$ input and \f$n\f$ output neurons the single
	 * values for \f$z\f$ are given as:
	 * <ul>
	 *     <li>\f$z_i = x_i,\ \mbox{for\ } 0 \leq i < c\f$</li>
	 *     <li>\f$z_i = g_{hidden}(x),\ \mbox{for\ } c \leq i < M - n\f$</li>
	 *     <li>\f$z_i = y_{i-M+n} = g_{output}(x),\ \mbox{for\ } M - n \leq 
	 *                  i < M\f$</li>
	 * </ul> 
	 *
	 * \sa g, gOutput
	 *
	 */
	double *z;


	/*!
	 *  \brief Used for temporary storage for the local gradient delta
	 *         values for all output neurons.
	 *
	 * Given a network with \f$M\f$ neurons, including
	 * \f$c\f$ input and \f$n\f$ output neurons, \f$d\f$ is a
	 * \f$n \times M\f$ Array. The delta values for each output
	 * neuron are calculated
	 * in a backwards pass, starting with the output neurons.
	 * The delta values for
	 * output neuron \f$k \mbox{,\ } 0 \leq k < n\f$ are then
	 * given as:
	 *
	 * \f$
	 * d(k, j) = g^*_{\mbox{output}_k}
	 * \f$
	 *
	 * for \f$j = M - n + k\f$ and
	 *
	 * \f$
	 * d(k, j) = 0
	 * \f$
	 *
	 * for \f$j \neq M - n + k\f$.
	 * Taken these values as base, the delta values for the hidden
	 * neurons with index \f$j \mbox{,\ } c \leq j < M - n \f$ are
	 * calculated as:
	 *
	 * \f$
	 * d(k, j) = g^*_{hidden_j} \ast \sum_{i=j+1}^{i < M} w_{ij} \ast d(k, i)
	 * \f$
	 *
	 * where \f$w_{ij}\f$ is the weight of an existing connection
	 * from neuron no. \f$j\f$ to neuron no. \f$i\f$.
	 * These temporary values are then used for the calculation
	 * of the partial derivatives of the outputs with respect to
	 * the parameters as performed in #backprop. 
	 *
	 * \sa dg, dgOutput, backprop
	 *
	 */
	Array< double > d;


	Array< double > delta;

};


//! Now, that connections to the bias must be explicitly set,
//! there is no (declaration) difference between a feed forward
//! net and a feed forward net with bias.
typedef  FFNet BFFNet;

#endif



