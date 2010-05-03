//===========================================================================
/*!
 *  \file FFNet.cpp
 *
 *  \brief Offers the functions to create and to work with
 *         a feed-forward network.
 *
 *  \author  C. Igel
 *  \date    2002
 *
 *  \par Copyright (c) 2002-2001:
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

#include <SharkDefs.h>
#include <ReClaM/FFNet.h>
#include <sstream>


using namespace std;

//===========================================================================
/*!
 *  \brief Destructor
 *
 *
 *  \warning none
 *  \bug     none
 *
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      none
 *
 */
FFNet::~FFNet()
{
	if (z) delete [] z;
}

//===========================================================================
/*!
 *  \brief Activation function \f$g_{hidden}(x)\f$ of the hidden neurons.
 *
 *  The activation function is used for the propagation of the input
 *  through the network.
 *  Given a network with \f$M\f$ neurons, including \f$c\f$
 *  input neurons and \f$n\f$ output neurons, the sigmoid activation
 *  function for the hidden neuron with index
 *  \f$i \mbox{,\ } c \leq i < M - n\f$    is given by
 *
 *  \f$
 *      z_i = g_{hidden}(x) = \frac{1}{1 + \exp (-a)}
 *  \f$
 *
 *  where \f$a\f$ as the propagated result of the input for
 *  the previous neurons is calculated as
 *
 *  \f$
 *      a = \left( \sum_{j=0}^{j<i} w_{ij} z_j + \Theta_i \right)
 *  \f$
 *
 *  and \f$ \Theta_i \f$ denotes the bias term.
 *
 *      \param  a Input for the activation function, see above.
 *      \return \f$ z_i \f$.
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
 *  \sa gOutput
 *
 */
double FFNet::g(double a)
{
	return 1 / (1 + exp(-a));
}


//===========================================================================
/*!
 *  \brief Computes the derivative of the activation function
 *         \f$g_{hidden}(x)\f$ for the hidden neurons given \f$g_{hidden}(x)\f$.
 *
 *  The derivative function \f$g'_{\mbox{hidden}}\f$ is defined
 *  as
 *
 *  \f$
 *      g'_{hidden}(g_{hidden}(x)) =
 *      \frac{\partial g_{hidden}(x)}{\partial x}
 *  \f$
 *      \param  ga The value of \f$g_{hidden}(x)\f$.
 *      \return The result of \f$g^*_{hidden}(g_{hidden}(x))\f$
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
 *  \sa g
 *
 */
double FFNet::dg(double ga)
{
	return ga *(1 - ga);
}


//===========================================================================
/*!
 *  \brief Activation function \f$g_{output}(x)\f$ of the output neurons.
 *
 *  The activation function is used for the propagation of the input
 *  through the network.
 *  Given a network with \f$M\f$ neurons, including \f$c\f$
 *  input neurons and \f$n\f$ output neurons, the sigmoid activation
 *  function for the output neuron with index
 *  \f$i \mbox{,\ } M - n \leq i < M \f$
 *  is given as
 *
 *  \f$
 *      z_i = g_{output}(x) = \frac{1}{1 + \exp (-a)}
 *  \f$
 *
 *  where \f$a\f$ as the propagated result of the input for
 *  the previous neurons is calculated as
 *
 *  \f$
 *      a = \left( \sum_{j=0}^{j<i} w_{ij} z_j + \Theta_i \right)
 *  \f$
 *
 *  and \f$ \Theta_i \f$ denotes the bias term.
 *
 *  \param  a Input for the activation function, see above.
 *  \return \f$ z_i \f$.
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
 *  \sa g
 *
 */
double FFNet::gOutput(double a)
{
	return 1 / (1 + exp(-a));
}

//===========================================================================
/*!
 *  \brief Computes the derivative of the activation function
 *         \f$g_{output}(x)\f$ for the output neurons.
 *
 *  The derivative function \f$g^*_{\mbox{output}}\f$ is defined
 *  as
 *
 *  \f$
 *      g^*_{output}(g_{output}(x)) =
 *      \frac{\partial g_{output}(x)}{\partial x}
 *  \f$
 *
 *  \param  ga The value of \f$g_{output}(x)\f$.
 *  \return The result of \f$g^*_{output}(g_{output}(x))\f$
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
 *  \sa gOutput, dg
 *
 */
double FFNet::dgOutput(double ga)
{
	return ga *(1 - ga);
}

//===========================================================================
/*!
 *  \brief Initializes the random number generator and the weights of the net.
 *
 *  After initializing the random number generator with a \em seed value,
 *  the weight values of the network are initialized with uniformly
 *  distributed random numbers.
 *
 *      \param  seed Initialization value for random number generator, default
 *                   value is "42".
 *      \param  l Lower bound for weight random numbers, default value
 *                is "-0.5".
 *      \param  h Upper bound for weight random numbers, default value
 *                is "0.5".
 *      \return None.
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
void FFNet::initWeights(long seed, double l, double h)
{
	Rng::seed(seed);
	for (unsigned i = inputDimension; i < numberOfNeurons; i++)
	{
		for (unsigned j = 0; j < i; j++)
			if (connection(i, j)) weight(i, j) = Rng::uni(l, h);
		if (connectionMatrix(i, bias)) weight(i, bias) = Rng::uni(l, h);
	}
	writeParameters();
}

//===========================================================================
/*!
 *  \brief Initializes the weights of the net.
 *
 *  The weight values of the network are initialized with uniformly
 *  distributed random numbers.
 *
 *      \param  l Lower bound for weight random numbers, default value
 *                is "-0.5".
 *      \param  h Upper bound for weight random numbers, default value
 *                is "0.5".
 *      \return None.
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
 *  \sa initWeights(long seed, double l, double h)
 *
 */
inline
void FFNet::initWeights(double l, double h)
{
	for (unsigned i = inputDimension; i < numberOfNeurons; i++)
	{
		for (unsigned j = 0; j < i; j++)
			if (connection(i, j)) weight(i, j) = Rng::uni(l, h);
		if (connectionMatrix(i, bias)) weight(i, bias) = Rng::uni(l, h);
	}
	writeParameters();
}

//===========================================================================
/*!
 *  \brief Uses the input pattern(s) "in" for the Feed Forward Net
 *         model to produce the output vector(s) "output".
 *
 *  The given input pattern(s) in \em input are propagated forward
 *  through the net by using the #activate function for all
 *  patterns sequentially. After each propagation the result
 *  in the output neurons is stored in the output vector(s) \em output.
 *
 *  \param  input Input pattern(s) for the model.
 *  \param  output Output vector(s).
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
 *  \sa activate
 *
 */
inline
void FFNet::model(const Array<double> &input, Array<double> &output)
{
	readParameters();
	if (input.ndim() == 1)
	{
		output.resize(outputDimension, false);
		activate(input);
		for (unsigned i = firstOutputNeuron, c = 0 ; i < numberOfNeurons; i++, c++)
			output(c) = z[i];
	}
	else
	{
		output.resize(input.dim(0), outputDimension, false);
		for (unsigned pattern = 0; pattern < input.dim(0); pattern++)
		{
			activate(input[pattern]);
			for (unsigned i = firstOutputNeuron, c = 0 ; i < numberOfNeurons; i++, c++)
				output(pattern, c) = z[i];
		}
	}
}

//===========================================================================
/*!
 *  \brief Performs backpropagation to calculate the derivatives of the
 *         outputs with respect to the parameters of the network.
 *
 *  \f$
 *  \nabla f(w) = \left( \frac{\partial f}{\partial w_1}, \dots ,
 *                \frac{\partial f}{\partial w_n} \right)^T
 *  \f$
 *
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
inline
void FFNet::backprop(Array<double>& derivative)
{
	unsigned i, j;  // denote neurons
	unsigned c;     // denotes output
	unsigned pos;   // the actual position in the Array derivative
	double sum;

	// 1. the 'local gradient' deltas for each
	//    output are calculated

	// output neurons
	for (j = firstOutputNeuron; j < numberOfNeurons; j++)
		for (c = 0; c < outputDimension; c++)
			if (c + firstOutputNeuron == j)  d(c, j) = dgOutput(z[j]);
			else d(c, j) = 0;

	// hidden units
	for (j = firstOutputNeuron - 1; j >= inputDimension; j--)
	{
		for (c = 0; c < outputDimension; c++)
		{
			sum = 0;
			for (i = j + 1; i < numberOfNeurons; i++)
			{
				if (connection(i, j)) sum += weight(i, j) * d(c, i);
			}
			d(c, j) = dg(z[j]) * sum;
		}
	}

	// 2. the gradient for each output is calculated from
	//    delta
	pos = 0;
	for (i = inputDimension; i < numberOfNeurons; i++)
	{
		for (j = 0; j < i; j++)
		{
			if (connection(i, j))
			{
				for (c = 0; c < outputDimension; c++)
				{
					derivative(c, pos) += d(c, i) * z[j];
				}
				pos++;
			}
		}
		if (connection(i, bias))
		{
			for (c = 0; c < outputDimension; c++)
			{ // bias
				derivative(c, pos) += d(c, i);
			}
			pos++;
		}
	}
}

//===========================================================================
/*!
 *  \brief Reads in one input pattern for the Feed Forward Net
 *         model and calculates the derivatives of the resulting network
 *         output with respect to the weights. Furthermore, the
 *         network output is given back.
 *
 *  Equal to method #modelDerivative(const Array<double> &input), but here
 *  the output of the network is not only used for the calculation,
 *  but also stored in \em output.
 *
 *  \param  input Input pattern for the model.
 *  \param  output Output for the input pattern.
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
void FFNet::modelDerivative(const Array<double> &input, Array<double> &output, Array<double>& derivative)
{
	unsigned i;  // denotes neurons
	unsigned c;  // denotes output

	derivative.resize(getOutputDimension(), getParameterDimension(), false);
	derivative = 0.;

	readParameters();

	if (input.ndim() == 1)
	{
		output.resize(getOutputDimension(), false);

		// calculate output
		activate(input);

		for (i = firstOutputNeuron, c = 0 ; i < numberOfNeurons; i++, c++) {
			output(c) = z[i];
		}

		// backpropagation step
		backprop(derivative);
	}
	else
	{
		output.resize(input.dim(0), getOutputDimension(), false);
		for (unsigned pattern = 0; pattern < input.dim(0); pattern++)
		{
			activate(input[pattern]);
			for (i = firstOutputNeuron, c = 0 ; i < numberOfNeurons; i++, c++) {
				output(pattern, c) = z[i];
			}
			backprop(derivative);
		}
	}
};

void FFNet::generalDerivative(const Array<double>& input, const Array<double>& coeffs, Array<double>& derivative)
{
	unsigned i, j;  // denote neurons
	unsigned c;     // denotes output
	unsigned pos;   // the actual position in the Array dedw
	double sum;

	for (i = firstOutputNeuron, c = 0 ; i < numberOfNeurons; i++, c++)
	{
		delta(i) =  coeffs(c) * dgOutput(z[i]);
	}

	// hidden units
	for (j = firstOutputNeuron - 1; j >= inputDimension; j--)
	{
		sum = 0;
		for (i = j + 1; i < numberOfNeurons; i++)
		{
			if (connection(i, j)) sum += weight(i, j) * delta(i);
		}
		delta(j) = dg(z[j]) * sum;
	}

	// calculate error gradient
	pos = 0;
	for (i = inputDimension; i < numberOfNeurons; i++)
	{
		for (j = 0; j < i; j++)
		{
			if (connection(i, j))
			{
				derivative(pos) -= delta(i) * z[j];
				pos++;
			}
		}
		if (connection(i, bias))
		{ // bias
			derivative(pos) -= delta(i);
			pos++;
		}
	}
};

//===========================================================================
/*!
 *  \brief Reads in one input pattern for the Feed Forward Net
 *         model and calculates the derivatives of the resulting network
 *         output with respect to the weights.
 *
 *  The single input pattern \em in is used to #activate the neurons
 *  of the network. The results, given in the output neurons
 *  are then used to calculate the derivatives of the output
 *  with respect to the weights of the network.
 *
 *  \param  input The single input pattern. If more than one
 *                pattern is given, the method exits with
 *                an exception.
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
 *  \sa backprop
 *
 */
void FFNet::modelDerivative(const Array<double>& input, Array<double>& derivative)
{
	readParameters();

	derivative.resize(getOutputDimension(), getParameterDimension(), false);
	derivative = 0.;

	if (input.ndim() == 1)
	{
		activate(input);
		// backpropagation step
		backprop(derivative);
	}
	else
	{
		stringstream s;
		s << "the derivative of the model with respect to more than one input is not defined" << endl;
		throw SHARKEXCEPTION(s.str().c_str());
	}
};


//===========================================================================
/*!
 *  \brief Reserves memory for all internal net data structures.
 *
 *  For the internal administration of a network several dynamic
 *  data structures are used. Based on the size of the #connectionMatrix
 *  and the values of Model::inputDimension and
 *  ModelInterface::outputDimension, the memory for all other data
 *  is reserved.
 *
 *  \return None.
 *
 *  \author  C. Igel
 *  \date    2002
 *
 *  \par Changes
 *      2002-01-08, ci <br>
 *      Some commands added to remove memory leackage,
 *      that lead to crashes.
 *
 *  \par Status
 *      stable
 *
 */
void FFNet::resize()
{
	numberOfNeurons   = connectionMatrix.dim(0);
	bias              = numberOfNeurons;
	firstOutputNeuron = numberOfNeurons - outputDimension;

	unsigned numberOfWeights = 0; //numberOfNeurons - inputDimension;
	for (unsigned i = 0; i < connectionMatrix.dim(0); i++)
	{
		for (unsigned j = 0; j < i; j++)
			if (connectionMatrix(i, j)) numberOfWeights++;
		if (connectionMatrix(i, bias)) numberOfWeights++;
	}

	parameter.resize(numberOfWeights, false);
	d.resize(outputDimension, numberOfNeurons, false);
	delta.resize(numberOfNeurons, false);

	if (z) delete [] z;
	if (numberOfNeurons) z = new double[numberOfNeurons];
	else z = NULL;
}


//===========================================================================
/*!
 *  Constructor no. 1
 *
 *  \brief Creates an empty feed-forward network with "in"
 *         input neurons and "out" output neurons.
 *
 *  Only the input and output dimensions are set, but the network
 *  will contain no neurons.
 *
 *      \param in Dimension of the input (no. of input neurons).
 *      \param out Dimension of the output (no. of output neurons).
 *      \return None.
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
FFNet::FFNet(const unsigned in, const unsigned out)
{
	inputDimension   = in;
	outputDimension  = out;

	numberOfNeurons   = 0;
	bias              = 0;
	firstOutputNeuron = 0;

	z = NULL;
}


//===========================================================================
/*!
 *  Constructor no. 2
 *
 *  \brief Creates a feed-forward network with "in" input neurons and
 *         "out" output neurons. Additionally, the array "cmat" determines
 *         the topology (i.e., number of neurons and their connections).
 *
 *
 *  A network with the given connections will be created, memory for
 *  the #weightMatrix reserved, but the weights for all connections
 *  will be set to zero.
 *
 *      \param in Dimension of the input (no. of input neurons).
 *      \param out Dimension of the output (no. of output neurons).
 *      \param cmat The #connectionMatrix.
 *      \return None.
 *
 *  \author  C. Igel
 *  \date    2002
 *
 *  \par Changes
 *      2002-01-09, ci <br>
 *      Memory leackage removed.
 *
 *  \par Status
 *      stable
 *
 */
FFNet::FFNet(const unsigned in, unsigned out,
			 const Array<int>& cmat)
{
	if (cmat.ndim() != 2)
	{
		stringstream s;
		s << "connection matrix has not dimension two" << endl;
		throw SHARKEXCEPTION(s.str().c_str());
	}
	if (cmat.dim(0) != (cmat.dim(1) - 1))
	{
		stringstream s;
		s << "connection matrix has to be (N + 1) x N" << endl;
		throw SHARKEXCEPTION(s.str().c_str());
	}

	inputDimension   = in;
	outputDimension  = out;
	connectionMatrix = cmat;

	weightMatrix.resize(connectionMatrix, false);
	weightMatrix     = 0;

	z = NULL;

	resize();
	writeParameters();
}


//===========================================================================
/*!
 *  Constructor no. 3
 *
 *  \brief Creates a feed-forward network with "in" input neurons and
 *         "out" output neurons. Additionally, the arrays "cmat" and
 *         "wmat" determine the topology (i.e., number of neurons and their
 *         connections) as well as the connection weights.
 *
 *  A network with the given connections and weights will be created.
 *
 *      \param in Dimension of the input (no. of input neurons).
 *      \param out Dimension of the output (no. of output neurons).
 *      \param cmat The #connectionMatrix.
 *      \param wmat The #weightMatrix.
 *      \return None.
 *
 *  \author  C. Igel
 *  \date    2002
 *
 *  \par Changes
 *      2002-01-09, ci <br>
 *      Memory leackage removed.
 *
 *  \par Status
 *      stable
 *
 */
FFNet::FFNet(const unsigned in, unsigned out,
			 const Array<int>& cmat, const Array<double>& wmat)
{
	if (cmat.ndim() != 2)
	{
		stringstream s;
		s << "connection matrix has not dimension two" << endl;
		throw SHARKEXCEPTION(s.str().c_str());
	}
	if (cmat.dim(0) != (cmat.dim(1) - 1))
	{
		stringstream s;
		s << "connection matrix has to be (N + 1) x N" << endl;
		throw SHARKEXCEPTION(s.str().c_str());
	}
	if (wmat.ndim() != 2)
	{
		stringstream s;
		s << "weight matrix has not dimension two" << endl;
		throw SHARKEXCEPTION(s.str().c_str());
	}
	if (wmat.dim(0) != (wmat.dim(1) - 1))
	{
		stringstream s;
		s << "weight matrix has to be (N + 1) x N" << endl;
		throw SHARKEXCEPTION(s.str().c_str());
	}

	inputDimension   = in;
	outputDimension  = out;
	connectionMatrix = cmat;
	weightMatrix     = wmat;

	z = NULL;
	resize();
	writeParameters();
}


//===========================================================================
/*!
 *  Constructor no. 4
 *
 *  \brief Creates a feed-forward network by reading the necessary
 *         information from a file named "filename".
 *
 *  A file is used to create a new feed-forward network. This
 *  file must have the following content:
 *  The first line of the file must contain two numbers that specify
 *  the number of input and the number of output neurons of the
 *  network, respectively.<BR>
 *  This line is followed by the values for the #connectionMatrix.<BR>
 *  The third and last part are the values for the #weightMatrix.
 *
 *  \param filename Name of the file that contains the information
 *                  for the creation of the network. If the file
 *                  doesn't exist, the method will with
 *                  failure.
 *  \return None.
 *
 *  \par Example
 *  <BR>
 *  1 1<BR>
 *  <BR>
 *  0 0 0<BR>
 *  1 0 0<BR>
 *  0 1 0<BR>
 *  <BR>
 *  0 0 0 2<BR>
 *  3 0 0 2<BR>
 *  0 3 0 2<BR>
 *  <BR>
 *
 *  A file with the content shown above will create a network
 *  with 1 input and 1 output neuron.<BR>
 *  A connection exists from the input neuron to the single
 *  hidden neuron of the network and from the hidden neuron
 *  to the output neuron. Each of the two connections has
 *  a weight of "3".<BR>
 *  The connection of each neuron to the bias value has a weight of "2".
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
FFNet::FFNet(const string &filename)
{
	ifstream input(filename.c_str());
	if (!input)
	{
		stringstream s;
		s << "cannot open net file " << filename << endl;
		throw SHARKEXCEPTION(s.str().c_str());
	}
	z = NULL;
	input >> *this;
	input.close();
}


//===========================================================================
/*!
 *  \brief Based on a given #connectionMatrix and a #weightMatrix the
 *         structure of the current network is modified.
 *
 *  This function will use the information in the
 *  given #connectionMatrix \em cmat and the #weightMatrix \em wmat to
 *  modify the network.
 *
 *      \param cmat The #connectionMatrix that provides the
 *                  basis information for the modification
 *                  of the network.
 *      \param wmat The #weightMatrix that provides the
 *                  basis information for the modification
 *                  of the network.
 *      \return     None.
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
void FFNet::setStructure(const Array<int>& cmat, const Array<double>& wmat)
{
	if ((cmat.ndim() != 2) ||
			(wmat.ndim() != 2))
	{
		stringstream s;
		s << "connection or weight matrix has not dimension two" << endl;
		throw SHARKEXCEPTION(s.str().c_str());
	}

	if ((cmat.dim(0) != (cmat.dim(1) - 1)) ||
			(cmat.dim(0) != wmat.dim(0)) ||
			(cmat.dim(1) != wmat.dim(1)))
	{
		stringstream s;
		s << "connection and weight matrices have to be (N + 1) x N" << endl;
		throw SHARKEXCEPTION(s.str().c_str());
	}

	connectionMatrix = cmat;
	weightMatrix     = wmat;

	resize();
	writeParameters();
}



//===========================================================================
/*!
 *  \brief Creates a connection matrix for a network.
 *
 *  Automatically creates a connection matrix with several layers, with
 *  the numbers of neurons for each layer defined by \em layers and
 *  (standard) connections defined by \em ff_layer, \em ff_in_out,
 *  \em ff_all and \em bias.
 *
 *      \param  layers     contains the numbers of neurons for each
 *                         layer of the network.
 *      \param  ff_layer   if set to \em true, connections from
 *                         each neuron of layer \f$i\f$ to each neuron
 *                         of layer \f$i+1\f$ will be set for all layers.
 *      \param  ff_in_out  if set to \em true, connections from
 *                         all input neurons to all output neurons
 *                         will be set.
 *      \param  ff_all     if set to \em true, connections from all
 *                         neurons of layer \f$i\f$ to all neurons of
 *                         layers \f$j\f$ with \f$j > i\f$ will be set
 *                         for all layers \f$i\f$.
 *      \param  bias       if set to \em true, connections from
 *                         all neurons (except the input neurons)
 *                         to the bias will be set.
 *      \return none
 *
 *  \author  C. Igel, M. H&uuml;sken
 *  \date    2002
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
void FFNet::setStructure(Array<unsigned> &layers,
												 bool ff_layer,    // all connections between layers?
												 bool ff_in_out,   // shortcuts from in to out?
												 bool ff_all,      // all shortcuts?
												 bool bias)        // bias?
{
	unsigned N = 0;  // total number of neurons:
	unsigned row,    // connection matrix row    (target neuron)
	column, // connection matrix column (start neuron)
	k,      // counter.
	z_pos,  // first target neuron in next layer
	s_pos;  // first start neuron in current layer


	//
	// Calculate total number of neurons from the
	// number of neurons per layer:
	//
	for (k = 0; k < layers.dim(0); k++) N += layers(k);
	connectionMatrix.resize(N, N + 1);
	connectionMatrix = 0;

	//
	// set connections from each neuron of layer i to each
	// neuron of layer i + 1 for all layers:
	//
	if (ff_layer)
	{
		z_pos = layers(0);
		s_pos = 0;
		for (k = 0; k < layers.dim(0) - 1; k++)
		{
			for (row = z_pos; row < z_pos + layers(k + 1); row++)
				for (column = s_pos; column < s_pos + layers(k); column++)
					connectionMatrix(row, column) = 1;
			s_pos += layers(k);
			z_pos += layers(k + 1);
		}
	}

	//
	// set connections from all input neurons to all output neurons:
	//
	if (ff_in_out)
	{
		for (row = N - layers(layers.dim(0) - 1); row < N; row++)
			for (column = 0; column < layers(0); column++)
				connectionMatrix(row, column) = 1;
	}

	//
	// set connections from all neurons of layer i to
	// all neurons of layers j with j > i for all layers i:
	//
	if (ff_all)
	{
		z_pos = layers(0);
		s_pos = 0;
		for (k = 0; k < layers.dim(0) - 1; k++)
		{
			for (row = z_pos; row < z_pos + layers(k + 1); row++)
				for (column = 0; column < s_pos + layers(k); column++)
					connectionMatrix(row, column) = 1;
			s_pos += layers(k);
			z_pos += layers(k + 1);
		}
	}

	//
	// set connections from all neurons (except the input neurons)
	// to the bias values:
	//
	if (bias)
		for (k = layers(0); k < N; k++) connectionMatrix(k, N) = 1;

	weightMatrix.resize(connectionMatrix.dim(0), connectionMatrix.dim(1), false);
	weightMatrix = 0.;


	inputDimension = layers(0);
	outputDimension = layers(layers.dim(0) - 1);

	resize();
	parameter = 0;

	writeParameters();
}


//===========================================================================
/*!
 *  \brief Creates a connection matrix for a network with a single
 *         hidden layer.
 *
 *  Automatically creates a connection matrix for a network with
 *  three different layers: An input layer with \em in input neurons,
 *  an output layer with \em out output neurons and a single hidden layer
 *  with \em hidden hidden neurons.
 *  (Standard) connections can be defined by \em ff_layer,
 *  \em ff_in_out, \em ff_all and \em bias.
 *
 *      \param  in         number of input neurons.
 *      \param  hidden     number of neurons of the single hidden layer.
 *      \param  out        number of output neurons.
 *      \param  ff_layer   if set to \em true, connections from
 *                         each neuron of layer \f$i\f$ to each neuron
 *                         of layer \f$i+1\f$ will be set for all layers.
 *      \param  ff_in_out  if set to \em true, connections from
 *                         all input neurons to all output neurons
 *                         will be set.
 *      \param  ff_all     if set to \em true, connections from all
 *                         neurons of layer \f$i\f$ to all neurons of
 *                         layers \f$j\f$ with \f$j > i\f$ will be set
 *                         for all layers \f$i\f$.
 *      \param  bias       if set to \em true, connections from
 *                         all neurons (except the input neurons)
 *                         to the bias will be set.
 *      \return none
 *
 *  \author  C. Igel, M. H&uuml;sken
 *  \date    2002
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
void FFNet::setStructure(unsigned in, unsigned hidden, unsigned out,
												 bool ff_layer,  // all connections beteween layers?
												 bool ff_in_out, // shortcuts from in to out?
												 bool ff_all,    // all shortcuts?
												 bool bias)      // bias?
{
	Array<unsigned> layer(3);
	layer(0) = in;
	layer(1) = hidden;
	layer(2) = out;
	setStructure(layer, ff_layer, ff_in_out, ff_all, bias);
}


//===========================================================================
/*!
 *  \brief Creates a connection matrix for a network with two
 *         hidden layers.
 *
 *  Automatically creates a connection matrix for a network with
 *  four different layers: An input layer with \em in input neurons,
 *  an output layer with \em out output neurons and two hidden layers
 *  with \em hidden1 and \em hidden2 hidden neurons, respectively.
 *  (Standard) connections can be defined by \em ff_layer,
 *  \em ff_in_out, \em ff_all and \em bias.
 *
 *      \param  in         number of input neurons.
 *      \param  hidden1    number of neurons of the first hidden layer.
 *      \param  hidden2    number of neurons of the second hidden layer.
 *      \param  out        number of output neurons.
 *      \param  ff_layer   if set to \em true, connections from
 *                         each neuron of layer \f$i\f$ to each neuron
 *                         of layer \f$i+1\f$ will be set for all layers.
 *      \param  ff_in_out  if set to \em true, connections from
 *                         all input neurons to all output neurons
 *                         will be set.
 *      \param  ff_all     if set to \em true, connections from all
 *                         neurons of layer \f$i\f$ to all neurons of
 *                         layers \f$j\f$ with \f$j > i\f$ will be set
 *                         for all layers \f$i\f$.
 *      \param  bias       if set to \em true, connections from
 *                         all neurons (except the input neurons)
 *                         to the bias will be set.
 *      \return none
 *
 *  \author  C. Igel, M. H&uuml;sken
 *  \date    2002
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
void FFNet::setStructure(unsigned in, unsigned hidden1, unsigned hidden2, unsigned out,
												 bool ff_layer,   // all connections beteween layers?
												 bool ff_in_out,  // shortcuts from in to out?
												 bool ff_all,     // all shortcuts?
												 bool bias)       // bias?
{
	Array<unsigned> layer(4);
	layer(0) = in;
	layer(1) = hidden1;
	layer(2) = hidden2;
	layer(3) = out;
	setStructure(layer, ff_layer, ff_in_out, ff_all, bias);
}

//===========================================================================
/*!
 *  \brief Based on a given #connectionMatrix the structure of the
 *         current network is modified.
 *
 *  This function will use the information in the
 *  given #connectionMatrix \em cmat to modify the network.
 *  The weights for all neurons of the new #connectionMatrix
 *  are initialized by zero.
 *
 *      \param cmat The #connectionMatrix that provides the
 *                  basis information for the creation
 *                  of the network.
 *      \return     None.
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
void FFNet::setStructure(const Array<int>& cmat)
{
	if (cmat.ndim() != 2)
	{
		stringstream s;
		s << "connection matrix has not dimension two" << endl;
		throw SHARKEXCEPTION(s.str().c_str());
	}
	if (cmat.dim(0) != (cmat.dim(1) - 1))
	{
		stringstream s;
		s << "connection matrix has to be (N + 1) x N" << endl;
		throw SHARKEXCEPTION(s.str().c_str());
	}

	connectionMatrix = cmat;
	weightMatrix.resize(cmat.dim(0), cmat.dim(1), false);

	resize();
	parameter = 0;

	writeParameters();
}


//===========================================================================
/*!
 *  \brief Based on a given #weightMatrix the structure of the current
 *         network is modified.
 *
 *  This function will use the values of ModelInterface::inputDimension,
 *  ModelInterface::outputDimension and the information in the
 *  given #weightMatrix \em wmat to change the structure of the network.
 *  An existing connection matrix (that is compatible to the
 *  the new #weightMatrix \em wmat) can be adopted or a new one
 *  will be created with the data of the #weightMatrix.
 *
 *      \param wmat     The #weightMatrix that provides the
 *                      basis information for the modification
 *                      of the network.
 *      \param preserve If set to "true", the existing
 *                      #connectionMatrix will be kept.
 *                      In case of inconsistency between
 *                      the #weightMatrix and the #connectionMatrix
 *                      the function will exit with an exception.
 *      \return         None.
 *
 *  \author  C. Igel
 *  \date    2002
 *
 *  \par Changes
 *      2002-01-08, ci <br>
 *      Added some commands to remove memory leackage,
 *      that lead to crashes, too.
 *
 *  \par Status
 *      stable
 *
 */
void FFNet::setStructure(const Array<double>& wmat, bool preserve)
{
	if (wmat.ndim() != 2)
	{
		stringstream s;
		s << "weight matrix has not dimension two" << endl;
		throw SHARKEXCEPTION(s.str().c_str());
	}
	if (wmat.dim(0) != (wmat.dim(1) - 1))
	{
		stringstream s;
		s << "weight matrix has to be (N + 1) x N" << endl;
		throw SHARKEXCEPTION(s.str().c_str());
	}

	weightMatrix = wmat;
	connectionMatrix.resize(wmat.dim(0), wmat.dim(1), false);

	numberOfNeurons   = connectionMatrix.dim(0);
	bias              = numberOfNeurons;
	firstOutputNeuron = numberOfNeurons - outputDimension;

	unsigned numberOfWeights = 0;
	for (unsigned i = 0; i < connectionMatrix.dim(0); i++)
	{
		for (unsigned j = 0; j < i; j++)
		{
			if (preserve)
			{
				if (weightMatrix(i, j)  != 0.)
				{
					if (connectionMatrix(i, j) == 0)
					{
						stringstream s;
						s << "weight specified for non-existing connection" << endl;
						throw SHARKEXCEPTION(s.str().c_str());
					}
					numberOfWeights++;
				}
				else if (connectionMatrix(i, j) != 0) numberOfWeights++;
				if (weightMatrix(i, bias)  != 0.)
				{
					if (connectionMatrix(i, bias) == 0)
					{
						stringstream s;
						s << "weight specified for non-existing bias" << endl;
						throw SHARKEXCEPTION(s.str().c_str());
					}
					numberOfWeights++;
				}
				else if (connectionMatrix(i, bias) != 0) numberOfWeights++;
			}
			else
			{
				if (weightMatrix(i, j)  != 0.)
				{
					connectionMatrix(i, j) = 1;
					numberOfWeights++;
				}
				else
				{
					connectionMatrix(i, j) = 0;
				}
				if (weightMatrix(i, bias)  != 0.)
				{
					connectionMatrix(i, bias) = 1;
					numberOfWeights++;
				}
				else
				{
					connectionMatrix(i, bias) = 0;
				}
			}
		}
	}

	resize();
	writeParameters();
}

//===========================================================================
/*!
 *  \brief Reads the parameters (weights) from the
 *         #weightMatrix and writes them to the parameter vector
 *         Model::parameter.
 *
 *  In feed forward networks connections and the corresponding weights
 *  can exist only from neurons with no. \f$j\f$ to neurons with no. \f$i\f$
 *  with \f$j < i\f$.
 *  So to save space, the parameters, i.e. the values for the connection
 *  weights are internally stored in a vector.
 *  But a matrix offers a more concise view so this function reads in the
 *  original (extended) #weightMatrix and stores the values in the
 *  parameter vector Model::parameter.
 *
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
void FFNet::writeParameters()
{
	for (unsigned k = 0, i = 0; i < connectionMatrix.dim(0); i++)
	{
		for (unsigned j = 0; j < i; j++)
		{
			if (connectionMatrix(i, j))
			{
				parameter(k) = weight(i, j);
				k++;
			}
		}
		if (i >= inputDimension)
		{
			if (connectionMatrix(i, bias))
			{
				parameter(k) = weight(i, bias);
				k++;
			}
		}
	}
}

//===========================================================================
/*!
 *  \brief Reads the parameters (weights) from the parameter
 *         vector Model::parameter and stores them in the #weightMatrix.
 *
 *  In feed forward networks connections and the corresponding weights
 *  can exist only from neurons with no. \f$j\f$ to neurons with no.
 *  \f$i\f$ with \f$j < i\f$.
 *  So to save space, the parameters, i.e. the values for the connection
 *  weights are internally stored in a vector.
 *  But a matrix offers a more concise view so this function reads in the
 *  parameter vector Model::parameter and writes the values to the
 *  #weightMatrix.
 *
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
void FFNet::readParameters()
{
	for (unsigned k = 0, i = 0; i < connectionMatrix.dim(0); i++)
	{
		for (unsigned j = 0; j < i; j++)
		{
			if (connectionMatrix(i, j))
			{
				weight(i, j) = parameter(k);
				k++;
			}
		}
		if (i >= inputDimension)
		{
			if (connectionMatrix(i, bias))
			{
				weight(i, bias) = parameter(k);
				k++;
			}
		}
	}
}


//===========================================================================
/*!
 *  \brief Returns the current #weightMatrix.
 *
 *  The weight values are read from the parameter vector
 *  Model::parameter and stored in the #weightMatrix,
 *  then the matrix is returned.
 *
 *  \return The #weightMatrix.
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
const Array< double > &FFNet::getWeights()
{
	readParameters();
	return weightMatrix;
}

//===========================================================================
/*!
 *  \brief Returns the current #connectionMatrix.
 *
 *  \return The #connectionMatrix.
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
const Array< int > &FFNet::getConnections()
{
	return connectionMatrix;
}

//===========================================================================
/*!
 *  \brief Writes the data of a network to an output stream.
 *
 *  The Model::inputDimension, Model::outputDimension,
 *  the #connectionMatrix and the #weightMatrix are written to an output
 *  stream.
 *  For the syntax of a possible output, see the example section of
 *  #FFNet(const std::string &filename).
 *
 *      \param os The output stream to where the data is written.
 *      \param net The net that will be written to the output stream.
 *      \return Reference to the output stream.
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
void FFNet::write(ostream& os) const
{
	os << inputDimension << " " << outputDimension << "\n";
	writeArray(connectionMatrix, os, "", "\n", ' ');
	writeArray(weightMatrix, os, "", "\n", ' ');
}


//===========================================================================
/*!
 *  \brief Reads the data for the creation of a network from an input stream.
 *
 *  The Model::inputDimension, Model::outputDimension,
 *  the #connectionMatrix and the
 *  #weightMatrix are read from the input stream,
 *  memory for the internal net data structures is reserved
 *  and the weight and bias values are written to the
 *  parameter vector Model::parameter.
 *  For the syntax of a possible input, see the example section of
 *  #FFNet(const std::string &filename).
 *
 *      \param is The input stream, from which the data is read.
 *      \param net The net object, to which the data will be copied.
 *      \return Reference to the input stream.
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
void FFNet::read(istream& is)
{
	unsigned i, j, n, pos;
	double v, dn;
	vector<double> l;

	is >> inputDimension;
	is >> outputDimension;

	is >> v;
	while (is)
	{
		l.push_back(v);
		is >> v;
	}

	if (!l.size())
	{
		stringstream s;
		s << "no matrices given in net file";
		throw SHARKEXCEPTION(s.str().c_str());
	}

	dn = (sqrt(1. + l.size() * 2.) - 1.) / 2.;
	n = unsigned(dn);
	if (double(n) == dn)
	{ // new style with bias
		connectionMatrix.resize(n, n + 1, false);
		weightMatrix.resize(n, n + 1, false);

		pos = 0;
		for (i = 0; i < n; i++)
		{
			for (j = 0; j < n + 1; j++)
			{
				connectionMatrix(i, j) = unsigned(l[pos]);
				pos++;
			}
		}

		for (i = 0; i < n; i++)
		{
			for (j = 0; j < n + 1; j++)
			{
				weightMatrix(i, j) = l[pos];
				pos++;
			}
		}
	}
	else
	{ // old style without bias
		dn = (sqrt(1. + l.size() * 8.) - 1.) / 4.;
		n =  unsigned(dn);
		if (double(n) == dn)
		{
			connectionMatrix.resize(n, n + 1, false);
			weightMatrix.resize(n, n + 1, false);

			pos = 0;
			for (i = 0; i < n; i++)
			{
				for (j = 0; j < n; j++)
				{
					connectionMatrix(i, j) = unsigned(l[pos]);
					pos++;
				}
				// handle bias
				if (i >= inputDimension) connectionMatrix(i, n) = 1;
				else connectionMatrix(i, n) = 0;
			}
			for (i = 0; i < n; i++)
			{
				for (j = 0; j < n + 1; j++)
				{
					weightMatrix(i, j) = l[pos];
					pos++;
				}
			}
		}
		else
		{ // no style
			stringstream s;
			s << "cannot parse network configuration file" << endl;
			throw SHARKEXCEPTION(s.str().c_str());
		}
	}

	resize();
	writeParameters();
}


void FFNet::replaceInString(const std::string &str, std::string &newString, const std::string &token, const std::string &newToken1, int newToken2, const std::string &newToken3)
{
	std::ostringstream buffer;

	buffer << newToken2;
	std::string newToken = newToken1 + buffer.str() + newToken3;

	std::string::size_type pos;
	newString = str;

	pos = newString.find(token, 0);
	while (pos != std::string::npos)
	{
		newString.replace(pos, token.length(), newToken);
		pos = newString.find(token, 0);
	}
}

void FFNet::writeSource(std::ostream &os, const char *g, const char *gOut, unsigned p)
{
	unsigned i, j;
	bool first;
	std::string str;

	os.precision(p);

	os << "void model(double *in, double *out) {\n";
	for (j = 0; j < inputDimension; j++)
		os << "  double z" << j << " = in[" << j << "];\n";

	for (i = inputDimension; i < firstOutputNeuron; i++)
	{
		first = true;
		for (j = 0; j < i; j++)
			if (connectionMatrix(i, j))
			{
				if (first)
				{
					os << "  double z" << i <<  " = z" << j << " * " << weightMatrix(i, j);
					first = false;
				}
				else
				{
					os << " + z" << j << " * " << weightMatrix(i, j);
				}
			}

		if( connectionMatrix(i, bias) ) {
			if (first)
			{
				os  << "double z" << i << " = " << weightMatrix(i, bias); // bias
				first = false;
			}
			else
			{
				os << " + " << weightMatrix(i, bias);
			}
		}

		if (!first)
		{
			os << ";\n";
			replaceInString(g, str, "#", "z", i, "");
			os << "  z" << i << " = " << str << ";\n";
		}
	}
	for (i = firstOutputNeuron; i < numberOfNeurons; i++)
	{
		first = true;

		for (j = 0; j < i; j++)
			if (connectionMatrix(i, j))
			{
				if (first)
				{
					os << "  out[" << i - firstOutputNeuron << "] = " << "z" << j
					<< " * " << weightMatrix(i, j);
					first = false;
				}
				else
				{
					os << " + z" << j << " * " << weightMatrix(i, j);
				}
			}
		if( connectionMatrix(i, bias) ) {
			if (first)
			{
				os << "  out[" << i - firstOutputNeuron << "] = "
				<< weightMatrix(i, bias); // bias
				first = false;
			}
			else
			{
				os << " + " << weightMatrix(i, bias); // bias
			}
		}
		
		if (!first)
		{
			os << ";\n";
			if((strlen(gOut) != 1) || (*gOut != '#')) { // skip for linear output neurons
				replaceInString(gOut, str, "#", "out[", i - firstOutputNeuron, "]");
				os <<  "  out[" << i - firstOutputNeuron << "] = " << str << ";\n";
			}
		}
	}
	os << "};" << std::endl;
}


