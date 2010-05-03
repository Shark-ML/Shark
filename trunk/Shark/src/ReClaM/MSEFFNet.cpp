//===========================================================================
/*!
 *  \file MSEFFNet.cpp
 *
 *  \brief Offers the functions to create and to work with a
 *         feed-forward network combined with the mean squared error
 *         measure. This combination is created due to computational
 *         efficiency.
 *
 *  \author  C. Igel
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
#include <ReClaM/MSEFFNet.h>
#include <sstream>


using namespace std;

//===========================================================================
/*!
 *  \brief Method is not implemented. Will always throw an exception.
 *
 *  \param  input
 *  \param  output
 *  \return None.
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
void MSEFFNet::modelDerivative(const Array<double> &input, Array<double> &output, Array<double>& derivative)
{
	throw SHARKEXCEPTION("modelDerivative(..) not implemented in MSEFFNet");
}


//===========================================================================
/*!
 *  \brief Method  is not implemented. Will always throw an exception.
 *
 *  \param  input
 *  \return None.
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
void MSEFFNet::modelDerivative(const Array<double> &input, Array<double>& derivative)
{
	throw SHARKEXCEPTION("modelDerivative(...) not implemented in MSEFFNet");
}


//===========================================================================
/*!
 *  \brief Calculates the mean squared error between the model output and
 *         the target values.
 *
 *  Measures the euklidian distance between the model output \f$model(in)\f$,
 *  calculated from the input pattern(s) \f$in\f$, and the target pattern(s)
 *  \f$target\f$.
 *  The result is then normalized to the number of output neurons.
 *  Consider the case of a N-dimensional output vector, i.e. a MSE feed
 *  forward network with \f$N\f$ output neurons, and a set of \f$P\f$
 *  patterns.
 *  In this case the function calculates
 *  \f[
 *      E = \frac{1}{N} \sum_{p=1}^P\sum_{i=1}^N(model(in)_{ip} -
 *      target_{ip})^{2}
 *  \f]
 *
 *      \param  input Input pattern(s) for the MSE feed forward net model.
 *      \param  target Target pattern(s).
 *      \return The mean squared error \f$E\f$.
 *
 *  \author  C. Igel
 *  \date    1999
 *
 *  \par Changes
 *      C. Igel, M. Toussaint, 2001-09-17:<BR>
 *      Normalising now with number of output neurons times number of patterns instead
 *      of just the number of patterns.
 *
 *  \par Status
 *      stable
 *
 */
double MSEFFNet::error(Model& model, const Array<double> &input, const Array<double> &target)
{
	double se = 0;
	if (input.ndim() == 1)
	{
		Array<double> output(target.dim(0));
		model.model(input, output);
		for (unsigned c = 0; c < target.dim(0); c++)
		{
			se += (target(c) - output(c)) * (target(c) - output(c));
		}
	}
	else
	{
		Array<double> output(target.dim(1));
		for (unsigned pattern = 0; pattern < input.dim(0); ++pattern)
		{
			model.model(input[pattern], output);
			for (unsigned c = 0; c < target.dim(1); c++)
			{
				se += (target(pattern, c) - output(c)) * (target(pattern, c) - output(c));
			}
		}
	}
	// normalise
	se /= (double) target.nelem();
	return se;
}

//===========================================================================
/*!
 *  \brief Calculates the mean squared error of the model output,
 *         compared to the target values, and the derivative of the
 *         mean squared error with respect to the weights.
 *
 *  Calculates
 *  \f$
 *      \frac{\partial E}{\partial w_i}
 *  \f$
 *  where \f$E\f$ is the mean squared error given in #error.
 *  The results are stored in the parameter derivative.
 *  Additionally, the mean squared error itself is returned.
 *
 *      \param  input Input pattern(s) for the model.
 *      \param  target The target pattern(s).
 *      \return The error \em E
 *
 *  \author  C. Igel
 *  \date    1999
 *
 *  \par Changes
 *      C. Igel, M. Toussaint, 2001-09-17:<BR>
 *      Normalising now with number of output neurons times number of patterns instead
 *      of just the number of patterns.
 *
 *  \par Status
 *      stable
 *
 */
double MSEFFNet::errorDerivative(Model& model, const Array<double> &input, const Array<double> &target, Array<double>& derivative)
{
	unsigned i, j;  // denote neurons
	unsigned c;     // denotes output
	unsigned pos;   // the actual position in the Array derivative
	double sum;
	double se = 0.;

	derivative  = 0;
	derivative.resize(model.getParameterDimension(), false);

	readParameters();

	if (input.ndim() == 1)
	{
		// calculate output
		activate(input);

		for (i = firstOutputNeuron, c = 0 ; i < numberOfNeurons; i++, c++)
		{
			d(i) =  2 * (target(c) - z[i]) * dgOutput(z[i]);
			se += (target(c) - z[i]) * (target(c) - z[i]);
		}

		// hidden units
		for (j = firstOutputNeuron - 1; j >= inputDimension; j--)
		{
			sum = 0;
			for (i = j + 1; i < numberOfNeurons; i++)
			{
				if (connection(i, j)) sum += weight(i, j) * d(i);
			}
			d(j) = dg(z[j]) * sum;
		}

		// calculate error gradient
		pos = 0;
		for (i = inputDimension; i < numberOfNeurons; i++)
		{
			for (j = 0; j < i; j++)
			{
				if (connection(i, j))
				{
					derivative(pos) -= d(i) * z[j];
					pos++;
				}
			}
			if (connection(i, bias))
			{ // bias
				derivative(pos) -= d(i);
				pos++;
			}
		}
	}
	else
	{ // more than one pattern
		for (unsigned pattern = 0; pattern < input.dim(0); ++pattern)
		{
			// calculate output
			activate(input[pattern]);

			for (i = firstOutputNeuron, c = 0 ; i < numberOfNeurons; i++, c++)
			{
				d(i) =  2 * (target(pattern, c) - z[i]) * dgOutput(z[i]);
			}

			// hidden units
			for (j = firstOutputNeuron - 1; j >= inputDimension; j--)
			{
				sum = 0;
				for (i = j + 1; i < numberOfNeurons; i++)
				{
					if (connection(i, j)) sum += weight(i, j) * d(i);
				}
				d(j) = dg(z[j]) * sum;
			}

			// calculate error gradient
			pos = 0;
			for (i = inputDimension; i < numberOfNeurons; i++)
			{
				for (j = 0; j < i; j++)
				{
					if (connection(i, j))
					{
						derivative(pos) -= d(i) * z[j];
						pos++;
					}
				}
				if (connection(i, bias))
				{ // bias
					derivative(pos) -= d(i);
					pos++;
				}
			}
		}
	}
	derivative /= (double) target.nelem(); // mt & igel 20010917

	se /= (double) target.nelem(); // mt & igel 20010917

	return se;
}


//===========================================================================
/*!
 *  Constructor no. 1
 *
 *  \brief Creates an empty MSE feed-forward network with "in"
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
 *  \date    1999
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
MSEFFNet::MSEFFNet(const unsigned in, const unsigned out) : FFNet(in, out)
{}


//===========================================================================
/*!
 *  Constructor no. 2
 *
 *  \brief Creates a MSE feed-forward network with "in" input neurons and
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
 *  \date    1999
 *
 *  \par Changes
 *      2002-01-09, ci <br>
 *      Memory leackage removed.
 *
 *  \par Status
 *      stable
 *
 */
MSEFFNet::MSEFFNet(const unsigned in, unsigned out,
				   const Array<int>& cmat) : FFNet(in, out, cmat)
{
	d.resize(numberOfNeurons);
}

//===========================================================================
/*!
 *  Constructor no. 3
 *
 *  \brief Creates a MSE feed-forward network with "in" input neurons and
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
 *  \date    1999
 *
 *  \par Changes
 *      2002-01-09, ci <br>
 *      Memory leackage removed.
 *
 *  \par Status
 *      stable
 *
 */
MSEFFNet::MSEFFNet(const unsigned in, unsigned out,
				   const Array<int>& cmat, const Array<double>& wmat) :
		FFNet(in, out, cmat, wmat)
{
	d.resize(numberOfNeurons, false);
}


//===========================================================================
/*!
 *  Constructor no. 4
 *
 *  \brief Creates a MSE feed-forward network by reading the necessary
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
 *                  doesn't exist, the method will exit with
 *                  failure.
 *  \return None.
 *
 *  \par Example
 *  <BR>
 *  1 1<BR>
 *  <BR>
 *  0 0 0 0<BR>
 *  1 0 0 1<BR>
 *  0 1 0 1<BR>
 *  <BR>
 *  0 0 0 0<BR>
 *  3 0 0 2<BR>
 *  0 3 0 2<BR>
 *  <BR>
 *
 *  A file with the content shown above will create a network
 *  with 1 input and 1 output neuron.<BR>
 *  A connection exists from the input neuron to the single
 *  hidden neuron of the network and from the hidden neuron
 *  to the output neuron, where each of the two connections
 *  has a weight of "3".<BR>.
 *  There is also a connection with weight "2" from each neuron
 *  (except the input neuron) to the bias value.
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
MSEFFNet::MSEFFNet(const string &filename) : FFNet(filename)
{
	d.resize(numberOfNeurons, false);
}

