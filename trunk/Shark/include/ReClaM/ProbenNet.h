//===========================================================================
/*!
 *  \file ProbenNet.h
 *
 *  \brief Offers a predefined feed-forward neural network.
 *
 *  \author  C. Igel, M. H&uuml;sken
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


#ifndef FFNETS_H
#define FFNETS_H

#include <SharkDefs.h>

#include "ReClaM/LinOutMSEFFNet.h"
#include "ReClaM/NetParams.h"



//===========================================================================
/*!
 *  \brief This special network is optimal for benchmark tests with
 *         the "proben 1" set.
 *
 *  The "proben 1" set is a set of benchmarks and benchmarking
 *  rules for neural networks, that is based on real data
 *  instead of artifical problems.
 *  This special predefined feed-forward network with an alternative
 *  activation function for the hidden units and linear output units is
 *  optimal for the problems offered in the proben set.<BR>
 *  For more information about the proben set, please refer to
 *  <a href="http://www.ubka.uni-karlsruhe.de/cgi-bin/psview?document=/ira/1994/21">the proben 1 website</a>.
 *
 *  \author  C. Igel, M. H&uuml;sken
 *  \date    1999
 *
 *  \par Changes:
 *      none
 *
 *  \par Status:
 *      stable
 */
class ProbenNet : public LinOutMSEFFNet // feed-forward network
{
public:


//===========================================================================
	/*!
	 *  Constructor no. 1
	 *
	 *  \brief Creates an empty proben network with "in"
	 *         input neurons and "out" output neurons.
	 *
	 *  Only the input and output dimensions are set, but the network
	 *  will contain no neurons.
	 *
	 *      \param in Dimension of the input (no. of input neurons), the
	 *                default value is zero.
	 *      \param out Dimension of the output (no. of output neurons),
	 *                 the default value is zero.
	 *      \return None.
	 *
	 *  \author  C. Igel, M. H&uuml;sken
	 *  \date    1999
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	ProbenNet(const unsigned in = 0, const unsigned out = 0) :
			LinOutMSEFFNet(in, out)
	{}


//===========================================================================
	/*!
	 *  Constructor no. 2
	 *
	 *  \brief Creates a proben network with "in" input
	 *         neurons and "out" output neurons. Additionally, the array "cmat"
	 *         determines the topology (i.e., number of neurons and their
	 *         connections).
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
	 *  \author  C. Igel, M. H&uuml;sken
	 *  \date    1999
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	ProbenNet(const unsigned in, const unsigned out, const Array<int> &cmat) :
			LinOutMSEFFNet(in, out, cmat)
	{
		;
	}





//===========================================================================
	/*!
	 *  Constructor no. 3
	 *
	 *  \brief Creates a proben network with "in" input
	 *         neurons and "out" output neurons. Additionally, the arrays
	 *         "cmat" and "wmat" determine the topology (i.e., number of neurons
	 *         and their connections) as well as the connection weights.
	 *
	 *  A network with the given connections and weights will be created.
	 *
	 *      \param in Dimension of the input (no. of input neurons).
	 *      \param out Dimension of the output (no. of output neurons).
	 *      \param cmat The #connectionMatrix.
	 *      \param wmat The #weightMatrix.
	 *      \return None.
	 *
	 *  \author  C. Igel, M. H&uuml;sken
	 *  \date    1999
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	ProbenNet(const unsigned in,
			  const unsigned out,
			  const Array<int> &cmat,
			  const Array<double>& wmat) :
			LinOutMSEFFNet(in, out, cmat, wmat)
	{
		;
	}


//===========================================================================
	/*!
	 *  Constructor no. 4
	 *
	 *  \brief Creates a proben network by reading the necessary
	 *         information from a file named "filename".
	 *
	 *  A file is used to create a new network. This
	 *  file must have the following content:
	 *  The first line of the file must contain two numbers that specify
	 *  the number of input and the number of output neurons of the
	 *  network, respectively.<BR>
	 *  This line is followed by the values for the #connectionMatrix.<BR>
	 *  The third and last part are the values for the #weightMatrix.<BR>
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
	 *  to the output neuron. Each of the two connections has
	 *  a weight of "3".<BR>
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
	ProbenNet(const std::string &filename) : LinOutMSEFFNet(filename)
	{
		;
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
	 *  \f$i \mbox{,\ } c \leq i < M - n\f$    is given as
	 *
	 *  \f$
	 *      z_i = g_{hidden}(x) = \frac{a}{1 + fabs(a)}
	 *  \f$
	 *
	 *  where \f$a\f$ as the propagated result of the input for
	 *  the previous neurons is calculated as
	 *
	 *  \f$
	 *      a = \left( \sum_{j=0}^{j<i} w_{ij} z_j + \Theta_i \right)
	 *  \f$
	 *
	 *  and \f$ \Theta_i \f$ denotes the bias term.<BR>
	 *  This alternative activation function maps inputs to the
	 *  values \f$-1, \dots, 1\f$ as the \f$tanh\f$ function does,
	 *  but in contrast to \f$tanh\f$ the alternative function
	 *  is more simple to calculate, more stable in a numerical
	 *  way and saturates more slowly what has a positive effect
	 *  when learning generalizations.
	 *
	 *      \param  a Input for the activation function, see above.
	 *      \return \f$ z_i \f$.
	 *
	 *  \author  M. H&uuml;sken, C. Igel
	 *  \date    1999
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
	double g(double a)
	{
		return a / (1 + fabs(a));
	}


//===========================================================================
	/*!
	 *  \brief Computes the derivative of the activation function
	 *         \f$g_{hidden}(x)\f$ for the hidden neurons.
	 *
	 *  The derivative function \f$g^*_{\mbox{hidden}}\f$ is defined
	 *  as
	 *
	 *  \f$
	 *      g^*_{hidden}(g_{hidden}(x)) =
	 *      \frac{\partial g_{hidden}(x)}{\partial x}
	 *  \f$
	 *      \param  ga The value of \f$g_{hidden}(x)\f$.
	 *      \return The result of \f$g^*_{hidden}(g_{hidden}(x))\f$
	 *
	 *  \author  M. H&uuml;sken, C. Igel
	 *  \date    1999
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
	double dg(double ga)
	{
		return (1 - sgn(ga) * ga) *(1 - sgn(ga) * ga);
	};
};
#endif

