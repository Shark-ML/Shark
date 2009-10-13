//===========================================================================
/*!
 *  \file LinOutFFNet.h
 *
 *  \brief Offers the functions to create and to work with
 *         (F)eed-(F)orward (Net)works with (Lin)ear (Out)put.
 *
 *  This file defines a class for a special type of feed-forward
 *  network, very similar to the networks defined in FFNet.h
 *  and MSEFFNet.h.
 *  The difference to the classes in these files is the activation
 *  function for the output neurons.
 *  Instead of sigmoid functions, linear activation functions are
 *  used here.
 *
 *  \author  M. H&uuml;sken
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


#ifndef LINOUTFFNET_H
#define LINOUTFFNET_H

#include <ReClaM/MSEFFNet.h>


//===========================================================================
/*!
 *  \brief Offers the functions to create and to work with
 *         (F)eed-(F)orward (Net)works with (Lin)ear (Out)put.
 *
 *  This class is very similar to the class defined in FFNet.h, but
 *  here linear activation functions for the output
 *  neurons are used instead of sigmoid functions.
 *
 *  \author  M. H&uuml;sken
 *  \date    1999
 *
 *  \par Changes:
 *      none
 *
 *  \par Status:
 *      stable
 */
class LinOutFFNet : public FFNet
{
public:

//===========================================================================
	/*!
	 *  Constructor no. 1
	 *
	 *  \brief Creates an empty linear output feed-forward network with "in"
	 *         input neurons and "out" output neurons.
	 *
	 *  Only the input and output dimensions are set, but the network
	 *  will contain no neurons.
	 *
	 *      \param in Dimension of the input (no. of input neurons).
	 *      \param out Dimension of the output (no. of output neurons).
	 *      \return None.
	 *
	 *  \author  M. H&uuml;sken
	 *  \date    1999
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	LinOutFFNet(const unsigned in = 0, const unsigned out = 0) : FFNet(in, out)
	{};


//===========================================================================
	/*!
	 *  Constructor no. 2
	 *
	 *  \brief Creates a linear output feed-forward network with "in" input
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
	 *  \author  M. H&uuml;sken
	 *  \date    1999
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	LinOutFFNet(const unsigned in,
				const unsigned out,
				const Array<int>& cmat) : FFNet(in, out, cmat)
	{};


//===========================================================================
	/*!
	 *  Constructor no. 3
	 *
	 *  \brief Creates a linear output feed-forward network with "in" input
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
	 *  \author  M. H&uuml;sken
	 *  \date    1999
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	LinOutFFNet(const unsigned in,
				const unsigned out,
				const Array<int>& cmat,
				const Array<double>& wmat) : FFNet(in, out, cmat, wmat)
	{};

//===========================================================================
	/*!
	 *  Constructor no. 4
	 *
	 *  \brief Creates a linear output feed-forward network by reading the
	 *         necessary information from a file named "filename".
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
	 *  to the output neuron. Each of the two connections has
	 *  a weight of "3".<BR>
	 *  There is also a connection with weight "2" from each neuron
	 *  (except the input neuron) to the bias value.
	 *
	 *  \author  M. H&uuml;sken
	 *  \date    1999
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	LinOutFFNet(const std::string & filename) : FFNet(filename)
	{};


//===========================================================================
	/*!
	 *  \brief Destructs an linear output feed forward network object.
	 *
	 *  This destructor is used to replace the standard destructor,
	 *  created by the compiler.
	 *
	 *  \warning none
	 *  \bug     none
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
	virtual ~LinOutFFNet()
	{ } ;

protected:

//===========================================================================
	/*!
	 *  \brief Activation function \f$g_{output}(x)\f$ of the output neurons.
	 *
	 *  The activation function is used for the propagation of the input
	 *  through the network.
	 *  Given a network with \f$M\f$ neurons, including \f$c\f$
	 *  input neurons and \f$n\f$ output neurons, the linear activation
	 *  function for the output neuron with index
	 *  \f$i \mbox{,\ } M - n \leq i < M \f$
	 *  is given as
	 *
	 *  \f$
	 *      z_i = g_{output}(x) = a
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
	 *  \param  a Input (and output) for the activation function, see above.
	 *  \return \f$ z_i \f$, because of linearity equal to the input itself.
	 *
	 *  \author  M. H&uuml;sken
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
	double gOutput(double a)
	{
		return a;
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
	 *      \frac{\partial g_{output}(x)}{\partial x} = 1.0
	 *  \f$
	 *
	 *  \param  ga The value of \f$g_{output}(x)\f$.
	 *  \return The result of \f$g^*_{output}(g_{output}(x))\f$, because
	 *          of linearity the constant "1.0".
	 *
	 *  \author  M. H&uuml;sken
	 *  \date    1999
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
	double dgOutput(double ga)
	{
		return 1.;
	}

};
#endif

