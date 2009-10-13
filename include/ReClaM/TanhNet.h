//===========================================================================
/*!
 *  \file TanhNet.h
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

#include "SharkDefs.h"

#include "ReClaM/FFNet.h"
#include "ReClaM/NetParams.h"


//===========================================================================
/*!
 *  \brief Offers a predefined feed-forward neural network, which
 *         uses the "tanh" activation function for hidden and
 *         output neurons.
 *
 *  The \f$tanh\f$ (tangens hyperbolicus) function replaces the
 *  default logistic function as defined in ModelInterface.
 *  The \f$tanh\f$ function differs from the logistic function
 *  only through a linear transformation, but empirically,
 *  it is often found that \f$tanh\f$ activation functions give
 *  rise to faster convergence of training algorithms than
 *  logistic functions.
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
class TanhNet : public FFNet,

{
public:

//===========================================================================
	/*!
	 *  \brief Creates a tanh feed-forward network by reading the necessary
	 *         information from a file named "filename".
	 *
	 *  A file is used to create a new network. This
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
	 *  The connection of each neuron (except the input neuron)
	 *  to the bias value has a weight of "2".
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
	TanhNet(const std::string &filename) : FFNet(filename)
	{}


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
	 *      z_i = g_{hidden}(x) = tanh(a) = \frac{2}{(1 + e^{-2a}) - 1}
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
	 *  The \f$tanh\f$ function maps input values to \f$-1, \dots, 1\f$.
	 *
	 *      \param  a Input for the activation function, see above.
	 *      \return \f$ z_i \f$.
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
	 *  \sa gOutput
	 *
	 */
	double g(double a)
	{
		return tanh(a);
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
	 *  \author  C. Igel, M. H&uuml;sken
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
		return 1 - ga * ga;
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
	 *      z_i = g_{output}(x) = tanh(a) = \frac{2}{(1 + e^{-2a}) - 1}
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
	 *  The \f$tanh\f$ function maps input values to \f$-1, \dots, 1\f$.
	 *
	 *  \param  a Input for the activation function, see above.
	 *  \return \f$ z_i \f$.
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
	 *  \sa g
	 *
	 */
	double gOutput(double a)
	{
		return tanh(a);
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
	 *  \author  C. Igel, M. H&uuml;sken
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
		return 1 - ga * ga;
	}

};
#endif

