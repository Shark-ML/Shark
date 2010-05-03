//===========================================================================
/*!
 *  \file LMSEFFNet.h
 *
 *  \brief Offers the functions to create and to work with a
 *         feed-forward network with explicit defined neuron-to-bias
 *         connections.
 *         The network is combined with the mean squared error
 *         measure. This combination is created due to computational
 *         efficiency.
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

#ifndef LMSEFFNET_H
#define LMSEFFNET_H

#include "ReClaM/MSEFFNet.h"

//===========================================================================
/*!
 *  \brief Offers the functions to create and to work with a
 *         feed-forward network with explicit defined
 *         neuron-to-bias connections.
 *         The network is combined with the mean squared error
 *         measure. This combination is created due to computational
 *         efficiency.
 *
 *  When you are using the classes FFNet or MSEFFNet it is assumed,
 *  that each neuron is connected with the (B)ias value.
 *  These classes are a specialization of this class, where you
 *  have the free choice to connect each neuron to the bias value
 *  or not.
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
class LMSEFFNet : public MSEFFNet
{

public:
	//! Creates an empty MSE Bias feed-forward network with "in"
	//! input neurons and "out" output neurons.
	LMSEFFNet(const unsigned in = 0, const unsigned out = 0, double min = 0., double max = 1.) : MSEFFNet(in, out)
	{
		tmin = min, tmax = max;
	};


	//! Creates a MSE Bias feed-forward network with "in" input neurons and
	//! "out" output neurons. Additionally, the array "cmat" determines
	//! the topology (i.e., number of neurons and their connections).
	LMSEFFNet(const unsigned in, const unsigned out,
			  const Array<int>& cmat, double min = 0., double max = 1.) : MSEFFNet(in, out, cmat)
	{
		tmin = min, tmax = max;
	};


	//! Creates a MSE Bias feed-forward network with "in" input neurons and
	//! "out" output neurons. Additionally, the arrays "cmat" and
	//! "wmat" determine the topology (i.e., number of neurons and their
	//! connections) as well as the connection weights.
	LMSEFFNet(const unsigned in, const unsigned out,
			  const Array<int>& cmat, const Array<double>& wmat, double min = 0., double max = 1.) : MSEFFNet(in, out, cmat, wmat)
	{
		tmin = min, tmax = max;
	};

	//! Creates a MSE Bias feed-forward network by reading the necessary
	//! information from a file named "filename".
	LMSEFFNet(const std::string &name, double min = 0., double max = 1.) : MSEFFNet(name)
	{
		tmin = min, tmax = max;
	};


	//! Destructs an MSE Bias Feed Forward Network object.
	virtual ~LMSEFFNet()
	{};


	//! Calculates the mean squared error of the model output, compared to
	//! the target values.
	double error(Model& model, const Array<double> &input, const Array<double> &target)
	{
		double se = 0;
		if (input.ndim() == 1)
		{
			Array<double> output(target.dim(0));
			model.model(input, output);
			for (unsigned c = 0; c < target.dim(0); c++)
			{
				if ((target(c) <= tmin) && (output(c) <= tmin)) continue;
				if ((target(c) >= tmax) && (output(c) >= tmax)) continue;
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
					if ((target(pattern, c) <= tmin) && (output(c) <= tmin)) continue;
					if ((target(pattern, c) >= tmax) && (output(c) >= tmax)) continue;
					se += (target(pattern, c) - output(c)) * (target(pattern, c) - output(c));
				}
			}
		}
		// normalise
		se /= (double) target.nelem();
		return se;
	};


	//! Calculates the mean squared Error of the model output,
	//! compared to the target values, and the derivative of the
	//! mean squared error with respect to the weights.
	double errorDerivative(Model& model, const Array<double> &input, const Array<double> &target, Array<double>& derivative)
	{
		unsigned i, j;  // denote neurons
		unsigned c;     // denotes output
		unsigned pos;   // the actual position in the Array derivative
		double sum;
		double se = 0.;

		derivative.resize(model.getParameterDimension(), false);
		derivative  = 0;

		readParameters();

		if (input.ndim() == 1)
		{
			// calculate output
			activate(input);

			for (i = firstOutputNeuron, c = 0 ; i < numberOfNeurons; i++, c++)
			{
				if (((target(c) <= tmin) && (z[i] <= tmin)) ||
						((target(c) >= tmax) && (z[i] >= tmax)))
				{
					d(i) = 0;
				}
				else
				{
					d(i) =  2 * (target(c) - z[i]) * dgOutput(z[i]);
					se += (target(c) - z[i]) * (target(c) - z[i]);
				}
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
			for (i = inputDimension; i < firstOutputNeuron; i++)
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
			for (i = firstOutputNeuron; i < numberOfNeurons; i++)
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
					if (((target(pattern, c) <= tmin) && (z[i] <= tmin)) ||
							((target(pattern, c) >= tmax) && (z[i] >= tmax)))
					{
						d(i) = 0;
					}
					else
					{
						d(i) =  2 * (target(pattern, c) - z[i]) * dgOutput(z[i]);
						se += (target(pattern, c) - z[i]) * (target(pattern, c) - z[i]);
					}
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
				for (i = inputDimension; i < firstOutputNeuron; i++)
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
				for (i = firstOutputNeuron; i < numberOfNeurons; i++)
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
	};

	double tmin;
	double tmax;
};


#endif









