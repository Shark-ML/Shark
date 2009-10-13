/*!
*  \file BinaryCriterion.h
*
*  \author C. Igel
*
*  \brief Uses thresholds for transforming output and target vectors
*         into binary vectors and then calculates the classification error
*
*  \par
*      Institut f&uuml;r Neuroinformatik<BR>
*      Ruhr-Universit&auml;t Bochum<BR>
*      D-44780 Bochum, Germany<BR>
*      Phone: +49-234-32-25558<BR>
*      Fax:   +49-234-32-14209<BR>
*      eMail: shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
*      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
*      <BR> 
*
*  \par Project:
*      ReClaM
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

#ifndef BINARY_CRITERION_H
#define BINARY_CRITERION_H


#include <SharkDefs.h>
#include <ReClaM/ErrorFunction.h>


//===========================================================================
//!  \brief Use thresholds for transforming output and target vectors
//!        into binary vectors and then calculates the classification error
class BinaryCriterion : public ErrorFunction
{
public:
	/*!
	 *  \brief Constructor
	 *
	 *  \param  t  Defines an interval of length 2\em t, output values
	 *             in this interval are not transformed into binary values.
	 */
	BinaryCriterion(double t = 0.0)
	{
		buffersize = t;
	}

	/*!
	 *  \brief Uses thresholds for transforming output and target vectors
	 *         into binary vectors and then calculating the error.
	 *
	 *  Used for qualification strategies. The target vectors are transformed
	 *  into binary vectors by using a threshold of "0.5". All target neurons
	 *  greater than "0.5" are set to "1", all other target neurons
	 *  are set to zero. The output vectors are transformed into binary
	 *  vectors in the same way, but by using the parameter \em t, you
	 *  can define a "buffer interval" of size \f$2 \ast t\f$.
	 *  All output neurons with a value
	 *  of 0.5 - \em t or less are set to zero, all output neurons with a value
	 *  greater than 0.5 + \em t are set to 1, output neurons with a
	 *  value in the interval \f$2 \ast t\f$ are ignored.
	 *  This make for these values counted as errors, because normally
	 *  the unmodified values of the output neurons are not "0" or "1".
	 *  After this transformation of the vectors the number of
	 *  vectors is counted for which the used model calculates
	 *  another output than given by the corresponding target vector.
	 *  Then the number of different vector-pairs is normalized
	 *  by the number of input patterns. Given a number of \f$P\f$
	 *  patterns the error is calculated by:
	 *
	 *  \f$
	 *    E = \frac{1}{P}\ |A|\ \mbox{with\ }
	 *    A = \{ in_i\ |\ binary(model(in_i)) \neq binary(out_i),\ i = 1, \dots,
	 *    P\}
	 *  \f$
	 *
	 *  \em binary is used here for the notation of the transformation
	 *  of the original vectors to binary vectors.
	 *
	 *  \par Example
	 *
	 *  \f$
	 *    \mbox{model(in)\ } =
	 *    \left(
	 *    \begin{array}{c}
	 *    0.1\\ 0.9\\ 0.5\\ 0.4\\ 0.3\\ 0.7\\ 0.5\\ 0.1\\ 0.8\\ 0.6\\
	 *    \end{array}
	 *    \right) \Rightarrow
	 *    \left(
	 *    \begin{array}{c}
	 *    0\\ 1\\ 0\\ 0\\ 0\\ 1\\ 0\\ 0\\ 1\\ 1\\
	 *    \end{array}
	 *    \right)
	 *    \mbox{\ \ \ \ \ out\ } =
	 *    \left(
	 *    \begin{array}{c}
	 *    0.6 \\ 0.6 \\ 0.1 \\ 0.2 \\ 0.1 \\ 0.9 \\ 0.7 \\ 0.9 \\ 0.1 \\ 0.8\\
	 *    \end{array}
	 *    \right) \Rightarrow
	 *    \left(
	 *    \begin{array}{c}
	 *    1\\ 1\\ 0\\ 0\\ 0\\ 1\\ 1\\ 1\\ 0\\ 1\\
	 *    \end{array}
	 *    \right)
	 *    \f$
	 *
	 *    \f$
	 *    \mbox{model(in)\ } =
	 *    \left(
	 *    \begin{array}{c}
	 *    0.1 \\ 0.9 \\ 0.5 \\ 0.4 \\ 0.3 \\ 0.7 \\ 0.5 \\ 0.1 \\ 0.8 \\ 0.6
	 *    \end{array}
	 *    \right) \Rightarrow
	 *    \left(
	 *    \begin{array}{c}
	 *    0 \\ 1 \\ 0.5 \\ 0.4 \\ 0 \\ 0.7 \\ 0.5 \\ 0 \\ 1 \\ 0.6
	 *    \end{array}
	 *    \right)
	 *    \mbox{\ \ \ \ \ out\ } =
	 *    \left(
	 *    \begin{array}{c}
	 *    0.6 \\ 0.6 \\ 0.1 \\ 0.2 \\ 0.1 \\ 0.9 \\ 0.7 \\ 0.9 \\ 0.1 \\ 0.8
	 *    \end{array}
	 *    \right) \Rightarrow
	 *    \left(
	 *    \begin{array}{c}
	 *    1 \\ 1 \\ 0 \\ 0 \\ 0 \\ 1 \\ 1 \\ 1 \\ 0 \\ 1
	 *    \end{array}
	 *    \right)\\
	 *  \f$
	 *
	 *  The first example shows the binary transformation for a
	 *  value for parameter \em t of "0" resulting in
	 *  4 differences between the output and the target vector,
	 *  the second shows the same transformation for a value of "0.2",
	 *  resulting in twice as many differences.
	 *
	 *      \param  model   The model used for prediction
	 *      \param  input   Input vector for the model.
	 *      \param  target  Target vector.
	 *      \return
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
	double error(Model& model, const Array<double>& input, const Array<double>& target)
	{
		double ce = 0.0;

		Array<double > o = target;
		binary(o, 0.0);

		if (input.ndim() == 1)
		{
			Array<double> output(target.dim(0));
			model.model(input, output);
			binary(output, buffersize);
			for (unsigned i = 0; i < output.nelem(); i++)
			{
				if (o(i) != output(i))
				{
					ce++;
					break;
				}
			}
		}
		else
		{
			Array<double> output(target.dim(1));
			for (unsigned pattern = 0; pattern < input.dim(0); ++pattern)
			{
				model.model(input[pattern], output);
				binary(output, buffersize);
				for (unsigned i = 0; i < output.nelem(); i++)
				{
					if (o[pattern](i) != output(i))
					{
						ce++;
						break;
					}
				}
			}
			ce /= (double) input.dim(0);
		}
		return ce;
	}

protected:
	//! Used to mark output neurons as set/unset.
	//! Normally the threshold for a neuron to be
	//! set is 0.5 and higher, but with parameter t
	//! a buffer interval between set and unset neurons
	//! is defined. All neurons lying in this interval
	//! are ignored.
	void binary(Array<double>& a, double t)
	{
		if (a.ndim() == 1)
		{
			for (unsigned i = 0; i < a.nelem(); i++)
			{
				if (a(i) > (0.5 + t)) a(i) = 1.;
				if (a(i) <= (0.5 - t)) a(i) = 0.;
			}
		}
		else
		{
			for (unsigned i = 0; i < a.dim(0); i++)
			{
				for (unsigned j = 0; j < a.dim(1); j++)
				{
					if (a(i, j) > (0.5 + t)) a(i, j) = 1.0;
					if (a(i, j) <= (0.5 - t)) a(i, j) = 0.0;
				}
			}
		}
	}

	//! Size of the buffer interval
	double buffersize;
};


#endif

