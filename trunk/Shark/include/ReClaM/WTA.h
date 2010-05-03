/*!
*  \file WTA.h
*
*  \author C. Igel
*
*  \brief Winner-takes-all error function
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
*
*
*  <BR><HR>
*  This file is part of Shark. This library is free software;
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

#ifndef WTA_H
#define WTA_H


#include <SharkDefs.h>
#include <ReClaM/ErrorFunction.h>
#include <Array/ArrayOp.h>


//! \brief Winner-takes-all error function
class WTA : public ErrorFunction
{
public:
	/*!
	 *  \brief Caculates the error based on "the winner takes it all" strategy.
	 *
	 *  This error measure is used for classification problems.
	 *  For each pattern only the output neuron with the best (maximum) value
	 *  is considered and set to 1, all other output neurons are set
	 *  to zero. This result is compared with the target vectors
	 *  and then the number of different vectors (= errors) is counted.
	 *  Finally, the number of errors is normalized with the number of patterns.
	 *
	 *      \param  model   The model used for prediction
	 *      \param  input   Input vector for the model.
	 *      \param  target  Target vector.
	 *      \return The number of errors, normalized with the number of patterns.
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
		if (input.ndim() == 1)
		{
			Array<double> output(target.dim(0));
			model.model(input, output);
			takeAll(output);
			if (target != output) ce++;
		}
		else
		{
			Array<double> output(target.dim(1));
			for (unsigned pattern = 0; pattern < input.dim(0); ++pattern)
			{
				model.model(input[pattern], output);
				takeAll(output);
				for (unsigned i = 0; i < output.nelem(); i++)
				{
					if (target[pattern](i) != output(i))
					{
						ce++;
						break;
					}
				}
			}
			ce /= input.dim(0);
		}
		return ce;
	}

protected:
	//! First searches for the best (maximum) output value,
	//! then sets this value to 1 and all others to 0.
	void takeAll(Array<double>& a)
	{
		unsigned i, bestIndex = 0;
		double best = a(bestIndex);
		for (i = 1; i < a.nelem(); i++)
		{
			if (a(i) > best)
			{
				bestIndex = i;
				best = a(bestIndex);
			}
		}
		for (i = 0; i < a.nelem(); i++)
		{
			if (i == bestIndex) a(i) = 1.0;
			else a(i) = 0.0;
		}
	}
};


#endif

