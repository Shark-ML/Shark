/*!
*  \file Variance.h
*
*  \brief Cariance of data in each column of a data set.
*
*  \author C. Igel
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


#ifndef VARIANCE_H
#define VARIANCE_H


#include <SharkDefs.h>


//===========================================================================


//! \brief Cariance of data in each column of a data set.
class Variance
{
public:

	/*!
	 *  \brief Caculates the variance of data in each column of a data set.
	 *
	 *  Assume a data set with \f$P\f$ rows (patterns) and \f$N\f$
	 *  columns (no. of input neurons), then the variance vector \em v
	 *  for \em data is defined by:
	 *
	 *  \f$
	 *  v = ( v_i ),\ \mbox{for\ } i = 1, \dots, N
	 *  \f$
	 *
	 *  \f$
	 *  v_i = \frac{1}{P}\sum_{p=1}^P(data_{ip})^{2} -
	 *        \left(\frac{1}{P}\sum_{p=1}^Pdata_{ip} \right) ^2
	 *  \f$
	 *
	 *      \param  data Set of data (matrix of input patterns).
	 *      \param  v The vector with the calculated variance values.
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
	static void variance(const Array<double> &data, Array<double> &v)
	{
		Array< double > m;
		m.resize(data.dim(1));
		v.resize(data.dim(1));
		m = 0.;
		v = 0.;
		for (unsigned i = 0; i < data.dim(0); ++i)
		{
			m += data[i];
			v += data[i] * data[i];
		}

		m /= double(data.dim(0));
		v /= double(data.dim(0));
		v -= m * m;
	}
};


#endif

