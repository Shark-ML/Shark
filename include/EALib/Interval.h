//===========================================================================
/*!
 *  \file Interval.h
 *
 *  \brief Interval [a, b] of the real numbers
 *
 *  \author  Martin Kreutz
 *  \date    01.01.1995
 *
 *  \par Copyright (c) 1995-2003:
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
 *      EALib
 *  <BR>
 *
 *
 *  <BR><HR>
 *  This file is part of EALib. This library is free software;
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


#ifndef __INTERVAL_H
#define __INTERVAL_H


//!
//! \brief Interval [a, b] of the real numbers
//!
class Interval
{
private:
	//! lower interval end
	double   lower;

	//! upper interval end
	double   upper;

public:
	//! \brief default constructor
	Interval()
	{
		lower = upper = 0;
	}

	//! \brief Constructor
	Interval(double lo)
	{
		lower = upper = lo;
	}

	//! \brief Constructor
	Interval(double lo, double hi)
	{
		lower = lo; upper = hi;
	}

	//! \brief return lower interval end
	double   lowerBound() const
	{
		return lower;
	}

	//! \brief return upper interval end
	double   upperBound() const
	{
		return upper;
	}

	//! \brief return interval width
	double   width() const
	{
		return upper - lower;
	}
};


#endif

