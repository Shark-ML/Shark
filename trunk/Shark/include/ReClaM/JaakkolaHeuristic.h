//===========================================================================
/*!
*  \file JaakkolaHeuristic.h
*
*  \brief Jaakkola's heuristic and related quantities for Gaussian kernel selection
*
*  \author  T. Glasmachers
*  \date    2007
*
*
*  \par Copyright (c) 1999-2007:
*      Institut f&uuml;r Neuroinformatik<BR>
*      Ruhr-Universit&auml;t Bochum<BR>
*      D-44780 Bochum, Germany<BR>
*      Phone: +49-234-32-25558<BR>
*      Fax:   +49-234-32-14209<BR>
*      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
*      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
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
//===========================================================================


#ifndef _JaakkolaHeuristic_H_
#define _JaakkolaHeuristic_H_


#include <SharkDefs.h>
#include <Array/Array.h>
#include <vector>


//! \brief Jaakkola's heuristic and related quantities for Gaussian kernel selection
class JaakkolaHeuristic
{
public:
	//! Constructor
	JaakkolaHeuristic(const Array<double>& input, const Array<double>& target);
	
	/*! Alternate constructor. Similar to regular JaakkolaHeuristic, but only using 
	 * 	a specified sub-set of the input data columns.
	 *  \param input same as in regular JaakkolaHeuristic
	 *  \param target same as in regular JaakkolaHeuristic
	 *  \param select an 1D-array containing indices of those columns the heuristic should operate on. */
	JaakkolaHeuristic(const Array<double>& input, const Array<double>& target,
							   const Array<unsigned>& select);

	//! Destructor
	~JaakkolaHeuristic();


	//! Compute the given quantile (usually the 0.5-quantile)
	//! of the empirical distribution of euklidean distances
	//! of data pairs with different labels.
	double sigma(double quantile = 0.5);

	//! Compute the given quantile (usually the 0.5-quantile)
	//! of the empirical distribution of euklidean distances
	//! of data pairs with different labels converted into
	//! a value usable as the gamma parameter of the #RBFKernel.
	double gamma(double quantile = 0.5);

protected:
	std::vector<double> stat;
};


#endif

