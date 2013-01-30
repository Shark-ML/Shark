//===========================================================================
/*!
 *  \file arrayoptimize.h
 *
 *  \brief functions for basic optimizers
 *
 *  \author O.Krause
 *  \date 2010-2011
 *
 *  \par Copyright (c) 1998-2007:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
 *
 *
 *  <BR><HR>
 *  This file is part of Shark. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 3, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, see <http://www.gnu.org/licenses/>.
 *
 */
//===========================================================================

#ifndef __ARRAYOPTIMIZE_H
#define __ARRAYOPTIMIZE_H

#include "shark/LinAlg/Base.h"

namespace shark{

/**
* \ingroup shark_globals
* 
* @{
*/

//! Given a nonlinear function, a starting point and a direction,
//! a new point is calculated where the function has
//! decreased "sufficiently".
template<class VectorT,class Function>
void lnsrch
(
	const VectorT& xold,
	double fold,
	VectorT& g,
	VectorT& p,
	VectorT& x,
	double& f,
	double stpmax,
	bool& check,
	Function func
);


//! Minimizes a function of "N" variables.
template<class VectorT,class Function>
void linmin
(
	VectorT& p,
	const VectorT& xi,
	double& fret,
	Function func,
	double ax = 0.0,
	double bx = 1.0
);

//! Minimizes a function of "N" variables by using
//! derivative information.
template<class VectorT,class VectorU,class DifferentiableFunction>
void dlinmin
(
	VectorT& p,
	const VectorU& xi,
	double& fret,
	DifferentiableFunction& func,
	double ax = 0.0,
	double bx = 1.0
);
/** @}*/
}
#endif /* !__ARRAYOPTIMIZE_H */

//implementations
#include "Impl/dlinmin.inl"
#include "Impl/linmin.inl"
#include "Impl/lnsrch.inl"







