//===========================================================================
/*!
 *  \file arrayoptimize.h
 *
 *  \brief Algorithms for the minimization of functions of
 *         N variables.
 *
 *  \author  M. Kreutz
 *  \date    1995
 *
 *  \par Copyright (c) 1998-2000:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
*
 *  \par Project:
 *      LinAlg
 *
 *
 *  <BR>
 *
 *
 *  <BR><HR>
 *  This file is part of LinAlg. This library is free software;
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

#ifndef __ARRAYOPTIMIZE_H
#define __ARRAYOPTIMIZE_H

#include "Array/Array.h"


//! Used to specify the type of line minimization
//! subroutine for #bfgs and #bfgs2
enum LinMinTypes
{
	//! Use the line search algorithm (#lnsrch)
	LNSRCH,
	//! Use the cubic line search algorithm (#cblnsrch)
	CBLNSRCH,
	//! Use the #linmin algorithm
	LINMIN,
	//! Use the #dlinmin algorithm
	DERIVLINMIN
};


//! Minimizes function "func" by using the
//! Broyden-Fletcher-Goldfarb-Shanno-algorithm.
void bfgs
(
	Array< double >& p,
	double    gtol,
	unsigned& iter,
	double&   fret,
	double(*func)(const Array< double >&),
	void(*dfunc)(const Array< double >&, Array< double >&),
	unsigned  iterMax = 200,
	LinMinTypes linmintype = DERIVLINMIN
);


//! Minimizes function "func" by using a modified
//! Broyden-Fletcher-Goldfarb-Shanno-algorithm.
void bfgs2
(
	Array< double >& p,
	double    gtol,
	unsigned& iter,
	double&   fret,
	double(*func)(const Array< double >&),
	void(*dfunc)(const Array< double >&, Array< double >&),
	unsigned  iterMax = 200,
	LinMinTypes linmintype = DERIVLINMIN
);



/*void bfgsk
(
    Array< double >& p,
    double    gtol,
    unsigned& iter,
    double&   fret,
    double  (*func)( const Array< double >& ),
    void   (*dfunc)( const Array< double >&, Array< double >& ),
    unsigned  iterMax = 200
);*/



//! Given a nonlinear function, a starting point and a direction,
//! a new point is calculated where the function has
//! decreased "sufficiently".
void lnsrch
(
	Array< double >& xold,
	double fold,
	Array< double >& g,
	Array< double >& p,
	Array< double >& x,
	double& f,
	double stpmax,
	bool& check,
	double(*func)(const Array< double >&)
);



//! Does a cubic line search, i.e. given a nonlinear function,
//! a starting point and a direction,
//! a new point is calculated where the function has
//! decreased "sufficiently".
void cblnsrch
(
	Array< double >& xold,
	double fold,
	Array< double >& g,
	Array< double >& p,
	Array< double >& x,
	double& f,
	double(*func)(const Array< double >&),
	double lambda = 0.25
);



//! Minimizes a function of "N" variables.
void linmin
(
	Array< double >& p,
	const Array< double >& xi,
	double& fret,
	double(*func)(const Array< double >&)
);



//! Minimizes a function of "N" variables by using
//! derivative information.
void dlinmin
(
	Array< double >& p,
	const Array< double >& xi,
	double& fret,
	double(*func)(const Array< double >&),
	void(*dfunc)(const Array< double >&, Array< double >&)
);



/*void nmsimplex
(
    Array< double >& p,
	Array< double >& y,
	double           ftol,
	unsigned&        nfunc,
	double           (*func)( const Array< double >& ),
	unsigned         nfuncMax = 5000
);*/



#endif /* !__ARRAYOPTIMIZE_H */








