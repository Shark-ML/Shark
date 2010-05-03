//===========================================================================
/*!
 *  \file UncertaintyQuantification.h
 *
 *  \brief Uncertainty quantification for rank-based selection
 *
 *  \author  T. Glasmachers, based on code by C. Igel
 *  \date    2008
 *
 * based on 
 *
 * "A Method for Handling Uncertainty in Evolutionary Optimization
 * with an Application to Feedback Control in Combustion"
 * N. Hansen, A.S.P. Niederberger, L. Guzzella, and P. Koumoutsakos
 *
 *  \par Copyright (c) 2008:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *
 *  \par Project:
 *      EALib
 *
 *
 *
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
 *
 *
*/
//===========================================================================


#ifndef _UNCERTAINTY_QUANTIFICATION_H_
#define _UNCERTAINTY_QUANTIFICATION_H_


#include <SharkDefs.h>
#include <Array/Array.h>
#include <EALib/Population.h>


//! \brief Wrapper class, which allows to reevaluate objective functions and
//! which counts the number of evaluations.
class NoisyFitnessFunction {
public:
	NoisyFitnessFunction() { evals = 1; count = 0; }
	virtual ~NoisyFitnessFunction() { }
	virtual double fitness(const std::vector<double>& v) = 0;
	void setN(unsigned n) {
		evals = n;
		if (! evals) evals = 1;
	}
	unsigned getN() { return evals; }
	unsigned getCount() { return count; }
	void resetCount() { count = 0; }

protected:
	unsigned evals;
	unsigned count;
};


//! compute uncertainty level s
double UncertaintyQuantification(Population& p, NoisyFitnessFunction& f, double theta = 0.2, double r_l = 0.);


#endif
