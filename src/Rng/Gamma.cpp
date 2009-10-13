//===========================================================================
/*!
*  \file Gamma.cpp
*
*  \brief Gamma distribution
*
*  \author  C. Igel
*  \date    2008-11-28
*
*  \par Copyright (c) 1995,1998:
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
*      Rng
*
*  <BR>
*
*  <BR><HR>
*  This file is part of Rng. This library is free software;
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

#include "Rng/Gamma.h"

Gamma::Gamma(double k, double theta) {
	pK = k;
	pTheta = theta;
}

Gamma::Gamma(double k, double theta, RNG& r) : RandomVar< double >(r) {
	pK = k;
	pTheta = theta;
};

void Gamma::seed(long s)
{
	RandomVar< double >::seed(s); 
}

double Gamma::mean() const
{
	return pK * pTheta;
}

double Gamma::variance() const
{
	return pK * pTheta * pTheta;
}

void   Gamma::shape(double s)
{
	pK = s;
}
void   Gamma::scale(double s)
{
	pTheta = s;
}

double Gamma::operator()(double k, double theta) {
	unsigned i;
	unsigned n = unsigned(k);
	double delta = k - double(n);
	double V_2, V_1, V;
	double v0 = exp(1.0) / (exp(1.0) + delta);
	double eta, xi;
	double Gn1 = 0; // Gamma(n, 1) distributed
	
	for(i=0; i<n; i++) Gn1 += -log(U01());
	
	do {
		V_2 = U01();
		V_1 = U01();
		V   = U01();
		if(V_2 <= v0) {
			xi = pow(V_1, 1./delta);
			eta = V * pow(xi, delta-1.);
		} else {
			xi = 1. - log(V_1);
			eta = V * exp(-xi);
		}
	} while(eta > (pow(xi, delta-1.) * exp(-xi)));
	
	return theta * (xi + Gn1);
};

double Gamma::operator()() {
	return (*this)(pK, pTheta);
}

double Gamma::p(const double &x) const {
	return pow(x, pK-1) * exp(-x / pTheta) / (Shark::gamma(pK) * pow(pTheta, pK)) ;
};
