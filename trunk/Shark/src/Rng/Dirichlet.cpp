//===========================================================================
/*!
*  \file Dirichlet.cpp
*
*  \brief Dirichlet distribution
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

#include "Rng/Dirichlet.h"

Dirichlet::Dirichlet(unsigned n, double alpha) {
	pAlpha.resize(n);
	std::vector<double>::iterator it;
	for(it=pAlpha.begin(); it!=pAlpha.end(); it++)
		*it = alpha;
}

Dirichlet::Dirichlet(const std::vector<double> &alpha) {
	pAlpha = alpha;
}

Dirichlet::Dirichlet(unsigned n, double alpha, RNG& r)  : RandomVar< std::vector<double> >(r) {
	pAlpha.resize(n);
	std::vector<double>::iterator it;
	for(it=pAlpha.begin(); it!=pAlpha.end(); it++)
		*it = alpha;
}

Dirichlet::Dirichlet(std::vector<double> &alpha, RNG& r)  : RandomVar< std::vector<double> >(r) {
	pAlpha = alpha;
}

void Dirichlet::seed(long s)
{
	RandomVar< std::vector<double> >::seed(s); 
}

void   Dirichlet::alpha(const std::vector<double> &a)
{
	pAlpha = a;
}

void   Dirichlet::alpha(double a) {
	std::vector<double>::iterator it;
	for(it=pAlpha.begin(); it!=pAlpha.end(); it++)
		*it = a;
}

std::vector <double> Dirichlet::operator()(const std::vector <double> &alpha) {
	Gamma gam(0, 1, rng);
	unsigned i, n = alpha.size();
	double *y = new double[n];
	double ysum = 0;
	std::vector<double> x;
	x.resize(n);
	for(i=0; i<n; i++) { 
		y[i] = gam(alpha[i], 1.);
		ysum += y[i];
	}
	for(i=0; i<n; i++)  
		(x)[i]= y[i] / ysum;
	delete [] y;
	return x;
};

std::vector <double> Dirichlet::operator()(unsigned n, double alpha) {
	std::vector<double> x; 
	Gamma gam(0, 1, rng);
	unsigned i;
	double *y = new double[n];
	double ysum = 0;
	x.resize(n);
	for(i=0; i<n; i++) { 
		y[i] = gam(alpha, 1.);
		ysum += y[i];
	}
	for(i=0; i<n; i++)  
		(x)[i]= y[i] / ysum;
	delete [] y;
	return x;
};

std::vector <double> Dirichlet::operator()() {
	return (*this)(pAlpha);
}

double Dirichlet::p(const std::vector<double> &x) const {
	unsigned i;
	double p = 1.;
	double sum = 0.;
	for(i=0; i<pAlpha.size(); i++) {
		p *= pow(x[i], pAlpha[i]-1) / Shark::gamma(pAlpha[i]);
		sum += pAlpha[i];
	}
	return p * Shark::gamma(sum);
};
