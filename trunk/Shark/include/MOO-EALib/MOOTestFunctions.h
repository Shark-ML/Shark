/*!
*  \file MOOTestFunctions.h
*
*  \brief Several Benchmark functions for algorithm analysis
* 
*  \author Tatsuya Okabe
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
*      MOO-EALib
*  <BR>
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


//
// 	Authors message
//======================================================================
/*	Thank you very much for your interest to MOO-EALib.

	Since our company's name was changed on 1st, January, 2003,
	my E-mail address in the source codes were also changed.
	The current E-mail address (6th,Feb.,2004) is as follows:

	tatsuya.okabe@honda-ri.de.

	If you cannot contact me with the above E-mail address,
	you can also use the following E-mail address:

	t_okabe_de@hotmail.com.

	If you have any questions, please don't hesitate to
	ask me. It's my pleasure.

	Best Regards,
	Tatsuya Okabe

	*********************************************************
	Tatsuya Okabe
	Honda Research Institute Europe GmbH
	Carl-Legien-Strasse 30, 63073 Offenbach/Main, Germany
	Tel: +49-69-89011-745
	Fax: +49-69-89011-749
	**********************************************************/

////////////////////////////////////////////////////////////////////// MOO


#ifndef __MOOTESTFUNCTIONS_H
#define __MOOTESTFUNCTIONS_H

#include <SharkDefs.h>
#include <Array/ArraySort.h>
#include <Array/ArrayIo.h>


//
//************************************************************************
// Tools for equidistant test function sampling in objective space
//************************************************************************
//

unsigned approxEquidistantFrontDoIt(Array<double> &v, Array<double> &w, unsigned n, double d)
{
	double dCurrent, dPrevious;
	unsigned i, k;

	k = 1;
	for (i = 1; i < v.dim(0) - 1; i++)
	{
		dCurrent =  sqrt(Shark::sqr(w(k - 1, 0) - v(i    , 0)) + Shark::sqr(w(k - 1, 1) - v(i    , 1)));
		dPrevious = sqrt(Shark::sqr(w(k - 1, 0) - v(i - 1, 0)) + Shark::sqr(w(k - 1, 1) - v(i - 1, 1)));
		if (dCurrent > d)
		{
			if (dPrevious == 0)
			{ // if the adjacent point is far, take it
				w[k] = v[i];
				k++;
			}
			else
			{ // take current
				if (fabs(dCurrent - d) < fabs(dPrevious - d))
				{
					if (k >= n)
					{
						return k + 1;
					}
					w[k] = v[i];
					k++;
				}
				else
				{ // take previous
					if (k >= n)
					{
						return k + 1;
					}
					w[k] = v[i - 1];
					k++;
					i--;
				}
			}
		}
	}
	return k;
}

void approxEquidistantFront(Array<double> &v, Array<double> &w, unsigned n = 500, double eps = 0.000001)
{
	double sum = 0;
	unsigned i, k;
	if (v.ndim() != 2)
	{
	  throw SHARKEXCEPTION("makeUpsilonFront: input Array has wrong dimension");
	}

	if (v.dim(0) < n)
	{
	  throw SHARKEXCEPTION("makeUpsilonFront: input Array has not enough sample points");
	}

	w.resize(n, 2, false);

	// sort w.r.t. first component
	sort2DBy1st(v);

	// compute length of front
	for (i = 0; i < v.dim(0) - 1; i++)
		if (v(i + 1, 0) - v(i, 0)) sum += (v(i + 1, 0) - v(i, 0)) *
											  sqrt(1 + Shark::sqr((v(i + 1, 1) - v(i, 1))  / (v(i + 1, 0) - v(i, 0))));

	// fix end
	w[0] = v[0];

	double dOld, d;
	dOld = d = sum / (n - 1.);

	k = approxEquidistantFrontDoIt(v, w,  n, d);
	while (k != n)
	{
		dOld = d;
		if (k < n) d *= (1. - eps);
		if (k > n) d *= (1. + eps);
		k = approxEquidistantFrontDoIt(v, w,  n, d);
		//cerr << k << " " << d << endl;
	}

	std::cout << "d = " << d << std::endl;
	std::cerr << "d = " << d << std::endl;

	w[n - 1] = v[v.dim(0) - 1];

}

//
//************************************************************************
// Tools for equidistant test function sampling in objective space
//************************************************************************
//


//
//************************************************************************
// Test function implementation of Tatsuya Okabe
//************************************************************************
//

//************************************************************************
// Sphere Test Function (Deb's SCH)
//************************************************************************
double SphereF1(const std::vector< double >& x)
{
	double    sum = 0.0;
	unsigned  n;

	n = x.size();
	for (unsigned i = n; i--;)
	{
		sum += x[ i ] * x[ i ];
	}

	return sum / (double) n;
}
//
double SphereF2(const std::vector< double >& x)
{
	double   sum = 0.0;
	unsigned n;

	n = x.size();
	for (unsigned i = n; i--;)
	{
		sum += (x[ i ] - 2.0) * (x[ i ] - 2.0);
	}

	return sum / (double) n;
}

//************************************************************************
// Deb's Convex Test Function (ZDT 1)
//************************************************************************
double DebConvexF1(const std::vector< double >& x)
{
	unsigned n;
	n = x.size();
	// Penalty
	for (unsigned i = 0; i < n; i++)
	{
		if (x[ i ] > 1.0 || x[ i ] < 0.0)
		{
			return 5.0;
		}
	}
	return x[ 0 ];
}
//
double DebConvexF2(const std::vector< double >& x)
{
	unsigned i;
	double   sum = 0.0;
	unsigned n;
	double   g, f1, f2;
	f1 = x[ 0 ];
	n  = x.size();
	// Penalty
	for (i = 0; i < n; i++)
	{
		if (x[ i ] > 1.0 || x[ i ] < 0.0)
		{
			return 10.0;
		}
	}
	//
	for (i = 1; i < n; i++)
	{
		sum += x[ i ];
	}
	g = 1.0 + 9.0 * sum / (double)(n - 1);
	f2 = g * (1.0 - sqrt(f1 / g));
	return f2;
}

//************************************************************************
// Deb's Concave Test Function (ZDT 2)
//************************************************************************
double DebConcaveF1(const std::vector< double >& x)
{
	unsigned n;
	unsigned i;
	n = x.size();
	// Penalty
	for (i = 0; i < n; i++)
	{
		if (x[ i ] > 1.0 || x[ i ] < 0.0)
		{
			return 5.0;
		}
	}

	return x[ 0 ];
}
//
double DebConcaveF2(const std::vector< double >& x)
{
	double   sum = 0.0;
	unsigned n;
	double   g, f1, f2;
	unsigned i;
	f1 = x[ 0 ];
	n  = x.size();
	// Penalty
	for (i = 0; i < n; i++)
	{
		if (x[ i ] > 1.0 || x[ i ] < 0.0)
		{
			return 10.0;
		}
	}
	//
	for (i = 1; i < n; i++)
	{
		sum += x[ i ];
	}
	g = 1.0 + 9.0 * sum / (double)(n - 1);
	f2 = g * (1.0 - (f1 / g) * (f1 / g));
	return f2;
}

//************************************************************************
// Deb's Discrete Test Function (ZDT 3)
//************************************************************************
double DebDiscreteF1(const std::vector< double >& x)
{
	unsigned n;
	unsigned i;
	n = x.size();
	// Penalty
	for (i = 0; i < n; i++)
	{
		if (x[ i ] > 1.0 || x[ i ] < 0.0)
		{
			return 5.0;
		}
	}

	return x[ 0 ];
}
//
double DebDiscreteF2(const std::vector< double >& x)
{
	double   sum = 0.0;
	unsigned n;
	double   g, f1, f2;
	unsigned i;
	f1 = x[ 0 ];
	n  = x.size();
	// Penalty
	for (i = 0; i < n; i++)
	{
		if (x[ i ] > 1.0 || x[ i ] < 0.0)
		{
			return 10.0;
		}
	}
	//
	for (i = 1; i < n; i++)
	{
		sum += x[ i ];
	}
	g = 1.0 + 9.0 * sum / (double)(n - 1);
	f2 = g * (1.0 - sqrt(f1 / g) - (f1 / g) * sin(10 * M_PI * f1));
	return f2;
}

//************************************************************************
// Fonseca's Concave Test Function (Deb's FON)
//************************************************************************
double FonsecaConcaveF1(const std::vector< double >& x)
{
	double   sum = 0.0;
	unsigned n;
	unsigned i;
	n = x.size();
	for (i = n; i--;)
	{
		sum += (x[ i ] - 1.0 / sqrt((double) n)) * (x[ i ] - 1.0 / sqrt((double) n));
	}



	return 1.0 - exp(-sum);
}
//
double FonsecaConcaveF2(const std::vector< double >& x)
{
	double   sum = 0.0;
	unsigned n;
	unsigned i;
	n = x.size();
	for (i = n; i--;)
	{
		sum += (x[ i ] + 1.0 / sqrt((double) n)) * (x[ i ] + 1.0 / sqrt((double) n));
	}

	return 1.0 - exp(-sum);
}

void FonsecaConcaveSampleFront(Array< double > &pf, unsigned dimension, unsigned n  = 500)
{
	double xmin = -1. / sqrt((double) dimension);
	double xmax = 1. / sqrt((double) dimension);
	unsigned i  = 0, ii = 0;
	std::vector< double > xv(dimension);

	pf.resize(n, 2u);
	unsigned raw = 10 * n;
	Array<double> raw_pf(raw, 2u);

	double it = (xmax - xmin) / (raw - 1.);

	for (double x = xmin;  x < xmax - 0.1 * it; x += it, i++)
	{
		for (ii = 0; ii < xv.size(); ii++)
			xv[ii] = x;

		raw_pf(i, 0) = FonsecaConcaveF1(xv);
		raw_pf(i, 1) = FonsecaConcaveF2(xv);
	}
	i = raw - 1;
	for (ii = 0; ii < xv.size(); ii++)
	{
		xv[ii] = xmax;
	}
	raw_pf(i, 0) = FonsecaConcaveF1(xv);
	raw_pf(i, 1) = FonsecaConcaveF2(xv);

	approxEquidistantFront(raw_pf, pf, n);
}

void FonsecaConcaveSample(Array< double > &pf, unsigned dimension, unsigned n  = 500)
{
	if (dimension != 2)
	{
		std::cerr << "sorry, method implemented for two dimensions only ..." << std::endl;
		return;
	}
	double xmin = -1. / sqrt((double) dimension);
	double xmax = 1. / sqrt((double) dimension);
	unsigned i  = 0;
	std::vector< double > xv(dimension);

	pf.resize(n*n, 2u);

	double it = (xmax - xmin) / (n - 1.);

	for (double x = xmin; x < xmax - 0.1 * it; x += it)
	{
		xv[0] = x;
		for (double y = xmin; y < xmax - 0.1 * it; y += it)
		{
			xv[1] = y;
			pf(i, 0) = FonsecaConcaveF1(xv);
			pf(i, 1) = FonsecaConcaveF2(xv);
			i++;
		}
		xv[1] = xmax;
		pf(i, 0) = FonsecaConcaveF1(xv);
		pf(i, 1) = FonsecaConcaveF2(xv);
		i++;
	}
	xv[0] = xmax;
	for (double y = xmin; y < xmax - 0.1 * it; y += it)
	{
		xv[1] = y;
		pf(i, 0) = FonsecaConcaveF1(xv);
		pf(i, 1) = FonsecaConcaveF2(xv);
		i++;
	}
	xv[1] = xmax;
	pf(i, 0) = FonsecaConcaveF1(xv);
	pf(i, 1) = FonsecaConcaveF2(xv);
}

//************************************************************************
// Messac's Concave Test Function
//************************************************************************
double MessacConcaveF1(const std::vector< double >& x)
{
	double   sum = 0.0;
	unsigned n, i;

	n = x.size();
	for (i = n; i--;)
	{
		sum += exp(-x[ i ]) + 1.4 * exp(-x[ i ] * x[ i ]);
	}

	return sum;
}
//
double MessacConcaveF2(const std::vector< double >& x)
{
	double   sum = 0.0;
	unsigned n, i;

	n = x.size();
	for (i = n; i--;)
	{
		sum += exp( + x[ i ]) + 1.4 * exp(-x[ i ] * x[ i ]);
	}

	return sum;
}

void MessacConcaveSampleFront(Array< double > &pf, unsigned dimension, double lower, double upper, unsigned n  = 500)
{
	if (dimension != 1)
	{
		std::cerr << "sorry, method implemented for one dimension only ..." << std::endl;
		return;
	}

	double xmin = lower;
	double xmax = upper;
	unsigned i = 0, ii = 0;
	std::vector< double > xv(dimension);

	pf.resize(n, 2u);

	double it = (xmax - xmin) / (n - 1.);

	for (double x = xmin; x < xmax - 0.1 * it; x += it, i++)
	{
		for (ii = 0; ii < xv.size(); ii++) xv[ii] = x;
		pf(i, 0) = MessacConcaveF1(xv);
		pf(i, 1) = MessacConcaveF2(xv);
	}
	i = n - 1;
	for (ii = 0; ii < xv.size(); ii++) xv[ii] = xmax;
	pf(i, 0) = MessacConcaveF1(xv);
	pf(i, 1) = MessacConcaveF2(xv);
}

void MessacConcaveSample(Array< double > &pf, unsigned dimension, double lower, double upper)
{
	if (dimension != 2)
	{
		std::cerr << "sorry, method implemented for two dimensions only ..." << std::endl;
		return;
	}

	double xmin = lower;
	double xmax = upper;
	unsigned n  = 100, i = 0;
	std::vector< double > xv(dimension);

	pf.resize(n*n, 2u);

	double it = (xmax - xmin) / (n - 1.);

	for (double x = xmin; x < xmax - 0.1 * it; x += it)
	{
		xv[0] = x;
		for (double y = xmin; y < xmax - 0.1 * it; y += it)
		{
			xv[1] =  y;
			pf(i, 0) = MessacConcaveF1(xv);
			pf(i, 1) = MessacConcaveF2(xv);
			i++;
		}
		xv[1] =  xmax;
		pf(i, 0) = MessacConcaveF1(xv);
		pf(i, 1) = MessacConcaveF2(xv);
		i++;
	}
	xv[0] = xmax;
	for (double y = xmin; y < xmax - 0.1 * it; y += it)
	{
		xv[1] =  y;
		pf(i, 0) = MessacConcaveF1(xv);
		pf(i, 1) = MessacConcaveF2(xv);
		i++;
	}
	xv[1] =  xmax;
	pf(i, 0) = MessacConcaveF1(xv);
	pf(i, 1) = MessacConcaveF2(xv);
}

//
//************************************************************************
// End of test function implementation of Tatsuya Okabe
//************************************************************************
//

//
//************************************************************************
// Rotated test functions (by Stefan Roth)
//************************************************************************
//

//************************************************************************
// Helpers
//************************************************************************

double norm(const Array<double> &a)
{
	double sum = 0;
	unsigned i ;
	for (i = 0; i < a.nelem(); i++) sum += a(i) * a(i);
	return sqrt(sum);
}

double scalarprod(const Array<double> &a, const Array<double> &b)
{
	double sum = 0;
	unsigned i;
	for (i = 0; i < a.nelem(); i++)
		sum += a(i) * b(i);
	return sum;
}

double scalarprod(const Array<double> &a, const std::vector<double> &b)
{
	double sum = 0;
	unsigned i;

	if (a.ndim() != 1 || a.nelem() != b.size())
	{
	  throw SHARKEXCEPTION("check size of vector or Array");
	}

	for (i = 0; i < a.nelem(); i++)
		sum += a(i) * b[i];
	return sum;
}


void generateBasis(unsigned d, Array<double> &B)
{
	unsigned i, j, c;
	double normB;
	Array<double> H;
	B.resize(d, d);
	H.resize(d, d);
	for (i = 0; i < d; i++)
		for (c = 0; c < d; c++)
			B(i, c) = Rng::gauss(0, 1);

	for (i = 0; i < d; i++)
	{
		for (j = 0; j < i; j++)
		{
			H = B;
			for (c = 0; c < d; c++)
				B(i, c) -= scalarprod(H[i], H[j]) * H(j, c);
		}
		normB = norm(B[i]);
		for (j = 0; j < d; j++) B(i, j) = B(i, j) / normB;
	}
}

//************************************************************************
// Rotated paraboloid
//************************************************************************

double RotParF1(const std::vector<double> &_v, Array<double> &coord, double cond)
{
	unsigned i;
	double sum = 0.;

	Array<double> v(_v.size());

	for (i = 0; i < v.dim(0); i++)
	{
		v(i) = scalarprod(coord.col(i), _v);
		sum += Shark::sqr(pow(cond, (double(i) / double(v.dim(0) - 1))) * v(i)) ;
	}

	return sum / (cond * cond * v.dim(0));
}

double RotParF2(const std::vector<double> &_v, Array<double> &coord, double cond1, double cond2 = 2)
{
	unsigned i;
	double sum = 0.;

	Array<double> v(_v.size());

	for (i = 0; i < v.dim(0); i++)
	{
		v(i) = scalarprod(coord.col(i), _v);
		sum += Shark::sqr(pow(cond1, (double(i) / double(v.dim(0) - 1))) * (v(i) - cond2)) ;
	}

	return sum / (cond1 * cond1 * v.dim(0));
}

double RotParF1(const std::vector<double> &_v, Array<double> &coord, const std::vector<double> &lower, const std::vector<double> &upper, double cond)
{
	unsigned i;

	for (i = 0; i < _v.size();i++)
	{
		// Penalty
		if (_v[i] > upper[i] || _v[i] < lower[i])
			return MAXDOUBLE;
		//
	}

	return RotParF1(_v, coord, cond);
}

double RotParF2(const std::vector<double> &_v, Array<double> &coord, const std::vector<double> &lower, const std::vector<double> &upper, double cond1, double cond2 = 2)
{
	unsigned i;
	for (i = 0; i < _v.size();i++)
	{
		// Penalty
		if (_v[i] > upper[i] || _v[i] < lower[i])
			return MAXDOUBLE;
		//
	}

	return RotParF2(_v, coord, cond1, cond2);
}

void RotParSampleFront(Array< double > &pf, unsigned dimension, double cond , double cond2 = 2, unsigned n  = 100000)
{
	double xmin = 0;
	double xmax = cond2;
	unsigned  i = 0, ii = 0;
	std::vector< double > xv(dimension);

	pf.resize(n, 2u);
	unsigned raw = 40 * n;
	Array<double> raw_pf(raw, 2u);

	Array<double> B(dimension, dimension);
	B = 0; for (i = 0;i < B.dim(0);i++) B(i, i) = 1;

	i = 0;
	double it = (xmax - xmin) / (raw - 1.);

	for (double x = xmin; x < xmax - 0.1 * it; x += it, i++)
	{
		for (ii = 0; ii < xv.size(); ii++) xv[ii] = x;
		raw_pf(i, 0) = RotParF1(xv, B, cond);
		raw_pf(i, 1) = RotParF2(xv, B, cond, cond2);
	}
	i = raw - 1;
	for (ii = 0; ii < xv.size(); ii++) xv[ii] = xmax;
	raw_pf(i, 0) = RotParF1(xv, B, cond);
	raw_pf(i, 1) = RotParF2(xv, B, cond, cond2);

	approxEquidistantFront(raw_pf, pf, n);
}

void RotParSample(Array< double > &pf, unsigned dimension, double cond , double cond2 = 2)
{
	if (dimension != 2)
	{
		std::cerr << "sorry, method implemented for one dimension only ..." << std::endl;
		return;
	}

	double xmin = 0, x, y;
	double xmax = cond2;
	unsigned n  = 100, i = 0;
	std::vector< double > xv(dimension);

	pf.resize(n*n, 2u);

	Array<double> B(dimension, dimension);
	B = 0; for (i = 0;i < B.dim(0);i++) B(i, i) = 1;

	i = 0;

	double it = (xmax - xmin) / (n - 1.);

	for (x = xmin; x < xmax - 0.1 * it; x += it)
	{
		xv[0] = x;
		for (y = xmin; y < xmax - 0.1 * it; y += it)
		{
			xv[1] = y;
			pf(i, 0) = RotParF1(xv, B, cond);
			pf(i, 1) = RotParF2(xv, B, cond, cond2);
			i++;
		}
		xv[1] = xmax;
		pf(i, 0) = RotParF1(xv, B, cond);
		pf(i, 1) = RotParF2(xv, B, cond, cond2);
		i++;
	}
	xv[0] = xmax;
	for (y = xmin; y < xmax - 0.1 * it; y += it)
	{
		xv[1] = y;
		pf(i, 0) = RotParF1(xv, B, cond);
		pf(i, 1) = RotParF2(xv, B, cond, cond2);
		i++;
	}
	xv[1] = y;
	pf(i, 0) = RotParF1(xv, B, cond);
	pf(i, 1) = RotParF2(xv, B, cond, cond2);
}


//************************************************************************
// Rotated cigar
//************************************************************************

double RotCigarF1(const std::vector<double> &_v, Array<double> &coord, double cond)
{
	unsigned i;
	double sum = 0.;

	Array<double> v(_v.size());

	v(0) = scalarprod(coord.col(0), _v);
	sum += v(0) * v(0) ;

	for (i = 1; i < v.dim(0); i++)
	{
		v(i) = scalarprod(coord.col(i), _v);
		sum += Shark::sqr(cond * v(i)) ;
	}

	return sum / (cond * cond * v.dim(0));
}

double RotCigarF2(const std::vector<double> &_v, Array<double> &coord, double cond1, double cond2 = 2)
{
	unsigned i;
	double sum = 0.;

	Array<double> v(_v.size());

	v(0) = scalarprod(coord.col(0), _v);
	sum += (v(0) - cond2) * (v(0) - cond2) ;

	for (i = 1; i < v.dim(0); i++)
	{
		v(i) = scalarprod(coord.col(i), _v);
		sum += Shark::sqr(cond1 * (v(i) - cond2)) ;
	}

	return sum / (cond1 * cond1 * v.dim(0));
}

double RotCigarF1(const std::vector<double> &_v, Array<double> &coord, const std::vector<double> &lower, const std::vector<double> &upper, double cond)
{
	unsigned i;

	for (i = 0; i < _v.size();i++)
	{
		// Penalty
		if (_v[i] > upper[i] || _v[i] < lower[i])
			return MAXDOUBLE;
		//
	}

	return RotCigarF1(_v, coord, cond);
}

double RotCigarF2(const std::vector<double> &_v, Array<double> &coord, const std::vector<double> &lower, const std::vector<double> &upper, double cond1, double cond2 = 2)
{
	unsigned i;
	for (i = 0; i < _v.size();i++)
	{
		// Penalty
		if (_v[i] > upper[i] || _v[i] < lower[i])
			return MAXDOUBLE;
		//
	}

	return RotCigarF2(_v, coord, cond1, cond2);
}

void RotCigarSampleFront(Array< double > &pf, unsigned dimension, double cond , double cond2 = 2, unsigned n  = 100000)
{
	double xmin = 0;
	double xmax = cond2;
	unsigned i = 0, ii = 0;
	std::vector< double > xv(dimension);

	pf.resize(n, 2u);
	unsigned raw = 10 * n;
	Array<double> raw_pf(raw, 2u);

	Array<double> B(dimension, dimension);
	B = 0; for (i = 0;i < B.dim(0);i++) B(i, i) = 1;

	i = 0;

	double it = (xmax - xmin) / (raw - 1.);

	for (double x = xmin; x < xmax - 0.1 * it; x += it, i++)
	{
		for (ii = 0; ii < xv.size(); ii++) xv[ii] = x;
		raw_pf(i, 0) = RotCigarF1(xv, B, cond);
		raw_pf(i, 1) = RotCigarF2(xv, B, cond, cond2);
	}
	i = raw - 1;
	for (ii = 0; ii < xv.size(); ii++) xv[ii] = xmax;
	raw_pf(i, 0) = RotCigarF1(xv, B, cond);
	raw_pf(i, 1) = RotCigarF2(xv, B, cond, cond2);

	approxEquidistantFront(raw_pf, pf, n);
}

//************************************************************************
// Rotated cigtab
//************************************************************************

double RotCigTabF1(const std::vector<double> &_v, Array<double> &coord, double cond)
{
	unsigned i, n;
	double sum = 0.;

	n = _v.size();
	Array<double> v;
	v.resize(n, false);

	v(0) = scalarprod(coord.col(0), _v);
	sum += v(0) * v(0) ;
	v(n - 1) = scalarprod(coord.col(n - 1), _v);
	sum += Shark::sqr(cond * v(n - 1)) ;

	for (i = 1; i < n - 1; i++)
	{
		v(i) = scalarprod(coord.col(i), _v);
		sum += cond * Shark::sqr(v(i)) ;
	}

	return sum / (cond * cond * n);
}

double RotCigTabF2(const std::vector<double> &_v, Array<double> &coord, double cond1, double cond2 = 2)
{
	unsigned i, n;
	double sum = 0.;

	n = _v.size();
	Array<double> v;
	v.resize(n, false);

	v(0) = scalarprod(coord.col(0), _v);
	sum += Shark::sqr(v(0) - cond2);
	v(n - 1) = scalarprod(coord.col(n - 1), _v);
	sum += Shark::sqr(cond1 * (v(n - 1) - cond2));

	for (i = 1; i < n - 1; i++)
	{
		v(i) = scalarprod(coord.col(i), _v);
		sum += cond1 * Shark::sqr(v(i) - cond2) ;
	}

	return sum / (cond1 * cond1 * n);
}

double RotCigTabF1(const std::vector<double> &_v, Array<double> &coord, const std::vector<double> &lower, const std::vector<double> &upper, double cond)
{
	unsigned i;

	for (i = 0; i < _v.size();i++)
	{
		// Penalty
		if (_v[i] > upper[i] || _v[i] < lower[i])
			return MAXDOUBLE;
		//
	}

	return RotCigTabF1(_v, coord, cond);
}

double RotCigTabF2(const std::vector<double> &_v, Array<double> &coord, const std::vector<double> &lower, const std::vector<double> &upper, double cond1, double cond2 = 2)
{
	unsigned i;
	for (i = 0; i < _v.size();i++)
	{
		// Penalty
		if (_v[i] > upper[i] || _v[i] < lower[i])
			return MAXDOUBLE;
		//
	}

	return RotCigTabF2(_v, coord, cond1, cond2);
}

void RotCigTabSampleFront(Array< double > &pf, unsigned dimension, double cond , double cond2 = 2, unsigned n  = 100000)
{
	double xmin = 0;
	double xmax = cond2;
	unsigned i = 0, ii = 0;
	std::vector< double > xv(dimension);

	pf.resize(n, 2u);
	unsigned raw = 10 * n;
	Array<double> raw_pf(raw, 2u);

	Array<double> B(dimension, dimension);
	B = 0; for (i = 0;i < B.dim(0);i++) B(i, i) = 1;

	i = 0;

	double it = (xmax - xmin) / (raw - 1.);

	for (double x = xmin; x < xmax - 0.1 * it; x += it, i++)
	{
		for (ii = 0; ii < xv.size(); ii++) xv[ii] = x;
		raw_pf(i, 0) = RotCigTabF1(xv, B, cond);
		raw_pf(i, 1) = RotCigTabF2(xv, B, cond, cond2);
	}
	i = raw - 1;
	for (ii = 0; ii < xv.size(); ii++) xv[ii] = xmax;
	raw_pf(i, 0) = RotCigTabF1(xv, B, cond);
	raw_pf(i, 1) = RotCigTabF2(xv, B, cond, cond2);

	approxEquidistantFront(raw_pf, pf, n);
}

//************************************************************************
// Rotated tablet
//************************************************************************

double RotTabletF1(const std::vector<double> &_v, Array<double> &coord, double cond)
{
	unsigned i;
	double sum = 0.;

	Array<double> v(_v.size());

	v(0) = scalarprod(coord.col(0), _v);
	sum += Shark::sqr(cond * v(0));

	for (i = 1; i < v.dim(0); i++)
	{
		v(i) = scalarprod(coord.col(i), _v);
		sum += v(i) * v(i);
	}

	return sum / (cond * cond * v.dim(0));
}

double RotTabletF2(const std::vector<double> &_v, Array<double> &coord, double cond1, double cond2 = 2)
{
	unsigned i;
	double sum = 0.;

	Array<double> v(_v.size());

	v(0) = scalarprod(coord.col(0), _v);
	sum += Shark::sqr(cond1 * (v(0) - cond2)) ;

	for (i = 1; i < v.dim(0); i++)
	{
		v(i) = scalarprod(coord.col(i), _v);
		sum += (v(i) - cond2) * (v(i) - cond2) ;
	}

	return sum / (cond1 * cond1 * v.dim(0));
}

double RotTabletF1(const std::vector<double> &_v, Array<double> &coord, const std::vector<double> &lower, const std::vector<double> &upper, double cond)
{
	unsigned i;

	for (i = 0; i < _v.size();i++)
	{
		// Penalty
		if (_v[i] > upper[i] || _v[i] < lower[i])
			return MAXDOUBLE;
		//
	}

	return RotTabletF1(_v, coord, cond);
}

double RotTabletF2(const std::vector<double> &_v, Array<double> &coord, const std::vector<double> &lower, const std::vector<double> &upper, double cond1, double cond2 = 2)
{
	unsigned i;
	for (i = 0; i < _v.size();i++)
	{
		// Penalty
		if (_v[i] > upper[i] || _v[i] < lower[i])
			return MAXDOUBLE;
		//
	}

	return RotTabletF2(_v, coord, cond1, cond2);
}

void RotTabletSampleFront(Array< double > &pf, unsigned dimension, double cond , double cond2 = 2, unsigned  n  = 100000)
{
	double xmin = 0;
	double xmax = cond2;
	unsigned i = 0, ii = 0;
	std::vector< double > xv(dimension);

	pf.resize(n, 2u);
	unsigned raw = 20 * n;
	Array<double> raw_pf(raw, 2u);

	Array<double> B(dimension, dimension);
	B = 0; for (i = 0;i < B.dim(0);i++) B(i, i) = 1;

	i = 0;
	double it = (xmax - xmin) / (raw - 1.);

	for (double x = xmin; x < xmax - 0.1 * it; x += it, i++)
	{
		for (ii = 0; ii < dimension; ii++) xv[ii] = x;
		raw_pf(i, 0) = RotTabletF1(xv, B, cond);
		raw_pf(i, 1) = RotTabletF2(xv, B, cond, cond2);
	}
	i = raw - 1;
	for (ii = 0; ii < dimension; ii++) xv[ii] = xmax;
	raw_pf(i, 0) = RotTabletF1(xv, B, cond);
	raw_pf(i, 1) = RotTabletF2(xv, B, cond, cond2);

	approxEquidistantFront(raw_pf, pf, n);
}

//************************************************************************
// Fonseca's Concave Test Function (Deb's FON)
//************************************************************************
double RotFonsecaConcaveF1(const std::vector< double >& x, Array<double> &coord, double cond)
{
	unsigned n = x.size();
	unsigned i;
	Array<double> y(n);
	for (i = 0; i < n; i++) y(i) = pow(cond, double(i) / double(n - 1)) * scalarprod(coord.col(i), x);
	double   sum = 0.0;

	for (i = n; i--;)
		sum += (y(i) - 1.0 / sqrt((double) n)) * (y(i) - 1.0 / sqrt((double) n));

	return 1.0 - exp(-sum);
}
//
double RotFonsecaConcaveF2(const std::vector< double >& x, Array<double> &coord, double cond)
{
	unsigned n = x.size();
	unsigned i;
	Array<double> y(n);
	for (i = 0; i < n; i++) y(i) = pow(cond, double(i) / double(n - 1)) * scalarprod(coord.col(i), x);
	double   sum = 0.0;

	for (i = n; i--;) sum += (y(i) + 1.0 / sqrt((double) n)) * (y(i) + 1.0 / sqrt((double) n));

	return 1.0 - exp(-sum);
}


//************************************************************************
// Deb's Rotated problem (IEEE Transactions on EA 6(2), 2002)
//************************************************************************

double DebRotatedF1(const std::vector< double > &x, Array<double> &coord)
{
	double f1;//, penalty = 0;
	f1 = scalarprod(coord.col(0), x);

	// Penalty
	if (f1 > .3 || f1 < -.3)
		//penalty = fabs(f1) - .3;
		return MAXDOUBLE;
	//
	return f1 ;//+ penalty;
}


//
double DebRotatedF2(const std::vector< double >& x, Array<double> &coord)
{
	double   sum = 0.0;
	unsigned n;
	double   g, f1, f2;//, penalty=0;
	unsigned i;

	n  = x.size();

	Array<double> v(n);

	for (i = 0; i < n; i++)
		v(i) = scalarprod(coord.col(i), x);

	f1 = v(0);

	if (f1 > .3 || f1 < -.3)
		//penalty = fabs(f1) - .3;
		return MAXDOUBLE;

	//
	for (i = 1; i < n; i++)
		sum += v(i) * v(i) - 10. * cos(4 * M_PI * v(i));

	g = 1.0 + 10. * (double)(n - 1) + sum;

	f2 = g * exp(- 1. * (f1 / g));

	return f2 ;//+ penalty;
}


void DebRotatedSampleFront(Array< double > &pf, unsigned dimension, unsigned n  = 500)
{
	double xmin = -.3;
	double xmax = .3;
	unsigned i = 0;
	std::vector< double > xv(dimension);

	pf.resize(n, 2u);
	unsigned raw = 10 * n;
	Array<double> raw_pf(raw, 2u);

	Array<double> B(dimension, dimension);
	B = 0; for (i = 0;i < B.dim(0);i++) B(i, i) = 1;

	for (i = 1; i < xv.size(); i++) xv[i] = 0.;

	i = 0;
	double it = (xmax - xmin) / (raw - 1.);

	for (double x = xmin; x < xmax - 0.1 * it; x += it, i++)
	{
		xv[0] = x;
		raw_pf(i, 0) = DebRotatedF1(xv, B);
		raw_pf(i, 1) = DebRotatedF2(xv, B);
	}
	i = raw - 1;
	xv[0] = xmax;
	raw_pf(i, 0) = DebRotatedF1(xv, B);
	raw_pf(i, 1) = DebRotatedF2(xv, B);

	approxEquidistantFront(raw_pf, pf, n);
}

//
//************************************************************************
// End of paraboloid test functions
//************************************************************************
//

//
//************************************************************************
// ZDT test function implementation of Christian Igel
//************************************************************************
//

double CI1F1(const std::vector< double >& x)
{
	return x[0] * x[0];
}

double CI1F2(const std::vector< double >& x)
{
	return (x[1] - 2.) *(x[1] - 2.) + (x[2] - 2.) *(x[2] - 2.);
}

double CI2F1(const std::vector< double >& x)
{
	return x[0] * x[0];
}

double CI2F2(const std::vector< double >& x)
{
	return (x[0] - 2.) *(x[0] - 2.) + x[1] * x[1];
}

//************************************************************************
// ZDT 1
//************************************************************************

double ZDT1F1(const std::vector< double >& x)
{
	return fabs(x[0]);
}

double ZDT1G(const std::vector< double >& x)
{
	double g = 0;
	unsigned n = x.size();
	for (unsigned i = 1; i < n; i++)
		g += fabs((x[i]));
	g = 9 * g / (n - 1.) + 1.;
	return g;
}

double ZDT1F2(const std::vector< double >& x)
{
	return  ZDT1G(x) *(1 - sqrt(fabs(x[0]) /  ZDT1G(x)));
}

void ZDT1SampleFront(Array< double > &pf, unsigned n = 500)
{
	double xmin = 0;
	double xmax = 1.;

	pf.resize(n, 2u);
	unsigned raw = 10 * n;
	Array<double> raw_pf(raw, 2u);

	std::vector< double > xv(30);

	unsigned i = 0;
	for (i = 1; i < 30; i++) xv[i] = 0.;
	i = 0;

	double it = (xmax - xmin) / (raw - 1.);

	for (double x = xmin; x < xmax - 0.1 * it; x += it, i++)
	{
		xv[0] = x;
		raw_pf(i, 0) = ZDT1F1(xv);
		raw_pf(i, 1) = ZDT1F2(xv);
	}
	i = raw - 1;
	xv[0] = xmax;
	raw_pf(i, 0) = ZDT1F1(xv);
	raw_pf(i, 1) = ZDT1F2(xv);

	approxEquidistantFront(raw_pf, pf, n);
}


//************************************************************************
// ZDT 2
//************************************************************************

double ZDT2F1(const std::vector< double >& x)
{
	return x[0];
}

double ZDT2G(const std::vector< double >& x)
{
	double g = 0;
	unsigned n = x.size();
	for (unsigned i = 1; i < n; i++)
		g += x[i];
	g = 9 * g / (n - 1.) + 1.;
	return g;
}

double ZDT2F2(const std::vector< double >& x)
{
	return  ZDT2G(x) *(1 - Shark::sqr(x[0] /  ZDT2G(x)));
}

void ZDT2SampleFront(Array< double > &pf, unsigned n = 500)
{
	double xmin = 0;
	double xmax = 1.;

	pf.resize(n, 2u);
	unsigned raw = 10 * n;
	Array<double> raw_pf(raw, 2u);

	std::vector< double > xv(30);

	unsigned i = 0;
	for (i = 1; i < 30; i++) xv[i] = 0.;
	i = 0;
	double it = (xmax - xmin) / (raw - 1.);

	for (double x = xmin; x < xmax - 0.1 * it; x += it, i++)
	{
		xv[0] = x;
		raw_pf(i, 0) = ZDT2F1(xv);
		raw_pf(i, 1) = ZDT2F2(xv);
	}
	i = raw - 1;
	xv[0] = xmax;
	raw_pf(i, 0) = ZDT2F1(xv);
	raw_pf(i, 1) = ZDT2F2(xv);

	approxEquidistantFront(raw_pf, pf, n);
}

//************************************************************************
// ZDT 3
//************************************************************************

double ZDT3F1(const std::vector< double >& x)
{
	return x[0];
}

double ZDT3G(const std::vector< double >& x)
{
	double g = 0;
	unsigned n = x.size();
	for (unsigned i = 1; i < n; i++)
		g += x[i];
	g = 9 * g / (n - 1.) + 1.;
	return g;
}

double ZDT3F2(const std::vector< double >& x)
{
	double gx = ZDT3G(x);
	return  gx *(1 - sqrt(x[0] / gx) - x[0] / gx * sin(10 * M_PI *  x[0]));
}

void ZDT3SampleFront(Array< double > &pf, unsigned n = 500)
{
	double xmin = 0;
	double xmax = 1.;
	std::vector< double > xv(30);

	pf.resize(n, 2u);
	unsigned raw = 40 * n;
	Array<double> raw_pf(raw, 2u);

	unsigned i = 0;
	for (i = 1; i < 30; i++) xv[i] = 0.;
	i = 0;
	double it = (xmax - xmin) / (raw - 1.);

	for (double x = xmin; x < (xmax - 0.1 * it); x += it, i++)
	{
		xv[0] = x;
		raw_pf(i, 0) = ZDT3F1(xv);
		raw_pf(i, 1) = ZDT3F2(xv);
	}
	i = raw - 1;
	xv[0] = xmax;
	raw_pf(i, 0) = ZDT3F1(xv);
	raw_pf(i, 1) = ZDT3F2(xv);

	sort2DBy1st(raw_pf);
	for (i = 0;  i < (raw_pf.dim(0) - 1);)
	{
		if (raw_pf(i + 1, 1) > raw_pf(i, 1)) raw_pf.remove_row(i + 1);
		else i++;
	}

	approxEquidistantFront(raw_pf, pf, n);
}


//************************************************************************
// ZDT 4
//************************************************************************

// Scaled version
double ZDT4F1(const std::vector< double >& x)
{
	if (x[0] < -5.)
	{
	  throw SHARKEXCEPTION("ZDT4F1 called with invalid parameter x[0]");
	}
	return (x[0] + 5.) / 10.;// x[0];
}

double ZDT4G(const std::vector< double >& x)
{
	double g;
	unsigned n = x.size();
	g = 1 + 10 * (n - 1);
	for (unsigned i = 1; i < n; i++)
		g += Shark::sqr(x[i]) - 10 * cos(4 * M_PI * x[i]);
	return g;
}

double ZDT4F2(const std::vector< double >& x)
{
	return ZDT4G(x) *(1 - sqrt(((x[0] + 5.) / 10.) / ZDT4G(x)));
}

double ZDT4PrimeF2(const std::vector< double >& x , const Array<double> &coord)
{
	unsigned i, j, n = x.size();
	double sum;
	std::vector< double > y;
	y.push_back(x[0]);
	for (i = 0; i < n - 1; i++)
	{
		sum = 0.;
		for (j = 1; j < n; j++) sum += coord(i, j - 1) * x[j];
		y.push_back(sum);
	}
	return ZDT4G(y) *(1 - sqrt(((y[0] + 5.) / 10.) / ZDT4G(y)));
}

// Scaled version 2
double ZDT4FII1(const std::vector< double >& x)
{
	return (x[0] + 5.) / 10.;// x[0];
}

double ZDT4GII(const std::vector< double >& x)
{
	double g;
	unsigned n = x.size();
	g = 1 + 10 * (n - 1);
	for (unsigned i = 1; i < n; i++)
		g += Shark::sqr(x[i]) - 10 * cos(4 * M_PI * x[i]);
	return g;
}

double ZDT4FII2(const std::vector< double >& x)
{
	return ZDT4G(x) *(1 - sqrt(((x[0] + 5.) / 10.) / ZDT4G(x)));
}

// Unscaled version
double ZDT4FG(const std::vector< double >& x)
{
	double g;
	unsigned n = x.size();
	g = 1 + 10 * (n - 1);
	for (unsigned i = 1; i < n; i++)
		g += Shark::sqr(x[i]) - 10 * cos(4 * M_PI * x[i]);
	return g;
}

double ZDT4FF2(const std::vector< double >& x)
{
	return  ZDT4FG(x) *(1 - sqrt(x[0] /  ZDT4FG(x)));
}

double ZDT4FF1(const std::vector< double >& x)
{
	return  x[0];
}


void ZDT4SampleFront(Array< double > &pf, unsigned n = 500)
{
	double xmin = 0.;
	double xmax = 1.;

	pf.resize(n, 2u);
	unsigned raw = 10 * n;
	Array<double> raw_pf(raw, 2u);

	std::vector< double > xv(10);

	unsigned i = 0;
	for (i = 1; i < 10; i++) xv[i] = 0.;
	i = 0;
	double it = (xmax - xmin) / (raw - 1.);

	for (double x = xmin; x < xmax - 0.1 * it; x += it, i++)
	{
		xv[0] = x;
		raw_pf(i, 0) = ZDT4FF1(xv);
		raw_pf(i, 1) = ZDT4FF2(xv);
	}
	i = raw - 1;
	xv[0] = xmax;
	raw_pf(i, 0) = ZDT4FF1(xv);
	raw_pf(i, 1) = ZDT4FF2(xv);

	approxEquidistantFront(raw_pf, pf, n);
}

//************************************************************************
// ZDT 6
//************************************************************************

double ZDT6F1(const std::vector< double >& x)
{
	return 1 -exp(-4*x[0]) * pow(sin(6 * M_PI * x[0]), 6);
}

double ZDT6G(const std::vector< double >& x)
{
	double g;
	unsigned n = x.size();
	g = 0;
	for (unsigned i = 1; i < n; i++)
		g += x[i];
	g /= (n - 1);
	g = pow(g, 0.25);
	g *= 9.;
	g += 1.;
	return g;
}

double ZDT6F2(const std::vector< double >& x)
{
	return  ZDT6G(x) *(1 - (ZDT6F1(x) /  ZDT6G(x)) *(ZDT6F1(x) /  ZDT6G(x)));
}

void ZDT6SampleFront(Array< double > &pf, unsigned n = 500)
{
	double xmin = 0.;
	double xmax = 1.;

	pf.resize(n, 2u);
	unsigned raw = 50 * n;
	Array<double> raw_pf(raw, 2u);

	std::vector< double > xv(10);

	unsigned i = 0;
	for (i = 1; i < 10; i++) xv[i] = 0.;
	i = 0;
	double it = (xmax - xmin) / (raw - 1.);

	for (double x = xmin; x < xmax - 0.1 * it; x += it, i++)
	{
		xv[0] = x;
		raw_pf(i, 0) = ZDT6F1(xv);
		raw_pf(i, 1) = ZDT6F2(xv);
	}
	i = raw - 1;
	xv[0] = xmax;
	raw_pf(i, 0) = ZDT6F1(xv);
	raw_pf(i, 1) = ZDT6F2(xv);

	approxEquidistantFront(raw_pf, pf, n);
}
//
//************************************************************************
// End of ZDT test function implementation of Christian Igel
//************************************************************************
//

////////////////////////////////////////////////////////////////////// SOO
//
//************************************************************************
// SOO test function implementation of Tatsuya Okabe
//************************************************************************
//

//************************************************************************
// Sphere Test Function
//************************************************************************
double sphere(const std::vector< double >& x)
{
	double a;
	unsigned i, n;
	for (i = 0, a = 0, n = x.size(); i < n; i++)
	{
		a += x[ i ] * x[ i ];
	}
	return a;
}

//************************************************************************
// DeJong F2 Test Function
//************************************************************************
double DeJongF2(const std::vector< double >& x)
{
	double f;
	unsigned i, n;
	f = 0.0;
	n = x.size();
	for (i = 0; i < n - 1; i++)
	{
		f += 100.0 * pow(x[i] * x[i] - x[i+1], 2.0) + pow(1.0 - x[0], 2.0);
	}
	return f;
}

//************************************************************************
// DeJong F3 Test Function
//************************************************************************
double DeJongF3(const std::vector< double >& x)
{
	double f;
	unsigned i, n;
	f = 0.0;
	n = x.size();
	for (i = 0; i < n; i++)
	{
		f += (double)(floor(x[ i ]));
	}
	return f;
}

//************************************************************************
// Schaffer F7 Test Function
//************************************************************************
double SchafferF7(const std::vector< double >& x)
{
	double f, t;
	unsigned i, n;
	f = 0.0;
	n = x.size();
	for (i = 0; i < n - 1; i++)
	{
		t = x[i] * x[i] + x[i+1] * x[i+1];
		f += pow(t, 0.25) * (pow(sin(50 * pow(t, 0.1)), 2.0) + 1.0);
	}
	return f;
}

//************************************************************************
// Schwefel F1 Test Function
//************************************************************************
double SchwefelF1(const std::vector< double >& x)
{
	double f;
	unsigned i, n;
	f = 0.0;
	n = x.size();
	for (i = 0; i < n; i++)
	{
		f += -x[i] * sin(pow(fabs(x[i]), 0.5));
	}
	return f;
}

//************************************************************************
// Schwefel F2 Test Function
//************************************************************************
double SchwefelF2(const std::vector< double >& x)
{
	double f;
	unsigned i, j, n;
	f = 0.0;
	n = x.size();
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < i + 1; j++)
		{
			f += x[j] * x[j];
		}
	}
	return f;
}

//************************************************************************
// Rastrigin Test Function
//************************************************************************
double rastrigin(const std::vector< double >& x)
{
	unsigned i, n;
	double sum;
	const double C = M_2PI;
	double A = 10.; // suitable for dim = 10
	double B = 10.; // suitable for dim = 10
	for (i = 0, n = x.size(), sum = n * A; i < n; i++)
	{
		sum += (x[ i ] * x[ i ]) - B * cos(C * x[ i ]);
	}
	return sum;
}

//************************************************************************
// Rosenbrock Test Function
//************************************************************************
double rosenbrock(const std::vector< double >& x)
{
	const double A = 100.;
	const double B = 1.;
	unsigned i, n;
	double   a;
	for (i = 0, a = 0, n = x.size(); i < n; i++)
	{
		a += A * ((((x[ i ]) - x[ i+1 ]) * ((x[ i ]) - x[ i+1 ])) + ((x[ i ] - B) * (x[ i ] - B))) * ((((x[ i ]) - x[ i+1 ]) * ((x[ i ]) - x[ i+1 ])) + ((x[ i ] - B) * (x[ i ] - B)));
	}
	return a;
}

//************************************************************************
// Ackley Test Function
//************************************************************************
double ackley(const std::vector< double >& x)
{
	const double A        = 20.;
	const double B        = 0.2;
	const double C        = M_2PI;
	unsigned i, n;
	double   a, b;
	for (a = b = 0., i = 0, n = x.size(); i < n; ++i)
	{
		a += x[ i ] * x[ i ];
		b += cos(C * x[ i ]);
	}
	return -A * exp(-B * sqrt(a / n)) - exp(b / n) + A + M_E;
}

//************************************************************************
// Func1 Test Function ( no name )
//************************************************************************
double func1(const std::vector< double >& x)
{
	const double A = 5.;
	const double B = 31.4159265359;
	unsigned i, n;
	double   a;
	for (i = 0, a  = 0, n = x.size(); i < n; i++)
	{
		if ((x[ i ] >= -5) && (x[ i ] <= 5))
		{
			a += (A - fabs(x [ i ])) * fabs(cos(B * x [ i ]));
		}
	}
	return a;
}




//
//************************************************************************
// IHR
//************************************************************************
//

//////////
// IHR1 //
//////////
inline double h(double x, double n)
{
	return 1 / (1 + exp(-x / sqrt(n)));
}

inline double hf(double x, double y0, double ymax)
{
	if (fabs(y0) <=  ymax) return x;
	return fabs(y0) + 1;
}

inline double hg(double x)
{
	return Shark::sqr(x) / (fabs(x) + 0.1);
}

double IHR1F1(const std::vector<double> &x, const Array<double> &coord)
{
	return fabs(scalarprod(coord.row(0), x));
}

double IHR1F2(const std::vector<double> &x, const Array<double> &coord)
{
	unsigned i;
	double g = 0;
	unsigned n = x.size();
	Array<double> y(n);

	for (i = 0; i < n; i++) y(i) = scalarprod(coord.row(i), x);

	double ymax = fabs(coord.col(0)(0));
	for (i = 1; i < n; i++) if (fabs(coord.row(0)(i)) > ymax) ymax = fabs(coord.row(0)(i));
	ymax = 1 / ymax;

	for (i = 1; i < n; i++) g += hg(y(i));
	g = 9 * g / (n - 1.) + 1.;

	if (fabs(y(0)) <= ymax) return g *(1. - sqrt(h(y(0), n) / g));
	return g *(1 + fabs(y(0)));
}

//////////
// IHR2 //
//////////
double IHR2F1(const std::vector<double> &x, Array<double> &coord)
{
	return fabs(scalarprod(coord.col(0), x));
}

double IHR2F2(const std::vector<double> &x, Array<double> &coord)
{
	unsigned i;
	double g = 0;
	unsigned n = x.size();
	Array<double> y;
	y.resize(n);

	for (i = 0; i < n; i++) y(i) = scalarprod(coord.col(i), x);
	double ymax = fabs(coord.col(0)(0));
	for (i = 1; i < n; i++) if (fabs(coord.row(0)(i)) > ymax) ymax = fabs(coord.row(0)(i));
	ymax = 1 / ymax;


	for (i = 1; i < n; i++) g += hg(y(i));
	g = 9 * g / (n - 1.) + 1.;

	if (fabs(y(0)) <= ymax) return g *(1. - Shark::sqr(y(0) / g));
	return g *(1 + fabs(y(0)));
}

//////////
// IHR3 //
//////////
double IHR3F1(const std::vector<double> &x, Array<double> &coord)
{
	return fabs(scalarprod(coord.col(0), x));
}

double IHR3F2(const std::vector<double> &x, Array<double> &coord)
{
	unsigned i;
	double g = 0;
	unsigned n = x.size();
	Array<double> y;
	y.resize(n);

	for (i = 0; i < n; i++) y(i) = scalarprod(coord.col(i), x);
	double ymax = fabs(coord.col(0)(0));
	for (i = 1; i < n; i++) if (fabs(coord.row(0)(i)) > ymax) ymax = fabs(coord.row(0)(i));
	ymax = 1 / ymax;


	for (i = 1; i < n; i++) g += hg(y(i));
	g = 9 * g / (n - 1.) + 1.;

	if (fabs(y(0)) <= ymax) return g *(1. - sqrt(h(y(0), n) / g) - h(y(0), n) / g * sin(10 * M_PI * y(0)));
	return g *(1 + fabs(y(0)));
}


//////////
// IHR4 //
//////////
double IHR4F1(const std::vector<double> &x, Array<double> &coord)
{
	return fabs(scalarprod(coord.col(0), x));
}

double IHR4F2(const std::vector<double> &x, Array<double> &coord)
{
	unsigned i;
	double g = 0;
	unsigned n = x.size();
	Array<double> y;
	y.resize(n);

	for (i = 0; i < n; i++) y(i) = scalarprod(coord.row(i), x);

	double ymax = fabs(coord.col(0)(0));
	for (i = 1; i < n; i++) if (fabs(coord.row(0)(i)) > ymax) ymax = fabs(coord.row(0)(i));
	ymax = 1 / ymax;

	for (i = 1; i < n; i++) g += Shark::sqr(y(i)) - 10 * cos(4 * M_PI * y(i));
	g += 10 * (n - 1.) + 1.;

	if (fabs(y(0)) <= ymax) return g *(1. - sqrt(h(y(0), n) / g));
	return g *(1 + fabs(y(0)));
}


//////////
// IHR6 //
//////////
double IHR6F1(const std::vector<double> &x, Array<double> &coord)
{
	double y0 = scalarprod(coord.col(0), x);
	return 1 -exp(-4*fabs(y0)) * pow(sin(6 * M_PI * y0), 6);
}

double IHR6F2(const std::vector<double> &x, Array<double> &coord)
{
	unsigned i;
	double g = 0;
	unsigned n = x.size();
	Array<double> y;
	y.resize(n);

	for (i = 0; i < n; i++) y(i) = scalarprod(coord.row(i), x);
	double ymax = fabs(coord.col(0)(0));
	for (i = 1; i < n; i++) if (fabs(coord.row(0)(i)) > ymax) ymax = fabs(coord.row(0)(i));
	ymax = 1 / ymax;

	for (i = 1; i < n; i++) g += hg(y(i));
	g = 1 + 9 * pow(g / (n - 1.), .25);

	if (fabs(y(0)) <= ymax) return g *(1. - Shark::sqr(IHR6F1(x, coord) / g));
	return g *(1 + fabs(y(0)));
}

//**********************************************************************************
// Scalable Test Problems for Evolutionary Multi-Objective Optimization,
// Deb, Thiele, Laumanns, Zitzler
//**********************************************************************************

#include <Array/Array.h>
#include <Array/Array2D.h>
#include <Array/ArrayOp.h>
#include <Rng/Uniform.h>

//! \brief Renders a fitness function unseparable.
template<typename DTLZ_T>
class Linkage_2
{
public:
	void operator()(const Array<double> & transformationMatrix,
					std::vector<double> & out,
					const std::vector<double> & in
				   )
	{
		std::vector<double> y(in.size());

		for (unsigned i = 0; i < in.size(); i++)
		{
			y[i] = scalarprod(transformationMatrix.col(i), in);
		}

		DTLZ_T dtlz; dtlz(out, y);
	}
};

//! \brief Renders a fitness function unseparable.
template<typename DTLZ_T>
class Linkage_3
{
public:
	void operator()(const Array<double> & transformationMatrix,
					std::vector<double> & out,
					const std::vector<double> & in)
	{
		unsigned i;

		std::vector<double> y(in), z(in);

		for (i = 0; i < z.size(); i++)
		{
			z[i] *= z[i];
		}

		for (i = 0; i < y.size(); i++)
		{
			y[i] = scalarprod(transformationMatrix.col(i), z);
		}

		DTLZ_T dtlz; dtlz(out, y);
	}
};


/* DTLZ1
   Defined for: \forall x_i, 1 <= i <= n: x_i \in [0,1]
*/
//!
//! \brief DTLZ-1 test function for algorithm analysis
//!
//! \par
//! This class realizes the DTLZ-1 standard test fitness function.
//!
struct DTLZ1
{
	void operator()(std::vector<double> & out, const std::vector<double> & in)
	{
		if (out.size() >= in.size())
			return;

		int k = in.size() - out.size() + 1 ;

		double g = 0.0 ;

		int i;
		for (i = in.size() - k + 1; i <= (int)in.size(); i++)
		{
			g += Shark::sqr(in[i-1] - 0.5) - cos(20 * 3.1416 * (in[i-1] - 0.5));
		}

		g = 100 * (k + g);

		for (i = 1; i <= (int)out.size(); i++)
		{
			double f = 0.5 * (1 + g);
			for (int j = out.size() - i; j >= 1; j--)
				f *= in[j-1];

			if (i > 1)
				f *= 1 - in[(out.size() - i + 1) - 1];

			out[i-1] = f;
		}
	}
};
/* !DTLZ1 */

/* DTLZ2
   Defined for: \forall x_i, 1 <= i <= n: x_i \in [0,1]
*/
//!
//! \brief DTLZ-2 test function for algorithm analysis
//!
//! \par
//! This class realizes the DTLZ-2 standard test fitness function.
//!
struct DTLZ2
{
	void operator()(std::vector<double> & out, const std::vector<double> & in)
	{
		if (out.size() >= in.size())
			return;

		int    i, k ;
		double g ;

		k = in.size() - out.size() + 1 ;
		g = 0.0 ;

		for (i = in.size() - k + 1; i <= (int)in.size(); i++)
			g += Shark::sqr(in[i-1] - 0.5);

		for (i = 1; i <= (int)out.size(); i++)
		{
			double f = (1 + g);
			for (int j = out.size() - i; j >= 1; j--)
				f *= cos(in[j-1] * 3.1416 / 2);

			if (i > 1)
				f *= sin(in[(out.size() - i + 1) - 1] * 3.1416 / 2);

			out[i-1] = f ;
		}
	}
};

typedef Linkage_2<DTLZ2> L_2_DTLZ2;
typedef Linkage_3<DTLZ2> L_3_DTLZ2;

/* !DTLZ2 */

/* DTLZ3 */
//!
//! \brief DTLZ-3 test function for algorithm analysis
//!
//! \par
//! This class realizes the DTLZ-3 standard test fitness function.
//!
struct DTLZ3
{
	void operator()(std::vector<double> & out, const std::vector<double> & in)
	{
		if (out.size() >= in.size())
			return;

		int    i, k ;
		double g ;

		k = in.size() - out.size() + 1 ;
		g = 0.0 ;

		for (i = in.size() - k + 1; i <= (int)in.size(); i++)
			g += Shark::sqr(in[i-1] - 0.5) - cos(20 * 3.141592653589793 * (in[i-1] - 0.5));

		g = 100 * (k + g);

		for (i = 1; i <= (int)out.size(); i++)
		{
			double f = (1 + g);
			for (int j = out.size() - i; j >= 1; j--)
				f *= cos(in[j-1] * 3.141592653589793 / 2);

			if (i > 1)
				f *= sin(in[(out.size() - i + 1) - 1] * 3.141592653589793 / 2);

			out[i-1] = f ;
		}
	}
};

typedef Linkage_2<DTLZ3> L_2_DTLZ3;
typedef Linkage_3<DTLZ3> L_3_DTLZ3;
/* !DTLZ3 */

/* DTLZ4 */
//!
//! \brief DTLZ-4 test function for algorithm analysis
//!
//! \par
//! This class realizes the DTLZ-4 standard test fitness function.
//!
struct DTLZ4
{
	void operator()(std::vector<double> & out, const std::vector<double> & in)
	{
		int    i, k ;
		double g ;

		double alpha ;

		k = in.size() - out.size() + 1 ;
		alpha = 100;
		g = 0.0 ;
		for (i = in.size() - k + 1; i <= (int)in.size(); i++)
			g += Shark::sqr(in[i-1] - 0.5);

		for (i = 1; i <= (int)out.size(); i++)
		{
			double f = (1 + g);
			for (int j = out.size() - i; j >= 1; j--)
				f *= cos(pow(in[j-1], alpha) * 3.141592653589793 / 2);

			if (i > 1)
				f *= sin(pow(in[(out.size() - i + 1) - 1], alpha) * 3.141592653589793 / 2);

			out[i-1] = f ;
		}
	}
};
/* !DTLZ4 */

/* DTLZ5 */
//!
//! \brief DTLZ-5 test function for algorithm analysis
//!
//! \par
//! This class realizes the DTLZ-5 standard test fitness function.
//!
struct DTLZ5
{
	void operator()(std::vector<double> & out, const std::vector<double> & in)
	{
		int    i, k ;
		double g ;

		double * theta = new double[out.size()] ;

		k = in.size() - out.size() + 1 ;
		g = 0.0 ;
		for (i = out.size() - k + 1; i <= (int)out.size(); i++)
			g += Shark::sqr(in[i-1] - 0.5);

		double t = M_PI  / (4 * (1 + g));

		theta[0] = in[0] * M_PI / 2;
		for (i = 2; i <= ((int)out.size() - 1); i++)
			theta[i-1] = t * (1 + 2 * g * in[i-1]);

		for (i = 1; i <= (int)out.size(); i++)
		{
			double f = (1 + g);

			for (int j = out.size() - i; j >= 1; j--)
				f *= cos(theta[j-1]);

			if (i > 1)
				f *= sin(theta[(out.size() - i + 1) - 1]);

			out[i-1] = f ;
		}

		delete [] theta ;
	}
};
/* !DTLZ5 */

/* DTLZ6 */
//!
//! \brief DTLZ-6 test function for algorithm analysis
//!
//! \par
//! This class realizes the DTLZ-6 standard test fitness function.
//!
struct DTLZ6
{
	void operator()(std::vector<double> & out, const std::vector<double> & in)
	{
		int    i, k ;
		double g ;
		double * theta = new double[out.size()] ;

		k = in.size() - out.size() + 1 ;
		g = 0.0 ;
		for (i = in.size() - k + 1; i <= (int)in.size(); i++)
			g += pow(in[i-1], 0.1);

		double t = M_PI  / (4 * (1 + g));

		theta[0] = in[0] * M_PI / 2;
		for (i = 2; i <= (int)out.size() - 1; i++)
			theta[i-1] = t * (1 + 2 * g * in[i-1]);

		for (i = 1; i <= (int)out.size(); i++)
		{
			double f = (1 + g);

			for (int j = out.size() - i; j >= 1; j--)
				f *= cos(theta[j-1]);

			if (i > 1)
				f *= sin(theta[(out.size() - i + 1) - 1]);

			out[i-1] = f ;
		} // for
		delete [] theta;
	}
};
/* !DTLZ6 */

/* DTLZ7 */
//!
//! \brief DTLZ-7 test function for algorithm analysis
//!
//! \par
//! This class realizes the DTLZ-7 standard test fitness function.
//!
struct DTLZ7
{
	void operator()(std::vector<double> & out, const std::vector<double> & in)
	{
		int    i, j, k ;
		double g ;
		double h ;

		k = in.size() - out.size() + 1 ;
		g = 0.0 ;
		for (i = in.size() - k + 1; i <= (int)in.size(); i++)
			g += in[i-1];

		g = 1 + 9 * g / k;

		for (i = 1; i <= (int)out.size() - 1; i++)
			out[i-1] = in[i-1];

		h = 0.0 ;
		for (j = 1; j <= (int)out.size() - 1; j++)
			h += in[j-1] / (1 + g) * (1 + sin(3 * M_PI * in[j-1]));

		h = out.size() - h ;

		out[out.size()-1] = (1 + g) * h;
	}
};
/* !DTLZ7 */

/* Igel1 - based on ELLI */
//!
//! \brief Igel-1 test function for algorithm analysis
//!
//! \par
//! This class realizes the Igel-1 standard test fitness function.
//!
struct Igel1
{
	Igel1() : a(1000)
	{}

	void operator()(const Array<double> & rotationMatrix, std::vector<double> & out, const std::vector<double> & x)
	{
		std::vector<double> y(x);

		unsigned int i, j;
		for (i = 0; i < x.size(); i++)
		{
			y[i] = scalarprod(rotationMatrix.col(i), x);
		}

		unsigned m = out.size(); unsigned n = y.size();

		c.resize(m, std::vector<double> (n));

		for (i = 0; i < m; i++)
		{
			for (j = 0; j < n; j++)
			{
				if (i == j)
				{
					c[i][j] = 1;
				}
				else
					c[i][j] = 0;

			}
		}

		double sum; double tmp = 1 / (a * a * n);

		for (i = 0; i < m; i++)
		{
			sum = 0;

			for (j = 0; j < n; j++)
			{
				sum += Shark::sqr(pow(a, (double) j / (n - 1)) * (y[j] - c[i][j]));
			}

			out[i] = tmp * sum;
		}
	}

	double a; std::vector<std::vector<double> > c;
};

/* Igel2 */

//!
//! \brief Igel-2 test function for algorithm analysis
//!
//! \par
//! This class realizes the Igel-2 standard test fitness function.
//!
struct Igel2
{
	Igel2() : a(1000)
	{}

	void operator()(const Array<double> & rotationMatrix,
					std::vector<double> & out,
					const std::vector<double> & x
				   )
	{
		std::vector<double> y(x);

		unsigned int i, j;
		for (i = 0; i < x.size(); i++)
		{
			y[i] = scalarprod(rotationMatrix.col(i), x);
		}

		unsigned m = out.size(); unsigned n = y.size();

		c.resize(m, std::vector<double> (n));

		for (i = 0; i < m; i++)
		{
			for (j = 0; j < n; j++)
			{
				if (i == j)
					c[i][j] = sqrt((long double)(m - 1) / m);
				else if (j > m - 1)
					c[i][j] = 0;
				else
					c[i][j] = -1 / sqrt((long double) m * (m - 1));
			}
		}

		double sum; double tmp = 1 / (a * a * n);

		for (i = 0; i < m; i++)
		{
			sum = 0;

			for (j = 0; j < n; j++)
			{
				sum += Shark::sqr(pow(a, (double) j / (n - 1)) * y[j] - c[i][j]);
			}

			out[i] = tmp * sum;
		}
	}

	double a; std::vector<std::vector<double> > c;
};

/* Igel3 */
//!
//! \brief Igel-3 test function for algorithm analysis
//!
//! \par
//! This class realizes the Igel-3 standard test fitness function.
//!
struct Igel3
{
	Igel3() : a(1000)
	{}

	void operator()(const Array<double> & rotationMatrix, std::vector<double> & out, const std::vector<double> & x)
	{
		std::vector<double> y(x);

		unsigned int i, j;
		for (i = 0; i < x.size(); i++)
		{
			y[i] = scalarprod(rotationMatrix.col(i), x);
		}

		unsigned m = out.size(); unsigned n = y.size();

		c.resize(m, std::vector<double> (n));

		for (i = 0; i < m; i++)
		{
			for (j = 0; j < n; j++)
			{
				c[i][j] = i * pow(2.0, (int) i);
			}
		}

		double sum; double tmp = 1 / (a * a * n);

		for (i = 0; i < m; i++)
		{
			sum = 0;

			for (j = 0; j < n; j++)
			{
				sum += Shark::sqr(pow(a, (double) j / (n - 1)) * (y[j] - c[i][j]));
			}

			out[i] = tmp * sum;
		}
	}

	double a; std::vector<std::vector<double> > c;
};

/* Igel4 */
//!
//! \brief Igel-4 test function for algorithm analysis
//!
//! \par
//! This class realizes the Igel-4 standard test fitness function.
//!
struct Igel4
{
	Igel4() : a(1000)
	{}

	void operator()(const std::vector<Array<double> > & rotationMatrices,
					std::vector<double> & out,
					const std::vector<double> & x
				   )
	{
		std::vector<double> y(x.size(), 0);

		unsigned m = out.size(); unsigned n = y.size();

		c.resize(m, std::vector<double> (n));

		unsigned int i, j;
		for (i = 0; i < m; i++)
		{
			for (j = 0; j < n; j++)
			{
				if (i == j)
					c[i][j] = 1;
				else
					c[i][j] = 0;
			}
		}

		double sum; double tmp = 1 / (a * a * n);

		for (i = 0; i < m; i++)
		{
			const Array<double> & rm = rotationMatrices[i];

			for (j = 0; j < x.size(); j++)
			{
				y[j] = scalarprod(rm.col(j), x);
			}

			sum = 0;

			for (j = 0; j < n; j++)
			{
				sum += Shark::sqr(pow(a, (double) j / (n - 1)) * (y[j] - c[i][j]));
			}

			out[i] = tmp * sum;
		}
	}

	double a; std::vector<std::vector<double> > c;
};

#endif /* !__TESTFUNCTIOMOO_H */

