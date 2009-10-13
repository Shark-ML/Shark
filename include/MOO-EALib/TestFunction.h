/*!
*  \file TestFunction.h
*
*  \brief Collection of benchmark fitness functions for algorithm analysis
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

/* ====================================================================== */
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


#ifndef __TESTFUNCTIONCMOO_H
#define __TESTFUNCTIONMOO_H

#include <SharkDefs.h>
#include <Array/ArraySort.h>
#include <Rng/GlobalRng.h>


//
//************************************************************************
// Tools for equidistant test function sampling in objective space
//************************************************************************
//

unsigned approxEquidistantFrontDoIt(Array<double> &v, Array<double> &w, unsigned n, double d)
{
	double dCurrent, dPrevious; //5.9976
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
	Array<double> H;
	B.resize(d, d);
	H.resize(d, d);
	for (i = 0; i < d; i++)
	{
		for (c = 0; c < d; c++)
		{
			B(i, c) = Rng::gauss(0, 1);
		}
	}

	for (i = 0; i < d; i++)
	{
		for (j = 0; j < i; j++)
		{
			H = B;
			for (c = 0; c < d; c++)
			{
				B(i, c) -= scalarprod(H[i], H[j]) * H(j, c);
			}
		}
		double normB = norm(B[i]);
		for (j = 0; j < d; j++)
		{
			B(i, j) = B(i, j) / normB;
		}
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
		sum += pow(cond, 2 * (double(i) / double(v.dim(0) - 1))) * v(i) * v(i) ;
	}

	return sum;
}

double RotParF2(const std::vector<double> &_v, Array<double> &coord, double cond1, double cond2 = 2)
{
	unsigned i;
	double sum = 0.;

	Array<double> v(_v.size());

	for (i = 0; i < v.dim(0); i++)
	{
		v(i) = scalarprod(coord.col(i), _v);
		sum += pow(cond1, 2 * (double(i) / double(v.dim(0) - 1))) * (v(i) - cond2) * (v(i) - cond2) ;
	}

	return sum;
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

void RotParSampleFront(Array< double > &pf, unsigned dimension, double cond , double cond2 = 2, unsigned n  = 500)
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
		sum += cond * v(i) * v(i) ;
	}

	return sum;
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
		sum += cond1 * (v(i) - cond2) * (v(i) - cond2) ;
	}

	return sum;
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

void RotCigarSampleFront(Array< double > &pf, unsigned dimension, double cond , double cond2 = 2, unsigned n  = 500)
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
// Rotated tablet
//************************************************************************

double RotTabletF1(const std::vector<double> &_v, Array<double> &coord, double cond)
{
	unsigned i;
	double sum = 0.;

	Array<double> v(_v.size());

	v(0) = scalarprod(coord.col(0), _v);
	sum += cond * v(0) * v(0);

	for (i = 1; i < v.dim(0); i++)
	{
		v(i) = scalarprod(coord.col(i), _v);
		sum += v(i) * v(i);
	}

	return sum;
}

double RotTabletF2(const std::vector<double> &_v, Array<double> &coord, double cond1, double cond2 = 2)
{
	unsigned i;
	double sum = 0.;

	Array<double> v(_v.size());

	v(0) = scalarprod(coord.col(0), _v);
	sum += cond1 * (v(0) - cond2) * (v(0) - cond2) ;

	for (i = 1; i < v.dim(0); i++)
	{
		v(i) = scalarprod(coord.col(i), _v);
		sum += (v(i) - cond2) * (v(i) - cond2) ;
	}

	return sum;
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

void RotTabletSampleFront(Array< double > &pf, unsigned dimension, double cond , double cond2 = 2, unsigned  n  = 500)
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
		raw_pf(i, 0) = RotTabletF1(xv, B, cond);
		raw_pf(i, 1) = RotTabletF2(xv, B, cond, cond2);
	}
	i = raw - 1;
	for (ii = 0; ii < xv.size(); ii++) xv[ii] = xmax;
	raw_pf(i, 0) = RotTabletF1(xv, B, cond);
	raw_pf(i, 1) = RotTabletF2(xv, B, cond, cond2);

	approxEquidistantFront(raw_pf, pf, n);
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
	return x[0];
}

double ZDT1G(const std::vector< double >& x)
{
	double g = 0;
	unsigned n = x.size();
	for (unsigned i = 1; i < n; i++)
		g += x[i];
	g = 9 * g / (n - 1.) + 1.;
	return g;
}

double ZDT1F2(const std::vector< double >& x)
{
	return  ZDT1G(x) *(1 - sqrt(x[0] /  ZDT1G(x)));
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
	return (x[0] + 3.) / 10.;// x[0];
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
	return ZDT4G(x) *(1 - sqrt(((x[0] + 3.) / 10.) / ZDT4G(x)));
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
	for (i = 0, a = 0, n = x.size(); i < n-1; i++)
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

#endif /* !__TESTFUNCTIOMOO_H */

