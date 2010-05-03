//===========================================================================
/*!
 *  \file SharkDefs.cpp
 *
 *  \brief basic definitions
 *
 *  \par
 *  This file serves as a minimal abstraction layer.
 *  Inclusion of this file makes some frequently used
 *  functions, constants, and header file inclusions
 *  OS-, compiler-, and version-independent.
 *
 *
 *  \par Copyright (c) 1998-2008:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
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


#include <SharkDefs.h>


using namespace Shark;


double Shark::round(double x)
{
	return floor(x + 0.5);
}

// erf(x) = 2/sqrt(pi)*integral(exp(-t^2),t,0,x)
//        = 2/sqrt(pi)*[x - x^3/3 + x^5/5*2! - x^7/7*3! + ...]
//        = 1-erfc(x)
double Shark::erf(double x)
{
	if (fabs(x) > 2.2)
	{
		return 1.0 - Shark::erfc(x);
	}

	double sum = x;
	double term = x;
	double xsqr = x*x;
	int j = 1;
	do
	{
		term *= xsqr / j;
		sum -= term / (2*j+1);
		j++;
		term *= xsqr / j;
		sum += term / (2*j+1);
		j++;
	}
	while (fabs(term / sum) > 1e-15);
	return (2.0 / SqrtPI) * sum;
}

// erfc(x) = 2/sqrt(pi)*integral(exp(-t^2),t,x,inf)
//         = exp(-x^2)/sqrt(pi) * [1/x+ (1/2)/x+ (2/2)/x+ (3/2)/x+ (4/2)/x+ ...]
//         = 1-erf(x)
// expression inside [] is a continued fraction so '+' means add to denominator only
double Shark::erfc(double x)
{
	if (fabs(x) < 2.2)
	{
		return 1.0 - Shark::erf(x);
	}
	if (x < 0.0)
	{
		return 2.0 - Shark::erfc(-x);
	}
	double a = 1.0;
	double b = x;
	double c = x;
	double d = x*x + 0.5;
	double q1;
	double q2 = b / d;
	double n = 1.0;
	double t;
	do
	{
		t = a*n + b*x;
		a = b;
		b = t;
		t = c*n + d*x;
		c = d;
		d = t;
		n += 0.5;
		q1 = q2;
		q2 = b / d;
	}
	while (fabs(q1-q2)/q2 > 1e-12);
	return exp(-x*x) * q2 / SqrtPI;
}


double Shark::gamma(double x)
{
    int i,k,m;
    double ga,gr,r=1.,z;

    static double g[] = {
        1.0,
        0.5772156649015329,
       -0.6558780715202538,
       -0.420026350340952e-1,
        0.1665386113822915,
       -0.421977345555443e-1,
       -0.9621971527877e-2,
        0.7218943246663e-2,
       -0.11651675918591e-2,
       -0.2152416741149e-3,
        0.1280502823882e-3,
       -0.201348547807e-4,
       -0.12504934821e-5,
        0.1133027232e-5,
       -0.2056338417e-6,
        0.6116095e-8,
        0.50020075e-8,
       -0.11812746e-8,
        0.1043427e-9,
        0.77823e-11,
       -0.36968e-11,
        0.51e-12,
       -0.206e-13,
       -0.54e-14,
        0.14e-14};

    if (x > 171.0) return 1e308;    // This value is an overflow flag.
    if (x == (int)x) {
        if (x > 0.0) {
            ga = 1.0;               // use factorial
            for (i=2;i<x;i++) {
               ga *= i;
            }
         }
         else
            ga = 1e308;
     }
     else {
        if (fabs(x) > 1.0) {
            z = fabs(x);
            m = (int)z;
            r = 1.0;
            for (k=1;k<=m;k++) {
                r *= (z-k);
            }
            z -= m;
        }
        else
            z = x;
        gr = g[24];
        for (k=23;k>=0;k--) {
            gr = gr*z+g[k];
        }
        ga = 1.0/(gr*z);
        if (fabs(x) > 1.0) {
            ga *= r;
            if (x < 0.0) {
                ga = -M_PI/(x*ga*sin(M_PI*x));
            }
        }
    }
    return ga;
}


////////////////////////////////////////////////////////////


SharkException::SharkException(const char* file, int line, const char* message)
{
	sprintf(msg, "exception in file %s in line %d error message: %s", file, line, message);
}
