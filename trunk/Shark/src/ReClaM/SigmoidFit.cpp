//===========================================================================
/*!
 *  \file SigmoidFit.cpp
 *
 *  \brief Optimization of the #SigmoidModel according to Platt, 1999
 *
 *  \author  T. Glasmachers
 *  \date    2006
 *
 *  \par Copyright (c) 1999-2006:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *
 *  \par Project:
 *      ReClaM
 *
 *
 *  <BR>
 *
 *
 *  <BR><HR>
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
 */
//===========================================================================


#include <ReClaM/SigmoidFit.h>
#include <ReClaM/SigmoidModel.h>


SigmoidFit::SigmoidFit()
{
}

SigmoidFit::~SigmoidFit()
{
}


void SigmoidFit::init(Model& model)
{
}

double SigmoidFit::optimize(Model& model, ErrorFunction& errorfunction, const Array<double>& input, const Array<double>& target)
{
	SigmoidModel* pSig = dynamic_cast<SigmoidModel*>(&model);
	if (pSig == NULL) throw SHARKEXCEPTION("[SigmoidFit::init] invalid model");

	SIZE_CHECK(input.ndim() == 2);
	SIZE_CHECK(target.ndim() == 2);

	double a, b, c, d, e, d1, d2;
	double t = 0.0;
	double oldA, oldB, diff, scale, det;
	double err = 0.0;
	double value, p;
	double lambda = 0.001;
	double olderr = 1e100;
	int pos = 0;
	int neg = 0;
	int i, ic = input.dim(0);

	SIZE_CHECK(input.dim(1) == 1);
	SIZE_CHECK(target.dim(0) == (unsigned)ic);
	SIZE_CHECK(target.dim(1) == 1);

	for (i = 0; i < ic; i++) if (target(i, 0) > 0.0) pos++; else neg++;
	double A = 0.0;
	double B = log((neg + 1.0) / (pos + 1.0));
	double lowTarget = 1.0 / (neg + 2.0);
	double highTarget = (pos + 1.0) / (pos + 2.0);
	Array<double> pp(ic);
	pp = (pos + 1.0) / (pos + neg + 2.0);
	int count = 0;
	int it;
	for (it = 0; it < 100; it++)
	{
		a = b = c = d = e = 0.0;
		for (i = 0; i < ic; i++)
		{
			t = (target(i, 0) > 0.0) ? highTarget : lowTarget;
			d1 = pp(i) - t;
			d2 = pp(i) * (1.0 - pp(i));
			value = input(i, 0);
			a += d2 * value * value;
			b += d2;
			c += d2 * value;
			d += d1 * value;
			e += d1;
		}
		if (fabs(d) < 1e-9 && fabs(e) < 1e-9) break;
		oldA = A;
		oldB = B;
		err = 0.0;
		while (true)
		{
			det = (a + lambda) * (b + lambda) - c * c;
			if (det == 0.0)
			{
				lambda *= 10.0;
				continue;
			}
			A = oldA + ((b + lambda) * d - c * e) / det;
			B = oldB + ((a + lambda) * e - c * d) / det;
			err = 0.0;
			for (i = 0; i < ic; i++)
			{
				p = 1.0 / (1.0 + exp(A * input(i) + B));
				pp(i) = p;
				err -= t * mylog(p) + (1.0 - t) * mylog(1.0 - p);
			}
			if (err < 1.0000001 * olderr)
			{
				lambda *= 0.1;
				break;
			}
			lambda *= 10.0;
			if (lambda >= 1e6)
			{
				// Something is broken. Give up.
				break;
			}
		}
		diff = err - olderr;
		scale = 0.5 * (err + olderr + 1.0);
		if (diff > -1e-3 * scale && diff < 1e-7 * scale)
			count++;
		else
			count = 0;
		olderr = err;
		if (count == 3) break;
	}

	pSig->setParameter(0, A);
	pSig->setParameter(1, B);

	return err;
}

double SigmoidFit::mylog(double x)
{
	if (x < 1.38389652673673753e-87)
		return -200.0;
	else
		return log(x);
}

