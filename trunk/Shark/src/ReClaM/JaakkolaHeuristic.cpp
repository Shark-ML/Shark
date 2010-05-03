//===========================================================================
/*!
*  \file JaakkolaHeuristic.cpp
*
*  \brief Jaakkola's heuristic and related quantities for Gaussian kernel selection
*
*  \author  T. Glasmachers
*  \date    2007
*
*
*  \par Copyright (c) 1999-2008:
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


#include <ReClaM/JaakkolaHeuristic.h>
#include <algorithm>


JaakkolaHeuristic::JaakkolaHeuristic(const Array<double>& input, const Array<double>& target)
{
	SIZE_CHECK(input.ndim() == 2);
	SIZE_CHECK(target.ndim() == 2);
	SIZE_CHECK(target.dim(1) == 1);

	int i, j, ell = input.dim(0);
	int d, dim = input.dim(1);

	for (i = 0; i < ell; i++)
	{
		double ti = target(i, 0);
		for (j = i + 1; j < ell; j++)
		{
			if (target(j, 0) == ti) continue;
			double dist2 = 0.0;
			for (d = 0; d < dim; d++)
			{
				double a = input(i, d) - input(j, d);
				dist2 += a * a;
			}
			stat.push_back(dist2);
		}
	}

	std::sort(stat.begin(), stat.end());
}

JaakkolaHeuristic::~JaakkolaHeuristic()
{
}


double JaakkolaHeuristic::sigma(double quantile)
{
	int ic = stat.size();
	if (ic == 0) return 1.0;

	if (quantile < 0.0) return sqrt(stat[0]);
	if (quantile >= 1.0) return sqrt(stat[ic-1]);

	double t = quantile * (ic - 1);
	int i = (int)floor(t);
	double rest = t - i;
	return ((1.0 - rest) * sqrt(stat[i]) + rest * sqrt(stat[i+1]));
}

double JaakkolaHeuristic::gamma(double quantile)
{
	double s = sigma(quantile);
	return 0.5 / (s * s);
}
