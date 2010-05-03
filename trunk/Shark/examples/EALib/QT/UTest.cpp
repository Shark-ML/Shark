//===========================================================================
/*!
 *  \file UTest.cpp
 *
 *  \brief   Mann-Whitney U-Test
 *
 *  \author  T. Glasmachers
 *  \date    2008
 *
 *  \par Copyright (c) 2008:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-27974<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
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
#include <algorithm>

#include "UTest.h"


struct tEntry
{
	int pos;
	double value;
	double rank;
};

bool cmpEntries(const tEntry& e1, const tEntry& e2)
{
	return (e1.value < e2.value);
}

void ComputeRanks(const std::vector<double>& sampleA, const std::vector<double>& sampleB, std::vector<double>& ranksA, std::vector<double>& ranksB)
{
	unsigned int ac = sampleA.size();
	unsigned int bc = sampleB.size();
	unsigned int n, nc = ac + bc;
	unsigned int f, first = 0;

	// join the lists and sort
	std::vector<tEntry> data(nc);
	for (n=0; n<ac; n++) data[n].value = sampleA[n];
	for (n=0; n<bc; n++) data[ac + n].value = sampleB[n];
	for (n=0; n<nc; n++) data[n].pos = n;
	std::sort(data.begin(), data.end(), &cmpEntries);

	// assign ranks, average for ties
	for (n=1; n<nc; n++)
	{
		if (data[n-1].value < data[n].value)
		{
			if (first < n-1)
			{
				double rank = 0.5 * (first+1 + n);
				for (f=first; f<n; f++) data[f].rank = rank;
			}
			else data[n-1].rank = (double)n;
			first = n;
		}
	}
	if (first < nc-1)
	{
		double rank = 0.5 * (first+1 + nc);
		for (f=first; f<nc; f++) data[f].rank = rank;
	}
	else data[nc-1].rank = (double)nc;

	// return the ranks
	ranksA.resize(ac);
	ranksB.resize(bc);
	for (n=0; n<nc; n++)
	{
		unsigned int pos = data[n].pos;
		if (pos < ac) ranksA[pos] = data[n].rank;
		else ranksB[pos - ac] = data[n].rank;
	}
}

void ComputeU(const std::vector<double>& ranksA, const std::vector<double>& ranksB, double& Ua, double& Ub)
{
	unsigned int ac = ranksA.size();
	unsigned int bc = ranksB.size();
	double a_sum = 0.0;
	double b_sum = 0.0;
	unsigned int n;
	for (n=0; n<ac; n++) a_sum += ranksA[n];
	for (n=0; n<bc; n++) b_sum += ranksB[n];
	Ua = a_sum - ac * (ac + 1.0) / 2.0;
	Ub = b_sum - bc * (bc + 1.0) / 2.0;
	if (fabs(Ua + Ub - ac * bc) > 1e-6) throw SHARKEXCEPTION("[ComputeU] computation of statistic U is inconsistent");
}

// For too large (intermediate)
// results this function returns -1.
int BinCoeff(unsigned int upper, unsigned int lower)
{
	if (2*lower > upper) lower = upper - lower;

	SharkInt64 ret = 1;
	int i;
	for (i=0; i<(int)lower; i++)
	{
		SharkInt64 cand = ret * (upper - i);
		if (cand / (SharkInt64)(upper - i) != ret) return -1;
		ret = cand;
	}
	for (i=2; i<=(int)lower; i++) ret /= i;

	int cand = (int)ret;
	if ((SharkInt64)cand != ret) return -1;

	return cand;
}

// Return the number of cases for which U
// is smaller or equal to the given value.
//     N = total number of samples
//     n = number of samples of class 1
//     U = U-value for class 1
int CountSmallerOrEqualCases(int N, int n, double U)
{
	int i;
	int Q = n * (n-1) / -2;
	std::vector<int> pos(n+1);
	for (i=0; i<n; i++) pos[i] = i;
	pos[n] = N;

	int ret = 0;

	// loop through all configurations
	int active = 0;
	while (true)
	{
		// compute U for pos
		int posU = Q;
		for (i=0; i<n; i++) posU += pos[i];

		if (posU <= U)
		{
			// update number of "smaller" cases
			ret++;

			// proceed to the next configuration
			for (i=n-1; i>=0; i--)
			{
				if (pos[i] < pos[i+1] - 1)
				{
					active = i;
					pos[active]++;
					for (i=active+1; i<n; i++)
					{
						pos[i] = pos[active] + i - active;
					}
					break;
				}
			}
			if (i < 0) break;
		}
		else
		{
			// skip higher configurations
			active--;
			for (; active>=0; active--)
			{
				if (pos[active] < N - n + active) break;
			}
			if (active < 0) break;
			pos[active]++;
			for (i=active+1; i<n; i++)
			{
				pos[i] = pos[active] + i - active;
			}
		}
	}

	return ret;
}

void UTest(const std::vector<double>& sampleA, const std::vector<double>& sampleB, double& p_twosided, double& p_A_leftOf_B, double& p_B_leftOf_A)
{
	unsigned int ac = sampleA.size();
	unsigned int bc = sampleB.size();
	unsigned int nc = ac + bc;

	// compute ranks
	std::vector<double> ranksA;
	std::vector<double> ranksB;
	ComputeRanks(sampleA, sampleB, ranksA, ranksB);

	// compute U statistic
	double Ua;
	double Ub;
	ComputeU(ranksA, ranksB, Ua, Ub);

	// decide whether to use exact or approximate method
	double pa;		// probability to find Ua at least as large as it is
	double pb;		// probability to find Ub at least as large as it is
	int cases = BinCoeff(nc, ac);
	if (cases < 0 || cases > 1000000)
	{
		// normal approximation
		double mu = (0.5 * ac) * bc;
		double sigma = sqrt(((nc + 1.0) * ac) * bc / 12.0);
		double Za = (Ua - mu) / sigma;
		double Zb = (Ub - mu) / sigma;
		pa = 0.5 * Shark::erfc(-Za / M_SQRT2);
		pb = 0.5 * Shark::erfc(-Zb / M_SQRT2);
	}
	else
	{
		// exact method
		int s_a = CountSmallerOrEqualCases(nc, ac, Ua);
		int s_b = CountSmallerOrEqualCases(nc, bc, Ub);
		pa = (double)s_a / (double)cases;
		pb = (double)s_b / (double)cases;
	}

	p_B_leftOf_A = pb;
	p_A_leftOf_B = pa;
	p_twosided = Shark::min(2.0 * pa, 2.0 * pb);
}
