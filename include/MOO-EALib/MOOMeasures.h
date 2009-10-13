/*!
*  \file MOOMeasures.h
*
*  \brief Distance and diversity measurement for pareto sets
*
*  \author Stefan Roth
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


#ifndef _MOOMEASURES_H_
#define _MOOMEASURES_H_

#include <SharkDefs.h>
#include <Array/ArraySort.h>
#include <vector>
#include <algorithm>


//
// Delta Diversity Metric (Deb et al. 2002, IEEE TEC 6(2) )
//


bool cmpDelta(IndividualMOO *a, IndividualMOO *b)
{
	if (a->getMOOFitness(0) < b->getMOOFitness(0)) return true;
	return false;
}

double delta(ArchiveMOO &q, Array< double > &pf)
{
	if ((q.readArchive(0)).getNoOfObj() != 2)
		throw SHARKEXCEPTION("delta(...) implemented for two objectives only ...");

	double Delta = 0.;
	double dBar = 0.;
	unsigned i ;

	if (q.size() < 2) return 1.;

	std::vector<IndividualMOO *> v;
	std::vector<double> ds;
	unsigned n = q.size();

	for (i = 0; i < n; i++) v.push_back(&q.readArchive(i));
	std::sort(v.begin(), v.end(), cmpDelta);
	for (i = 0; i < n - 1; i++)
	{
		double f1 = v[i]->getMOOFitness(0);
		double f2 = v[i]->getMOOFitness(1);
		double g1 = v[i+1]->getMOOFitness(0);
		double g2 = v[i+1]->getMOOFitness(1);
		double d = sqrt(Shark::sqr(f1 - g1) + Shark::sqr(f2 - g2));
		ds.push_back(d);
		dBar += d;
	}

	dBar /= (n - 1.);

	double numerator = 0;
	double ex = 0.;
	for (i = 0; i < n - 1; i++) numerator += fabs(ds[i] - dBar);

	unsigned last = pf.dim(0) - 1;

	double e1 = sqrt(Shark::sqr(pf(0, 0) - v[0]->getMOOFitness(0)) +
					 Shark::sqr(pf(0, 1) - v[0]->getMOOFitness(1)));
	double e2 = sqrt(Shark::sqr(pf(0, 0) - v[n-1]->getMOOFitness(0)) +
					 Shark::sqr(pf(0, 1) - v[n-1]->getMOOFitness(1)));
	double e3 = sqrt(Shark::sqr(pf(last, 0) - v[0]->getMOOFitness(0)) +
					 Shark::sqr(pf(last, 1) - v[0]->getMOOFitness(1)));
	double e4 = sqrt(Shark::sqr(pf(last, 0) - v[n-1]->getMOOFitness(0)) +
					 Shark::sqr(pf(last, 1) - v[n-1]->getMOOFitness(1)));

	if ((e1 + e4) < (e2 + e3)) ex = (e1 + e4);
	else ex = (e2 + e3);

	Delta = (numerator + ex) / (ex + (n - 1) * dBar);

	return Delta;
}

//
// Upsilon Convergence Metric (Deb et al. 2002, IEEE TEC 6(2) )
//


double upsilonDistanceElement(IndividualMOO &i, Array< double > &pf)
{
	double x1 = i.getMOOFitness(0);
	double x2 = i.getMOOFitness(1);
	unsigned j;
	double d, e;
	d = sqrt(Shark::sqr(x1 - pf(0, 0)) + Shark::sqr(x2 - pf(0, 1)));

	for (j = 1; j < pf.dim(0); j++)
	{
		e = sqrt(Shark::sqr(x1 - pf(j, 0)) + Shark::sqr(x2 - pf(j, 1)));
		if (e < d) d = e;
	}
	return d;
}

double upsilonDistanceFront(ArchiveMOO &q, Array< double > &pf, double &stdv)
{

	if ((q.readArchive(0)).getNoOfObj() != 2)
		throw SHARKEXCEPTION("upsilonDistanceFront(...) implemented for two objectives only ...");

	unsigned nq = q.size();
	double square = 0.;
	double mean = 0;
	for (unsigned i = 0; i < nq; i++)
	{
		double d = upsilonDistanceElement(q.readArchive(i), pf);
		mean += d;
		square += d * d;
	}
	mean = mean / (double)(nq);
	if (nq > 1)
		stdv = (square - (double) nq * mean * mean) / ((double) nq - 1.);
	else
		stdv = 0;
	stdv = sqrt(stdv);
	return mean;
}



//
// Bosman's Convergence/Diversity Metric (Bosman et al. 2003, IEEE TEC 7(2) )
//

double bosmanDistanceElement(Array< double > &fitness, Array< double > &pf)
{
	unsigned j;
	double d, e;

	d = sqrt(Shark::sqr(fitness(0, 0) - pf(0)) + Shark::sqr(fitness(0, 1) - pf(1)));

	for (j = 1; j < fitness.dim(0); j++)
	{
		e = sqrt(Shark::sqr(fitness(j, 0) - pf(0)) + Shark::sqr(fitness(j, 1) - pf(1)));
		if (e < d) d = e;
	}
	return d;
}


double bosmanDistanceFront(ArchiveMOO &q, Array< double > &pf, double &stdv)
{

	if ((q.readArchive(0)).getNoOfObj() != 2)
		throw SHARKEXCEPTION("bosmanDistanceFront(...) implemented for two objectives only ...");

	unsigned i;
	unsigned samplesize = pf.dim(0);

	Array<double> fitness(q.size(), 2);
	Array<double> buffer;

	for (i = 0; i < fitness.dim(0); i++)
	{
		fitness(i, 0) = q.readArchive(i).getMOOFitness(0);
		fitness(i, 1) = q.readArchive(i).getMOOFitness(1);
	}

	double square = 0.;
	double mean = 0;
	for (i = 0; i < samplesize; i++)
	{
		buffer = pf.row(i);
		double d = bosmanDistanceElement(fitness, buffer);
		mean += d;
		square += d * d;
	}
	mean = mean / (double)(samplesize);
	if (samplesize > 1)
		stdv = (square - (double)(samplesize) * mean * mean) / ((double) samplesize - 1.);
	else
		stdv = 0;
	stdv = sqrt(stdv);
	return mean;
}


//
// Zitzler's Hypervolume Metric (Zitzler 1999 )
//

void minmax(Array< double > &pf, double &min1, double &max1, double &min2, double &max2, bool init = false)
{
	unsigned j;
	double MAX;

	MAX = MAXDOUBLE;

	if (init)
		min1 = MAX , min2 = MAX, max1 = -MAX, max2 = -MAX;

	for (j = 0; j < pf.dim(0); j++)
	{
		if (pf(j, 0) < min1) min1 = pf(j, 0);
		if (pf(j, 0) > max1) max1 = pf(j, 0);
		if (pf(j, 1) < min2) min2 = pf(j, 1);
		if (pf(j, 1) > max2) max2 = pf(j, 1);
	}
}

void minmax(ArchiveMOO &q, double &min1, double &max1, double &min2, double &max2, bool init = false)
{
	if (q.size() < 2) return;

	Array<double>	af(q.size(), 2);

	for (unsigned i = 0;i < af.dim(0);i++)
	{
		af(i, 0) = q.readArchive(i).getMOOFitness(0);
		af(i, 1) = q.readArchive(i).getMOOFitness(1);
	}

	minmax(af, min1, max1, min2, max2, init);
}


double hyperVolumeElement(double x1, double x2, double y, double bound)
{
	return (x2 - x1) *(bound - y);
}

double hyperVolumeValue(Array<double> &pf, double min1, double max1, double min2, double max2, bool minimizeall)
{

	if (pf.ndim() != 2)
		throw SHARKEXCEPTION("hyperVolumeValue(...) implemented for two objectives only ...");

	if (!minimizeall)
	{
		throw SHARKEXCEPTION("Error at runtime:: hyperVolumeValue(...) implemented for minimization of all objectives only ...");
		abort();
	}
	unsigned i;
	unsigned np = pf.dim(0);

	if (np < 2)
		return 0;

	Array<double>   p = pf.col(0);
	Array<unsigned> pi(np);

	sort(p, pi);

	double sum = 0.;

	for (i = 1; i < np; i++)
		sum += hyperVolumeElement(pf(pi(i - 1), 0), pf(pi(i), 0), pf(pi(i - 1), 1), max2);

	sum += hyperVolumeElement(pf(pi(np - 1), 0), max1, pf(pi(np - 1), 1), max2);

	sum = sum / ((max1 - min1) * (max2 - min2));

	return sum;
}

double hyperVolumeValue(ArchiveMOO &q, double min1, double max1, double min2, double max2, bool minimizeall)
{

	if ((q.readArchive(0)).getNoOfObj() != 2)
		throw SHARKEXCEPTION("hyperVolumeValue(...) implemented for two objectives only ...");

	if (q.size() < 2) return 0;

	Array<double>	af(q.size(), 2);

	for (unsigned i = 0;i < af.dim(0);i++)
	{
		af(i, 0) = q.readArchive(i).getMOOFitness(0);
		af(i, 1) = q.readArchive(i).getMOOFitness(1);
	}

	return hyperVolumeValue(af, min1, max1, min2, max2, minimizeall);
}

//
// Aggregation Metric
//

double aggregationElement(Array< double > &pf, double tradeoff)
{
	unsigned j;
	double d, e;

	d = (1 - tradeoff) * pf(0, 0) + tradeoff * pf(0, 1);

	for (j = 1; j < pf.dim(0); j++)
	{
		e = (1 - tradeoff) * pf(j, 0) + tradeoff * pf(j, 1);
		if (e < d) d = e;
	}

	return d;
}

double aggregationValue(Array<double> &pf, unsigned resolution)
{

	if (pf.ndim() != 2)
		throw SHARKEXCEPTION("Warning: aggregationValue(...) implemented for two objectives only ...");

	unsigned i;

	double sum = 0.;

	for (i = 0; i < resolution; i++)
		sum += aggregationElement(pf, ((double) i / ((double) resolution - 1)));

	sum = sum / ((double) resolution - 1);
	return sum;
}

double aggregationValue(ArchiveMOO &q, unsigned resolution)
{

	if ((q.readArchive(0)).getNoOfObj() != 2)
		throw SHARKEXCEPTION("Warning: aggregationValue(...) implemented for two objectives only ...");

	Array<double>	af(q.size(), 2);

	for (unsigned i = 0;i < af.dim(0);i++)
	{
		af(i, 0) = q.readArchive(i).getMOOFitness(0);
		af(i, 1) = q.readArchive(i).getMOOFitness(1);
	}

	return aggregationValue(af, resolution);
}

#endif

