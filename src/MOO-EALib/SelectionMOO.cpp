/*!
*
*  \file SelectionMOO.cpp
*
*  \brief Several classes and interfaces for (indicator-based) multi-objective selection.
*
*  \author Thomas Voss
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
*  This file is part of MOO-EALib. This library is free software;
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


#define _SCL_SECURE_NO_WARNINGS

#include <SharkDefs.h>
#include <algorithm>
#include <list>
#include <vector>

#include <MOO-EALib/Hypervolume.h>
#include <MOO-EALib/IndividualMOO.h>
#include <MOO-EALib/PopulationMOO.h>
#include <MOO-EALib/SelectionMOO.h>

bool ObjectiveSort::operator()(Individual * a, Individual * b)
{
	IndividualMOO * x = (IndividualMOO*) a;
	IndividualMOO * y = (IndividualMOO*) b;

	return(x->getMOOFitnessValues(m_bUnpenalizedFitness)[m_objective] < y->getMOOFitnessValues(m_bUnpenalizedFitness)[m_objective]);
}

bool ObjectiveSort::operator()(unsigned a, unsigned b)
{
	if (pop == 0)
		return(false);

	return((*this)(&(*pop)[a], &(*pop)[b]));
}

void FastNonDominatedSort::operator()(PopulationMOO & pop)
{
	std::vector<unsigned> r(pop.size());
	std::vector<unsigned> n(pop.size()); std::fill(n.begin(), n.end(), 0);
	std::vector<std::list<unsigned> > s(pop.size());

	std::list<unsigned> f; std::list<unsigned>::iterator it, itE, its, itsE;

	unsigned i, j;
	for (i = 0; i < pop.size(); i++)
	{
		for (j = 0; j < pop.size(); j++)
		{
			if (i == j)
				continue;

			if (pop.Dominate(i, j, m_bUnpenalizedFitness) > 1)
				s[i].push_back(j);
			else if (pop.Dominate(i, j) < -1)
				n[i]++;
		}

		if (n[i] == 0)
			f.push_back(i);
	}

	it = f.begin();
	itE = f.end();

	while (it != itE)
	{
		n[*it] = 1;
		++it;
	}

	unsigned frontCounter = 2; std::list<unsigned> h;

	while (!f.empty())
	{
		it = f.begin();
		itE = f.end();

		h.clear();

		while (it != itE)
		{
			its = s[*it].begin();
			itsE = s[*it].end();

			while (its != itsE)
			{
				n[*its]--;

				if (n[*its] == 0)
					h.push_back(*its);

				++its;
			}

			++it;
		}

		its = h.begin();
		itsE = h.end();
		while (its != itsE)
		{
			n[*its] = frontCounter;
			++its;
		}

		f = h;
		frontCounter++;
	}

	for (i = 0; i < pop.size(); i++)
	{
		pop[i].setMOORank(n[i]);
	}
}

void CrowdingDistance::operator()(PopulationMOO & pop)
{
	unsigned m_maxMOORank = pop.maxMOORank();

	unsigned i, j; unsigned k;

	const unsigned objCount = pop[0].getNoOfObj();

	for (i = 1; i <= m_maxMOORank; i++)
	{
		std::vector<unsigned> p(0);

		for (j = 0; j < pop.size(); j++)
		{
			if (pop[j].getMOORank() == i)
			{
				p.push_back(j);
			}
		}

		std::vector<double> distance(pop.size(), 0);

		ObjectiveSort objSort(m_bUnpenalizedFitness, 0);
		objSort.pop = &pop;

		double factor;

		for (j = 0; j < objCount; j++)
		{
			objSort.m_objective = j;
			std::sort(p.begin(), p.end(), objSort);

			distance[p.front()] = distance[p.back()] = std::numeric_limits<double>::max();
			factor = pop[p.back()].getMOOFitnessValues(m_bUnpenalizedFitness)[j] - pop[p.front()].getMOOFitnessValues(m_bUnpenalizedFitness)[j];
			factor = Shark::max(1.0, static_cast<double>(factor));	// required for Macintosh compiler
			for (k = 1; k < p.size() - 1; k++)
			{
				std::vector<double> & a = pop[p[k+1]].getMOOFitnessValues(m_bUnpenalizedFitness);
				std::vector<double> & b = pop[p[k-1]].getMOOFitnessValues(m_bUnpenalizedFitness);

				distance[p[k]] += (a[j] - b[j]) / factor;
			}
		}

		for (j = 0; j < p.size(); j++)
		{
			pop[p[j]].setMOOShare(distance[p[j]]);
		}
	}
}

/*---- Binary Quality Indicator ----*/

double IBinaryQualityIndicator::operator()(const std::vector<std::vector<double> > & set, unsigned idx)
{
	std::vector<std::vector<double> > setB(set); setB.erase(setB.begin() + idx);
	return((*this)(setB, set));
}

/*---- Indicator Based Selection Strategy ----*/
//!
//! \brief Predicate for std::sort
//!
struct LastObjectiveSort
{
	LastObjectiveSort(bool UnpenalizedFitness) : m_bUnpenalizedFitness(UnpenalizedFitness)
	{}

	bool operator()(const Individual * a, const Individual * b)
	{
		IndividualMOO * x = (IndividualMOO*) a;
		IndividualMOO * y = (IndividualMOO*) b;

		return(x->getMOOFitnessValues(m_bUnpenalizedFitness).back() <
			   y->getMOOFitnessValues(m_bUnpenalizedFitness).back()
			  );
	}

	bool m_bUnpenalizedFitness;
};

MinMax calc_boundary_elements(PopulationMOO & pop,
							  const std::vector<unsigned> & popView,
							  unsigned noObjectives,
							  bool unpenalizedFitness
							 )
{
	std::vector<double> regLow(noObjectives, std::numeric_limits<double>::max());
	std::vector<double> regUp(noObjectives, std::numeric_limits<double>::min());

	std::vector<unsigned> min(noObjectives), max(noObjectives);

	unsigned j, k;

	for (j = 0; j < popView.size(); j++)
	{
		for (k = 0; k < noObjectives; k++)
		{
			if (pop[popView[j]].getMOOFitnessValues(unpenalizedFitness)[k] < regLow[k])
			{
				regLow[k] = pop[popView[j]].getMOOFitnessValues(unpenalizedFitness)[k];
				min[k] = j;
			}

			if (pop[popView[j]].getMOOFitnessValues(unpenalizedFitness)[k] > regUp[k])
			{
				regUp[k] = pop[popView[j]].getMOOFitnessValues(unpenalizedFitness)[k];
				max[k] = j;
			}
		}
	}

	return(MinMax(min, max));
}

// template void IndicatorBasedSelectionStrategy<AdditiveEpsilonIndicator>::operator()(PopulationMOO & pop);
// template void IndicatorBasedSelectionStrategy<HypervolumeIndicator>::operator()(PopulationMOO & pop);

/*---- Additive Epsilon Indicator ----*/
double AdditiveEpsilonIndicator::operator()(const std::vector<std::vector<double> > & a,
		const std::vector<std::vector<double> > & b)
{
	unsigned i, j, k;

	double epsilon = 0;
	double tmp1, tmp2;

	for (i = 0; i < a.size(); i++)
	{
		tmp1 = std::numeric_limits<double>::max();

		for (j = 0; j < b.size(); j++)
		{
			tmp2 = 0;

			for (k = 0; k < a[i].size(); k++)
			{
				tmp2 = Shark::max(tmp2, b[j][k] - a[i][k]);
			}

			tmp1 = Shark::min(tmp1, tmp2);
		}

		epsilon = Shark::max(epsilon, tmp1);
	}

	return(epsilon);
}

double AdditiveEpsilonIndicator::operator()(const std::vector<std::vector<double> > & a, unsigned idx)
{
	unsigned i, j;

	double epsilon = std::numeric_limits<double>::max();
	double tmp;

	for (i = 0; i < a.size(); i++)
	{
		if (i == idx)
			continue;

		tmp = 0;

		for (j = 0; j < a[i].size(); j++)
		{
			tmp = Shark::max(tmp,
							 a[i][j] - a[idx][j]
							);
		}

		epsilon = Shark::min(epsilon, tmp);
	}

	return(epsilon);
}

/*---- Hypervolume Indicator ----*/

double HypervolumeIndicator::operator()(const std::vector<std::vector<double> > & a,
										const std::vector<std::vector<double> > & b)
{
	unsigned i;

	//Population B
	double * pop = new double[Shark::max( a.size(), b.size() ) * m_noObjectives];

	double * p = pop;

	double * regLow = new double[m_noObjectives]; //(double*) malloc(m_noObjectives * sizeof(double));
	double * regUp = new double[m_noObjectives];//(double*) malloc(m_noObjectives * sizeof(double));
	
	std::fill( regUp, regUp + m_noObjectives, -std::numeric_limits<double>::max() );
	std::fill( regLow, regLow + m_noObjectives, std::numeric_limits<double>::max() );
	
	for (i = 0; i < b.size(); i++) {
		std::copy(b[i].begin(), b[i].end(), p);

		for( unsigned int j = 0; j < m_noObjectives; j++ ) {
			regUp[j] 	= Shark::max( regUp[j], *p );
			regLow[j] 	= Shark::min( regLow[j], *p );
			++p;
		}
	}
	
	for (i = 0; i < m_noObjectives; i++)
		m_bAscending ? regUp[i] += 1.0 : regLow[i] -= 1.0;

	// m_noSqrtPoints = (unsigned) sqrt((double)b.size());
	double volB = 0;
	/*if( m_noObjectives == 3 ) {
		volB = fonseca( pop, regUp, m_noObjectives, b.size() );
	} else if( m_noObjectives > 3 ) {
		volB = overmars_yap( pop, regUp, m_noObjectives, b.size() );
	}*/
	if( m_noObjectives >= 3 ) {
		volB = overmars_yap( pop, regUp, m_noObjectives, b.size() );
	}
	
	std::fill( regLow, regLow + m_noObjectives, std::numeric_limits<double>::max() );
	
	//double * popA = new double[a.size() * m_noObjectives];//(doublep*) malloc(a.size() * sizeof(doublep));
	p = pop;
	
	for (i = 0; i < a.size(); i++) {
		std::copy(a[i].begin(), a[i].end(), p);
		
		for( unsigned int j = 0; j < m_noObjectives; j++ ) {
			regLow[j] = Shark::min( regLow[j], *p );
			++p;
		}
	}
	
	double volA = 0;
	
	/*if( m_noObjectives == 3 ) {
		volA = fonseca( pop, regUp, m_noObjectives, a.size() );
	} else if( m_noObjectives > 3 ) {							
		volA = overmars_yap( pop, regUp, m_noObjectives, a.size() );
	}*/
	if( m_noObjectives >= 3 ) {							
		volA = overmars_yap( pop, regUp, m_noObjectives, a.size() );
	}

	delete [] pop;
	delete [] regLow;
	delete [] regUp;

	//printf( "(II) volA: %f | volB: %f \n", volA, volB );

	return(volB - volA);
}

double HypervolumeIndicator::operator()(const std::vector<std::vector<double> > & set, unsigned idx)
{
	std::vector<std::vector<double> > setB(set); setB.erase(setB.begin() + idx);
	return((*this)(setB, set));
}

