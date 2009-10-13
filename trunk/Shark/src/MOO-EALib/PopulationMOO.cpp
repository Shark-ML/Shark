/*!
*  \file PopulationMOO.cpp
*
*  \brief Population of individuals with vector-valued fitness
*
*  \author Tatsuya Okabe <tatsuya.okabe@honda-ri.de>
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


#include <SharkDefs.h>
#include <EALib/Population.h>
#include <MOO-EALib/PopulationMOO.h>
#include <MOO-EALib/SelectionMOO.h>
#include <MOO-EALib/SelectionMOO-Impl.h>
#include <algorithm>
#include <MOO-EALib/RMMEDA.h>

using namespace std;

// ----------------------------------------------------------------------------
// TO-PM-001
PopulationMOO::PopulationMOO()
		: Population()
{}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-002
PopulationMOO::PopulationMOO(unsigned n)
		: Population(n)
{
	subPop    = false;
	for (unsigned i = n; i--;)
	{
		delete *(begin() + i);
		*(begin() + i) = new IndividualMOO;
	}
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-003
PopulationMOO::PopulationMOO(const IndividualMOO& indmoo)
		: Population()
{
	subPop    = false;
	IndividualMOO *dummy = new IndividualMOO(indmoo);
	//std::vector< Individual * >::push_back( dummy );
	vector< Individual * >::push_back(dummy);
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-005
PopulationMOO::PopulationMOO(unsigned n, const IndividualMOO& indmoo)
		: Population()
{
	subPop    = false;
	for (unsigned i = n; i--;)
	{
		IndividualMOO *dummy = new IndividualMOO(indmoo);
		vector< Individual * >::push_back(dummy);
	}
}
// ----------------------------------------------------------------------------


// ----------------------------------------------------------------------------
// TO-PM-007
PopulationMOO::PopulationMOO(unsigned n, const Chromosome& chrom0)
		: Population(n, chrom0)
{
	subPop    = false;
	for (unsigned i = n; i--;)
	{
		delete *(begin() + i);
		*(begin() + i) = new IndividualMOO(chrom0);
	}
}
// ----------------------------------------------------------------------------


// ----------------------------------------------------------------------------
// TO-PM-008
PopulationMOO::PopulationMOO(unsigned n, const Chromosome& chrom0,
							 const Chromosome& chrom1)
		: Population(n, chrom0, chrom1)
{
	subPop    = false;
	for (unsigned i = n; i--;)
	{
		delete *(begin() + i);
		*(begin() + i) = new IndividualMOO(chrom0, chrom1);
	}
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-009
PopulationMOO::PopulationMOO(unsigned n, const Chromosome& chrom0,
							 const Chromosome& chrom1,
							 const Chromosome& chrom2)
		: Population(n, chrom0, chrom1, chrom2)
{
	subPop    = false;
	for (unsigned i = n; i--;)
	{
		delete *(begin() + i);
		*(begin() + i) = new IndividualMOO(chrom0, chrom1, chrom2);
	}
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-010
PopulationMOO::PopulationMOO(unsigned n, const Chromosome& chrom0,
							 const Chromosome& chrom1,
							 const Chromosome& chrom2,
							 const Chromosome& chrom3)
		: Population(n, chrom0, chrom1, chrom2, chrom3)
{
	subPop    = false;
	for (unsigned i = n; i--;)
	{
		delete *(begin() + i);
		*(begin() + i) = new IndividualMOO(chrom0, chrom1, chrom2, chrom3);
	}
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-011
PopulationMOO::PopulationMOO(unsigned n, const Chromosome& chrom0,
							 const Chromosome& chrom1,
							 const Chromosome& chrom2,
							 const Chromosome& chrom3,
							 const Chromosome& chrom4)
		: Population(n, chrom0, chrom1, chrom2, chrom3, chrom4)
{
	subPop    = false;
	for (unsigned i = n; i--;)
	{
		delete *(begin() + i);
		*(begin() + i) = new IndividualMOO(chrom0, chrom1, chrom2, chrom3,
										   chrom4);
	}
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-012
PopulationMOO::PopulationMOO(unsigned n, const Chromosome& chrom0,
							 const Chromosome& chrom1,
							 const Chromosome& chrom2,
							 const Chromosome& chrom3,
							 const Chromosome& chrom4,
							 const Chromosome& chrom5)
		: Population(n, chrom0, chrom1, chrom2, chrom3, chrom4, chrom5)
{
	subPop    = false;
	for (unsigned i = n; i--;)
	{
		delete *(begin() + i);
		*(begin() + i) = new IndividualMOO(chrom0, chrom1, chrom2, chrom3,
										   chrom4, chrom5);
	}
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-013
PopulationMOO::PopulationMOO(unsigned n, const Chromosome& chrom0,
							 const Chromosome& chrom1,
							 const Chromosome& chrom2,
							 const Chromosome& chrom3,
							 const Chromosome& chrom4,
							 const Chromosome& chrom5,
							 const Chromosome& chrom6)
		: Population(n, chrom0, chrom1, chrom2, chrom3, chrom4, chrom5, chrom6)
{
	subPop    = false;
	for (unsigned i = n; i--;)
	{
		delete *(begin() + i);
		*(begin() + i) = new IndividualMOO(chrom0, chrom1, chrom2, chrom3,
										   chrom4, chrom5, chrom6);
	}
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-014
PopulationMOO::PopulationMOO(unsigned n, const Chromosome& chrom0,
							 const Chromosome& chrom1,
							 const Chromosome& chrom2,
							 const Chromosome& chrom3,
							 const Chromosome& chrom4,
							 const Chromosome& chrom5,
							 const Chromosome& chrom6,
							 const Chromosome& chrom7)
		: Population(n, chrom0, chrom1, chrom2, chrom3, chrom4, chrom5, chrom6,
					 chrom7)
{
	subPop    = false;
	for (unsigned i = n; i--;)
	{
		delete *(begin() + i);
		*(begin() + i) = new IndividualMOO(chrom0, chrom1, chrom2, chrom3,
										   chrom4, chrom5, chrom6, chrom7);
	}
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-015
PopulationMOO::PopulationMOO(unsigned n, const std::vector< Chromosome* >& chrom)
{
	subPop    = false;
	ascending = false;
	spinOnce  = true;
	IndividualMOO indiv(chrom);

	for (unsigned i = 0; i < n; i++)
	{
		IndividualMOO *dummy = new IndividualMOO(indiv);
		vector< Individual * >::push_back(dummy);
	}
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-016
PopulationMOO::PopulationMOO(const PopulationMOO& popmoo)
		: Population()
{
	for (unsigned i = 0; i < popmoo.size(); i++)
	{
		IndividualMOO *dummy = new IndividualMOO(popmoo[i]);
		//std::vector< Individual * >::push_back( dummy );
		vector< Individual * >::push_back(dummy);
	}
	// copy internal variables
	index         = popmoo.index;
	subPop        = popmoo.subPop;
	ascending     = popmoo.ascending;
	spinOnce      = popmoo.spinOnce;
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-017
PopulationMOO::PopulationMOO(const Population& pop)
		: Population(pop)
{
	for (unsigned i = pop.size(); i--;)
	{
		delete *(begin() + i);
		*(begin() + i) = new IndividualMOO(pop[i]);
	}
	// copy internal variables
	index     = pop.getIndex();
	subPop    = pop.getSubPop();
	ascending = pop.ascendingFitness();
	spinOnce  = pop.getSpinOnce();
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-IM-018
PopulationMOO::~PopulationMOO()
{ }

// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-030
unsigned PopulationMOO::size() const
{
	return Population::size();
}

// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-031
vector< Individual* >::iterator PopulationMOO::begin()
{
	return vector< Individual * >::begin();
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-032
vector< Individual* >::iterator PopulationMOO::end()
{
	return vector< Individual * >::end();
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-033
void PopulationMOO::resize(unsigned n)
{
	unsigned i;
	unsigned s = size();

	subPop    = false;

	// delete individualMOOs
	if (n < s)
	{
		for (i = n; i < s; i++)
		{
			delete *(begin() + i);
		}
		vector< Individual * >::erase(begin() + n, end());
	}
	// add individualMOOs
	else if (n > s)
	{
		if (s > 0)
		{
			for (i = s; i < n; i++)
			{
				IndividualMOO *dummy = new IndividualMOO((*this)[0]);
				vector< Individual * >::push_back(dummy);
			}
		}
		else
		{
			for (i = s; i < n; i++)
			{
				IndividualMOO *dummy = new IndividualMOO;
				vector< Individual * >::push_back(dummy);
			}
		}
	}
}

// ----------------------------------------------------------------------------
// TO-PM-040
IndividualMOO& PopulationMOO::operator [ ](unsigned i)
{
	RANGE_CHECK(i < size())
	return *(static_cast< IndividualMOO * >(
				 //std::vector< Individual * >::operator[ ]( i ) ));
				 vector< Individual * >::operator[ ](i)));
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-041
const IndividualMOO& PopulationMOO::operator [ ](unsigned i) const
{
	RANGE_CHECK(i < size())
	return *(static_cast< IndividualMOO * >(
				 //std::vector< Individual * >::operator[ ]( i ) ));
				 vector< Individual * >::operator[ ](i)));
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-042
PopulationMOO& PopulationMOO::operator = (const IndividualMOO & indmoo)
{
	for (unsigned i = size(); i--;)
	{
		(*this)[ i ] = indmoo;
	}
	return *this;
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-043
PopulationMOO& PopulationMOO::operator = (const Individual & ind)
{
	for (unsigned i = size(); i--;)
	{
		(*this)[ i ] = IndividualMOO(ind);
	}
	return *this;
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-044
PopulationMOO& PopulationMOO::operator = (const PopulationMOO& popmoo)
{
	if (size() != popmoo.size())
	{
		cerr << "Error in TO-PM-044" << endl;
		throw SHARKEXCEPTION("size mismatch");
	}
	for (unsigned i = size(); i--;)
	{
		(*this)[i] = popmoo[i];
	}
	// copy internal variables
	index         = popmoo.index;
	subPop        = popmoo.subPop;
	ascending     = popmoo.ascending;
	spinOnce      = popmoo.spinOnce;

	return *this;
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-045
PopulationMOO& PopulationMOO::operator = (const Population& pop)
{
	// check the size
	if (size() != pop.size())
	{
		cerr << "Error in TO-PM-044" << endl;
		throw SHARKEXCEPTION("size mismatch");
	}
	// copy individuals
	for (unsigned i = size(); i--;)
	{
		(*this)[i] = pop[i];
	}
	// copy internal variables
	index     = pop.getIndex();
	subPop    = pop.getSubPop();
	ascending = pop.ascendingFitness();
	spinOnce  = pop.getSpinOnce();

	return *this;
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-046
bool PopulationMOO::operator == (const PopulationMOO& popmoo) const
{
	if (size() == popmoo.size())
	{
		for (unsigned i = size(); i--;)
		{
			if ((*this)[i] != popmoo[i])
			{
				return false;
			}
		}

		return	index         == popmoo.index        &&
			   subPop        == popmoo.subPop       &&
			   ascending     == popmoo.ascending    &&
			   spinOnce      == popmoo.spinOnce;
	}

	return false;
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-052
void PopulationMOO::setAscending(bool strategy)
{
	if (strategy)
	{
		Population::setMinimize();
	}
	else
	{
		Population::setMaximize();
	}
}
// ----------------------------------------------------------------------------


// ----------------------------------------------------------------------------
// TO-PM-057
void PopulationMOO::setSpinOnce(bool spin)
{
	if (spin)
	{
		Population::spinWheelOneTime();
	}
	else
	{
		Population::spinWheelMultipleTimes();
	}
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-070
void PopulationMOO::replace(unsigned i, const Individual& ind)
{
	RANGE_CHECK(i < size())
	delete *(begin() + i);
	*(begin() + i) = new IndividualMOO(ind);
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-071
void PopulationMOO::replace(unsigned i,  const IndividualMOO& indmoo)
{
	RANGE_CHECK(i < size())
	delete *(begin() + i);
	*(begin() + i) = new IndividualMOO(indmoo);
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-072
void PopulationMOO::replace(unsigned i,  const Population& pop)
{
	RANGE_CHECK(i + pop.size() <= size())
	PopulationMOO popmoo(pop);
	for (unsigned j = popmoo.size(); j--;)
	{
		(*this)[ i + j ] = popmoo[ j ];
	}
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-073
void PopulationMOO::replace(unsigned i,  const PopulationMOO& popmoo)
{
	RANGE_CHECK(i + popmoo.size() <= size())
	for (unsigned j = popmoo.size(); j--;)
	{
		(*this)[ i + j ] = popmoo[ j ];
	}
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-074
void PopulationMOO::insert(unsigned i,  const Individual& ind)
{
	RANGE_CHECK(i <= size())
	vector< Individual * >::insert(begin() + i, new IndividualMOO(ind));
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-075
void PopulationMOO::insert(unsigned i,  const IndividualMOO& indmoo)
{
	RANGE_CHECK(i <= size())
	vector< Individual * >::insert(begin() + i, new IndividualMOO(indmoo));
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-076
void PopulationMOO::insert(unsigned i,  const Population& pop)
{
	RANGE_CHECK(i <= size())
	for (unsigned j = pop.size(); j--;)
	{
		vector< Individual * >::insert(begin() + i, new IndividualMOO(pop[j]));
	}
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-077
void PopulationMOO::insert(unsigned i,  const PopulationMOO& popmoo)
{
	RANGE_CHECK(i <= size())
	for (unsigned j = popmoo.size(); j--;)
	{
		vector< Individual * >::insert(begin() + i,
									   new IndividualMOO(popmoo[j]));
	}
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-078
void PopulationMOO::append(const Individual& ind)
{
	vector< Individual * >::push_back(new IndividualMOO(ind));
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-079
void PopulationMOO::append(const IndividualMOO& indmoo)
{
	vector< Individual * >::push_back(new IndividualMOO(indmoo));
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-080
void PopulationMOO::append(const Population& pop)
{
	PopulationMOO popmoo(pop);
	insert(size(), popmoo);
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-081
void PopulationMOO::append(const PopulationMOO& popmoo)
{
	insert(size(), popmoo);
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-082
void PopulationMOO::remove(unsigned i)
{
	RANGE_CHECK(i < size())
	delete *(begin() + i);
	vector< Individual * >::erase(begin() + i);
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-083
void PopulationMOO::remove(unsigned i, unsigned k)
{
	if (i < k)
	{
		RANGE_CHECK(k < size())
		for (unsigned j = i; j < k; j++)
		{
			delete *(begin() + j);
		}
		vector< Individual * >::erase(begin() + i, begin() + k);
	}
}
// ----------------------------------------------------------------------------

// static comparison function
bool PopulationMOO::compareFitnessAscending(Individual*const& pInd1, Individual*const& pInd2)
{
	return pInd1->getFitness() < pInd2->getFitness();
}

// static comparison function
bool PopulationMOO::compareFitnessDescending(Individual*const& pInd1, Individual*const& pInd2)
{
	return pInd1->getFitness() > pInd2->getFitness();
}

// ----------------------------------------------------------------------------
// TO-PM-093-b
void PopulationMOO::sortIndividuals(std::vector< IndividualMOO* >& indvec)
{
	// sort in time ~ n log(n)
	if (ascendingFitness())
	{
		std::sort(indvec.begin(), indvec.end(), compareFitnessAscending);
	}
	else
	{
		std::sort(indvec.begin(), indvec.end(), compareFitnessDescending);
	}
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-094
void PopulationMOO::sort()
{
	// sort in time ~ n log(n)
	if (ascendingFitness())
	{
		std::sort(begin(), end(), compareFitnessAscending);
	}
	else
	{
		std::sort(begin(), end(), compareFitnessDescending);
	}
}

// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-095
void PopulationMOO::shuffle()
{
	Population::shuffle();
}

// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-098

void PopulationMOO::exchange(PopulationMOO& popmoo)
{
	// check the size
	SIZE_CHECK(size() == popmoo.size())
	// swap Individuals
	for (unsigned i = size(); i--;)
	{
		std::swap((*this)[i], popmoo[i]);
	}
	// swap internal variables
	std::swap(index    , popmoo.index);
	std::swap(subPop   , popmoo.subPop);
	std::swap(ascending, popmoo.ascending);
	std::swap(spinOnce , popmoo.spinOnce);
}
// ---------------------------------------------------------------------------


// ----------------------------------------------------------------------------
// TO-PM-102
IndividualMOO& PopulationMOO::oneOfBest()
{
	RANGE_CHECK(size() > 0)
	std::vector< unsigned > bestIndices;

	// store best value
	double best = (ascending ? minFitness() : maxFitness());
	// find all indices of best individuals ( unsorted PopulationMOO )
	for (unsigned i = 0; i < size(); i++)
	{
		if ((*this)[i].fitnessValue() == best)
		{
			bestIndices.push_back(i);
		}
	}
	// return one of best individuals using random.
	return (*this)[bestIndices[ Rng::discrete(0, bestIndices.size() - 1)]];
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-103
const IndividualMOO& PopulationMOO::oneOfBest() const
{
	RANGE_CHECK(size() > 0)
	std::vector< unsigned > bestIndices;

	// store best value
	double best = (ascending ? minFitness() : maxFitness());
	// find all indices of best individuals ( unsorted PopulationMOO )
	for (unsigned i = 0; i < size(); i++)
	{
		if ((*this)[i].fitnessValue() == best)
		{
			bestIndices.push_back(i);
		}
	}
	// return one of best individuals using random.
	return (*this)[bestIndices[ Rng::discrete(0, bestIndices.size() - 1)]];
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-104
IndividualMOO& PopulationMOO::best()
{
	return (*this)[ bestIndex()];
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-105
const IndividualMOO& PopulationMOO::best() const
{
	return (*this)[ bestIndex()];
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-106
IndividualMOO& PopulationMOO::worst()
{
	return (*this)[ worstIndex()];
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-107
const IndividualMOO& PopulationMOO::worst() const
{
	return (*this)[ worstIndex()];
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-108
IndividualMOO& PopulationMOO::random()
{
	return (*this)[ Rng::discrete(0, size() - 1)];
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-109
const IndividualMOO& PopulationMOO::random() const
{
	return (*this)[ Rng::discrete(0, size() - 1)];
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-110
IndividualMOO& PopulationMOO::best(IndividualMOO& i1, IndividualMOO& i2) const
{
	if ((ascending && i1.fitnessValue() < i2.fitnessValue()) ||
			(!ascending && i1.fitnessValue() > i2.fitnessValue()))
	{
		return i1;
	}
	else
	{
		return i2;
	}
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-111
IndividualMOO& PopulationMOO::best(unsigned i, unsigned j)
{
	if ((ascending && (*this)[i].fitnessValue() < (*this)[j].fitnessValue()) ||
			(!ascending && (*this)[i].fitnessValue() > (*this)[j].fitnessValue()))
	{
		return (*this)[i];
	}
	else
	{
		return (*this)[j];
	}
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-112
IndividualMOO& PopulationMOO::worst(IndividualMOO& i1, IndividualMOO& i2) const
{
	if ((ascending && i1.fitnessValue() > i2.fitnessValue()) ||
			(!ascending && i1.fitnessValue() < i2.fitnessValue()))
	{
		return i1;
	}
	else
	{
		return i2;
	}
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-113
IndividualMOO& PopulationMOO::worst(unsigned i, unsigned j)
{
	if ((ascending && (*this)[i].fitnessValue() > (*this)[j].fitnessValue()) ||
			(!ascending && (*this)[i].fitnessValue() < (*this)[j].fitnessValue()))
	{
		return (*this)[i];
	}
	else
	{
		return (*this)[j];
	}
}
// ----------------------------------------------------------------------------


void PopulationMOO::getNonDominated(PopulationMOO& nondom, bool unpenalized)
{
	nondom.resize(0);
	nondom.setSubPop(true);

	int i, j, ic = size();
	if (size() == 0) return;

	int o, oc = (*this)[0].getNoOfObj();
	std::vector<bool> dom(ic);
	for (i=0; i<ic; i++) dom[i] = false;
	for (i=1; i<ic; i++)
	{
		std::vector<double>& fit_i = (*this)[i].getMOOFitnessValues(unpenalized);
		for (j=0; j<i; j++)
		{
			std::vector<double>& fit_j = (*this)[j].getMOOFitnessValues(unpenalized);

			// check strict dominance
			bool isdom_i = true;
			bool isdom_j = true;
			for (o=0; o<oc; o++)
			{
				if (fit_i[o] <= fit_j[o]) isdom_i = false;
				if (fit_j[o] <= fit_i[o]) isdom_j = false;
			}
			dom[i] = dom[i] || isdom_i;
			dom[j] = dom[j] || isdom_j;
		}
	}
	// int n, nc = 0;
	// for (i=0; i<ic; i++) if (! dom[i]) nc++;
	int n;

	n = 0;
	for (i=0; i<ic; i++)
	{
		if (! dom[i])
		{
			((std::vector<Individual*>&)nondom).push_back(&((*this)[i]));
			n++;
		}
	}
}


// ----------------------------------------------------------------------------
// TO-PM-200
void PopulationMOO::setNoOfObj(unsigned NOO)
{
	for (unsigned i = size(); i--;)
	{
		(*this)[i].setNoOfObj(NOO);
	}
	
	m_sMeasure.m_binaryQualityIndicator.m_noObjectives = NOO;
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-201
void PopulationMOO::setMOOFitness(double obj)
{
	for (unsigned i = size(); i--;)
	{
		for (unsigned j = (*this)[i].getNoOfObj(); j--;)
		{
			(*this)[i].setMOOFitness(j, obj);
		}
	}
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-202
void PopulationMOO::setMOORank(unsigned MOOR)
{
	for (unsigned i = size(); i--;)
	{
		(*this)[i].setMOORank(MOOR);
	}
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-203
void PopulationMOO::setMOOShare(double MOOS)
{
	for (unsigned i = size(); i--;)
	{
		(*this)[i].setMOOShare(MOOS);
	}
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-210
int PopulationMOO::Dominate(unsigned i1, unsigned i2, bool unpenalized)
{
	// check the number of objective functions
	if ((*this)[i1].getNoOfObj() != (*this)[i2].getNoOfObj())
	{
		cerr << "\n ***** The number of objective functions are different in TO-PM-211 *****\n" << endl;
		throw SHARKEXCEPTION("wrong number of objectives");
	}
	unsigned NoOfObj = (*this)[i1].getNoOfObj();

	// set flag for calculation
	unsigned flag1 = 0;
	unsigned flag2 = 0;
	unsigned flag3 = 0;

	// calculation
	if (unpenalized)
	{
		for (unsigned i = NoOfObj; i--;)
		{
			if ((*this)[i1].getUnpenalizedMOOFitness(i) > (*this)[i2].getUnpenalizedMOOFitness(i))
			{
				flag1++;
			}
			else if ((*this)[i1].getUnpenalizedMOOFitness(i) < (*this)[i2].getUnpenalizedMOOFitness(i))
			{
				flag3++;
			}
			else
			{
				flag2++;
			}
		}
	}
	else
	{
		for (unsigned i = NoOfObj; i--;)
		{
			if ((*this)[i1].getMOOFitness(i) > (*this)[i2].getMOOFitness(i))
			{
				flag1++;
			}
			else if ((*this)[i1].getMOOFitness(i) < (*this)[i2].getMOOFitness(i))
			{
				flag3++;
			}
			else
			{
				flag2++;
			}
		}
	}
	if (!ascending)
	{
		std::swap(flag1, flag3);
	}

	// relationship
	if (flag1 + flag2 + flag3 != NoOfObj)
	{
		return 0; // abnormal
	}
	else if (flag3 == NoOfObj)
	{
		return 3; // i1 dominates i2 completely
	}
	else if (flag3 != 0 && flag1 == 0)
	{
		return 2; // i1 dominates i2 imcompletely
	}
	else if (flag2 == NoOfObj)
	{
		return 1; // i1 equals i2
	}
	else if (flag1 == NoOfObj)
	{
		return -3; // i2 dominates i1 completely
	}
	else if (flag1 != 0 && flag3 == 0)
	{
		return -2; // i2 dominates i1 imcompletely
	}
	else
	{
		return -1; // trade off
	}
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-211
void PopulationMOO::MOGAFonsecaRank()
{
	unsigned i, j, k;
	int      l;
	// initialize rank
	setMOORank(0);
	// calculation
	for (i = size(); i--;)
	{
		k = 1;
		for (j = size(); j--;)
		{
			l = (*this).Dominate(i, j);
			if (l < -1)
			{
				k++;
			}
		}
		(*this)[i].setMOORank(k);
	}
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-212
void PopulationMOO::MOGAGoldbergRank()
{
	unsigned i, j, k;
	int      l;
	unsigned CurrentRank = 0;
	unsigned *StoreRank;
	StoreRank = new unsigned[size()];
	unsigned NoAssign = size();
	// initialize rank
	setMOORank(0);
	// calculation
	while (NoAssign)
	{
		CurrentRank++;
		for (i = size(); i--;)
		{
			if ((*this)[i].getMOORank() == 0)
			{
				k = 1;
				for (j = size(); j--;)
				{
					l = (*this).Dominate(i, j);
					if (l < -1 && (*this)[j].getMOORank() == 0)
					{
						k++;
					}
				}
			}
			else
			{
				k = 0;
			}
			StoreRank[ i ] = k;
		}
		for (unsigned i = size(); i--;)
		{
			if (StoreRank[ i ] == 1)
			{
				(*this)[i].setMOORank(CurrentRank);
				NoAssign--;
			}
		}

	}
	delete [] StoreRank;
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-213
void PopulationMOO::NSGAIIRank()
{
	unsigned i, j;
	int      l;
	unsigned CurrentRank = 0;
	unsigned NoAssign = size();
	unsigned check;
	unsigned **RankData;
	RankData = new unsigned* [size()];
	for (i = size(); i--;)
		RankData[i] = new unsigned[size()];
	unsigned *Rank;
	Rank = new unsigned[size()];
	for (i = size(); i--;)
	{
		Rank[ i ] = 0;
		for (j = size(); j--;)
		{
			RankData[ i ][ j ] = 0;
		}
	}
	// initiailize rank
	setMOORank(0);
	// MOGAFonseca flow
	for (i = size(); i--;)
	{
		Rank[ i ] = 1;
		for (j = size(); j--;)
		{
			l = (*this).Dominate(i, j);
			if (l < -1)
			{
				Rank[ i ]++;
				RankData[ i ][ j ] = 1;
			}
		}
	}
	// assign the rank
	while (NoAssign)
	{
		check = NoAssign;
		CurrentRank++;
		for (i = size(); i--;)
		{
			if (Rank[ i ] == 1)
			{
				(*this)[i].setMOORank(CurrentRank);
				NoAssign--;
				for (j = size(); j--;)
				{
					RankData[ j ][ i ] = 0;
				}
			}
		}
		// correct Rank[ ]
		for (i = size(); i--;)
		{
			if ((*this)[i].getMOORank() == 0)
			{
				Rank[ i ] = 1;
				for (j = size(); j--;)
				{
					Rank[ i ] += RankData[ i ][ j ];
				}
			}
			else
			{
				Rank[ i ] = 0;
			}
		}
		if (check == NoAssign)
		{
			cout << "\n ***** Calculation Error in TO-PM-213 *****\n" << endl;
			NoAssign = 0;
		}
	}
	for (i = size(); i--;)
		delete [] RankData[i];
	delete [] RankData;
	delete [] Rank;
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-250
void PopulationMOO::MOORankToFitness()
{
	unsigned i, r, m;
	if (ascendingFitness())
	{
		for (i = 0; i < size(); i++)
		{
			r = (*this)[i].getMOORank();
			(*this)[i].setFitness(static_cast< double >(r));
		}
	}
	else
	{
		m = 0;
		for (i = 0; i < size(); i++)
		{
			r = (*this)[i].getMOORank();
			if (r > m)
			{
				m = r;
			}
		}
		for (i = 0; i < size(); i++)
		{
			r = (*this)[i].getMOORank();
			r = m - r + 1;
			(*this)[i].setFitness(static_cast< double >(r));
		}
	}
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-251
void PopulationMOO::aggregation(const std::vector< double >& weight)
{
	double result;
	for (unsigned i = size(); i--;)
	{
		result = (*this)[i].aggregation(weight);
		(*this)[i].setFitness(result);
	}
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-252
void PopulationMOO::simpleSum()
{
	double result;
	for (unsigned i = size(); i--;)
	{
		result = (*this)[i].simplesum();
		(*this)[i].setFitness(result);
	}
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-253
void PopulationMOO::simpleTransferFitness(unsigned i)
{
	double result;
	for (unsigned j = size(); j--;)
	{
		result = (*this)[j].getMOOFitness(i);
		(*this)[j].setFitness(result);
	}
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-400 ( OK )
void PopulationMOO::selectInit()
{
	Population::selectInit();
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-401 ( OK )
void PopulationMOO::selectElitists(PopulationMOO& offspring, unsigned numElitists)
{
	if (numElitists)
	{
		double   s;
		double   dtemp;
		unsigned utemp;
		unsigned i;
		PopulationMOO pop(*this);
		std::vector< IndividualMOO* > indvec(pop.size() + offspring.size());
		// Sort
		for (i = 0; i < pop.size(); i++)
		{
			indvec[ i ] = &(pop[ i ]);
		}
		for (i = 0; i < offspring.size(); i++)
		{
			indvec[ size() + i ] = &(offspring[ i ]);
		}
		sortIndividuals(indvec);
		// Copy elitists and correct selection probabilities
		for (i = 0; i < size() && i < numElitists; i++)
		{
			dtemp = indvec[i]->getSelProb();
			indvec[i]->setSelProb(dtemp - Shark::min(dtemp, 1.0 / size()));
			utemp = indvec[i]->getNumCopies();
			indvec[i]->setNumCopies(utemp + 1);
			indvec[i]->setElitist(true);
			(*this)[ i ] = *indvec[ i ];
		}
		s = 0;
		for (i = 0; i < offspring.size(); i++)
		{
			s += offspring[ i ].getSelProb();
		}
		if (s > 0)
		{
			for (i = 0; i < offspring.size(); i++)
			{
				dtemp = offspring[ i ].getSelProb();
				offspring[ i ].setSelProb(dtemp / s);
			}
		}
	}
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-402
void PopulationMOO::selectRouletteWheel(PopulationMOO& popmoo,
										unsigned numElitists)
{
	unsigned i, j;
	double p, r, s;
	unsigned temp;
	if (getSpinOnce())
	{
		p = size() - numElitists;
		r = Rng::uni(0, 1);
		s = 0.0;
		for (i = numElitists, j = 0; i < size() && j < popmoo.size(); j++)
		{
			s += p * popmoo[ j ].getSelProb();
			while (s > r && i < size())
			{
				temp = popmoo[ j ].getNumCopies();
				popmoo[ j ].setNumCopies(temp + 1);
				(*this)[ i++ ] = popmoo[ j ];
				r++;
			}
		}
	}
	else
	{
		for (i = numElitists; i < size();)
		{
			r = Rng::uni(0, 1);
			s = 0.0;
			for (j = 0; j < popmoo.size() && r >= (p = popmoo[ j ].getSelProb()) + s;
				s += p, j++) {}
			RANGE_CHECK(j < popmoo.size())
			temp = popmoo[ j ].getNumCopies();
			popmoo[ j ].setNumCopies(temp + 1);
		}
	}
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-405 ( OK )
void PopulationMOO::linearDynamicScaling(std::vector< double >& window,
		unsigned long t)
{
	Population::linearDynamicScaling(window, t);
}

// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-410 ( OK )
void PopulationMOO::selectMuLambda(PopulationMOO& offspring,
								   unsigned numElitists)
{
	unsigned i, j;
	unsigned utemp;
	// Clear number of copies and select elitists
	offspring.selectInit();
	selectElitists(offspring, numElitists);
	// Sort offspring ( only pointers )
	std::vector< IndividualMOO* > indvec(offspring.size());
	for (unsigned k = 0; k < offspring.size(); k++)
	{
		indvec[ k ] = &(offspring[ k ]);
	}
	sortIndividuals(indvec);
	// Select
	for (i = numElitists, j = 0;
			i < size() && j < indvec.size(); j++)
	{
		if (! indvec[ j ]->getElitist())
		{
			utemp = indvec[ j ]->getNumCopies();
			indvec[ j ]->setNumCopies(utemp + 1);
			(*this)[ i++ ] = *indvec[ j ];
		}
	}
	// Fill remaining slots with the worst individual
	while (i < size())
	{
		utemp = indvec[ j - 1 ]->getNumCopies();
		indvec[ j - 1 ]->setNumCopies(utemp + 1);
		(*this)[ i++ ] = *indvec[ j - 1 ];
	}
}

// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-412 ( OK )
void PopulationMOO::selectProportional(PopulationMOO& popmoo,
									   unsigned numElitists)
{
	unsigned i;
	double   s, t;
	double   temp;
	// clear number of copies and select elitists
	popmoo.selectInit();
	selectElitists(popmoo, numElitists);
	// selection probabilities are proportional to the (scaled) fitness
	for (t = 0., i = 0; i < popmoo.size(); i++)
	{
		if (popmoo[ i ].getScaledFitness() < t)
		{
			t = popmoo[ i ].getScaledFitness();
		}
	}
	for (s = 0., i = 0; i < popmoo.size(); i++)
	{
		temp = popmoo[ i ].getScaledFitness() - t;
		popmoo[ i ].setScaledFitness(temp);
		s += popmoo[ i ].getScaledFitness();
	}
	for (i = 0; i < popmoo.size(); i++)
	{
		temp = popmoo[ i ].getScaledFitness() / s;
		popmoo[ i ].setSelProb(temp);
	}
	// select individuals by roulette wheel selection
	selectRouletteWheel(popmoo, numElitists);
}

// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-413
IndividualMOO&  PopulationMOO::selectOneIndividual()
{
	double t, s, r;
	unsigned i;
	// calculate minimum scaled fitness value
	t = 0.0;
	for (i = 0; i < size(); i++)
	{
		if ((*this)[i].getScaledFitness() < t)
		{
			t = (*this)[i].getScaledFitness();
		}
	}
	// correct scaled fitness value
	for (i = 0; i < size(); i++)
	{
		s = (*this)[i].getScaledFitness() - t;
		(*this)[i].setScaledFitness(s);
	}
	// calculate sum of scaled fitness value
	s = 0.0;
	for (i = 0; i < size(); i++)
	{
		s += (*this)[i].getScaledFitness();
	}
	// calculate selection probabilities
	for (i = 0; i < size(); i++)
	{
		t = (*this)[i].getScaledFitness() / s;
		(*this)[i].setSelProb(t);
	}
	// select individual
	r = Rng::uni(0, 1);
	s = 0.0;
	for (i = 0; i < size() && r > (t = (*this)[i].getSelProb()) + s; s += t, i++);
	i = ((*this)[i].getNumCopies()) + 1;
	(*this)[i].setNumCopies(i);

	return (*this)[i];
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-450
void PopulationMOO::NormalizeSelectProb()
{
	double total = 0.0;
	double sp;
	unsigned i;
	for (i = size(); i--;)
	{
		total += (*this)[i].getSelProb();
	}
	if (total == 0.0)
	{
		cerr << "\n ***** Total probability is 0 in TO-PM-450 *****\n" << endl;
		throw SHARKEXCEPTION("cannot compute selection probability");
	}
	for (i = size(); i--;)
	{
		sp = (*this)[i].getSelProb() / total;
		(*this)[i].setSelProb(sp);
	}
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-451
void PopulationMOO::SelectProbMichalewicz(double c)
{
	// calculate selection probability
	double sp;
	for (unsigned i = size(); i--;)
	{
		if ((*this)[i].getMOORank() <= 0)
		{
			cerr << "\n ***** No rank data in TO-PM-451 *****\n" << endl;
			throw SHARKEXCEPTION("cannot compute selection probability");
		}
		sp = c * pow((double)(1.0 - c), (double)((*this)[i].getMOORank() - 1));
		(*this)[i].setSelProb(sp);
	}
}
// ----------------------------------------------------------------------------


// ----------------------------------------------------------------------------
// TO-PM-500
void PopulationMOO::printPM()
{
	cout << "\n\n********** PopulationMOO **********\n";
	cout << "No. of Individuals         : " << (*this).size() << "\n";
	cout << "Variable ( index )         : " << (*this).index << "\n";
	cout << "Variable ( subPop )        : " << (*this).subPop << "\n";
	cout << "Variable ( ascending )     : " << (*this).ascending << "\n";
	cout << "Variable ( spinOnce )      : " << (*this).spinOnce << "\n";
	cout << endl;
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-601
double PopulationMOO::PhenoFitDisN2(unsigned i1, unsigned i2)
{
	double distance = 0.0;
	for (unsigned i = (*this)[i1].getNoOfObj(); i--;)
	{
		distance += pow((*this)[i1].getMOOFitness(i) -
						(*this)[i2].getMOOFitness(i), 2);
	}
	distance = sqrt(distance);
	return distance;
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-704
void PopulationMOO::NicheCountPFN2(double sr)
{
	double distance;
	unsigned count;
	for (unsigned i1 = (*this).size(); i1--;)
	{
		count = 0;
		for (unsigned i2 = (*this).size(); i2--;)
		{
			distance = PhenoFitDisN2(i1, i2);
			if (distance <= sr)
			{
				count++;
			}
		}
		(*this)[i1].setMOOShare(static_cast< double >(count));
	}
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-790
void PopulationMOO::SharingSelProb()
{
	double share;
	double selprob;
	for (unsigned i = size(); i--;)
	{
		share = (*this)[i].getMOOShare();
		if (share == 0.0)
		{
			cout << "\n ***** Sharing value is 0 in TO-PM-790 *****\n" << endl;
			return;
		}
		selprob = (*this)[i].getSelProb() / share;
		(*this)[i].setSelProb(selprob);
	}
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-PM-800
void PopulationMOO::SelectByRoulette(PopulationMOO& offsprings, unsigned elit)
{
	double *pileProb;
	pileProb = new double[size()];
	double r;
	unsigned nc;
	unsigned i;
	offsprings.NormalizeSelectProb();
	if (!elit)
	{
		selectInit();
	}
	// calculate the piled-up probability
	for (i = 0; i < size(); i++)
	{
		if (i == 0)
		{
			pileProb[ i ] = offsprings[i].getSelProb();
		}
		else
		{
			pileProb[ i ] = offsprings[i].getSelProb() + pileProb[ i - 1 ];
		}
	}
	// selection
	for (i = elit; i < size(); i++)
	{
		r = Rng::uni(0.0, 1.0);
		if (r <= pileProb[ 0 ])
		{
			nc = offsprings[0].getNumCopies();
			offsprings[0].setNumCopies(nc++);
			(*this)[i] = offsprings[0];
		}
		else
		{
			for (unsigned j = 1; j < size(); j++)
			{
				if (r <= pileProb[ j ] && r > pileProb[ j-1 ])
				{
					nc = offsprings[j].getNumCopies();
					offsprings[j].setNumCopies(nc++);
					(*this)[i] = offsprings[j];
				}
			}
		}
	}
	delete [] pileProb;
}
// ----------------------------------------------------------------------------


// static comparison function
bool PopulationMOO::compareRankShare(Individual*const& pInd1, Individual*const& pInd2)
{
	IndividualMOO* pI1 = (IndividualMOO*)pInd1;
	IndividualMOO* pI2 = (IndividualMOO*)pInd2;

	unsigned Rank1 = pI1->getMOORank();
	unsigned Rank2 = pI2->getMOORank();
	double Share1 = pI1->getMOOShare();
	double Share2 = pI2->getMOOShare();

	return ((Rank1 < Rank2)	|| (Rank1 == Rank2 && Share1 > Share2));
}


// ----------------------------------------------------------------------------
// TO-PM-2000
void PopulationMOO::crowdedTournamentSelection(PopulationMOO& offsprings)
{
	// Combined Population
	PopulationMOO total(size() + offsprings.size());
	total.combinePopulationMOO((*this), offsprings);
	// Calculate Crowded Distance. In this function, the rank also is calculated.
	total.crowdedDistance();

	// sort in time ~ n log(n)
	std::sort(total.begin(), total.end(), compareRankShare);

	// Selection
	unsigned int j;
	for (j = 0; j < size(); j++)
	{
		(*this)[ j ] = total[ j ];
	}
}


// ----------------------------------------------------------------------------
// TO-PM-2001

void PopulationMOO::crowdingDistance(bool UnpenalizedFitness)
{
	m_fastNonDominatedSort(*this);
	m_crowdingDistance.m_bUnpenalizedFitness = UnpenalizedFitness;
	m_crowdingDistance(*this);
}

void PopulationMOO::EpsilonMeasure(bool UnpenalizedFitness)
{
	m_fastNonDominatedSort(*this);
	m_epsilonMeasure.m_bUnpenalizedFitness = UnpenalizedFitness;
	m_epsilonMeasure(*this);
}

void PopulationMOO::crowdedDistance(bool UnpenalizedFitness)
{
	unsigned counter;
	double max = 0.0;
	double min = 0.0;
	std::vector< unsigned > sort(size());
	std::vector< unsigned > global(size());
	//std::vector< unsigned > minisort( size( ) );
	double temp1;
	double temp2;
	unsigned i;

	// Rank calculation
	NSGAIIRank();
	// Maximum Rank
	const unsigned maxrank = maxMOORank();
	// Initialize sharing values
	setMOOShare(0.0);

	// Rank Loop
	for (unsigned rank = 1; rank <= maxrank; rank++)
	{
		// gather individuals with the same rank into a sub population
		// PopulationMOO subpop( numberOfMOORank( rank ) );
		counter = 0;
		for (i = 0; i < size(); i++)
		{
			if ((*this)[ i ].getMOORank() == rank)
			{
				global[ counter ] = i;
				counter++;
			}
		}
		// Object Loop
		for (unsigned obj = 0; obj < 2; obj++)
		{
			// Max and min
			if (UnpenalizedFitness)
			{
				max = (*this)[ 0 ].getUnpenalizedMOOFitness(obj);
				min = (*this)[ 0 ].getUnpenalizedMOOFitness(obj);
			}
			else
			{
				max = (*this)[ 0 ].getMOOFitness(obj);
				min = (*this)[ 0 ].getMOOFitness(obj);
			}
			for (i = 1; i < size(); i++)
			{
				if (UnpenalizedFitness)
				{
					if (max < (*this)[ i ].getUnpenalizedMOOFitness(obj))
						max = (*this)[ i ].getUnpenalizedMOOFitness(obj);
					if (min > (*this)[ i ].getUnpenalizedMOOFitness(obj))
						min = (*this)[ i ].getUnpenalizedMOOFitness(obj);
				}
				else
				{
					if (max < (*this)[ i ].getMOOFitness(obj))
						max = (*this)[ i ].getMOOFitness(obj);
					if (min > (*this)[ i ].getMOOFitness(obj))
						min = (*this)[ i ].getMOOFitness(obj);
				}
			}
			// Initialization of index vector
			for (i = 0; i < size(); i++)
				sort[ i ] = i;

			// sort in time ~ n log(n)
			SortMOO_CD sorter(&sort, &global, counter, this, obj, UnpenalizedFitness);
			sorter.sort();

			// Calculate Crowding Distance
			(*this)[ global[ sort[ 0 ] ] ].setMOOShare(MAXDOUBLE);
			(*this)[ global[ sort[ counter - 1 ] ] ].setMOOShare(MAXDOUBLE);

			for (i = 1; i < counter - 1; i++)
			{
				temp1 = (*this)[ global[ sort[ i ] ] ].getMOOShare();
				if (temp1 != MAXDOUBLE)
				{
					if (UnpenalizedFitness)
						temp2 = ((*this)[ global[ sort[ i + 1 ] ] ].getUnpenalizedMOOFitness(obj)
								 - (*this)[ global[ sort[ i - 1 ] ] ].getUnpenalizedMOOFitness(obj));
					else
						temp2 = ((*this)[ global[ sort[ i + 1 ] ] ].getMOOFitness(obj)
								 - (*this)[ global[ sort[ i - 1 ] ] ].getMOOFitness(obj));
					if (max - min != 0) temp2 /= (max - min);
					(*this)[ global[ sort[ i ] ] ].setMOOShare(temp1 + temp2);
				}
			}
		} // Object loop
	} // Rank Loop
}

void PopulationMOO::SMeasure(bool UnpenalizedFitness)
{
	if (size() == 0)
		return;

	if ((*this)[0].getNoOfObj() == 2)
		SMeasureTwoObjectives(UnpenalizedFitness);
	else
	{
		m_fastNonDominatedSort(*this);
		m_sMeasure.m_bUnpenalizedFitness = UnpenalizedFitness;
		m_sMeasure(*this);
	}
}

void PopulationMOO::SMeasureTwoObjectives(bool UnpenalizedFitness)
{
	unsigned counter;
	std::vector< unsigned > sort(size());
	std::vector< unsigned > global(size());
	std::vector< bool >     assigned(size());
	double temp1;
	double temp2;
	double v;
	unsigned i, left, right;

	// Rank calculation
	NSGAIIRank();
	// Maximum Rank
	const unsigned maxrank = maxMOORank();
	// Initialize sharing values
	setMOOShare(0.0);

	if ((*this)[0].getNoOfObj() != 2)
	{
		cerr << "SMeasure only defined for two objectives" << endl;
		throw SHARKEXCEPTION("wrong number of objectives");
	}

	// rank loop
	for (unsigned rank = 1; rank <= maxrank; rank++)
	{
		// gather individuals with the same rank into a sub population
		// PopulationMOO subpop( numberOfMOORank( rank ) );
		counter = 0;
		for (i = 0; i < size(); i++)
		{
			if ((*this)[ i ].getMOORank() == rank)
			{
				global[ counter ] = i;
				counter++;
			}
		}

		// sort according to 1st objective
		unsigned obj = 0;
		// initialization of index vector
		for (i = 0; i < size(); i++)
		{
			sort[ i ] = i; assigned[ i ] = false;
		}
		// sort
		bool changed;
		do
		{
			changed  = false;

			for (i = 0; i < counter - 1; i++)
			{
				if (UnpenalizedFitness)
				{
					temp1 = (*this)[ global[ sort[ i ] ] ]    .getUnpenalizedMOOFitness(obj);
					temp2 = (*this)[ global[ sort[ i + 1 ] ] ].getUnpenalizedMOOFitness(obj);
				}
				else
				{
					temp1 = (*this)[ global[ sort[ i ] ] ]    .getMOOFitness(obj);
					temp2 = (*this)[ global[ sort[ i + 1 ] ] ].getMOOFitness(obj);
				}

				if (temp1 > temp2)
				{
					unsigned temp = sort[ i ];
					sort[ i ] = sort[ i + 1 ];
					sort[ i + 1 ] = temp;
					changed  = true ;
				}
			}
		}
		while (changed);

		(*this)[ global[ sort[ 0 ] ] ].setMOOShare(MAXDOUBLE);
		(*this)[ global[ sort[ counter - 1 ] ] ].setMOOShare(MAXDOUBLE);

		for (unsigned e = 1; e < counter - 1; e++)
		{ // loop over all non-border elements
			for (i = 1; (assigned[ sort[ i ] ]); i++); // determine 1st not assigned, non-border element
			for (left = 0; i < counter - 1;)
			{   // loop over all not assigned elements
				// determine right not assigned neighbor
				for (right = i + 1; (assigned[ sort[ right ] ]); right++);

				if (UnpenalizedFitness)
				{
					v = ((*this)[ global[ sort[ right ] ] ].getUnpenalizedMOOFitness(0) -
						 (*this)[ global[ sort[ i  ] ] ].getUnpenalizedMOOFitness(0)) *
						((*this)[ global[ sort[ left ] ] ].getUnpenalizedMOOFitness(1) -
						 (*this)[ global[ sort[ i  ] ] ].getUnpenalizedMOOFitness(1));
				}
				else
				{
					v = ((*this)[ global[ sort[ right ] ] ].getMOOFitness(0) -
						 (*this)[ global[ sort[ i  ] ] ].getMOOFitness(0)) *
						((*this)[ global[ sort[ left ] ] ].getMOOFitness(1) -
						 (*this)[ global[ sort[ i  ] ] ].getMOOFitness(1));
				}
				(*this)[ global[ sort[ i ] ] ].setMOOShare(v);
				

				left = i;
				i = right;
			}
			unsigned minIndex = 0;
			double min = (*this)[ global[ sort[ minIndex ] ] ].getMOOShare();
			for (unsigned f = 1; f < counter - 1; f++)
			{
				if (!assigned[ sort[ f ] ])
					if ((*this)[ global[ sort[ f ] ] ].getMOOShare() < min)
					{
						min = (*this)[ global[ sort[ f ] ] ].getMOOShare();
						minIndex = f;
					}
			}
			assigned[ sort[ minIndex ] ] = true;
			(*this)[ global[ sort[ minIndex ] ] ].setMOOShare( e );
		}
	} // Rank Loop
}

// ----------------------------------------------------------------------------
// TO-PM-2002
unsigned PopulationMOO::maxMOORank()
{
	unsigned rank = 0;
	for (unsigned i = 0; i < size(); i++)
	{
		if ((*this)[i].getMOORank() > rank)
		{
			rank = (*this)[i].getMOORank();
		}
	}
	return rank;
}

// ----------------------------------------------------------------------------
// TO-PM-2003
unsigned PopulationMOO::numberOfMOORank(unsigned rank)
{
	unsigned counter = 0;
	for (unsigned i = 0; i < size(); i++)
	{
		if ((*this)[i].getMOORank() == rank)
		{
			counter++;
		}
	}
	return counter;
}


// static comparison function
bool PopulationMOO::compareScaledFitnessRankShare(Individual*const& pInd1, Individual*const& pInd2)
{
	IndividualMOO* pI1 = (IndividualMOO*)pInd1;
	IndividualMOO* pI2 = (IndividualMOO*)pInd2;

	return (pI2->getScaledFitness() > pI1->getScaledFitness())
		   || (pI2->getScaledFitness() == pI1->getScaledFitness()
			   && pI2->getMOORank() < pI1->getMOORank())
		   || (pI2->getScaledFitness() == pI1->getScaledFitness()
			   && pI2->getMOORank() == pI1->getMOORank()
			   && pI2->getMOOShare() > pI1->getMOOShare());
}


// ----------------------------------------------------------------------------
// SW-PM-2020
void PopulationMOO::selectCrowdedEPTournament(PopulationMOO& offsprings, unsigned q)
{
	unsigned i, j;

	Array<unsigned> nichedRanks;
	(*this).nichedComparisonRank(offsprings, nichedRanks);

	// Combined Population
	PopulationMOO total(size() + offsprings.size());
	total.combinePopulationMOO((*this), offsprings);
	uint opponent ;

	// perform tournament
	for (i = 0; i < total.size(); i++)
	{
		total[ i ].scaledFitness = 0;
		for (j = 0; j < q; j++)
		{
			opponent = Rng::discrete(0,  total.size() - 1);
			if (nichedRanks(i) <= nichedRanks(opponent))
				total[ i ].scaledFitness++;
		}
	}

	// sort in time ~ n log(n)
	std::sort(total.begin(), total.end(), compareScaledFitnessRankShare);

	// Selection
	for (i = 0; i < size(); i++)
	{
		(*this)[ i ] = total[ i ];
	}
}

// ----------------------------------------------------------------------------
// SW-PM-2021
void PopulationMOO::nichedComparisonRank(PopulationMOO& offsprings, Array<unsigned>& nichedRanks)
{
	// Combined Population
	PopulationMOO total(size() + offsprings.size());
	total.combinePopulationMOO((*this), offsprings);
	unsigned i;

	nichedRanks.resize(total.size());

	vector<unsigned> totinds(total.size());

	for (i = 0; i < total.size(); i++)
	{
		totinds[i] = i;
	}

	// Calculate Crowded Distance. In this function, the rank also is calculated.
	total.crowdedDistance();

	// sort in time ~ n log(n)
	SortMOO_NCR sorter(&totinds, &total);
	sorter.sort();

	for (i = 0; i < total.size(); i++) nichedRanks(totinds[i]) = i;
}


// ----------------------------------------------------------------------------
// TO-PM-2501 ( check ok )
PopulationMOO& PopulationMOO::combinePopulationMOO(PopulationMOO& p1, PopulationMOO& p2)
{
	if (p1.checkData(p2) != 0)
	{
		cerr << "difference between parent and offspring" << endl;
		throw SHARKEXCEPTION("cannot combine populations");
	}

	setAscending(p1.ascendingFitness());
	setNoOfObj(p1[0].getNoOfObj());

	unsigned no = p1.size() + p2.size();
	for (unsigned i = no; i--;)
	{
		delete *(begin() + i);
		if (i < p1.size())
		{
			*(begin() + i) = new IndividualMOO(p1[ i ]);
		}
		else
		{
			*(begin() + i) = new IndividualMOO(p2[ i - p1.size()]);
		}
	}

	return (*this);
}

// ----------------------------------------------------------------------------
// TO-PM-2502 ( check ok )
PopulationMOO& PopulationMOO::combinePopulationMOO(PopulationMOO& p1, ArchiveMOO& a1)
{
	setAscending(p1.ascendingFitness());
	setNoOfObj(p1[0].getNoOfObj());

	unsigned no = p1.size() + a1.size();
	for (unsigned i = no; i--;)
	{
		delete *(begin() + i);
		if (i < p1.size())
		{
			*(begin() + i) = new IndividualMOO(p1[ i ]);
		}
		else
		{
			*(begin() + i) = new IndividualMOO(a1.readArchive(i - p1.size()));
		}
	}

	return (*this);
}

// ----------------------------------------------------------------------------
// TO-PM-3001
int PopulationMOO::checkData(PopulationMOO& offsprings)
{
	if (ascendingFitness() != offsprings.ascendingFitness()) return -1;
	if ((*this)[0].getNoOfObj() != offsprings[0].getNoOfObj()) return -1;
	return 0;
}
// ----------------------------------------------------------------------------
//  added by Stefan Wiegand at 28.1.2004

// SW-PM-4000( pvm pack routine )
int PopulationMOO::pvm_pkpop()
{
	unsigned i;

	unsigned *s = new unsigned;
	*s = this->size();
	pvm_pkuint(s, 1, 1);
	delete s;

	for (i = 0; i < this->size(); i++)
	{
		((*this)[i]).IndividualMOO::pvm_pkind();
	}

	uint *u = new uint[4];
	u[0] = index;
	u[1] = subPop;
	u[2] = ascending;
	u[3] = spinOnce;
	pvm_pkuint(u, 4, 1);
	delete[] u;

	return 1;
}

// SW-PM-4001( pvm unpack routine )
int PopulationMOO::pvm_upkpop()
{
	unsigned i;

	unsigned *s = new unsigned;
	pvm_upkuint(s, 1, 1);
	if (this->size() != *s)
	{
		cerr << "PopulationMOO.cpp: the population which has called pvm_upkpop() is of unexpected size!" << endl;
		throw SHARKEXCEPTION("wrong population size");
	}
	delete s;

	for (i = 0; i < this->size(); i++)((*this)[i]).IndividualMOO::pvm_upkind();

	uint *u   = new uint[4];
	pvm_upkuint(u, 4, 1);
	index     = u[0] ;
	subPop    = u[1] ? 1 : 0 ;
	ascending = u[2] ? 1 : 0 ;
	spinOnce  = u[3] ? 1 : 0 ;
	delete[] u;

	return 1;
}

void PopulationMOO::selectCrowdedMuPlusLambda(PopulationMOO& offsprings, bool unpenalized)
{
	unsigned i;

	// Combined Population
	PopulationMOO total(size() + offsprings.size());
	total.combinePopulationMOO((*this), offsprings);

	// Calculate Crowded Distance. In this function, the rank also is calculated.
	total.crowdedDistance(unpenalized);

	// Sort
	std::sort(total.begin(), total.end(), compareRankShare);

	// Selection
	for (i = 0; i < size(); i++)
	{
		(*this)[ i ] = total[ i ];
	}
}


void PopulationMOO::selectCrowdedMuCommaLambda(PopulationMOO& offsprings, bool unpenalized)
{
	unsigned i;

	// Calculate Crowded Distance. In this function, the rank also is calculated.
	offsprings.crowdedDistance(unpenalized);

	// Sort
	std::sort(offsprings.begin(), offsprings.end(), compareRankShare);

	// Selection
	for (i = 0; i < size(); i++)
	{
	  std::cout<<" here "<<i<<std::endl;
		(*this)[ i ] = offsprings[ i ];
	}
}

void PopulationMOO::selectBinaryTournamentMOO(PopulationMOO& parents)
{
	unsigned i;
	unsigned np =  parents.size();
	unsigned no =  size();
	unsigned a, b;
	for (i = 0; i < no; i++)
	{
		a = Rng::discrete(0, np - 1);
		b = Rng::discrete(0, np - 1);
		if (compareRankShare(&parents[a], &parents[b]))
			(*this)[i] = parents[a];
		else
			(*this)[i] = parents[b];
	}
}



void PopulationMOO::SelectPop(std::vector<std::vector<double> >&PopX,std::vector<std::vector<double> >&PopF,std::vector<std::vector<double> >&OffX, std::vector<std::vector<double> >&OffF)
{
	unsigned int i;
	std::vector< std::vector<double> > tmpF(OffF.size()+PopF.size()), tmpX(OffF.size()+PopF.size());
	for(i=0; i<PopF.size(); i++) {tmpF[i] = PopF[i]; tmpX[i] = PopX[i];}
	for(i=0; i<OffF.size(); i++) {tmpF[i+PopF.size()] = OffF[i]; tmpX[i+PopF.size()] = OffX[i];}
	az::mea::sel::NDS sel;
	sel.Select(PopX.size(), PopF, PopX, tmpF, tmpX);
}

void PopulationMOO::selectRMMuPlusLambda(PopulationMOO& offspring){
  std::vector<std::vector<double> >PopF,PopX, OffF,OffX;
  PopX.resize(size());
  PopF.resize(size());
  OffX.resize(offspring.size());
  OffF.resize(offspring.size());
  unsigned int i,j;
  for(i=0; i<offspring.size(); i++) {
    PopX[i].resize((*this)[0][0].size());
    PopF[i].resize(2);
    OffX[i].resize(offspring[0][0].size());
    OffF[i].resize(2);
    for (j=0;j<offspring[0][0].size();j++){
      PopX[i][j] = dynamic_cast<ChromosomeT<double>&>((*this)[i][0])[j];
      OffX[i][j] = dynamic_cast<ChromosomeT<double>&>(offspring[i][0])[j];
    }
      for (j=0;j<2;j++){
	PopF[i][j] = (*this)[i].getMOOFitness(j);
	OffF[i][j] = offspring[i].getMOOFitness(j);
      }
    }
    SelectPop(PopX, PopF,OffX,OffF);
    for(i=0; i<offspring.size(); i++) {
      for (j=0;j<offspring[0][0].size();j++){
	dynamic_cast<ChromosomeT<double>&>((*this)[i][0])[j]=PopX[i][j];
	dynamic_cast<ChromosomeT<double>&>(offspring[i][0])[j]=OffX[i][j];
      }
      for (j=0;j<2;j++){
	offspring[i].setMOOFitness(j,OffF[i][j]);
	(*this)[i].setMOOFitness(j,PopF[i][j]);
      }    
    }
}


















