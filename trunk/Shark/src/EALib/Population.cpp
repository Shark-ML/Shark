/*!
*  \file Population.cpp
*
*  \author Martin Kreutz
*
*  \brief Base class for populations of individuals in an evolutionary algorithm
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
*      EALib
*
*
*  <BR>
*
* 
*  <BR><HR>
*  This file is part of the EALib. This library is free software;
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

#include <SharkDefs.h>
#include <EALib/Population.h>


using namespace std;

//===========================================================================
//
// constructors
//

Population::Population()
{
	subPop    = false;
	ascending = false;
	spinOnce  = true;
}

Population::Population(const vector< Individual * >& indvec)
		: vector< Individual * >(indvec)
{
	subPop    = true;
	ascending = false;
	spinOnce  = true;
}

Population::Population(unsigned n)
		: vector< Individual * >(n)
{
	for (unsigned i = size(); i--;)
		*(begin() + i) = new Individual;
	subPop    = false;
	ascending = false;
	spinOnce  = true;
}

Population::Population(const Individual& indiv)
		: vector< Individual * >(1)
{
	*begin() = new Individual(indiv);
	subPop    = false;
	ascending = false;
	spinOnce  = true;
}

Population::Population(unsigned n, const Individual& indiv)
		: vector< Individual * >(n)
{
	for (unsigned i = size(); i--;)
		*(begin() + i) = new Individual(indiv);
	subPop    = false;
	ascending = false;
	spinOnce  = true;
}

/*!
*
* \example countingOnes.cpp
*
*/
Population::Population(unsigned n, const Chromosome& chrom0)
		: vector< Individual * >(n)
{
	for (unsigned i = size(); i--;)
		*(begin() + i) = new Individual(chrom0);
	subPop    = false;
	ascending = false;
	spinOnce  = true;
}


Population::Population(unsigned n, const Chromosome& chrom0,
					   const Chromosome& chrom1)
		: vector< Individual * >(n)
{
	for (unsigned i = size(); i--;)
		*(begin() + i) = new Individual(chrom0, chrom1);
	subPop    = false;
	ascending = false;
	spinOnce  = true;
}

Population::Population(unsigned n, const Chromosome& chrom0,
					   const Chromosome& chrom1,
					   const Chromosome& chrom2)
		: vector< Individual * >(n)
{
	for (unsigned i = size(); i--;)
		*(begin() + i) = new Individual(chrom0, chrom1, chrom2);
	subPop    = false;
	ascending = false;
	spinOnce  = true;
}

Population::Population(unsigned n, const Chromosome& chrom0,
					   const Chromosome& chrom1,
					   const Chromosome& chrom2,
					   const Chromosome& chrom3)
		: vector< Individual * >(n)
{
	for (unsigned i = size(); i--;)
		*(begin() + i) = new Individual(chrom0, chrom1, chrom2,
										chrom3);
	subPop    = false;
	ascending = false;
	spinOnce  = true;
}

Population::Population(unsigned n, const Chromosome& chrom0,
					   const Chromosome& chrom1,
					   const Chromosome& chrom2,
					   const Chromosome& chrom3,
					   const Chromosome& chrom4)
		: vector< Individual * >(n)
{
	for (unsigned i = size(); i--;)
		*(begin() + i) = new Individual(chrom0, chrom1, chrom2,
										chrom3, chrom4);
	subPop    = false;
	ascending = false;
	spinOnce  = true;
}

Population::Population(unsigned n, const Chromosome& chrom0,
					   const Chromosome& chrom1,
					   const Chromosome& chrom2,
					   const Chromosome& chrom3,
					   const Chromosome& chrom4,
					   const Chromosome& chrom5)
		: vector< Individual * >(n)
{
	for (unsigned i = size(); i--;)
		*(begin() + i) = new Individual(chrom0, chrom1, chrom2,
										chrom3, chrom4, chrom5);
	subPop    = false;
	ascending = false;
	spinOnce  = true;
}

Population::Population(unsigned n, const Chromosome& chrom0,
					   const Chromosome& chrom1,
					   const Chromosome& chrom2,
					   const Chromosome& chrom3,
					   const Chromosome& chrom4,
					   const Chromosome& chrom5,
					   const Chromosome& chrom6)
		: vector< Individual * >(n)
{
	for (unsigned i = size(); i--;)
		*(begin() + i) = new Individual(chrom0, chrom1, chrom2,
										chrom3, chrom4, chrom5,
										chrom6);
	subPop    = false;
	ascending = false;
	spinOnce  = true;
}

Population::Population(unsigned n, const Chromosome& chrom0,
					   const Chromosome& chrom1,
					   const Chromosome& chrom2,
					   const Chromosome& chrom3,
					   const Chromosome& chrom4,
					   const Chromosome& chrom5,
					   const Chromosome& chrom6,
					   const Chromosome& chrom7)
		: vector< Individual * >(n)
{
	for (unsigned i = size(); i--;)
		*(begin() + i) = new Individual(chrom0, chrom1, chrom2,
										chrom3, chrom4, chrom5,
										chrom6, chrom7);
	subPop    = false;
	ascending = false;
	spinOnce  = true;
}

Population::Population(unsigned n, const vector< Chromosome * >& chrom)
		: vector< Individual * >(n)
{
	Individual indiv(chrom);
	for (unsigned i = size(); i--;)
		*(begin() + i) = new Individual(indiv);
	subPop    = false;
	ascending = false;
	spinOnce  = true;
}

Population::Population(const Population& pop)
		: vector< Individual * >(pop.size())
{
	for (unsigned i = size(); i--;)
		*(begin() + i) = new Individual(pop[ i ]);
	subPop    = false;
	ascending = pop.ascending; //false;
	spinOnce  = pop.spinOnce;  //true;
}

//===========================================================================
//
// destructor
//
Population::~Population()
{
	if (! subPop)
		for (unsigned i = size(); i--;)
			delete *(begin() + i);
}

//===========================================================================
//
// change size of a population
//
void Population::resize(unsigned n)
{
	unsigned i, s = size();
	if (n < s) {
		for (i = n; i < s; ++i)
			delete *(begin() + i);
		vector< Individual * >::erase(begin() + n, end());
	}
	else if (n > size()) {
		vector< Individual * >::insert(end(), n - s, (Individual *)NULL);
		if (s > 0)
			for (i = s; i < n; ++i)
				*(begin() + i) = new Individual(*(*(begin())));
		else
			for (i = s; i < n; ++i)
				*(begin() + i) = new Individual;
	}
}

//===========================================================================

Population& Population::operator = (const Individual& ind)
{
	for (unsigned i = size(); i--;)
		(*this)[ i ] = ind;

	return *this;
}

//===========================================================================

Population& Population::operator = (const Population& pop)
{
        if(this == &pop) return *this;
	SIZE_CHECK(size() == pop.size() || ! subPop)

	unsigned i;

	if (size() != pop.size()) {
		for (i = size(); i--;)
			delete *(begin() + i);
		vector< Individual * >::operator = (pop);
		for (i = size(); i--;)
			*(begin() + i) = new Individual(pop[ i ]);
	}
	else
		for (i = size(); i--;)
			(*this)[ i ] = pop[ i ];

	return *this;
}

//===========================================================================

vector< const Chromosome* > Population::matingPool(unsigned chrom,
		unsigned from,
		unsigned to) const
{
	if (from == 0 && to == 0)
		to = size() - 1;

	vector< const Chromosome* > pool;

	for (unsigned i = from; i <= to; ++i)
		pool.push_back(&(*this)[ i ][ chrom ]);

	return pool;
}

//===========================================================================

bool Population::lessFitness(Individual*const& i1, Individual*const& i2)
{
	return i1->fitness < i2->fitness;
}

bool Population::greaterFitness(Individual*const& i1, Individual*const& i2)
{
	return i1->fitness > i2->fitness;
}

void Population::sortIndividuals(vector< Individual * >& indvec)
{
	std::sort(indvec.begin(), indvec.end(),
			  ascending ? Population::lessFitness : Population::greaterFitness);
}

//===========================================================================

void Population::sort()
{
	sortIndividuals(*this);
}

//===========================================================================

void Population::shuffle()
{
	for (unsigned i = size(); i--;)
		swap(i, Rng::discrete(0, size() - 1));
}

//===========================================================================

void Population::replace(unsigned i, const Individual& ind)
{
	RANGE_CHECK(i < size())
	delete *(begin() + i);
	*(begin() + i) = new Individual(ind);
}

//===========================================================================

void Population::replace(unsigned i, const Population& pop)
{
	RANGE_CHECK(i + pop.size() <= size())
	for (unsigned j = pop.size(); j--;)
		(*this)[ i+j ] = pop[ j ];
}

//===========================================================================

void Population::insert(unsigned i, const Individual& ind)
{
	RANGE_CHECK(i <= size())
	vector< Individual * >::insert(begin() + i, new Individual(ind));
}

//===========================================================================

void Population::insert(unsigned i, const Population& pop)
{
	RANGE_CHECK(i <= size())
	vector< Individual * >::insert(begin() + i, pop.size(),
								   (Individual *)NULL);
	for (unsigned j = pop.size(); j--;)
		*(begin() + i + j) = new Individual(pop[ j ]);
}

//===========================================================================

void Population::append(const Individual& ind)
{
	vector< Individual * >::push_back(new Individual(ind));
}

//===========================================================================

void Population::append(const Population& pop)
{
	insert(size(), pop);
}

//===========================================================================

void Population::remove(unsigned i)
{
	RANGE_CHECK(i < size())
	delete *(begin() + i);
	vector< Individual * >::erase(begin() + i);
}

//===========================================================================

void Population::remove(unsigned i, unsigned k)
{
	if (i <= k) {
		RANGE_CHECK(k < size())
		for (unsigned j = i; j < k; j++)
			delete *(begin() + j);
		vector< Individual * >::erase(begin() + i, begin() + k);
	}
}

//===========================================================================

void Population::exchange(Population& pop)
{
	unsigned buffer1;
	bool buffer2;

	SIZE_CHECK(size() == pop.size())

	for (unsigned i = size(); i--;)
		std::swap(*(begin() + i), *(pop.begin() + i));

	// Exchanging internal variables:

	buffer1 = index;
	index = pop.index;
	pop.index = buffer1;
	buffer2 = subPop;
	subPop = pop.subPop;
	pop.subPop = buffer2;
	buffer2 = ascending;
	ascending = pop.ascending;
	pop.ascending = buffer2;
	buffer2 = spinOnce;
	spinOnce = pop.spinOnce;
	pop.spinOnce = buffer2;
}

//===========================================================================

Individual& Population::best(Individual& ind0, Individual& ind1) const
{
	if ((ascending && ind0.fitness < ind1.fitness) ||
			(! ascending && ind0.fitness > ind1.fitness)) {
		return ind0;
	}
	else {
		return ind1;
	}
}

Individual& Population::worst(Individual& ind0, Individual& ind1) const
{
	if ((ascending && ind0.fitness < ind1.fitness) ||
			(! ascending && ind0.fitness > ind1.fitness)) {
		return ind1;
	}
	else {
		return ind0;
	}
}

//===========================================================================

Individual&
Population::oneOfBest()
{
	RANGE_CHECK(size() > 0)
	vector< unsigned > bestIndices;

	// find indices of all best individuals
	// assume unsorted population
	for (unsigned i = 0; i < size(); i++) {
		if (ascending) {
			if ((*this)[ i ].fitness == maxFitness())
				bestIndices.push_back(i);
		}
		else {
			if ((*this)[ i ].fitness == minFitness())
				bestIndices.push_back(i);
		}
	}

	// return random "best" individual
	return((*this)[ bestIndices[ Rng::discrete(0, bestIndices.size() - 1)] ]);
}

const Individual&
Population::oneOfBest() const
{
	RANGE_CHECK(size() > 0)
	vector< unsigned > bestIndices;

	// find indices of all best individuals
	// assume unsorted population
	for (unsigned i = 0; i < size(); i++) {
		if (ascending) {
			if ((*this)[ i ].fitness == maxFitness())
				bestIndices.push_back(i);
		}
		else {
			if ((*this)[ i ].fitness == minFitness())
				bestIndices.push_back(i);
		}
	}

	// return random "best" individual
	return((*this)[ bestIndices[ Rng::discrete(0, bestIndices.size() - 1)] ]);
}

//===========================================================================

unsigned Population::bestIndex() const
{
	RANGE_CHECK(size() > 0)

	unsigned ind = 0;

	if (ascending) {
		for (unsigned i = 1; i < size(); i++)
			if ((*this)[ i ].fitness < (*this)[ ind ].fitness)
				ind = i;
	}
	else {
		for (unsigned i = 1; i < size(); i++)
			if ((*this)[ i ].fitness > (*this)[ ind ].fitness)
				ind = i;
	}

	return ind;
}

//===========================================================================

unsigned Population::worstIndex() const
{
	RANGE_CHECK(size() > 0)

	unsigned ind = size() - 1;

	if (ascending) {
		for (unsigned i = size() - 1; i--;)
			if ((*this)[ i ].fitness > (*this)[ ind ].fitness)
				ind = i;
	}
	else {
		for (unsigned i = size() - 1; i--;)
			if ((*this)[ i ].fitness < (*this)[ ind ].fitness)
				ind = i;
	}

	return ind;
}

//===========================================================================

void Population::selectInit()
{
	for (unsigned i = size(); i--;) {
		(*this)[ i ].numCopies = 0;
		(*this)[ i ].elitist   = false;
	}
}

void Population::selectElitists(Population& parents,
								unsigned numElitists)
{
	if (numElitists) {
		unsigned   i;
		double     s;
		Population pop(*this);
		vector< Individual * > indvec(pop.size() + parents.size());

		//
		// sort populations
		//
		std::copy(pop.begin(), pop.end(), indvec.begin());
		std::copy(parents.begin(), parents.end(), indvec.begin() + pop.size());
		sortIndividuals(indvec);

		//
		// copy elitists and correct selection probabilities
		//
		for (i = 0; i < size() && i < numElitists; i++) {
			indvec[ i ]->selProb -= Shark::min(indvec[i]->selProb, 1. / size());
			indvec[ i ]->numCopies++;
			indvec[ i ]->elitist = true;
			(*this)[ i ] = *indvec[ i ];
		}

		s = 0;
		for (i = 0; i < parents.size(); i++)
			s += parents[ i ].selProb;
		if (s > 0)
			for (i = 0; i < parents.size(); i++)
				parents[ i ].selProb /= s;
	}
}

//===========================================================================
//
// stochastic universal sampling according to Baker,87
//
void Population::selectRouletteWheel(Population& parents,
									 unsigned numElitists)
{
	unsigned i, j;
	double   p, r, s;

	if (spinOnce) {
		//
		// spin the wheel one time, derandomized variant
		// (stochastic remainder method according to Baker '87)
		//
		p = size() - numElitists;
		r = Rng::uni(0, 1);
		s = 0.;
		for (i = numElitists, j = 0;
				i < size() && j < parents.size(); j++) {
			s += p * parents[ j ].selProb;
			while (s > r && i < size()) {
				parents[ j ].numCopies++;
				(*this)[ i++ ] = parents[ j ];
				r++;
			}
		}
	}
	else {
		//
		// full stochastic method
		//
		for (i = numElitists; i < size();) {
			r = Rng::uni(0, 1);
			s = 0.;
			for (j = 0;
					j < parents.size() &&
					r >= (p = parents[ j ].selProb) + s;
					s += p, j++);
			RANGE_CHECK(j < parents.size())
			parents[ j ].numCopies++;
			(*this)[ i++ ] = parents[ j ];
		}
	}
}

//===========================================================================
//
// Standard: Die mu besten Individuen aus parents werden in die aktuelle
//           Population uebernommen
// Elitist : Die no_elitists besten Individuen aus parents und der aktuellen
//           Population werden uebernommen und der Rest stammt aus parents,
//           aber keine doppelten aus der Elite.
//
void Population::selectMuLambda(Population& parents,
								unsigned numElitists)
{
	unsigned i, j;

	//
	// clear number of copies and select elitists
	//
	parents.selectInit();
	selectElitists(parents, numElitists);

	//
	// sort parents (only pointers)
	//
	vector< Individual * > indvec(parents);
	sortIndividuals(indvec);

	for (i = numElitists, j = 0;
			i < size() && j < indvec.size(); j++) {
		if (! indvec[ j ]->elitist) {
			indvec[ j ]->numCopies++;
			(*this)[ i++ ] = *indvec[ j ];
		}
	}

	//
	// fill remaining slots with the last/worst individual
	//
	while (i < size()) {
		indvec[ j-1 ]->numCopies++;
		(*this)[ i++ ] = *indvec[ j-1 ];
	}
}

//===========================================================================
//
//
//
void Population::selectMuLambdaKappa(Population& parents,
									 unsigned lifespan,
									 unsigned adolescence)
{
	unsigned i, j;
	Population pop(*this);
	vector< Individual * > indvec;

	//
	// clear number of copies
	//
	parents.selectInit();

	//
	// copy individuals, discard old ones, save young ones
	//
	for (i = j = 0; i < size() && j < pop.size(); ++j) {
		//
		// if individual is a youngster copy it directly to next generation
		//
		if (pop[ j ].age < adolescence) {
			pop[ j ].numCopies++;
			(*this)[ i++ ] = pop[ j ];
			//
			// otherwise check whether individual exceeds its lifespan
			//
		}
		else if (pop[ j ].age < lifespan)
			indvec.push_back(&pop[ j ]);
	}

	//
	// same procedure for the parent population
	//
	for (j = 0; i < size() && j < parents.size(); ++j) {
		if (parents[ j ].age < adolescence) {
			parents[ j ].numCopies++;
			(*this)[ i++ ] = parents[ j ];
		}
		else if (parents[ j ].age < lifespan)
			indvec.push_back(&parents[ j ]);
	}

	//
	// sort remaining individuals
	//
	sortIndividuals(indvec);

	//
	// copy the best individuals to the current population
	//
	for (j = 0; i < size() && j < indvec.size(); ++j) {
		indvec[ j ]->numCopies++;
		(*this)[ i++ ] = *indvec[ j ];
	}

	//
	// fill remaining slots with old individuals
	//
	pop.sort();
	for (j = 0; i < size() && j < pop.size(); ++j) {
		pop[ j ].numCopies++;
		(*this)[ i++ ] = pop[ j ];
	}

	//
	// increment age of all individuals
	//
	incAge();
}


//===========================================================================

//
// added by Stefan Wiegand at 20.11.2002
//

int Population::pvm_pkpop()
{
	//cout << "\tPopulation_pk" << endl;

	unsigned i;

	unsigned *s = new unsigned;
	*s = this->size();
	pvm_pkuint(s, 1, 1);
	delete s;

	for (i = 0; i < this->size(); i++)
		((*this)[i]).pvm_pkind();

	unsigned *u = new unsigned[4];
	u[0] = index;
	u[1] = subPop;
	u[2] = ascending;
	u[3] = spinOnce;
	pvm_pkuint(u, 4, 1);
	delete[] u;

	return 1;
}

int Population::pvm_upkpop()
{
	// cout << "\tPopulation_upk" << endl;

	unsigned i;

	unsigned *s = new unsigned;
	pvm_upkuint(s, 1, 1);
	if (this->size() != *s) {
	  throw SHARKEXCEPTION("EALib/Population.cpp: the population which has called pvm_upkpop() is of unexpected size!\nPlease initialize this population prototypically.");
	}
	delete s;

	for (i = 0; i < this->size(); i++)
		((*this)[i]).pvm_upkind();

	unsigned *u   = new unsigned[4];
	pvm_upkuint(u, 4, 1)     ;
	index     = u[0]       ;
	subPop    = (u[1] != 0);
	ascending = (u[2] != 0);
	spinOnce  = (u[3] != 0);
	delete[] u;

	return 1;
}



//===========================================================================
//
// linear dynamic scaling
//
//
void Population::linearDynamicScaling(vector< double >& window,
									  unsigned long t)
{
	unsigned omega;
	double   f;

	omega = window.size();

	if (t == 0) {   // fill scaling window
		f = worst().fitness;
		for (unsigned i = omega; i--; window[ i ] = f);
	}
	else {
		f = window[(t + omega) % omega ];
		window[ t % omega ] = worst().fitness;
	}

	if (ascending)
		for (unsigned i = size(); i--;)
			(*this)[ i ].scaledFitness = f - (*this)[ i ].fitness;
	else
		for (unsigned i = size(); i--;)
			(*this)[ i ].scaledFitness = (*this)[ i ].fitness - f;
}

//===========================================================================
//
// Standard: Proportionale Selektion, die Fitnesswerte muessen positiv sein
// Elitist : Die no_elitists besten Individuen aus parents und der aktuellen
//           Population werden uebernommen und der Rest stammt aus parents
//
// stochastic universal sampling according to Baker,87
//
void Population::selectProportional(Population& parents,
									unsigned numElitists)
{
	unsigned i;
	double   s, t;

	//
	// clear number of copies and select elitists
	//
	parents.selectInit();
	selectElitists(parents, numElitists);

	//
	// selection probabilities are proportional to the (scaled) fitness
	//
	for (t = 0., i = 0; i < parents.size(); i++)
		if (parents[ i ].scaledFitness < t)
			t = parents[ i ].scaledFitness;
	for (s = 0., i = 0; i < parents.size(); i++)
		s += (parents[ i ].scaledFitness -= t);
	for (i = 0; i < parents.size(); i++)
		parents[ i ].selProb = parents[ i ].scaledFitness / s;

	//
	// select individuals by roulette wheel selection
	//
	selectRouletteWheel(parents, numElitists);
}

//===========================================================================
//
// Standard: Proportionale Selektion, die Fitnesswerte muessen positiv sein
//
// stochastic universal sampling according to Baker,87
//
Individual& Population::selectOneIndividual()
{
	unsigned i;
	double   p, r, s, t;

	//
	// selection probabilities are proportional to the (scaled) fitness
	//
	for (t = 0., i = 0; i < size(); ++i)
		if ((*this)[ i ].scaledFitness < t)
			t = (*this)[ i ].scaledFitness;
	for (s = 0., i = 0; i < size(); ++i)
		s += ((*this)[ i ].scaledFitness -= t);
	for (i = 0; i < size(); i++)
		(*this)[ i ].selProb = (*this)[ i ].scaledFitness / s;

	//
	// select individuals by roulette wheel selection
	// (full stochastic method)
	//
	r = Rng::uni(0, 1);
	s = 0.;
	for (i = 0; i < size() &&
			r > (p = (*this)[ i ].selProb) + s;
			s += p, ++i);
	(*this)[ i ].numCopies++;
	return (*this)[ i ];
}

//===========================================================================
//
// stochastic universal sampling according to Baker,87
//
void Population::selectLinearRanking(Population& parents,
									 double etaMax,
									 unsigned numElitists)
{
	if (size() == 0 || parents.size() == 0)
		return;
	else if (parents.size() == 1)
		//
		// if the parent population has only one individual it
		// receives selection probability 1
		//
		parents[ 0 ].selProb = 1;
	else {
		//
		// sort parents (only pointers)
		//
		vector< Individual * > indvec(parents);
		sortIndividuals(indvec);

		//
		// selection probabilities
		//
		double a = 2 * (etaMax - 1) / (indvec.size() - 1);
		for (unsigned i = 0; i < indvec.size(); i++)
			indvec[ i ]->selProb = (etaMax - a * i) / indvec.size();
	}

	//
	// clear number of copies and select elitists
	//
	parents.selectInit();
	selectElitists(parents, numElitists);

	//
	// select individuals by roulette wheel selection
	//
	selectRouletteWheel(parents, numElitists);
}

//===========================================================================
//
// numberOfIndividuals == parents.numberOfIndividuals => random walk
//
//
void Population::selectUniformRanking(Population& parents,
									  unsigned numElitists)
{
	//
	// selection probabilities
	//
	double p = 1. / parents.size();
	for (unsigned i = parents.size(); i--;)
		parents[ i ].selProb = p;

	//
	// clear number of copies and select elitists
	//
	parents.selectInit();
	selectElitists(parents, numElitists);

	//
	// select individuals by roulette wheel selection
	//
	selectRouletteWheel(parents, numElitists);
}

//===========================================================================
//
// no ( real ) correction for elitists
//
void Population::selectTournament(Population& parents,
								  unsigned q,
								  unsigned numElitists)
{
	unsigned i, j;
	Individual* ind;

	//
	// clear number of copies and select elitists
	//
	parents.selectInit();
	selectElitists(parents, numElitists);

	//
	// clear number of copies once again
	//
	for (i = parents.size(); i--;)
		parents[ i ].numCopies = 0;

	//
	// loop for all individuals to be replaced
	//
	for (i = numElitists; i < size();) {
		//
		// first candidate for tournament
		//
		ind = &parents.random();

		//
		// loop for tournament size - 1
		//
		for (j = 1; j < q; j++)
			//
			// select competitor and save the winner
			//
			ind = &parents.best(*ind, parents.random());

		//
		// copy winner of tournament to next generation
		// if an elitist is selected the first time then skip it
		//
		if (! ind->elitist || ind->numCopies > 0)
			(*this)[ i++ ] = *ind;
		ind->numCopies++;
	}
}

//===========================================================================
//
// no ( real ) correction for elitists
//
void Population::selectLinearRankingWhitley(Population& parents,
		double a,
		unsigned numElitists)
{
	unsigned i, j;

	//
	// clear number of copies and select elitists
	//
	parents.selectInit();
	selectElitists(parents, numElitists);

	//
	// clear number of copies once again
	//
	for (i = parents.size(); i--;)
		parents[ i ].numCopies = 0;

	//
	// sort parents (only pointers)
	//
	vector< Individual * > indvec(parents);
	sortIndividuals(indvec);

	double b = 2 * (a - 1);
	double c = a * a;
	double d = 2 * b;

	for (i = numElitists; i < size();) {
		j = unsigned(indvec.size() *
					 (a - sqrt(c - Rng::uni(0, d))) / b);

		//
		// if an elitist is selected the first time then skip it
		//
		if (! indvec[ j ]->elitist || indvec[ j ]->numCopies > 0)
			(*this)[ i++ ] = *indvec[ j ];
		indvec[ j ]->numCopies++;
	}
}

//===========================================================================
//
// EP-style Tournament Selection according to D. B. Fogel
//
bool Population::greaterScoreAscending(Individual*const& i1, Individual*const&
									   i2)
{
	if (i1->scaledFitness > i2->scaledFitness) return true;
	if (i1->scaledFitness < i2->scaledFitness) return false;
	if (i1->fitness < i2->fitness) return true;
	return false;
}

bool Population::greaterScoreDescending(Individual*const& i1, Individual*const&
										i2)
{
	if (i1->scaledFitness > i2->scaledFitness) return true;
	if (i1->scaledFitness < i2->scaledFitness) return false;
	if (i1->fitness > i2->fitness) return true;
	return false;
}

//  void Population::selectEPTournament( Population& parents,
//                                       unsigned q )
//  {
//    unsigned opponent;

//    Population pop( *this );
//    pop.append( parents );

//    // perform tournament
//    for( unsigned i = 0; i < pop.size(); i++ ) {
//      //cout << pop[ i ].scaledFitness << " " << pop[ i ].fitnessValue() << endl;
//      pop[ i ].scaledFitness = 0;
//      for( unsigned j = 0; j < q; j++ ) {
//        opponent = Rng::discrete( 0, pop.size( ) - 1 );
//        if( ascending ) {
//          if( pop[ i ].fitnessValue( ) <= pop[ opponent ].fitnessValue( ) )
//            pop[ i ].scaledFitness++;
//        } else {
//          if( pop[ i ].fitnessValue( ) >= pop[ opponent ].fitnessValue( ) )
//            pop[ i ].scaledFitness++;
//        }
//      }
//    }

//    // sort parents and offspring
//    vector< Individual * > indvec( pop );
//    std::sort( indvec.begin( ), indvec.end( ),
//               ascending
//  	       ? Population::greaterScoreAscending
//  	       : Population::greaterScoreDescending );
//    for( unsigned j = 0; j < size(); j++ )
//      ( *this )[ j ] = *indvec[ j ];
//    /*
//    for( unsigned j = 0; j < pop.size( ); j++ )
//      cout << indvec[ j ]->fitnessValue( ) << " " << indvec[ j ]->scaledFitness << endl;
//    */
//  }

void Population::selectEPTournament(Population& offspring,
									unsigned q)
{
	Population pop(*this);

	unsigned i;
	unsigned opponent;

	unsigned poolSize = this->size() + offspring.size();

	vector< Individual * > indvec(poolSize);

	for (i = 0; i < this->size(); i++) indvec[i] = &(pop[ i ]);
	for (i = 0; i < offspring.size(); i++) indvec[this->size() + i] = &(offspring[ i ]);

	// perform tournament
	for (i = 0; i < poolSize; i++) {
		indvec[ i ]->scaledFitness = 0;
		for (unsigned j = 0; j < q; j++) {
			opponent = Rng::discrete(0, poolSize - 1);
			if (ascending) {
				if (indvec[i]->fitnessValue() <= indvec[ opponent ]->fitnessValue())
					indvec[ i ]->scaledFitness++;
			}
			else {
				if (indvec[ i ]->fitnessValue() >= indvec[ opponent ]->fitnessValue())
					indvec[ i ]->scaledFitness++;
			}
		}
	}

	// sort parents and offspring
	std::sort(indvec.begin(), indvec.end(),
			  ascending
			  ? Population::greaterScoreAscending
			  : Population::greaterScoreDescending);

	for (unsigned j = 0; j < size(); j++)
		(*this)[ j ] = *indvec[ j ];
}

//===========================================================================

/*
void replaceUnconditional( Population& pop )
{
    Index index( 0, numElements-1 );
    index.permute( );

    for( unsigned i = Shark::min( numElements, pop.numElements ); i--; )
        *elements[ index[ i ] ] = *pop.elements[ i ];
}

//===========================================================================

void replaceConditional( Population& pop )
{
    Index index( 0, numElements-1 );
    index.permute( );

    for( unsigned i = Shark::min( numElements, pop.numElements ); i--; )
        *elements[ index[ i ] ] = best( *elements[ index[ i ] ],
				        *pop.elements[ i ] );
}
*/

//===========================================================================

double Population::minFitness() const
{
	double f = 0;

	if (size()) {
		f = (*this)[ 0 ].fitness;
		for (unsigned i = 1; i < size(); i++)
			f = Shark::min(f, (*this)[ i ].fitness);
	}

	return f;
}

//===========================================================================

double Population::maxFitness() const
{
	double f = 0;

	if (size()) {
		f = (*this)[ 0 ].fitness;
		for (unsigned i = 1; i < size(); i++)
			f = Shark::max(f, (*this)[ i ].fitness);
	}

	return f;
}

//===========================================================================

double Population::meanFitness() const
{
	double f = 0;

	if (size()) {
		for (unsigned i = size(); i--;)
			f += (*this)[ i ].fitness;
		f /= size();
	}

	return f;
}

//===========================================================================

double Population::stdDevFitness() const
{
	double m = 0;
	double s = 0;

	if (size()) {
		for (unsigned i = size(); i--;) {
			m += (*this)[ i ].fitness;
			s += (*this)[ i ].fitness * (*this)[ i ].fitness;
		}
		m /= size();
		s /= size();
		s -= m * m;
	}

	//
	// due to rounding problems the value of 's' may be negative
	// in this case 's' is supposed to be zero
	//
	return s > 0 ? sqrt(s) : 0;
}

//===========================================================================

void Population::setAge(unsigned a)
{
	for (unsigned i = size(); i--;)
		(*this)[ i ].setAge(a);
}

void Population::incAge()
{
	for (unsigned i = size(); i--;)
		(*this)[ i ].incAge();
}

//===========================================================================

bool Population::operator == (const Population& pop) const
{
	if (size() == pop.size()) {
		for (unsigned i = 0; i < size(); ++i)
			if ((*this)[ i ] != pop[ i ]) return false;

		return index     == pop.index     &&
			   subPop    == pop.subPop    &&
			   ascending == pop.ascending &&
			   spinOnce  == pop.spinOnce;
	}

	return false;
}

bool Population::operator < (const Population& pop) const
{
	if (size() == pop.size()) {
		bool less = false;

		for (unsigned i = 0; i < size(); ++i)
			if (pop[ i ] < (*this)[ i ])
				return false;
			else if (! less && (*this)[ i ] < pop[ i ])
				less = true;

		return less/*                     &&
			   	       index     <= pop.index     &&
			   	       subPop    <= pop.subPop    &&
			   	       ascending <= pop.ascending &&
			   	       spinOnce  <= pop.spinOnce*/;
	}

	return size() < pop.size();
}


//===========================================================================

