//===========================================================================
/*!
 *  \file Population.h
 *
 *  \brief Base class for populations of individuals in an evolutionary algorithm
 *
 *  \date    01.01.1995
 *
 *  \par Copyright (c) 1995-2003:
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
 *  This file is part of EALib. This library is free software;
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




#ifndef __POPULATION_H
#define __POPULATION_H

#include <EALib/Individual.h>


//!
//! \brief Base class for populations of individuals in an evolutionary algorithm
//!
class Population : protected std::vector< Individual * >
{
public:
	Population();
	explicit Population(unsigned);
	Population(const Individual&);
	Population(unsigned, const Individual&);
	Population(unsigned, const Chromosome&);
	Population(unsigned, const Chromosome&,
			   const Chromosome&);
	Population(unsigned, const Chromosome&,
			   const Chromosome&,
			   const Chromosome&);
	Population(unsigned, const Chromosome&,
			   const Chromosome&,
			   const Chromosome&,
			   const Chromosome&);
	Population(unsigned, const Chromosome&,
			   const Chromosome&,
			   const Chromosome&,
			   const Chromosome&,
			   const Chromosome&);
	Population(unsigned, const Chromosome&,
			   const Chromosome&,
			   const Chromosome&,
			   const Chromosome&,
			   const Chromosome&,
			   const Chromosome&);
	Population(unsigned, const Chromosome&,
			   const Chromosome&,
			   const Chromosome&,
			   const Chromosome&,
			   const Chromosome&,
			   const Chromosome&,
			   const Chromosome&);
	Population(unsigned, const Chromosome&,
			   const Chromosome&,
			   const Chromosome&,
			   const Chromosome&,
			   const Chromosome&,
			   const Chromosome&,
			   const Chromosome&,
			   const Chromosome&);
	Population(unsigned, const std::vector< Chromosome * >&);
	Population(const Population&);
	Population(const std::vector< Individual * >&);
	virtual ~Population();

	unsigned          size() const
	{
		return static_cast< const std::vector< Individual * > * >(this)->size();
	}

	void              resize(unsigned n);

	void              setMaximize()
	{
		ascending = false;
	}
	void              setMinimize()
	{
		ascending = true;
	}

	void              spinWheelOneTime()
	{
		spinOnce  = true;
	}
	void              spinWheelMultipleTimes()
	{
		spinOnce  = false;
	}

	bool              ascendingFitness() const
	{
		return ascending;
	}

	Individual&       operator [ ](unsigned i)
	{
		RANGE_CHECK(i < size())
		return *(*(begin() + i));
	}

	const Individual& operator [ ](unsigned i) const
	{
		RANGE_CHECK(i < size())
		return *(*(begin() + i));
	}

	Population        operator()(unsigned from, unsigned to) const
	{
		RANGE_CHECK(from <= to && to < size())
		return Population(std::vector< Individual * >(
							  begin() + from, begin() + to + 1));
	}

	Population&       operator = (const Individual& ind);
	Population&       operator = (const Population& pop);

	std::vector< const Chromosome* > matingPool(unsigned chrom,
			unsigned from = 0,
			unsigned to   = 0) const;

	void              swap(unsigned i, unsigned j)
	{
		RANGE_CHECK(i < size() && j < size())
		std::swap(*(begin() + i), *(begin() + j));
	}

	void              sort();
	void              shuffle();

	//
	// note that individual pop[ i ] may change type and size
	// in contrast to pop[ i ] = ind, where type and size of pop[ i ] and ind
	// must be the same
	//
	void              replace(unsigned i, const Individual& ind);
	void              replace(unsigned i, const Population& pop);
	void              insert(unsigned i, const Individual& ind);
	void              insert(unsigned i, const Population& pop);
	void              append(const Individual& ind);
	void              append(const Population& pop);
	void              remove (unsigned i);
	void              remove (unsigned from, unsigned to);

	//
	// exchange pointer to individuals
	//
	void              exchange(Population& pop);

	unsigned          bestIndex() const;
	unsigned          worstIndex() const;

	//
	// it is possible that there are several "best" individuals in the
	// population, i.e., several individuals with the same "best" fitness
	// value.
	// oneOfBest() randomly returns one of these "best" individuals.
	//
	Individual&       oneOfBest();
	const Individual& oneOfBest() const;

	Individual&       best()
	{
		return (*this)[ bestIndex()];
	}
	const Individual& best() const
	{
		return (*this)[ bestIndex()];
	}

	Individual&       worst()
	{
		return (*this)[ worstIndex()];
	}
	const Individual& worst() const
	{
		return (*this)[ worstIndex()];
	}

	Individual&       random()
	{
		return (*this)[ Rng::discrete(0, size()-1)];
	}
	const Individual& random() const
	{
		return (*this)[ Rng::discrete(0, size()-1)];
	}

	void              reproduce(Population& parents,
								unsigned nelitists = 0)
	{
		selectUniformRanking(parents, nelitists);
	}

	void              linearDynamicScaling(std::vector< double >& window,
										   unsigned long t);
	Individual&       selectOneIndividual();
	void              selectMuLambda(Population& parents,
									 unsigned nelitists = 0);
	void              selectMuLambdaKappa(Population& parents,
										  unsigned lifespan = 1,
										  unsigned adolescence = 0);
	void              selectProportional(Population& parents,
										 unsigned nelitists = 0);
	void              selectLinearRanking(Population& parents,
										  double etaMax = 1.1,
										  unsigned nelitists = 0);
	void              selectUniformRanking(Population& parents,
										   unsigned nelitists = 0);
	void              selectTournament(Population& parents,
									   unsigned q = 2,
									   unsigned nelitists = 0);
	void              selectLinearRankingWhitley(Population& parents,
			double a = 1.1,
			unsigned nelitists = 0);
	void              selectEPTournament(Population& parents, unsigned q);
	void              replaceUnconditional(Population& parents);
	void              replaceConditional(Population& parents);
	void              replaceRanking(Population& parents);
	void              replaceRankingConditional(Population& parents);

	double            minFitness() const;
	double            maxFitness() const;
	double            meanFitness() const;
	double            stdDevFitness() const;

	void              setAge(unsigned a = 0);
	void              incAge();

	bool operator == (const Population&) const;
	bool operator < (const Population&) const;

	void              setIndex(unsigned i)
	{
		index = i;
	}

	unsigned          getIndex() const
	{
		return index;
	}

	void              setSubPop(bool sub)
	{
		subPop = sub;
	}

	bool              getSubPop() const
	{
		return subPop;
	}

	bool              getSpinOnce() const
	{
		return spinOnce;
	}

	//=======================================================================

	//
	// Added by Stefan Wiegand at 20.11.2002
	//

	/*! Part of PVM-send routine for populations */
	int pvm_pkpop();

	/*! Part of PVM-receive routine for populations */
	int pvm_upkpop();


	//=======================================================================

protected:
	unsigned index;
	bool     subPop;
	bool     ascending;
	bool     spinOnce;

	static bool lessFitness(Individual*const&, Individual*const&);
	static bool greaterFitness(Individual*const&, Individual*const&);

	static bool       greaterScoreAscending(Individual*const& i1, Individual*const& i2);
	static bool       greaterScoreDescending(Individual*const& i1, Individual*const& i2);

	Individual& best(Individual&, Individual&) const;
	Individual& worst(Individual&, Individual&) const;

	void        sortIndividuals(std::vector< Individual * >&);

	void        selectInit();
	void        selectElitists(Population&, unsigned);
	void        selectRouletteWheel(Population&, unsigned);

#ifndef __NO_GENERIC_IOSTREAM
	friend std::ostream& operator << (std::ostream& os,
									  const Population& pop)
	{
		os << "Population(" << pop.size() << ")\n"
		<< pop.index     << '\n'
		<< pop.subPop    << '\n'
		<< pop.ascending << '\n'
		<< pop.spinOnce  << std::endl;
		for (unsigned i = 0; i < pop.size(); ++i)
			os << pop[ i ];
		return os;
	}

	friend std::istream& operator >> (std::istream& is,
									  Population& pop)
	{
		unsigned i, popSize(0);
		std::string s, t;

		is >> s;
		is.get();    // skip end of line

		if ((!pop.subPop || pop.size() == 0) && is.good() &&
				s.substr(0, 11) == "Population(" &&
				s.find(')') != std::string::npos) {

			// Extract the size indication from the string:
			t = s.substr(s.find('(') + 1, s.find(')') - s.find('(') - 1);
			popSize = atoi(t.c_str());

			// Adapt size of Individual:
			pop.resize(popSize);

			is >> pop.index
			>> pop.subPop
			>> pop.ascending
			>> pop.spinOnce;

			for (i = 0; i < pop.size(); ++i)
				is >> pop[ i ];
		}

		return is;
	}
#endif // !__NO_GENERIC_IOSTREAM

};

//===========================================================================

#endif /* !__POPULATION_H */

