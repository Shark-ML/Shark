//===========================================================================
/*!
 *  \file Individual.h
 *
 *  \brief Base class for individuals in a population
 *
 *  \author  Martin Kreutz
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
 *
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
//===========================================================================


#ifndef __INDIVIDUAL_H
#define __INDIVIDUAL_H

#include <EALib/ChromosomeT.h>
#include <EALib/PVMinterface.h>


//===========================================================================
//!
//! \brief Base class for individuals in a population
//!
class Individual : protected std::vector< Chromosome * >
{
public:
	explicit Individual(unsigned noChromosomes = 0);
	Individual(unsigned, const Chromosome&);
	Individual(const Chromosome&);
	Individual(const Chromosome&,
			   const Chromosome&);
	Individual(const Chromosome&,
			   const Chromosome&,
			   const Chromosome&);
	Individual(const Chromosome&,
			   const Chromosome&,
			   const Chromosome&,
			   const Chromosome&);
	Individual(const Chromosome&,
			   const Chromosome&,
			   const Chromosome&,
			   const Chromosome&,
			   const Chromosome&);
	Individual(const Chromosome&,
			   const Chromosome&,
			   const Chromosome&,
			   const Chromosome&,
			   const Chromosome&,
			   const Chromosome&);
	Individual(const Chromosome&,
			   const Chromosome&,
			   const Chromosome&,
			   const Chromosome&,
			   const Chromosome&,
			   const Chromosome&,
			   const Chromosome&);
	Individual(const Chromosome&,
			   const Chromosome&,
			   const Chromosome&,
			   const Chromosome&,
			   const Chromosome&,
			   const Chromosome&,
			   const Chromosome&,
			   const Chromosome&);
	Individual(const std::vector< Chromosome* >&);
	Individual(const Individual&);
	virtual ~Individual();

	unsigned size() const
	{
		return static_cast< const std::vector< Chromosome * > * >(this)->size();
	}

	unsigned totalSize() const;
	double   fitnessValue() const
	{
		return fitness;
	}
	double   selectionProbability() const
	{
		return selProb;
	}
	bool     isFeasible() const
	{
		return feasible;
	}
	unsigned numberOfCopies() const
	{
		return numCopies;
	}
	bool     isElitist() const
	{
		return elitist;
	}

	void     setFitness(double fit)
	{
		fitness = scaledFitness = fit;
	}
	void     setFeasible(bool f)
	{
		feasible = f;
	}
	void     setSelectionProbability(double ps)
	{
		selProb = ps;
	}

	Chromosome& operator [ ](unsigned i)
	{
		RANGE_CHECK(i < size())
		return *(*(begin() + i));
	}

	const Chromosome& operator [ ](unsigned i) const
	{
		RANGE_CHECK(i < size())
		return *(*(begin() + i));
	}

	//=======================================================================

	//
	// Changed by Marc Toussaint & Stefan Wiegand at 20.11.2002
	// refer to Individual& Individual::operator = ( const Individual& );
	// in Individual.cpp

	/*! The Chromosome method 'registerIndividual(Ind&,int)' was added,
	  refer also to 'Chromosome.h'.*/

	Individual& operator = (const Individual&);

	//=======================================================================

	void replace(unsigned i, const Chromosome& chrom);
	void insert(unsigned i, const Chromosome& chrom);
	void append(const Chromosome& chrom);
	void remove (unsigned i);
	void remove (unsigned from, unsigned to);

	void setAge(unsigned a = 0)
	{
		age = a;
	}
	void incAge()
	{
		++age;
	}
	unsigned getAge() const
	{
		return age;
	}

	//=======================================================================

	//  Added by Marc Toussaint & Stefan Wiegand at 20.11.2002
	/*! Appends Chromosomes to the Individual, refer also to Chromosome.h. */

	template<class ChromosomeTemplate> void append(const ChromosomeTemplate& chrom) {
		ChromosomeTemplate* newChrom = new ChromosomeTemplate(chrom);
		std::vector< Chromosome * >::push_back(newChrom);
		newChrom->registerIndividual(*this, size() - 1);
	}


	// Added by Stefan Wiegand at 20.11.2002
	/*! Interface for a buffer that stores how many learning iterations
	    an individual conducts per generation */

	void setLearnTime(unsigned lt = 0)
	{
		learnTime = lt;
	}
	unsigned getLearnTime() const
	{
		return learnTime;
	}


	// Added by Stefan Wiegand at 24.02.2003
	/*! Interface for a flag that indicates the neccessity of an individual to become evaluated */

	void setEvaluationFlag()
	{
		evalFlg = true;
	}
	void clearEvaluationFlag()
	{
		evalFlg = false;
	}
	bool needEvaluation() const {
		return evalFlg;
	}

	bool operator == (const Individual&) const;
	bool operator < (const Individual&) const;

	double   getFitness() const
	{
		return fitness;
	}

	double   getScaledFitness() const
	{
		return scaledFitness;
	}

	void     setScaledFitness(double sf)
	{
		scaledFitness = sf;
	}

	void     setSelProb(double sp)
	{
		selProb = sp;
	}

	double   getSelProb() const
	{
		return selProb;
	}

	void     setNumCopies(unsigned snc)
	{
		numCopies = snc;
	}

	unsigned getNumCopies() const
	{
		return numCopies;
	}

	void     setEvalFlg(bool ef)
	{
		evalFlg = ef;
	}

	bool     getEvalFlg() const
	{
		return evalFlg;
	}

	bool     getFeasible() const
	{
		return feasible;
	}

	void     setElitist(bool e)
	{
		elitist = e;
	}

	bool     getElitist() const
	{
		return elitist;
	}


	//=======================================================================

	//
	// Added by Marc Toussaint & Stefan Wiegand at 20.11.2002
	//

	/*! Part of PVM-send routine for individuals */
	int pvm_pkind();

	/*! Part of PVM-rceive routine for individuals */
	int pvm_upkind();


	//=======================================================================

protected:
	double   fitness;
	double   scaledFitness;
	bool     evalFlg;
	bool     feasible;
	double   selProb;
	unsigned numCopies;
	bool     elitist;
	unsigned age;

	// Added by Stefan Wiegand at 20.11.2002
	unsigned learnTime;

#ifndef __NO_GENERIC_IOSTREAM
	friend std::ostream& operator << (std::ostream& os,
									  const Individual& ind)
	{
		os << "Individual(" << ind.size() << ")\n"
		<< ind.fitness       << '\n'
		<< ind.scaledFitness << '\n'
		<< ind.evalFlg       << '\n'
		<< ind.feasible      << '\n'
		<< ind.selProb       << '\n'
		<< ind.numCopies     << '\n'
		<< ind.elitist       << '\n'
		<< ind.age           << '\n'
		<< ind.learnTime     << '\n';

		for (unsigned i = 0; i < ind.size(); ++i)
			os << '\n' << ind[ i ];
		os << std::endl;
		return os;
	}

	friend std::istream& operator >> (std::istream& is, Individual& ind)
	{
		unsigned i, indSize(0);
		std::string s, t;

		is >> s;
		is.get();    // skip end of line

		if (is.good() &&
				s.substr(0, 11) == "Individual(" &&
				s.find(')') != std::string::npos) {

			// Extract the size indication from the string:
			t = s.substr(s.find('(') + 1, s.find(')') - s.find('(') - 1);
			indSize = atoi(t.c_str());

			// Adapt size of Individual:
			ind.resize(indSize);

			is >> ind.fitness
			>> ind.scaledFitness
			>> ind.evalFlg
			>> ind.feasible
			>> ind.selProb
			>> ind.numCopies
			>> ind.elitist
			>> ind.age
			>> ind.learnTime;
			for (i = 0; i < ind.size(); ++i)
				is >> ind[ i ];
		}
		return is;
	}
#endif // !__NO_GENERIC_IOSTREAM

	friend class Population;
	friend class PopulationMOO;
	friend class IndividualMOO;
};

//===========================================================================

#endif /* !__INDIVIDUAL_H */

