//===========================================================================
/*!
 *  \file ElitistCMA.h
 *
 *  \par Copyright (c) 2006:
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


#ifndef _ElitistCMA_H_
#define _ElitistCMA_H_


#include <EALib/ChromosomeCMA.h>
#include <EALib/IndividualT.h>
#include <EALib/PopulationT.h>
#include <EALib/SearchAlgorithm.h>


/*! \brief Implements the elitist version of the CMA-ES
*
* The ElitistCMA class is a simple implementation
* of the elitist CMA. However, Most of the work
* is done by the ChromosomeCMA class.
*/
class ElitistCMA
{
public:
	//!
	//! \brief initialization of the elitist CMA
	//!
	//! \param  parent  parent individual with ChromosoleCMA as its first chromosome
	//! \param  sigma   initial step width
	//! \param  lambda  size of the offspring population
	//!
	static void init(IndividualCT<ChromosomeCMA>& parent, double sigma, int lambda = 1);

	//!
	//! \brief initialization of the elitist CMA with individual step sizes
	//!
	//! \param  parent  parent individual with ChromosoleCMA as its first chromosome
	//! \param  sigma   initial step width for each coordinate
	//! \param  lambda  size of the offspring population
	//!
	static void init(IndividualCT<ChromosomeCMA>& parent, const Array<double>& sigma, int lambda = 1);

	//!
	//! \brief Overwrite the offspring population with mutations of the parent
	//!
	//! \param  parent     parent individual
	//! \param  offspring  population of new offspring
	//!
	static void Mutate(IndividualCT<ChromosomeCMA>& parent, PopulationCT<ChromosomeCMA>& offspring);

	//!
	//! \brief Update of the CMA search strategy
	//!
	//! It is assumed that all individuals, that is, parent
	//! and offspring, have a valid fitness value.
	//!
	//! \param  parent     parent individual
	//! \param  offspring  offspring population
	//!
	static void SelectAndUpdateStrategyParameters(IndividualCT<ChromosomeCMA>& parent, PopulationCT<ChromosomeCMA>& offspring);
};

/*! \brief Elitist CMA-ES that implements the interface EvolutionaryAlgorithm. */
class CMAElitistSearch : public EvolutionaryAlgorithm<double*>
{
public:
	CMAElitistSearch();
	~CMAElitistSearch();

	inline const PopulationCT<ChromosomeCMA>* parents() const { return m_parents; }
	inline const PopulationCT<ChromosomeCMA>* offspring() const { return m_offspring; }

	void init(ObjectiveFunctionVS<double>& fitness, unsigned int lambda = 1);
	void init(ObjectiveFunctionVS<double>& fitness, const Array<double>& start, double stepsize, unsigned int lambda = 1);
	void run();
	void bestSolutions(std::vector<double*>& points);
	void bestSolutionsFitness(Array<double>& fitness);

protected:
	ObjectiveFunctionVS<double>* m_fitness;
	PopulationCT<ChromosomeCMA>* m_parents;
	PopulationCT<ChromosomeCMA>* m_offspring;
	bool m_bIsParentFitnessValid;
};


#endif
