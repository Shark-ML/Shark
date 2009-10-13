/*! ======================================================================
 *
 *  \file SelectionMOO.h
 *
 *  \brief Several classes and interfaces for (indicator-based) multi-objective selection.
 * 
 *  \author Thomas Voß <thomas.voss@rub.de>
 *
 *  \par 
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
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
#ifndef _SELECTIONMOO_H_
#define _SELECTIONMOO_H_

typedef double * doublep;

class Individual;
class IndividualMOO;
class PopulationMOO;

#include <vector>
#include <SharkDefs.h>


typedef std::pair<std::vector<unsigned>, std::vector<unsigned> > MinMax;


extern MinMax calc_boundary_elements(PopulationMOO & pop,
		const std::vector<unsigned> & popView,
		unsigned noObjectives,
		bool unpenalizedFitness);


//!
//! \brief Sorting of a MOO population
//!
//! \par
//! A predicate for std::sort. Sorts a population according to the objective m_objective.
//!
struct ObjectiveSort
{
	ObjectiveSort(bool unpenalizedFitness, unsigned objective) : m_bUnpenalizedFitness(unpenalizedFitness),
			m_objective(objective),
			pop(0)
	{}

	bool operator()(Individual * a, Individual * b);
	bool operator()(unsigned a, unsigned b);

	bool m_bUnpenalizedFitness;
	unsigned m_objective;
	PopulationMOO * pop;
};

//!
//! \brief Level of non-dominance
//!
//!
//! Calculates the level of non-dominance of each member of the population pop.
//!
struct FastNonDominatedSort
{
	FastNonDominatedSort(bool unpenalizedFitness = false)
	: m_bUnpenalizedFitness(unpenalizedFitness)
	{ }

	void operator()(PopulationMOO & pop);

	bool m_bUnpenalizedFitness; //*< If true, the unpenalized fitness is used for calculation. Otherwise, the penalized fitness is used. */
};


//!
//! \brief Crowding distance
//!
//! Calculates the crowding distance for each member of the population pop.
//!
struct CrowdingDistance
{
	CrowdingDistance(bool unpenalizedFitness = false) : m_bUnpenalizedFitness(unpenalizedFitness)
	{}

	void operator()(PopulationMOO & pop);

	bool m_bUnpenalizedFitness; //*< If true, the unpenalized fitness is used for calculation. Otherwise, the penalized fitness is used. */
};

//!
//! \brief Abstract binary quality indicator for the comparison of individuals
//!
//! This class models a binary quality indicator, i.e. a function that calculates a real value for two sets \f$ A,B \subset R^m\f$. More formally, a binary quality indicator I is given by:
//! \f[
//!    I: P(R^m) \times P(R^m) \to R\enspace.
//! \f]
//!
class IBinaryQualityIndicator
{
public:
	IBinaryQualityIndicator(bool unpenalizedFitness = false, bool ascending = true) : m_bUnpenalizedFitness(unpenalizedFitness),
			m_bAscending(ascending)
	{}

	virtual ~IBinaryQualityIndicator() {};
	/*!
	   Models the normal function of a binary quality indicator.
	 */
	virtual double operator()(const std::vector<std::vector<double> > & a,
							  const std::vector<std::vector<double> > & b
							 ) = 0;

	/*!
	   Most often, binary quality indicator are applied within the indicator based selection strategy. Roughly spoken, given a population P and a binary quality indicator I, every individual p in P is evaluated according to I( P, P \ {p}). Apparently, optimizations might be possible for this special case. 
	   @param set a set of objective vectors
	   @param idx the index that needs to be evaluated
	 */
	virtual double operator()(const std::vector<std::vector<double> > & set, unsigned idx);

	bool m_bUnpenalizedFitness; //*< If true, the unpenalized fitness is used for calculation. Otherwise, the penalized fitness is used. */
	bool m_bAscending; /**< If true, a minimization goal is assumed. Otherwise, a maximization goal is assumed. */
};

//!
//! \brief Additive epsilon quality indicator
//!
//! A binary quality indicator. Given two sets \f$A,B \subset R^m\f$, the distance \f$\epsilon\f$ by that each member of B needs to be translated such that each member of A is weakly covered by B is calculated.
//!
class AdditiveEpsilonIndicator : public IBinaryQualityIndicator
{
public:
	AdditiveEpsilonIndicator(bool unpenalizedFitness = false) : IBinaryQualityIndicator(unpenalizedFitness)
	{}

	double operator()(const std::vector<std::vector<double> > & a,
					  const std::vector<std::vector<double> > & b
					 );

	double operator()(const std::vector<std::vector<double> > & set,
						  unsigned idx
						 );

};

/*!
 *
 * \brief Hypercolume quality indicator
 *
 * A binary quality indicator. The hypervolume \f$H(A)\f$ of a set
 * \f$A \subset R^m\f$ is the space exclusively covered by A
 * (severally covered regions are counted once. Given two sets \f$A,B \in R^m\f$,
 * the indicator calculates the difference H(A) - H(B). Refer to the following reference for further details.
 * <PRE>
 * author = {Nicola Beume and G\"unther Rudolph},
 * title = {Faster {S}-Metric Calculation By Considering Dominated Hypervolume as {Klee}'s Measure Problem},
 * booktitle = {IASTED International Conference on Computational Intelligence},
 * publisher = {ACTA Press},
 * pages = {231-236},
 * year = {2006},
 * </PRE>
 */
class HypervolumeIndicator : public IBinaryQualityIndicator
{
public:
	HypervolumeIndicator(bool unpenalizedFitness = false) : IBinaryQualityIndicator(unpenalizedFitness)
	{}

	double operator()(const std::vector<std::vector<double> > & a,
					  const std::vector<std::vector<double> > & b
					 );

	double operator()(const std::vector<std::vector<double> > & set, unsigned idx);

	unsigned m_noObjectives; //*< The number of objectives of the underlying multi-objective optimization problem.
	//unsigned m_noSqrtPoints; //*< The square root of the number of points. Stored for performance reasons.

//protected:
	void calcBoundingBox(doublep * pop, unsigned nPoints, double * regLow, double * regUp);

	/*
	int covers(double * cuboid, double * regionLow);
	int partCovers(double cuboid[], double regionUp[]);
	int containsBoundary(double * cub, double * regLow, int split);
	double getMeasure(double * regionLow, double * regionUp);
	int isPile(double * cuboid, double * regionLow, double * regionUp);
	int binaryToInt(int * bs);
	void intToBinary(int i, int * result);
	double computeTrellis(double * regLow, double * regUp, double * trellis);
	double getMedian(double * bounds, int length);

	void calcBoundingBox(doublep * pop, unsigned nPoints, double * regLow, double * regUp);

	double stream(double * regionLow,
				  double * regionUp,
				  doublep * points,
				  unsigned noPoints,
				  int split,
				  double cover);
	*/
};

/*!
 *
 * \brief Abstract selection based on binary indicators
 *
 * This class models the indicator based selection strategy proposed by Zitzler, Brockhoff and Thiele (2007) in
 * <PRE>
 * author = {Eckart Zitzler and Dimo Brockhoff and Lothar Thiele},
 * title = {The Hypervolume Indicator Revisited: {O}n The Design of {Pareto}-Compliant Indicators Via Weighted Integration},
 * year = {2007},
 * booktitle = {Fourth International Conference on Evolutionary Multi-Criterion Optimization (EMO 2007)},
 * publisher = {Springer-Verlag},
 * volume = 4403,
 * series = {LNCS}
 * </PRE>
*/
template<typename Indicator_T>
class IndicatorBasedSelectionStrategy
{
public:
	IndicatorBasedSelectionStrategy(bool unpenalizedFitness = false, bool ascending = true)
		: m_bUnpenalizedFitness(unpenalizedFitness)
		, m_bAscending(ascending)
	{
		m_binaryQualityIndicator.m_bUnpenalizedFitness = m_bUnpenalizedFitness;
	}

	void operator()(PopulationMOO & pop);

	bool m_bUnpenalizedFitness; /**< If true, the unpenalized fitness is used for calculation. Otherwise, the penalized fitness is used. */
	bool m_bAscending; /**< If true, a minimization goal is assumed. Otherwise, a maximization goal is assumed. */

	Indicator_T m_binaryQualityIndicator; /**< The indicator that is used for the selection of individuals. */
};

#endif /* __SELECTIONMOO_H__ */

