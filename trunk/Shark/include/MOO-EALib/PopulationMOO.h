/*! ======================================================================
 *
 *  \file PopulationMOO.h
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
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
 *
 *
 *
 *  <BR><HR>
 *  This file is part of the Shark. This library is free software;
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

#ifndef _POPULATIONMOO_H_
#define _POPULATIONMOO_H_

#include <SharkDefs.h>
#include <Rng/DiscreteUniform.h>
#include <EALib/Population.h>
#include <EALib/Individual.h>
#include <MOO-EALib/IndividualMOO.h>
#include <MOO-EALib/SelectionMOO.h>
#include <MOO-EALib/ArchiveMOO.h>


//!
//! \brief Population of individuals with vector-valued fitness
//!
class PopulationMOO : public Population
{
public:

	// ---------------------------------------------------
	// Constructor
	// ---------------------------------------------------
	// TO-PM-001
	PopulationMOO();
	// TO-PM-002
	PopulationMOO(unsigned);
	// TO-PM-003
	PopulationMOO(const IndividualMOO&);
	// TO-PM-004
	//PopulationMOO( const Individual& );
	// TO-PM-005
	PopulationMOO(unsigned, const IndividualMOO&);
	// TO-PM-006
	//PopulationMOO( unsigned, const Individual& );
	// TO-PM-007
	PopulationMOO(unsigned, const Chromosome&);
	// TO-PM-008
	PopulationMOO(unsigned, const Chromosome&,
				  const Chromosome&);
	// TO-PM-009
	PopulationMOO(unsigned, const Chromosome&,
				  const Chromosome&,
				  const Chromosome&);
	// TO-PM-010
	PopulationMOO(unsigned, const Chromosome&,
				  const Chromosome&,
				  const Chromosome&,
				  const Chromosome&);
	// TO-PM-011
	PopulationMOO(unsigned, const Chromosome&,
				  const Chromosome&,
				  const Chromosome&,
				  const Chromosome&,
				  const Chromosome&);
	// TO-PM-012
	PopulationMOO(unsigned, const Chromosome&,
				  const Chromosome&,
				  const Chromosome&,
				  const Chromosome&,
				  const Chromosome&,
				  const Chromosome&);
	// TO-PM-013
	PopulationMOO(unsigned, const Chromosome&,
				  const Chromosome&,
				  const Chromosome&,
				  const Chromosome&,
				  const Chromosome&,
				  const Chromosome&,
				  const Chromosome&);
	// TO-PM-014
	PopulationMOO(unsigned, const Chromosome&,
				  const Chromosome&,
				  const Chromosome&,
				  const Chromosome&,
				  const Chromosome&,
				  const Chromosome&,
				  const Chromosome&,
				  const Chromosome&);
	// TO-PM-015
	PopulationMOO(unsigned, const std::vector< Chromosome * >&);
	// TO-PM-016
	PopulationMOO(const PopulationMOO&);
	// TO-PM-017
	PopulationMOO(const Population&);

	// ---------------------------------------------------
	// Destructor
	// ---------------------------------------------------
	// TO-PM-018
	~PopulationMOO();

	// ---------------------------------------------------
	// Structure
	// ---------------------------------------------------
	// TO-PM-030
	unsigned size() const;
	// TO-PM-031
	std::vector< Individual* >::iterator begin();

	// TO-PM-032
	std::vector< Individual* >::iterator end();

	// TO-PM-033
	void resize(unsigned);

	// ---------------------------------------------------
	// Operator
	// ---------------------------------------------------
	// TO-PM-040
	IndividualMOO& operator [ ](unsigned);
	// TO-PM-041
	const IndividualMOO& operator [ ](unsigned) const;
	// TO-PM-042
	PopulationMOO& operator = (const IndividualMOO&);
	// TO-PM-043
	PopulationMOO& operator = (const Individual&);
	// TO-PM-044
	PopulationMOO& operator = (const PopulationMOO&);
	// TO-PM-045
	PopulationMOO& operator = (const Population&);
	// TO-PM-046
	bool operator == (const PopulationMOO&) const;

	// ---------------------------------------------------
	// Internal variables
	// ---------------------------------------------------
	// TO-PM-052
	void setAscending(bool);
	// TO-PM-057
	void setSpinOnce(bool);

	// ---------------------------------------------------
	// Structure changing
	// ---------------------------------------------------
	// TO-PM-070
	void replace(unsigned, const Individual&);
	// TO-PM-071
	void replace(unsigned, const IndividualMOO&);
	// TO-PM-072
	void replace(unsigned, const Population&);
	// TO-PM-073
	void replace(unsigned, const PopulationMOO&);
	// TO-PM-074
	void insert(unsigned, const Individual&);
	// TO-PM-075
	void insert(unsigned, const IndividualMOO&);
	// TO-PM-076
	void insert(unsigned, const Population&);
	// TO-PM-077
	void insert(unsigned, const PopulationMOO&);
	// TO-PM-078
	void append(const Individual&);
	// TO-PM-079
	void append(const IndividualMOO&);
	// TO-PM-080
	void append(const Population&);
	// TO-PM-081
	void append(const PopulationMOO&);
	// TO-PM-082
	void remove(unsigned);
	// TO-PM-083
	void remove(unsigned, unsigned);

	// ---------------------------------------------------
	// Order changing
	// ---------------------------------------------------
	// TO-PM-093-b
	void sortIndividuals(std::vector< IndividualMOO* >&);
	// TO-PM-094
	void sort();
	// TO-PM-095
	void shuffle();
	// TO-PM-098
	void exchange(Population&);
	// TO-PM-099
	void exchange(PopulationMOO&);

	// ---------------------------------------------------
	// Selection of Individuals 1
	// ---------------------------------------------------
	// TO-PM-102
	IndividualMOO& oneOfBest();
	// TO-PM-103
	const IndividualMOO& oneOfBest() const;
	// TO-PM-104
	IndividualMOO& best();
	// TO-PM-105
	const IndividualMOO& best() const;
	// TO-PM-106
	IndividualMOO& worst();
	// TO-PM-107
	const IndividualMOO& worst() const;
	// TO-PM-108
	IndividualMOO& random();
	// TO-PM-109
	const IndividualMOO& random() const;
	// TO-PM-110
	IndividualMOO& best(IndividualMOO&, IndividualMOO&) const;
	// TO-PM-111
	IndividualMOO& best(unsigned, unsigned);
	// TO-PM-112
	IndividualMOO& worst(IndividualMOO&, IndividualMOO&) const;
	// TO-PM-113
	IndividualMOO& worst(unsigned, unsigned);

	//! fill population nondom with the non-dominated
	//! individuals according to either penalized or
	//! unpenalized fitness
	void getNonDominated(PopulationMOO& nondom, bool unpenalized = false);

	// ---------------------------------------------------
	// Control internal variables in class IndividualMOO
	// ---------------------------------------------------
	// TO-PM-200
	void setNoOfObj(unsigned);
	// TO-PM-201
	void setMOOFitness(double);
	// TO-PM-202
	void setMOORank(unsigned);
	// TO-PM-203
	void setMOOShare(double);

	// ---------------------------------------------------
	// Rank
	// ---------------------------------------------------
	// TO-PM-210
	int Dominate(unsigned i, unsigned j, bool unpenalized = false);
	// TO-PM-211
	void MOGAFonsecaRank();
	// TO-PM-212
	void MOGAGoldbergRank();
	// TO-PM-213
	void NSGAIIRank();

	// ---------------------------------------------------
	// Fitness from other values
	// ---------------------------------------------------
	// TO-PM-250
	void MOORankToFitness();
	// TO-PM-251
	void aggregation(const std::vector< double >&);
	// TO-PM-252
	void simpleSum();
	// TO-PM-253
	void simpleTransferFitness(unsigned);

	// ---------------------------------------------------
	// selection of Individuals 2
	// ---------------------------------------------------
	// TO-PM-400 ( protected in class Population )
	void selectInit();
	// TO-PM-401 ( protected in class Population )
	void selectElitists(PopulationMOO&, unsigned);
	// TO-PM-402 ( protected in class Population )
	void selectRouletteWheel(PopulationMOO&, unsigned);
	// TO-PM-405
	void linearDynamicScaling(std::vector< double >&, unsigned long);
	// TO-PM-410
	void selectMuLambda(PopulationMOO&, unsigned);
	// TO-PM-411
	void selectMuLambdaKappa(PopulationMOO&, unsigned, unsigned);
	// TO-PM-412
	void selectProportional(PopulationMOO&, unsigned);
	// TO-PM-413
	IndividualMOO& selectOneIndividual();
	// TO-PM-414
	void selectTournament(PopulationMOO&, unsigned, unsigned);
	// TO-PM-415
	void selectEPTournament(PopulationMOO&, unsigned);
	// TO-PM-416
	//void selectUniformRanking( PopulationMOO&, unsigned );
	// TO-PM-417
	//void reproduce( PopulationMOO&, unsigned );
	// TO-PM-418
	void selectLinearRanking(PopulationMOO&, double, unsigned);
	// TO-PM-419
	void selectLinearRankingWhitley(PopulationMOO&, double, unsigned);

	// ---------------------------------------------------
	// selection of Individuals 3
	// ---------------------------------------------------
	// TO-PM-425
	void selectElitistsMOO(PopulationMOO& offspring, unsigned numElitists);

	// ---------------------------------------------------
	// selection probability
	// ---------------------------------------------------
	// TO-PM-450
	void     NormalizeSelectProb();
	// TO-PM-451
	void     SelectProbMichalewicz(double = 0.075);

	// ---------------------------------------------------
	// print data
	// ---------------------------------------------------
	// TO-PM-500
	void     printPM();

	// ---------------------------------------------------
	// Distance
	// ---------------------------------------------------
	// TO-PM-600
	double   PhenoFitDisN1(unsigned, unsigned);
	// TO-PM-601
	double   PhenoFitDisN2(unsigned, unsigned);
	// TO-PM-602
	double   PhenoFitDisNM(unsigned, unsigned);

	// ---------------------------------------------------
	// Sharing
	// ---------------------------------------------------
	// TO-PM-700
	unsigned NicheCountPFN1(unsigned, double);
	// TO-PM-701
	unsigned NicheCountPFN2(unsigned, double);
	// TO-PM-702
	unsigned NicheCountPFNM(unsigned, double);
	// TO-PM-703
	void     NicheCountPFN1(double);
	// TO-PM-704
	void     NicheCountPFN2(double);
	// TO-PM-705
	void     NicheCountPFNM(double);
	// TO-PM-710
	double   SharingTriPFN1(unsigned, double);
	// TO-PM-711
	double   SharingTriPFN2(unsigned, double);
	// TO-PM-712
	double   SharingTriPFNM(unsigned, double);
	// TO-PM-713
	void     SharingTriPFN1(double);
	// TO-PM-714
	void     SharingTriPFN2(double);
	// TO-PM-715
	void     SharingTriPFNM(double);
	// TO-PM-720
	double   SharingPowPFN1(unsigned, double, double);
	// TO-PM-721
	double   SharingPowPFN2(unsigned, double, double);
	// TO-PM-722
	double   SharingPowPFNM(unsigned, double, double);
	// TO-PM-723
	void     SharingPowPFN1(double, double);
	// TO-PM-724
	void     SharingPowPFN2(double, double);
	// TO-PM-725
	void     SharingPowPFNM(double, double);

	// TO-PM-790
	void     SharingSelProb();
	// TO-PM-791
	void     SharingFitness();
	// TO-PM-792
	void     SharingMOOFitness();

	// ---------------------------------------------------
	// Selection in MOO
	// ---------------------------------------------------
	// TO-PM-800
	void     SelectByRoulette(PopulationMOO&, unsigned = 0);
	// TO-PM-801
	void     SelectTournamentNCPFN1(PopulationMOO&, double,
									unsigned = 0);
	// TO-PM-802
	void     SelectTournamentNCPFN2(PopulationMOO&, double,
									unsigned = 0);
	// TO-PM-803
	void     SelectTournamentNCPFNM(PopulationMOO&, double,
									unsigned = 0);
	// TO-PM-804
	void     SelectTournamentSTPFN1(PopulationMOO&, double,
									unsigned = 0);
	// TO-PM-805
	void     SelectTournamentSTPFN2(PopulationMOO&, double,
									unsigned = 0);
	// TO-PM-806
	void     SelectTournamentSTPFNM(PopulationMOO&, double,
									unsigned = 0);
	// TO-PM-807
	void     SelectTournamentSPPFN1(PopulationMOO&, double,
									double, unsigned = 0);
	// TO-PM-808
	void     SelectTournamentSPPFN2(PopulationMOO&, double,
									double, unsigned = 0);
	// TO-PM-809
	void     SelectTournamentSPPFNM(PopulationMOO&, double,
									double, unsigned = 0);
	// TO-PM-810
	void     SelectComparisonNCPFN1(PopulationMOO&,
									unsigned, double,
									unsigned = 0);
	// TO-PM-811
	void     SelectComparisonNCPFN2(PopulationMOO&,
									unsigned, double,
									unsigned = 0);
	// TO-PM-812
	void     SelectComparisonNCPFNM(PopulationMOO&,
									unsigned, double,
									unsigned = 0);
	// TO-PM-813
	void     SelectComparisonSTPFN1(PopulationMOO&,
									unsigned, double,
									unsigned = 0);
	// TO-PM-814
	void     SelectComparisonSTPFN2(PopulationMOO&,
									unsigned, double,
									unsigned = 0);
	// TO-PM-815
	void     SelectComparisonSTPFNM(PopulationMOO&,
									unsigned, double,
									unsigned = 0);
	// TO-PM-816
	void     SelectComparisonSPPFN1(PopulationMOO&,
									unsigned, double,
									double, unsigned = 0);
	// TO-PM-817
	void     SelectComparisonSPPFN2(PopulationMOO&,
									unsigned, double,
									double, unsigned = 0);
	// TO-PM-818
	void     SelectComparisonSPPFNM(PopulationMOO&,
									unsigned, double,
									double, unsigned = 0);

	// ---------------------------------------------------
	// New concept of MOO-Elitist
	// ---------------------------------------------------
	// TO-PM-850
	bool     EvaluationMOOElitists(unsigned);
	// TO-PM-851
	unsigned getNoOfRankOne();
	// TO-PM-852
	unsigned SelectMOOElitists(PopulationMOO&, unsigned, unsigned,
							   double, double = 1);
	// TO-PM-853
	unsigned SelectMOOElitistsSub(PopulationMOO&, unsigned, unsigned,
								  double, double = 1);
	// TO-PM-854
	unsigned SelectMOOElitistsFromRankOne(PopulationMOO&, unsigned,
										  unsigned, double,
										  double = 1);


	// NSGA 2
	// TO-PM-2000
	void crowdedTournamentSelection(PopulationMOO& offsprings);
	// TO-PM-2001
	void crowdedDistance(bool UnpenalizedFitness = false);
	void crowdingDistance(bool UnpenalizedFitness = false);
	void EpsilonMeasure(bool UnpenalizedFitness = false);
	void SMeasure(bool UnpenalizedFitness = false);
	void SMeasureTwoObjectives(bool UnpenalizedFitness = false);
	// TO-PM-2002
	unsigned maxMOORank();
	// TO-PM-2003
	unsigned numberOfMOORank(unsigned rank);

	// SPEA2
	// TO-PM-2010
	void SPEA2Strengthen(ArchiveMOO& archive);
	// TO-PM-2011
	void SPEA2Density(ArchiveMOO& archive);
	// TO-PM-2012
	void SPEA2BinaryTournamentSelection(ArchiveMOO& archive);
	// TO-PM-2013
	double kthDistance(unsigned i, unsigned k);
	// TO-PM-2014
	void SPEA2Sort();
	// TO-PM-2015
	void environmentalSelection(PopulationMOO& pop, ArchiveMOO& archive);
	// TO-PM-2016
	unsigned SPEA2NoOfNonDominated();

	// SW-PM-2020
	/*! EP-tournament style crowding distance based selection*/
	void selectCrowdedEPTournament(PopulationMOO& offsprings, unsigned q);
	// SW-PM-2021
	void nichedComparisonRank(PopulationMOO& pop, Array<unsigned>& nichedRanks);

	// TO-PM-2501
	PopulationMOO& combinePopulationMOO(PopulationMOO& p1, PopulationMOO& p2);

	// TO-PM-2502
	PopulationMOO& combinePopulationMOO(PopulationMOO& p1, ArchiveMOO& a1);

	// TO-PM-3001
	int checkData(PopulationMOO& offsprings);

	// ---------------------------------------------------
	// Christian's selection methods
	// ---------------------------------------------------
	void selectCrowdedMuPlusLambda(PopulationMOO& offsprings, bool unpenalized = false);
	void selectCrowdedMuCommaLambda(PopulationMOO& offsprings, bool unpenalized = false);
	void selectBinaryTournamentMOO(PopulationMOO& offspring);

	// --------------------------------------------------
	// RMMeda selection method
	// --------------------------------------------------
	void SelectPop(std::vector<std::vector<double> >&PopX,
		       std::vector<std::vector<double> >&PopF,
		       std::vector<std::vector<double> >&OffX, 
		       std::vector<std::vector<double> >&OffF);
	void selectRMMuPlusLambda(PopulationMOO& offspring);

	//=======================================================================
	// ---------------------------------------------------
	// PVM Interface
	// ---------------------------------------------------

	// SW-PM-4000
	/*! Part of PVM-send routine for MOO-populations */
	int pvm_pkpop();

	// SW-PM-4001
	/*! Part of PVM-receive routine for MOO-populations */
	int pvm_upkpop();

	//=======================================================================


#ifndef __NO_GENERIC_IOSTREAM
	friend std::ostream& operator << (std::ostream& os, const PopulationMOO& pop)
	{
		os << "PopulationMOO(" << pop.size() << ")\n"
		<< pop.index     << '\n'
		<< pop.subPop    << '\n'
		<< pop.ascending << '\n'
		<< pop.spinOnce  << std::endl;
		for (unsigned i = 0; i < pop.size(); ++i)
			os << pop[ i ];
		return os;
	}

	friend std::istream& operator >> (std::istream& is, PopulationMOO& pop)
	{
		unsigned i, popSize(0);
		std::string s, t;

		is >> s;
		is.get();    // skip end of line

		if ((!pop.subPop || pop.size() == 0) && is.good() &&
				s.substr(0, 14) == "PopulationMOO(" &&
				s.find(')') != std::string::npos)
		{
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

	FastNonDominatedSort m_fastNonDominatedSort;
	CrowdingDistance m_crowdingDistance;
	IndicatorBasedSelectionStrategy<AdditiveEpsilonIndicator> m_epsilonMeasure;
	IndicatorBasedSelectionStrategy<HypervolumeIndicator> m_sMeasure;

	// static comparison functions for std::sort
	static bool compareFitnessAscending(Individual*const& pInd1, Individual*const& pInd2);
	static bool compareFitnessDescending(Individual*const& pInd1, Individual*const& pInd2);
	static bool compareRankShare(Individual*const& pInd1, Individual*const& pInd2);
	static bool compareScaledFitnessRankShare(Individual*const& pInd1, Individual*const& pInd2);
};

//!
//! \brief Quicksort implementation for the crowding distance computation
//!
class SortMOO_CD
{
public:
	SortMOO_CD(std::vector<unsigned>* pSort, std::vector<unsigned>* pGlobal, int size_global, std::vector<Individual*>* pIndvec, int obj, bool bUnpenalized)
	{
		m_sort = pSort;
		m_global = pGlobal;
		m_size_global = size_global;
		m_indvec = pIndvec;
		m_obj = obj;
		m_bUnpenalized = bUnpenalized;
	}

	void sort()
	{
		quicksort(0, m_size_global - 1);
	}

private:
	std::vector<unsigned>* m_sort;
	std::vector<unsigned>* m_global;
	int m_size_global;
	std::vector<Individual*>* m_indvec;
	int m_obj;
	bool m_bUnpenalized;

	bool lessUnpenalizedMOOFitness(unsigned i1, unsigned i2)
	{
		return ((IndividualMOO*)(*m_indvec)[(*m_global)[(*m_sort)[ i1 ] ] ])->getUnpenalizedMOOFitness(m_obj) < ((IndividualMOO*)(*m_indvec)[(*m_global)[(*m_sort)[ i2 ] ] ])->getUnpenalizedMOOFitness(m_obj);
	}
	bool lessMOOFitness(unsigned i1, unsigned i2)
	{
		return ((IndividualMOO*)(*m_indvec)[(*m_global)[(*m_sort)[ i1 ] ] ])->getMOOFitness(m_obj) < ((IndividualMOO*)(*m_indvec)[(*m_global)[(*m_sort)[ i2 ] ] ])->getMOOFitness(m_obj);
	}

	void Xchg(int a, int b)
	{
		unsigned temp = (*m_sort)[a];
		(*m_sort)[a] = (*m_sort)[b];
		(*m_sort)[b] = temp;
	}

	void partition(int left, int right, int& lp, int &rp)
	{
		int i = left + 1;
		int j = left + 1;
		while (j <= right)
		{
			if (m_bUnpenalized)
			{
				if (lessUnpenalizedMOOFitness(j, left))
				{
					Xchg(i, j);
					i++;
				}
			}
			else
			{
				if (lessMOOFitness(j, left))
				{
					Xchg(i, j);
					i++;
				}
			}
			j++;
		}
		Xchg(left, i - 1);
		lp = i - 2;
		rp = i;
	}

	void quicksort(int left, int right)
	{
		if (left < right)
		{
			int lp, rp;
			partition(left, right, lp, rp);
			quicksort(left, lp);
			quicksort(rp, right);
		}
	}
};


//!
//! \brief Quicksort implementation for the niched comparison rank computation
//!
class SortMOO_NCR
{
public:
	SortMOO_NCR(std::vector<unsigned>* pSort, PopulationMOO* pPop)
	{
		m_sort = pSort;
		m_pPop = pPop;
	}

	void sort()
	{
		quicksort(0, m_sort->size() - 1);
	}

private:
	std::vector<unsigned>* m_sort;
	PopulationMOO* m_pPop;

	bool less(unsigned i1, unsigned i2)
	{
		unsigned Rank1 = (*m_pPop)[ i1 ].getMOORank();
		unsigned Rank2 = (*m_pPop)[ i2 ].getMOORank();
		double Share1 = (*m_pPop)[ i1 ].getMOOShare();
		double Share2 = (*m_pPop)[ i2 ].getMOOShare();

		return ((Rank1 < Rank2)
				|| (Rank1 == Rank2 && Share1 >= Share2));
	}

	void Xchg(int a, int b)
	{
		unsigned temp = (*m_sort)[a];
		(*m_sort)[a] = (*m_sort)[b];
		(*m_sort)[b] = temp;

		m_pPop->swap(a, b);
	}

	void partition(int left, int right, int& lp, int &rp)
	{
		int i = left + 1;
		int j = left + 1;
		while (j <= right)
		{
			if (less(j, left))
			{
				Xchg(i, j);
				i++;
			}
			j++;
		}
		Xchg(left, i - 1);
		lp = i - 2;
		rp = i;
	}

	void quicksort(int left, int right)
	{
		if (left < right)
		{
			int lp, rp;
			partition(left, right, lp, rp);
			quicksort(left, lp);
			quicksort(rp, right);
		}
	}
};


#endif

