#ifndef SHARK_ALGORITHMS_DIRECTSEARCH_FASTNONDOMINATEDSORT_H
#define SHARK_ALGORITHMS_DIRECTSEARCH_FASTNONDOMINATEDSORT_H

#include <shark/Algorithms/DirectSearch/TypedIndividual.h>
#include <shark/Algorithms/DirectSearch/ParetoDominanceComparator.h>

#include <list>
#include <vector>

namespace shark {

/**
 * \brief Implements the well-known non-dominated sorting algorithm.
 *
 * Assembles subsets/fronts of mututally non-dominating individuals.
 *
 * \tparam Comparator Predicate for comparing set members.
 */
template<typename Comparator>
struct BaseFastNonDominatedSort {

	/**
	 * \brief Executes the algorithm.
	 *
	 * \tparam PopulationType Container type, needs to be random accessible.
	 *
	 * \param pop [in,out] Population to subdivide into fronts of non-dominated individuals.
	 */
	template<typename PopulationType>
	void operator()(PopulationType &pop) {
		std::vector<unsigned> r(pop.size());
		std::vector<unsigned> n(pop.size(), 0);
		std::vector<std::vector<unsigned> > s(pop.size());

		std::vector<unsigned> f;
		f.reserve(pop.size());
		std::vector<unsigned>::iterator it, itE, its, itsE;

		Comparator pdc;

		unsigned i, j;
		for (i = 0; i < pop.size(); i++) {
			for (j = 0; j < pop.size(); j++) {
				if (i == j)
					continue;

				// if (pop.Dominate(i, j, m_bUnpenalizedFitness) > 1)
				if (pdc(pop[i], pop[j]) > 1)
					s[i].push_back(j);
				//else if (pop.Dominate(i, j) < -1)
				else if (pdc(pop[i], pop[j]) < -1)
					n[i]++;
			}

			if (n[i] == 0)
				f.push_back(i);
		}

		it = f.begin();
		itE = f.end();

		while (it != itE) {
			n[*it] = 1;
			++it;
		}

		unsigned frontCounter = 2;
		std::vector<unsigned> h;

		while (!f.empty()) {
			it = f.begin();
			itE = f.end();

			h.clear();

			while (it != itE) {
				its = s[*it].begin();
				itsE = s[*it].end();

				while (its != itsE) {
					n[*its]--;

					if (n[*its] == 0)
						h.push_back(*its);

					++its;
				}

				++it;
			}

			its = h.begin();
			itsE = h.end();
			while (its != itsE) {
				n[*its] = frontCounter;
				++its;
			}

			f = h;
			frontCounter++;
		}

		for (i = 0; i < pop.size(); i++) {
			pop[i].rank() = n[i];
		}
	}

	/**
	 * \brief Executes the algorithm.
	 *
	 * \tparam PopulationType Container type, needs to be random accessible.
	 * \tparam Extractor Mapping operator for extracting fitness values from individuals.
	 * \param pop [in,out] Population to subdivide into fronts of non-dominated individuals.
	 * \param e [in,out] Extractor instance.
	 */
	template<typename PopulationType, typename Extractor>
	void operator()(PopulationType &pop, Extractor &e) {
		// std::vector<unsigned> r(pop.size());
		std::vector<unsigned> n(pop.size(), 0);
		std::vector<std::vector<unsigned> > s(pop.size());

		std::vector<unsigned> f;
		f.reserve(pop.size());
		std::vector<unsigned>::iterator it, itE, its, itsE;

		//ParetoDominanceComparator<PENALIZED_FITNESS_TYPE> pdc;
		Comparator pdc;

		unsigned i, j;
		for (i = 0; i < pop.size(); i++) {
			for (j = 0; j < pop.size(); j++) {
				if (i == j)
					continue;

				// if (pop.Dominate(i, j, m_bUnpenalizedFitness) > 1)
				if (pdc(pop[i], pop[j], e) > 1)
					s[i].push_back(j);
				//else if (pop.Dominate(i, j) < -1)
				else if (pdc(pop[i], pop[j], e) < -1)
					n[i]++;
			}

			if (n[i] == 0)
				f.push_back(i);
		}

		it = f.begin();
		itE = f.end();

		while (it != itE) {
			n[*it] = 1;
			++it;
		}

		unsigned frontCounter = 2;
		std::vector<unsigned> h;

		while (!f.empty()) {
			it = f.begin();
			itE = f.end();

			h.clear();

			while (it != itE) {
				its = s[*it].begin();
				itsE = s[*it].end();

				while (its != itsE) {
					n[*its]--;

					if (n[*its] == 0)
						h.push_back(*its);

					++its;
				}

				++it;
			}

			its = h.begin();
			itsE = h.end();
			while (its != itsE) {
				n[*its] = frontCounter;
				++its;
			}

			f = h;
			frontCounter++;
		}

		for (i = 0; i < pop.size(); i++) {
			pop[i].setRank(n[i]);
		}
	}
};

/** \brief Default fast non-dominated sorting based on the Pareto-dominance relation. */
typedef BaseFastNonDominatedSort< ParetoDominanceComparator<tag::PenalizedFitness> > FastNonDominatedSort;

}
#endif // FASTNONDOMINATEDSORT_H
