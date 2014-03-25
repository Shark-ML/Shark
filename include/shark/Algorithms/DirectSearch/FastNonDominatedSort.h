#ifndef SHARK_ALGORITHMS_DIRECTSEARCH_FASTNONDOMINATEDSORT_H
#define SHARK_ALGORITHMS_DIRECTSEARCH_FASTNONDOMINATEDSORT_H

#include <shark/Algorithms/DirectSearch/TypedIndividual.h>
#include <shark/Algorithms/DirectSearch/ParetoDominanceComparator.h>
#include <shark/Algorithms/DirectSearch/FitnessExtractor.h>

#include <list>
#include <vector>

namespace shark {

/**
 * \brief Implements the well-known non-dominated sorting algorithm.
 *
 * Assembles subsets/fronts of mututally non-dominating individuals.
 * Afterwards every individual is assigned a rank by pop[i].rank() = fronNumber.
 * The front of dominating points has the value 1. 
 * 
 * The algorithm is dscribed in Deb et al, 
 * A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II
 * IEEE Transactions on Evolutionary Computation, 2002
 *
 * \tparam Extractor returning the fitness vector of an individual
 */
template<typename Extractor>
struct BaseFastNonDominatedSort {

	/**
	 * \brief Executes the algorithm.
	 *
	 * Afterwards every individual is assigned a rank by pop[i].rank() = fronNumber.
	 * The front of dominating points has the value 1.
	 *
	 * \param pop [in,out] Population to subdivide into fronts of non-dominated individuals.
	 */
	template<typename PopulationType>
	void operator()(PopulationType &pop) {
		
		//dominance relation
		ParetoDominanceComparator<Extractor> pdc;

		//stors for the i-th point which points are dominated by i
		std::vector<std::vector<unsigned> > s(pop.size());
		//stores for every point how many points are dominating it
		std::vector<unsigned> numberOfDominatingPoints(pop.size(), 0);
		//stores initially the front of non-dominated points
		std::vector<unsigned> front;
		
		for (std::size_t i = 0; i < pop.size(); i++) {
			//check which points j are dominated by i and add them to s[i]
			//also increment n[j] for every i dominating j
			for (std::size_t  j = 0; j < pop.size(); j++) {
				if (i == j)
					continue;
				
				int domination = pdc(pop[i], pop[j]);
				if ( domination > 1)//pop[i]> pop[j]
					s[i].push_back(j);
				else if (domination < -1)//pop[i]< pop[j]
					numberOfDominatingPoints[i]++;
			}
			//all non-dominated points form the first front
			if (numberOfDominatingPoints[i] == 0){
				front.push_back(i);
				pop[i].rank() = 1;//non-dominated points have rank 1
			}
		}

		//find subsequent fronts.
		unsigned frontCounter = 2;
		std::vector<unsigned> nextFront;

		//as long as we can find fronts
		//visit all points of the last front found and remove them from the
		//set. All points which are not dominated anymore form the next front
		while (!front.empty()) {
			//visit all points of the current front and remove them
			// if any point is not dominated, it is part the next front.
			for(std::size_t element = 0; element != front.size(); ++element) {
				//visit all points dominated by the element
				std::vector<unsigned int> const& dominatedPoints = s[front[element]];
				for (std::size_t  j = 0; j != dominatedPoints.size(); ++j){
					std::size_t point = dominatedPoints[j];
					numberOfDominatingPoints[point]--;
					// if no more points are dominating this, add to the next front.
					if (numberOfDominatingPoints[point] == 0){
						nextFront.push_back(point);
						pop[point].rank() = frontCounter;
					}
				}
			}
			
			//make the new found front the current
			front.swap(nextFront);
			nextFront.clear();
			frontCounter++;
		}
	}
};

/** \brief Default fast non-dominated sorting based on the Pareto-dominance relation. */
typedef BaseFastNonDominatedSort< FitnessExtractor > FastNonDominatedSort;

}
#endif // FASTNONDOMINATEDSORT_H
