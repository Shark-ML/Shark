#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_RECOMBINATION_PARTIALLYMAPPEDCROSSOVER_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_RECOMBINATION_PARTIALLYMAPPEDCROSSOVER_H

#include <shark/Core/Random.h>
#include <shark/Core/Exception.h>

namespace shark {

/// @brief Implements partially mapped crossover
/// 
/// PartiallyMappedCrossover recombines points representing
/// permutations, i.e. it ensures that the results are also permutations.
struct PartiallyMappedCrossover {

	/// \brief Mates the supplied individuals.
	/// 
	/// \param [in,out] individual1 Individual to be mated.
	/// \param [in,out] individual2 Individual to be mated.
	template<class Rng, typename IndividualType>
	void operator()(Rng& rng, IndividualType & individual1, IndividualType & individual2 )const{	
		SIZE_CHECK(individual1.searchPoint().size() == individual2.searchPoint().size());
		
		typedef typename IndividualType::SearchPointType PointType;
		PointType& t1 = individual1.searchPoint();
		PointType& t2 = individual2.searchPoint();
		
		std::size_t n = t1.size();
		unsigned int unset = static_cast<unsigned int>(n + 1);
		
		
		//compute cuttingpoints 0 <= cuttingPoint1 < cuttingPoint2 <= n
		std::size_t cuttingPoint1 = random::discrete(rng, std::size_t(0), n - 2);
		std::size_t cuttingPoint2 = random::discrete(rng,cuttingPoint1+1,n-1);

		PointType r1(n, unset), r2(n, unset);
		PointType p1(n, unset), p2(n, unset);

		//swap ranges [cuttingPoint1,cuttingPoint2] and store in p1,p2
		//also keep track which elements are already taken and which one are free
		for( std::size_t i = cuttingPoint1; i <= cuttingPoint2; i++ ) {
			p1[i] = t2[i];
			p2[i] = t1[i];

			r1[ t2[i] ] = t1[i];
			r2[ t1[i] ] = t2[i];
		}

		for( std::size_t i = 0; i < t1.size(); i++) {
			if ((i >= cuttingPoint1) && (i <= cuttingPoint2)) continue;

			std::size_t n1 = t1[i] ;
			std::size_t m1 = r1[n1] ;

			std::size_t n2 = t2[i] ;
			std::size_t m2 = r2[n2] ;

			while (m1 != unset) {
				n1 = m1 ;
				m1 = r1[m1] ;
			}
			while (m2 != unset) {
				n2 = m2 ;
				m2 = r2[m2] ;
			}
			p1[i] = n1 ;
			p2[i] = n2 ;
		}

		t1 = p1;
		t2 = p2;

	}

	/// \brief Serializes this instance to the supplied archive.
	template<typename Archive>
	void serialize( Archive &, const unsigned int ) {}
};
}

#endif
