#ifndef SHARK_EA_POPULATION_H
#define SHARK_EA_POPULATION_H

#include <shark/Algorithms/DirectSearch/EA.h>
#include <shark/Algorithms/DirectSearch/Individual.h>
#include <shark/Algorithms/DirectSearch/FitnessComparator.h>

#include <algorithm>
#include <vector>

namespace shark {

    typedef std::vector< Individual > Population;

    template<typename Population>
    void shuffle( Population & p ) {
        std::random_shuffle( p.begin(), p.end() );
    }

    template<typename FitnessTag>
    Population::const_iterator best_individual( const Population & p ) {
        return( std::min_element( p.begin(), p.end(), FitnessComparator<FitnessTag>() ) );
    }

    template<typename FitnessTag>
    Population::const_iterator worst_individual( const Population & p ) {
        return( std::max_element( p.begin(), p.end(), FitnessComparator<FitnessTag>() ) );
    }

}

#endif // SHARK_EA_POPULATION_H
