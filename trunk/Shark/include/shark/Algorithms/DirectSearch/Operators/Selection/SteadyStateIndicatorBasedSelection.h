/*!
 * 
 *
 * \brief       Steady state (+1) Indicator-based selection strategy for multi-objective selection.
 * 
 * 
 *
 * \author      T.Voss
 * \date        2010
 *
 *
 * \par Copyright 1995-2014 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://image.diku.dk/shark/>
 * 
 * Shark is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Shark is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with Shark.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_SELECTION_SS_INDICATOR_BASED_SELECTION_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_SELECTION_SS_INDICATOR_BASED_SELECTION_H

#include <shark/Algorithms/DirectSearch/FitnessExtractor.h>
#include <shark/Algorithms/DirectSearch/BoundingBoxCalculator.h>

#include <shark/Algorithms/DirectSearch/Traits/QualityIndicatorTraits.h>
#include <shark/Core/OpenMP.h>
#include <iterator>
#include <limits>
#include <map>
#include <vector>

namespace shark {

    /**
     * \brief Steady state (+1) Indicator-based selection strategy for multi-objective selection.
     */
    template<typename Indicator>
	struct SteadyStateIndicatorBasedSelection {

	    typedef SteadyStateIndicatorBasedSelection< Indicator > this_type;

	    /**
	     * \brief Value that marks the best share value.
	     */
	    static double BEST_SHARE() {
		return( 1E10 );
	    }

	    /**
	     * \brief Value that marks the worst share value.
	     */
	    static double WORST_SHARE() {
		return( 0 );
	    }

	    /** \cond */
	    template<typename T>
	    struct view_reference {
		T * mep_value;
		bool m_isBoundaryElement;
	    public:
	    view_reference() : mep_value( NULL ),
		    m_isBoundaryElement( false ) {

	    }

	    view_reference(T & value) : mep_value( &value ),
		    m_isBoundaryElement( false ) {
	    }

		operator T & () {
		    return( *mep_value );
		}

		operator const T & () const {
		    return( *mep_value );
		}

		view_reference<T> operator=( const T & rhs ) {
		    *mep_value = rhs;
		    return( *this );
		}
	    };
	    /** \endcond */

	    /**
	     * \brief Assigns the maximum share to every member of the population.
	     */
	    template<typename PopulationType>
	    struct MaxShareOperator {

		MaxShareOperator() {
		}

		template<typename Member>
		void operator()( Member & m ) {
		    static_cast<typename PopulationType::value_type &>( m ).setShare( this_type::BEST_SHARE() );
		}

	    };

	    /**
	     * \brief Default c'tor.
	     * \param [in] mu The target size mu of the population.
	     * \param [in] noObjectives The number of objectives to consider.
	     */
	SteadyStateIndicatorBasedSelection( unsigned int mu = 0, unsigned int noObjectives = 0 ) : m_mu( mu ),
		m_noObjectives( noObjectives ),
		m_useLogHyp( false ) {
	}

	    /**
	     * \brief Accesses the target size mu of the population.
	     */
	    unsigned int mu() const {
		return( m_mu );
	    }

	    /**
	     * \brief Adjusts the target size mu of the population.
	     * \param [in] mu The new target size mu.
	     */
	    void setMu( unsigned int mu ) {
		m_mu = mu;
	    }

	    /**
	     * \brief Accesses the number of objectives.
	     */
	    unsigned int noObjectives() const {
		return( m_noObjectives );
	    }

	    /**
	     * \brief Adjusts the number of objectives.
	     * \param [in] noObjectives The new number of objectives.
	     */
	    void setNoObjectives( unsigned int noObjectives ) {
		m_noObjectives = noObjectives;
	    }

	    /**
	     * \brief Executes the algorithm and assigns each member of the population
	     * its level non-dominance (rank) and its individual contribution to the front
	     * it belongs to (share).
	     * 
	     * \param [in,out] population The population to be ranked.
	     */
	    template<typename PopulationType>
	    void operator()( PopulationType & population ) {

		typedef std::vector< view_reference<typename PopulationType::value_type> > View;

		RealVector utopianFitness( m_noObjectives, std::numeric_limits<double>::max() );
		RealVector nadirFitness( m_noObjectives, -std::numeric_limits<double>::max() );		

		CastingFitnessExtractor<typename PopulationType::value_type> cExtractor;

		BoundingBoxCalculator< CastingFitnessExtractor<typename PopulationType::value_type> > bbCalculator( cExtractor, utopianFitness, nadirFitness );

		// BoundingBoxCalculator< FitnessExtractor > bbCalculator( extractor, utopianFitness, nadirFitness );

		unsigned int maxRank = 0;
		std::map<unsigned int, View > fronts;

		for( unsigned int i = 0; i < population.size(); i++ ) {
		    maxRank = std::max( maxRank, static_cast<unsigned int>( population[i].rank() ) );
		    fronts[population[i].rank()].push_back( population[i] );
		    population[i].share() = this_type::BEST_SHARE();
		}

		View & front = fronts[maxRank];

		if( front.size() == 1 ) {
		    front[0].mep_value->share() = this_type::WORST_SHARE();
		    return;
		}

		for( unsigned int i = 0; i < front.size(); i++ )
		    bbCalculator( front[i] );

		m_indicator.setUtopianFitness( utopianFitness );
		m_indicator.setNadirFitness( nadirFitness );

		unsigned int lc = leastContributor( 
						   cExtractor, 
						   front, 
						   typename QualityIndicatorTraits< Indicator >::type()
						    );

		front[lc].mep_value->share() = this_type::WORST_SHARE();
	    }

	    /**
	     * \brief Determines the individual contributing the least to the front it belongs to using a unary quality-indicator.
	     * 
	     * \param [in, out] extractor Maps the individuals to the objective space.
	     * \param [in] pop The front of non-dominated individuals.
	     * \param [in] t Marks the function for considering unary performance indicators.
	     */
	    template<typename Extractor, typename PopulationType>
		unsigned int leastContributor( Extractor & extractor, const PopulationType & pop, tag::Unary t ) {
		double total = m_indicator( extractor, pop, m_noObjectives ) + 10.0;
		std::vector<double> indicatorValues( pop.size() );

		SHARK_PARALLEL_FOR( int i = 0; i < static_cast< int >( pop.size() ); i++ ) {

		    Indicator ind( m_indicator );

		    PopulationType copy( pop );
		    copy.erase( copy.begin() + i );	

		    double indicator;

		    indicator = ind( extractor, copy, m_noObjectives );
		    indicatorValues[i] = total - indicator;
		}

		std::vector<double>::iterator it = std::min_element( indicatorValues.begin(), indicatorValues.end() );

		return( std::distance( indicatorValues.begin(), it ) );
	    }

	    /**
	     * \brief Determines the individual contributing the least to the front it belongs to using a binary quality-indicator.
	     * 
	     * \param [in, out] extractor Maps the individuals to the objective space.
	     * \param [in] pop The front of non-dominated individuals.
	     * \param [in] t Marks the function for considering binary performance indicators.
	     */
	    template<typename Extractor, typename PopulationType>
		unsigned int leastContributor( Extractor & extractor, const PopulationType & pop, tag::Binary t ) {

		std::vector<double> indicatorValues( pop.size() );
		SHARK_PARALLEL_FOR( int i = 0; i < static_cast< int >( pop.size() ); i++ ) {

		    Indicator ind( m_indicator );

		    PopulationType copy( pop );
		    copy.erase( copy.begin() + i );	

		    double indicator;

		    indicator = ind( copy.begin(), copy.end(), pop.begin(), pop.end(), extractor );
		    indicatorValues[i] = indicator;
		}

		std::vector<double>::iterator it = std::min_element( indicatorValues.begin(), indicatorValues.end() );

		return( std::distance( indicatorValues.begin(), it ) );
	    }

	    /**
	     * \brief Serializes the state of the selection operator.
	     */
	    template<typename Archive>
	    void serialize( Archive & archive, const unsigned int version ) {
		archive & BOOST_SERIALIZATION_NVP( m_indicator );
		archive & BOOST_SERIALIZATION_NVP( m_extractor );
		archive & BOOST_SERIALIZATION_NVP( m_mu );
		archive & BOOST_SERIALIZATION_NVP( m_noObjectives );
		archive & BOOST_SERIALIZATION_NVP( m_useLogHyp );
	    }

	    Indicator m_indicator;
	    FitnessExtractor m_extractor;

	    unsigned int m_mu;
	    unsigned int m_noObjectives;
	    bool m_useLogHyp;
	};

}

#endif
