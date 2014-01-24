/*!
 * 
 * \file        ApproximatedHypervolumeSelection.h
 *
 * \brief       Hypervolume selection based on an approximation scheme.
 * 
 * The algorithm is presented in:
 * K. Bringmann, T. Friedrich. 
 * Approximating the least hypervolume contributor: NP-hard in general, but fast in practice. 
 * Proc. of the 5th International Conference on Evolutionary Multi-Criterion Optimization (EMO 2009), 
 * Vol. 5467 of LNCS, pages 6-20, Springer-Verlag, 2009.
 * 
 * 
 *
 * \author      T.Voss, T. Glasmachers, O.Krause
 * \date        2010-2011
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
#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_SELECTION_APPROXIMATED_HYPERVOLUME_SELECTION_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_SELECTION_APPROXIMATED_HYPERVOLUME_SELECTION_H

#include <shark/Algorithms/DirectSearch/BoundingBoxCalculator.h>
#include <shark/Algorithms/DirectSearch/FitnessExtractor.h>
#include <shark/Algorithms/DirectSearch/HypervolumeCalculator.h>
#include <shark/Algorithms/DirectSearch/LeastContributorApproximator.hpp>

#include <iterator>
#include <limits>
#include <map>
#include <vector>

namespace shark {

    /**
     * \brief Implements an approximated hypervolume selection scheme.
     *
     * See Bringmann, Friedrich. 
     * Approximating the Least Hypervolume Contributor: NP-hard in General, but Fast in Practice. 
     * EMO 2009.
     */
    struct ApproximatedHypervolumeSelection {

	static double BEST_SHARE;
	static double WORST_SHARE;

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

	template<typename PopulationType>
	struct MinShareOperator {

	    template<typename Member>
	    void operator()( Member & m ) {
		static_cast<typename PopulationType::value_type &>( m ).setShare( WORST_SHARE );
	    }

	};

	template<typename PopulationType>
	struct MaxShareOperator {
	    MaxShareOperator() {
	    }


	    template<typename Member>
	    void operator()( Member & m ) {
		static_cast<typename PopulationType::value_type &>( m ).setShare( BEST_SHARE );
	    }

	};

	template<typename PopulationType>
	struct FrontAssembler {
	FrontAssembler( const PopulationType & pop ) : m_pop( pop ) {
	    // m_front.reserve( m_pop.size() );
	}

	    void operator()( unsigned int idx ) {
		m_front.append( m_pop[idx] );
	    }

	    const PopulationType & m_pop;
	    PopulationType m_front;
	};

	template<typename PopulationType>
	struct MinObjective {

	MinObjective( unsigned int objective ) : m_objective( objective ) {}

	    template<typename Member>
	    bool operator()( Member & i1, Member & i2 ) {
		return( static_cast<typename PopulationType::value_type &>( i1 ).fitness( tag::PenalizedFitness() )[m_objective] < 
			static_cast<typename PopulationType::value_type &>( i2 ).fitness( tag::PenalizedFitness() )[m_objective] );
	    }

	    unsigned int m_objective;
	};
	/** \endcond */

	/**
	 * \brief Default c'tor.
	 * 
	 * \param [in] mu Target size of the population, default value: 0.
	 */
    ApproximatedHypervolumeSelection( unsigned int mu = 0 ) : m_mu( mu ),
	    m_noObjectives( 0 ),
	    m_errorBound( 1E-2 ),
	    m_errorProbability( 1E-2 ) {
    }

	/**
	 * \brief Accesses the target size mu of the population.
	 */
	unsigned int mu() const {
	    return( m_mu );
	}

	/**
	 * \brief Adjusts the target size mu of the population.
	 */
	void setMu( unsigned int mu ) {
	    m_mu = mu;
	}

	/**
	 * \brief Accesses the dimensionality of the objective space.	   
	 */ 
	unsigned int noObjectives() const {
	    return( m_noObjectives );
	}

	/**
	 * \brief Adjusts the dimensionality of the objective space.
	 */
	void setNoObjectives( unsigned int noObjectives ) {
	    m_noObjectives = noObjectives;
	}

	/** \cond */
	template<typename Extractor>
	struct LastObjectiveComparator {

	LastObjectiveComparator( Extractor & extractor ) : m_extractor( extractor ) {}

	    template<typename VectorType>
	    bool operator()( const VectorType & lhs, const VectorType & rhs ) {
		return( m_extractor( lhs ).back() < m_extractor( rhs ).back() );
	    }

	    Extractor & m_extractor;
	};

	template<typename Extractor>
	struct FitnessComparator {

	    template<typename Individual>
	    FitnessComparator( Extractor & extractor, const Individual & individual ) : m_extractor( extractor ),
		m_fitness( extractor( individual ) ) {
	    }


	    template<typename VectorType>
	    bool operator()( const VectorType & lhs, const VectorType & rhs ) {
		return( m_extractor( lhs ) == m_extractor( rhs ) );
	    }

	    template<typename VectorType>
	    bool operator()( const VectorType & lhs ) {
		for( unsigned int i = 0; i < m_extractor( lhs ).size(); i++ )
		    if( m_extractor( lhs )( i ) != m_fitness( i ) )
			return( false );
		return( true );
	    }

	    Extractor & m_extractor;
	    typename Extractor::fitness_type m_fitness;
	};
	/** \endcond */

	/**
	 * \brief Executes the algorithm and assigns each individual a share according to its hypervolume contribution.
	 */
	template<typename PopulationType>
	void operator()( PopulationType & population ) {

	    typedef std::vector< view_reference<typename PopulationType::value_type > > View;

	    RealVector utopianFitness( m_noObjectives, std::numeric_limits<double>::max() );
	    RealVector nadirFitness( m_noObjectives, -std::numeric_limits<double>::max() );		

	    CastingFitnessExtractor<typename PopulationType::value_type> cExtractor;			

	    BoundingBoxCalculator< CastingFitnessExtractor<typename PopulationType::value_type> > bbCalculator( cExtractor, utopianFitness, nadirFitness );

	    unsigned int maxRank = 0;
	    std::map<unsigned int, View > fronts;

	    for( unsigned int i = 0; i < population.size(); i++ ) {
		maxRank = std::max( maxRank, static_cast<unsigned int>( population[i].rank() ) );

		// Take care of clones.
		FitnessComparator< CastingFitnessExtractor<typename PopulationType::value_type> > fitnessComparator( cExtractor, population[ i ] );
		if( std::find_if( fronts[population[i].rank()].begin(), fronts[population[i].rank()].end(), fitnessComparator ) != fronts[population[i].rank()].end() ) {
		    population[ i ].share() = WORST_SHARE;
		    continue;
		}

		fronts[population[i].rank()].push_back( population[i] );
		population[i].share() = BEST_SHARE;
	    }

	    View & front = fronts[maxRank];

	    if( front.size() == 0 )
		return;

	    if( front.size() == 1 ) {
		front[0].mep_value->share() = WORST_SHARE;
		return;
	    }			


	    for( unsigned int i = 0; i < front.size(); i++ )
		bbCalculator( front.at( i ) );

	    for( unsigned int i = 0; i < nadirFitness.size(); i++ )
		nadirFitness[i] += 1.0;

	    LastObjectiveComparator< CastingFitnessExtractor<typename PopulationType::value_type> > comp( cExtractor );

	    std::size_t lc = leastContributor( cExtractor, front, nadirFitness );
	    front[std::min( front.size()-1, lc )].mep_value->share() = WORST_SHARE;

	    m_leastContributor = lc;
	    m_nadirPoint = nadirFitness;
	}


	/**
	 * \brief Determines the individual contributing the least amount of hypervolume in the population.
	 * \param [in] extractor Function object for projecting individuals in the space \f$ \mathbb{R}^m\f$.
	 * \param [in] pop The population.
	 * \param [in] refPoint The reference point for the hypervolume computation (note: minimization is assumed here).
	 * \returns An index i such that 0 <= i <= |pop|.
	 */
	template<typename Extractor, typename PopulationType>
	    unsigned int leastContributor( Extractor & extractor, const PopulationType & pop, const RealVector & refPoint ) {
	    if( pop.size() == 1 )
		return( 0 );


	    Extractor e;
	    typename PopulationType::const_iterator it = m_lca( e, pop, refPoint, m_noObjectives, m_errorProbability, m_errorBound );
	    return( std::distance( pop.begin(), it ) );		
	}

	template <typename Node>
	void configure( const Node & node ) {
	    m_lca.configure( node );
	}

	/**
	 * \brief Stores/restores the serializer's state.
	 * \tparam Archive Type of the archive.
	 * \param [in,out] archive The archive to serialize to.
	 * \param [in] version number, currently unused.
	 */
	template<typename Archive>
	void serialize( Archive & archive, const unsigned int version ) {
	    archive & BOOST_SERIALIZATION_NVP( m_extractor );
	    archive & BOOST_SERIALIZATION_NVP( m_mu );
	    archive & BOOST_SERIALIZATION_NVP( m_noObjectives );
	    archive & BOOST_SERIALIZATION_NVP( m_errorBound );
	    archive & BOOST_SERIALIZATION_NVP( m_errorProbability );
	    archive & BOOST_SERIALIZATION_NVP( m_leastContributor );
	    archive & BOOST_SERIALIZATION_NVP( m_nadirPoint );
	}

	/** \cond */
	FitnessExtractor m_extractor;

	unsigned int m_mu;
	unsigned int m_noObjectives;

	double m_errorBound;
	double m_errorProbability;

	unsigned int m_leastContributor;
	RealVector m_nadirPoint;

	LeastContributorApproximator<FastRng,HypervolumeCalculator> m_lca;

	/** \endcond */
    };

    double ApproximatedHypervolumeSelection::BEST_SHARE = 1E10;

    double ApproximatedHypervolumeSelection::WORST_SHARE = -1E10;
}

#endif 
