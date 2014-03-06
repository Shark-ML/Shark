/*!
 *
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
			m_isBoundaryElement( false )
		{

		}

		view_reference(T & value) : mep_value( &value ),
			m_isBoundaryElement( false )
		{
		}

		operator T & ()
		{
			return( *mep_value );
		}

		operator const T & () const
		{
			return( *mep_value );
		}

		view_reference<T> operator=( const T & rhs )
		{
			*mep_value = rhs;
			return( *this );
		}
	};
	/** \endcond */

	/**
	 * \brief Default c'tor.
	 *
	 * \param [in] mu Target size of the population, default value: 0.
	 */
	ApproximatedHypervolumeSelection( unsigned int mu = 0 ) 
	: m_errorBound( 1E-2 )
	, m_errorProbability( 1E-2 )
	{}

	/** \cond */

	template<typename Extractor>
	struct FitnessComparator {

		template<typename Individual>
		FitnessComparator( Extractor & extractor, const Individual & individual ) : m_extractor( extractor ),
			m_fitness( extractor( individual ) )
		{
		}


		template<typename VectorType>
		bool operator()( const VectorType & lhs, const VectorType & rhs )
		{
			return( m_extractor( lhs ) == m_extractor( rhs ) );
		}

		template<typename VectorType>
		bool operator()( const VectorType & lhs )
		{
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
	void operator()( PopulationType & population )
	{
		if(population.empty()) return;
		
		//get the number of objectives
		std::size_t noObjectives =  population[0].fitness( tag::PenalizedFitness() ).size();
		
		typedef std::vector< view_reference<typename PopulationType::value_type > > View;

		RealVector utopianFitness( noObjectives, std::numeric_limits<double>::max() );
		RealVector nadirFitness( noObjectives, -std::numeric_limits<double>::max() );

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
			bbCalculator( front[i] );

		for( unsigned int i = 0; i < nadirFitness.size(); i++ )
			nadirFitness[i] += 1.0;

		std::size_t lc = m_indicator.leastContributor( cExtractor, front, nadirFitness, m_errorProbability, m_errorBound );
		
		front[lc].mep_value->share() = WORST_SHARE;
	}

	template <typename Node>
	void configure( const Node & node )
	{
		m_indicator.configure( node );
	}
	
	/**
	* \brief Adjusts the target size mu of the population(unused as SteadyState uses mu=1)
	* \param [in] mu The new target size mu.
	*/
	void setMu( unsigned int mu )
	{
		(void)mu;
	}

	/**
	 * \brief Stores/restores the serializer's state.
	 * \tparam Archive Type of the archive.
	 * \param [in,out] archive The archive to serialize to.
	 * \param [in] version number, currently unused.
	 */
	template<typename Archive>
	void serialize( Archive & archive, const unsigned int version )
	{
		archive & BOOST_SERIALIZATION_NVP( m_errorBound );
		archive & BOOST_SERIALIZATION_NVP( m_errorProbability );
	}

	/** \cond */

	double m_errorBound;
	double m_errorProbability;

	LeastContributorApproximator<FastRng,HypervolumeCalculator> m_indicator;

	/** \endcond */
};

double ApproximatedHypervolumeSelection::BEST_SHARE = 1E10;

double ApproximatedHypervolumeSelection::WORST_SHARE = -1E10;
}

#endif
