/*!
 * 
 * \file        LeastContributorApproximator.hpp
 *
 * \brief       Approximately determines the individual contributing the least
 * hypervolume.
 * 
 * 
 *
 * \author      T.Voss
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
#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_LEASTCONTRIBUTORAPPROXIMATOR_HPP
#define SHARK_ALGORITHMS_DIRECT_SEARCH_LEASTCONTRIBUTORAPPROXIMATOR_HPP

#include <shark/Core/Exception.h>
#include <shark/Core/Math.h>

#include <boost/assign.hpp>
#include <boost/bimap.hpp>
#include <boost/cstdint.hpp>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <limits>
#include <vector>
#include <cmath>

namespace shark {

    /**
     * \brief Samples a random point.
     */
    template<typename Rng>
    struct Sampler {

	/**
	 * \brief Samples a random point and checks whether it is dominated or not.
	 *
	 * \tparam Set The type of the set of points.
	 *
	 * \param [in] s The set of individuals to check the sampled point against.
	 * \param [in] point Iterator to the point in the supplied that shall serve as basis for the random point.
	 *
	 * \returns true if the sample was successful (i.e. non-dominated) and false otherwise.
	 * 
	 */
	template<typename Set>
	bool operator()( const Set & s, typename Set::iterator point ) {
	    point->m_sample = point->m_point;
	    for( unsigned int i = 0; i < point->m_sample.size(); i++ ) {
		point->m_sample[ i ] += Rng::uni( 0., 1. ) * ( point->m_boundingBox[ i ] - point->m_point[ i ] );
	    }

	    bool successful = true;
	    for( unsigned int i = 0; i < point->m_influencingPoints.size(); i++ ) {
		point->m_noOperations += point->m_sample.size() + 1;
		bool inside = true;
		for( unsigned int j = 0; j < point->m_sample.size(); j++ ) {
		    if( point->m_influencingPoints[i]->m_point[ j ] > point->m_sample[ j ] ) {
			inside = false;
			break;
		    }
		}

		if( inside ) {
		    successful = false;
		    break;
		}
	    }

	    /*if( successful ) {
	      point->m_successfulSamples.push_back( point->m_sample );
	      } else
	      point->m_nonSuccessfulSamples.push_back( point->m_sample );
	    */
	    return( successful );
	}
    };

    /**
     * \brief Calculates bounding boxes.
     * \tparam Set The type of the set of points.
     */
    template<typename Set>
    struct BoundingBoxComputer {

	/**
	 * \brief Compares points based on their contributed volume.
	 */
	struct VolumeComparator {
	    /**
	     * \brief Compares points based on their contributed volume.
	     * \returns true if the volume contributed by i1 is larger than the volume contributed by i2, false otherwise.
	     */
	    template<typename Iterator>
	    bool operator()( Iterator i1, Iterator i2 ) {
		return( i1->m_overlappingVolume > i2->m_overlappingVolume );
	    }
	};

	/**
	 * \brief Computes bounding boxes and their volume for the range of points defined by the iterators.
	 * \param [in] begin Iterator pointing to the first valid point.
	 * \param [in] end Iterator pointing right after the last valid point.
	 */
	void operator()( typename Set::iterator begin, typename Set::iterator end ) {

	    for( typename Set::iterator it = begin; it != end; ++it ) {
		typename Set::value_type & p1 = *it;
		for( typename Set::iterator itt = begin; itt != end; ++itt ) {

		    if( itt == it )
			continue;

		    typename Set::value_type & p2 = *itt;

		    unsigned int coordCounter = 0;
		    unsigned int coord = 0;
		    for( unsigned int o = 0; o < p1.m_point.size(); o++ ) {
			if( p2.m_point[ o ] > p1.m_point[ o ] ) {
			    coordCounter++;
			    if( coordCounter == 2 )
				break;
			    coord = o;                            
			}
		    }

		    if( coordCounter == 1 && p1.m_boundingBox[ coord ] > p2.m_point[ coord ] )
			p1.m_boundingBox[ coord ] = p2.m_point[ coord ];
		}

		it->m_boundingBoxVolume = 1.;
		for( unsigned int i = 0 ; i < it->m_boundingBox.size(); i++ ) {

		    it->m_boundingBoxVolume *= it->m_boundingBox[ i ] - it->m_point[ i ];

		}

		for( typename Set::iterator itt = begin; itt != end; ++itt ) {
		    if( itt == it )
			continue;

		    bool isInfluencing = true;
		    double vol = 1.;
		    for( unsigned int i = 0; i < it->m_point.size(); i++ ) {

			if( itt->m_point[ i ] >= it->m_boundingBox[ i ] ) {
			    isInfluencing = false;
			    break;
			}
			vol *= std::max( itt->m_point[ i ], it->m_point[ i ] ) - it->m_boundingBox[ i ];//min_dbl(_alc_front[k][l],_alc_front[i][l]) - _alc_boundBoxLower[_alc_noObjectives*i+l];
		    }
		    if( isInfluencing ) {
			itt->m_overlappingVolume = vol;
			it->m_influencingPoints.push_back( itt );
		    }
		}

		std::sort( it->m_influencingPoints.begin(), it->m_influencingPoints.end(), VolumeComparator() );
	    }
	}
    };

    /**
     * \brief Approximately determines the point of a set contributing the least hypervolume. 
     *
     * See K. Bringmann, T. Friedrich. Approximating the least hypervolume contributor: NP-hard in general, but fast in practice. Proc. of the 5th International Conference on Evolutionary Multi-Criterion Optimization (EMO 2009), Vol. 5467 of LNCS, pages 6-20, Springer-Verlag, 2009.
     *
     * \tparam Rng The type of the Rng for sampling random points.
     * \tparam ExactHypervolume Exact hypervolume calculator that is used to speed up the approximation scheme.
     */
    template<typename Rng, typename ExactHypervolume>
    struct LeastContributorApproximator {

	/** 
	 * \brief Typedefs the type including all templates. 
	 */
	typedef LeastContributorApproximator< Rng, ExactHypervolume > this_type;

	/**
	 * \brief Models the sampling strategy.
	 */
	enum Strategy {
	    ENSURE_UNION_BOUND, ///< Ensures that the selected epsilon and delta are reached.
	    FAST ///< Puts a threshold on the number of samples.
	};

	typedef boost::bimap< Strategy, std::string > registry_type;
	static registry_type & STRATEGY_REGISTRY() {
	    static registry_type registry = boost::assign::list_of< typename registry_type::relation >( ENSURE_UNION_BOUND, "EnsureUnionBound" )( FAST, "Fast" );
	    return( registry );
	}

	friend std::ostream & operator<<( std::ostream & stream, Strategy strategy ) {
	    stream << STRATEGY_REGISTRY().left.find( strategy )->second;
	    return( stream );
	}

	friend std::istream & operator>>( std::istream & stream, Strategy & strategy ) {
	    static std::string s;
	    stream >> s;
	    strategy = STRATEGY_REGISTRY().right.find( s )->second;
	    return( stream );
	}

	/**
	 * \brief Default sampling strategy.
	 */
	static Strategy DEFAULT_STRATEGY() {
	    return( FAST );
	}

	/**
	 * \brief Default sampling strategy.
	 */
	static boost::uint_fast64_t DEFAULT_SAMPLE_THRESHOLD() {
	    return( 1000000 );
	}

	/**
	 * \brief Default delta value at start of algorithm.
	 */
	static const double DEFAULT_START_DELTA() { 
	    return( 0.1 ); 
	}

	/**
	 * \brief Default multiplier value for adjusting delta.
	 */
	static const double DEFAULT_MULTIPLIER_DELTA() { 
	    return( 0.775 ); 
	}

	/**
	 * \brief Default multiplier value for adjusting delta.
	 */
	static const double DEFAULT_MINIMUM_MULTIPLIER_DELTA() { 
	    return( 0.2 ); 
	}

	/**
	 * \brief Default threshold for sample count.
	 */
	static const boost::uint_fast64_t DEFAULT_MAX_NUM_SAMPLES() { 
	    return( std::numeric_limits<boost::uint_fast64_t>::max() ); 
	}

	/**
	 * \brief Default gamma value.
	 */
	static const double DEFAULT_GAMMA() { 
	    return( 0.25 ); 
	}

	/**
	 * \brief Models a point and associated information for book-keeping purposes.
	 */ 
	template<typename VectorType>
	struct Point {

	    typedef typename std::list< VectorType >::iterator sample_iterator;
	    typedef typename std::list< VectorType >::const_iterator const_sample_iterator;

	    Point( unsigned int noObjectives, const VectorType & point, const VectorType & refPoint ) : m_point( point ),
													m_sample( point.size() ),
													m_boundingBox( refPoint ),
													m_boundingBoxVolume( 0. ),
													m_approximatedContribution( 0. ),
													m_noOperations( 0 ),
													m_noSamples( 0 ),
													m_noSuccessfulSamples( 0 ) {
	    }

	    VectorType m_point;
	    VectorType m_sample;
	    VectorType m_boundingBox;

	    std::list< VectorType > m_successfulSamples;
	    std::list< VectorType > m_nonSuccessfulSamples;

	    std::vector< typename std::vector<Point>::const_iterator > m_influencingPoints;

	    double m_boundingBoxVolume;
	    double m_approximatedContribution;
	    double m_overlappingVolume;

	    unsigned long long m_noOperations;
	    unsigned long long m_noSamples;
	    unsigned long long m_noSuccessfulSamples;

	    template<typename Stream>
	    void print( Stream & s ) const {

		std::copy( m_point.begin(), m_point.end(), std::ostream_iterator<double>( s, " " ) );
		// s << " || ";
		std::copy( m_boundingBox.begin(), m_boundingBox.end(), std::ostream_iterator<double>( s, " " ) );
		// s << " || ";
		s << m_noOperations << " " << m_noSamples << " " << m_noSuccessfulSamples << " " << m_boundingBoxVolume << std::endl;

	    }
	};

	/**
	 * \brief Returns the supplied argument.
	 */
	struct IdentityFitnessExtractor {
	    template<typename Member>
	    const Member & operator()( const Member & member ) const {
		return( member );
	    }
	};

	double m_logFactor;
	double m_startDelta;
	double m_multiplierDelta;
	double m_minimumMultiplierDelta;
	unsigned long long m_maxNumSamples;
	double m_gamma;

	unsigned int m_round;

	Strategy m_strategy;
	boost::uint_fast64_t m_sampleCounter;
	boost::uint_fast64_t m_sampleCountThreshold;

	/**
	 * \brief C'tor
	 * \param [in] startDelta Initial delta value.
	 * \param [in] multiplierDelta Multiplier for adjusting the delta value.
	 * \param [in] minimumDeltaMultiplier Multiplier for adjusting the delta value.
	 * \param [in] maxNumSamples The maximum number of samples. If reached, the algorithm aborts.
	 * \param [in] gamma 
	 */
	LeastContributorApproximator( double startDelta = this_type::DEFAULT_START_DELTA(),
				      double multiplierDelta = this_type::DEFAULT_MULTIPLIER_DELTA(),
				      double minimumDeltaMultiplier = this_type::DEFAULT_MINIMUM_MULTIPLIER_DELTA(),
				      unsigned long long maxNumSamples = this_type::DEFAULT_MAX_NUM_SAMPLES(),
				      double gamma = this_type::DEFAULT_GAMMA() 
				      ) : m_startDelta( startDelta ),
					  m_multiplierDelta( multiplierDelta ),
					  m_minimumMultiplierDelta( minimumDeltaMultiplier ),
					  m_maxNumSamples( maxNumSamples ),
					  m_gamma( gamma ),
					  m_round( 0 ),
					  m_strategy( this_type::DEFAULT_STRATEGY() ),
					  m_sampleCounter( 0 ),
					  m_sampleCountThreshold( this_type::DEFAULT_SAMPLE_THRESHOLD() ){

	}

	template<typename Node>
	void configure( const Node & node ) {
	    m_startDelta = node.template get<double>( "StartDelta", this_type::DEFAULT_START_DELTA() );
	    m_multiplierDelta = node.template get< double >( "MultiplierDelta", this_type::DEFAULT_MULTIPLIER_DELTA());
	    m_minimumMultiplierDelta = node.template get< double >( "MinimumDeltaMultiplier", this_type::DEFAULT_MINIMUM_MULTIPLIER_DELTA() );
	    m_maxNumSamples = node.get( "MaxNumSamples", static_cast<unsigned long long>(this_type::DEFAULT_MAX_NUM_SAMPLES()) );
	    m_gamma = node.template get< double >( "Gamma", this_type::DEFAULT_GAMMA() );
	    m_strategy = node.template get< Strategy >( "Strategy", this_type::DEFAULT_STRATEGY() );
	    m_sampleCountThreshold = node.get( "SampleCountThreshold", static_cast<unsigned long long>(this_type::DEFAULT_SAMPLE_THRESHOLD()) );

	    std::cout << "Configure: " << m_strategy << std::endl;
	    std::cout << "Configure: " << m_sampleCountThreshold << std::endl;
	}

	/** 
	 * \brief Samples in the bounding box of the supplied point until a pre-defined threshold is reached.
	 * \param [in] s Set of points.
	 * \param [in] point Iterator to the point that should be sampled.
	 * \param [in] r The current round.
	 * \param [in] delta The delta that should be reached.
	 * \param [in] refPoint Reference point for hypervolume calculation/approximation.
	 */ 
	template<typename Set, typename VectorType>
	void sample( const Set & s, typename Set::iterator point, unsigned int r, double delta, const VectorType & refPoint ) {

	    if( point->m_noSamples >= m_maxNumSamples )
		return;


	    if( point->m_noOperations >= ExactHypervolume::runtime( point->m_influencingPoints.size(), refPoint.size() ) ) {

		point->m_noSamples = point->m_noSuccessfulSamples = m_maxNumSamples;

		std::vector< VectorType > neighborhood( point->m_influencingPoints.size(), refPoint );
		for( unsigned int i = 0; i < point->m_influencingPoints.size(); i++ ) {
		    for( unsigned int j = 0; j < refPoint.size(); j++  )
			neighborhood[i][j] = std::max( point->m_point[ j ], (point->m_influencingPoints[i])->m_point[ j ] ) + ( refPoint[ j ] - point->m_boundingBox[ j ] );
		}

		IdentityFitnessExtractor e;
		ExactHypervolume hv;
		point->m_approximatedContribution = point->m_boundingBoxVolume - hv( e, neighborhood, refPoint, refPoint.size() );
		point->m_boundingBoxVolume = point->m_approximatedContribution;
		return;
	    }

	    Sampler<Rng> sampler;

	    double threshold = 0.5 * ( (1. + m_gamma) * ::log( static_cast<double>( r ) ) + m_logFactor ) * sqr( point->m_boundingBoxVolume / delta );
	    for( ; point->m_noSamples == 0 || point->m_noSamples < threshold; point->m_noSamples++ ) {
		if( m_strategy == FAST )
		    if( m_sampleCounter > m_sampleCountThreshold )
			throw( SHARKEXCEPTION( "LeastContributorApproximator::sample(): Maximum number of total samples reached." ) );
		if( sampler( s, point ) )
		    point->m_noSuccessfulSamples++;

		m_sampleCounter++;
	    }

	    point->m_approximatedContribution = point->m_boundingBoxVolume * ( static_cast<double>( point->m_noSuccessfulSamples ) / static_cast<double>( point->m_noSamples ) );
	}

	template<typename iterator>
	double deltaForPoint( iterator point, unsigned int R ) {

	    return( ::sqrt( 0.5 * ((1. + m_gamma) * ::log( static_cast<double>( R ) ) + m_logFactor ) / point->m_noSamples ) * point->m_boundingBoxVolume );

	}

	/**
	 * \brief Determines the point contributing the least hypervolume to the overall set of points.
	 * \param [in] e Extracts point information from set elements.
	 * \param [in] s Set of points.
	 * \param [in] refPoint The reference point to consider for calculating individual points' contributions.
	 * \param [in] noObjectives The dimension of the objective space.
	 * \param [in] delta The error probability.
	 * \param [in] eps The error bound.
	 */
	template<typename Extractor, typename Set, typename VectorType>
	typename Set::const_iterator operator()( Extractor & e, const Set & s, const VectorType & refPoint, unsigned int noObjectives, double delta, double eps ) {

	    std::vector< Point<VectorType> > front;
	    front.reserve( s.size() );

	    std::vector< typename std::vector< Point<VectorType> >::iterator > activePoints;

	    for( typename Set::const_iterator it = s.begin(); it != s.end(); ++it ) {
		front.push_back( Point<VectorType>( noObjectives, e( *it ), refPoint ) );
	    }

	    for( typename std::vector< Point<VectorType> >::iterator it = front.begin(); it != front.end(); ++it )
		activePoints.push_back( it );

	    BoundingBoxComputer< std::vector< Point<VectorType> > > bbc;
	    bbc( front.begin(), front.end() );

	    double maxBoundingBoxVolume = 0.;

	    for( typename std::vector< Point<VectorType> >::iterator it = front.begin(); it != front.end(); ++it )
		maxBoundingBoxVolume = std::max( maxBoundingBoxVolume, it->m_boundingBoxVolume );

	    double _delta = m_startDelta * maxBoundingBoxVolume;
	    m_logFactor = ::log( 2. * front.size() * (1. + m_gamma) / (delta * m_gamma) );
	    double minApprox = std::numeric_limits<double>::max();

	    typename std::vector<Point< VectorType > >::iterator minimalElement;
	    m_round = 0;
	    m_sampleCounter = 0;

	    while( activePoints.size() > 1 ) {
		m_round++;

		minApprox = std::numeric_limits<double>::max();
		minimalElement = front.end();

		for( int i = 0; i < static_cast<int>( activePoints.size() ); i++ )
		    try {
			sample( front, activePoints[i], m_round, _delta, refPoint );
		    } catch( ... ) {
			std::size_t idx = std::distance( front.begin(), minimalElement );
			return( s.begin() + idx );
		    }

		for( typename std::vector< typename std::vector< Point<VectorType> >::iterator >::iterator it = activePoints.begin(); it != activePoints.end(); ++it ) {
		    if( (*it)->m_approximatedContribution < minApprox ) {
			minApprox = (*it)->m_approximatedContribution;
			minimalElement = *it;
		    }                    
		}

		if( activePoints.size() > 2 ) {
		    try {
			sample( front, minimalElement, m_round, m_minimumMultiplierDelta * _delta, refPoint );
		    } catch( ... ) {
			std::size_t idx = std::distance( front.begin(), minimalElement );
			return( s.begin() + idx );
		    }
		}

		minApprox = std::numeric_limits<double>::max();
		for( typename std::vector< typename std::vector< Point<VectorType> >::iterator >::iterator it = activePoints.begin(); it != activePoints.end(); ++it ) {
		    if( (*it)->m_approximatedContribution < minApprox ) {
			minApprox = (*it)->m_approximatedContribution;
			minimalElement = *it;
		    }
		}

		typename std::vector< typename std::vector< Point<VectorType> >::iterator >::iterator it = activePoints.begin();
		while( it != activePoints.end() ) {
		    if( (*it)->m_approximatedContribution - minApprox > deltaForPoint( *it, m_round ) + deltaForPoint( minimalElement, m_round ) )
			it = activePoints.erase( it );
		    else
			++it;
		}

		if( activePoints.size() <= 1 )
		    break;

		double d = 0;
		for( it = activePoints.begin(); it != activePoints.end(); ++it ) {
		    if( *it == minimalElement )
			continue;
		    double nom = minApprox + deltaForPoint( minimalElement, m_round );
		    double den = (*it)->m_approximatedContribution - deltaForPoint( *it, m_round );

		    if( den <= 0. )
			d = std::numeric_limits<double>::max();
		    else if( d < nom/den )
			d = nom/den;
		}

		if( d < 1. + eps )
		    break;

		_delta *= m_multiplierDelta;
	    }

	    std::size_t idx = std::distance( front.begin(), minimalElement );
	    return( s.begin() + idx );
	}
    };
}

#endif
