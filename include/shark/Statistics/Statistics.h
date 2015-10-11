/*!
 * 
 *
 * \brief       Calculate statistics given a range of values.
 * 
 * 
 *
 * \author      T.Voss, T. Glasmachers, O.Krause
 * \date        2010-2011
 *
 *
 * \par Copyright 1995-2015 Shark Development Team
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
#ifndef SHARK_STATISTICS_H
#define SHARK_STATISTICS_H

#include <shark/Core/Flags.h>

#include <boost/range/iterator_range.hpp>
#include <boost/optional.hpp>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>

#include <boost/accumulators/statistics/count.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/median.hpp>
#include <boost/accumulators/statistics/moment.hpp>
#include <boost/accumulators/statistics/p_square_quantile.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <iostream>
#include <vector>

namespace ba = boost::accumulators;

namespace shark {

    /**
     * \brief Calculate pre-defined statistics given a range of values.
     *
     * \sa examples/Statistics/StatisticsMain.cpp
     *
     * Calculate statistics from standard in:
     * \code
     * shark::Statistics stats;
     * stats = std::for_each( std::istream_iterator<double>( std::cin ), std::istream_iterator<double>(), stats );
     * std::cout << stats << std::endl;
     * \endcode
     * Implemented in terms of boost::accumulators.
     */
    struct Statistics {

	/** \cond IMPL */
	typedef ba::accumulator_set<
	double, 
	    ba::stats<
	    ba::tag::median(ba::with_p_square_quantile), 		   
		   ba::tag::density,
		   ba::tag::mean, 
		   ba::tag::variance,
		   ba::tag::min, 
		   ba::tag::max, 
		   ba::tag::count 
		   > 
		   > AccumulatorType;
	typedef ba::accumulator_set<double, ba::stats<ba::tag::p_square_quantile> > QuartileAccumulatorType;

	typedef double LowerQuantileProbability;
	typedef double UpperQuantileProbability;
	/** \endcond IMPL */

	/** \brief Histogram type */
	typedef boost::iterator_range< 
	std::vector< 
	std::pair<double,double> 
	    >::iterator 
	    > histogram_type;

	/**
	 * \brief Tags the mean value.
	 */
	struct Mean 				{};
	/**
	 * \brief Tags the variance.
	 */
	struct Variance 			{};
	/**
	 * \brief Tags the unbiased variance (not implemented).
	 */
	struct UnbiasedVariance 	{};
	/**
	 * \brief Tags the histogram.
	 */ 
	struct Histogram 			{};
	/**
	 * \brief Tags the median.
	 */
	struct Median 				{};
	/**
	 * \brief Tags the lower quartile.
	 */
	struct LowerQuartile 		{};
	/**
	 * \brief Tags the upper quartile.
	 */
	struct UpperQuartile 		{};
	/**
	 * \brief Tags the minimum value.
	 */
	struct Min					{};
	/**
	 * \brief Tags the maximum value.
	 */
	struct Max					{};
	/**
	 * \brief Tags the number of samples.
	 */
	struct NumSamples			{};

	/**
	 * \brief Default c'tor.
	 * \param [in] lowerQuantileProbability Probability for the lower quantile, default value: 0.25.
	 * \param [in] upperQuantileProbability Probability for the upper quantile, default value: 0.75.
	 */
    Statistics( double lowerQuantileProbability = 0.25, double upperQuantileProbability = 0.75 ) : m_acc( ba::density_cache_size = 5, ba::density_num_bins = 20 ),
	    m_accLowerQuartile( ba::quantile_probability = lowerQuantileProbability ),
	    m_accUpperQuartile( ba::quantile_probability = upperQuantileProbability ) {
    }

	/**
	 * \brief Accesses the mean value of the supplied values.
	 */
	double operator()( Mean mean ) const { return( ba::mean( m_acc ) ); }

	/**
	 * \brief Accesses the variance of the supplied values.
	 */
	double operator()( Variance variance ) const { return( ba::variance( m_acc ) ); }

	/**
	 * \brief Accesses the histogram of the supplied values.
	 */
	histogram_type operator()( Histogram histogram ) const { return( ba::density( m_acc ) ); }
	/**
	 * \brief Accesses the median of the supplied values.
	 */
	double operator()( Median median ) const { return( ba::median( m_acc ) ); }

	/**
	 * \brief Accesses the lower quartile of the supplied values.
	 */
	double operator()( LowerQuartile lq ) const { return( ba::p_square_quantile( m_accLowerQuartile ) ); }

	/**
	 * \brief Accesses the upper quartile of the supplied values.
	 */
	double operator()( UpperQuartile uq ) const { return( ba::p_square_quantile( m_accUpperQuartile ) ); }

	/**
	 * \brief Accesses the minimum of the supplied values.
	 */
	double operator()( Min min ) const { return( ba::min( m_acc ) ); }

	/**
	 * \brief Accesses the maximum of the supplied values.
	 */
	double operator()( Max max ) const { return( ba::max( m_acc ) ); }

	/**
	 * \brief Accesses the total number of samples.
	 */
	std::size_t operator()( NumSamples numSamples ) const { return( ba::count( m_acc ) ); }

	/**
	 * \brief Updates statistics with the supplied value.
	 * \param [in] d The value.
	 */
	void operator()( double d ) {
	    m_acc( d );
	    m_accLowerQuartile( d );
	    m_accUpperQuartile( d );
	}

	/**
	 * \brief Calculates statistics for the supplied range of values.
	 * \tparam InputIterator Iterator type, needs to be a model of forward iterator.
	 * \param [in] begin Iterator pointing to the first valid element of the range.
	 * \param [in] end Iterator pointing behind the last valid element of the range.
	 */
	template<class InputIterator>
	void operator()( InputIterator begin , InputIterator end ) {
	    for(;begin != end; ++begin){
		(*this)(*begin);
	    }
	}

	/** \cond IMPL */
	AccumulatorType m_acc;
	QuartileAccumulatorType m_accLowerQuartile;
	QuartileAccumulatorType m_accUpperQuartile;
	/** \endcond IMPL */
    };

    /**
     * \brief Writes statistics to the supplied stream.
     */
    template<typename CharT, typename Traits>
	static std::basic_ostream<CharT,Traits> & operator<<( std::basic_ostream<CharT,Traits> & s, const Statistics & stats ) {
	s << "Sample size: " 		<< stats( shark::Statistics::NumSamples() ) << std::endl;
	s << "Min: " 				<< stats( shark::Statistics::Min() ) << std::endl;
	s << "Max: " 				<< stats( shark::Statistics::Max() ) << std::endl;
	s << "Mean: " 				<< stats( shark::Statistics::Mean() ) << std::endl;
	s << "Variance: " 			<< stats( shark::Statistics::Variance() ) << std::endl;
	s << "Median: " 			<< stats( shark::Statistics::Median() ) << std::endl;
	s << "Lower Quantile: " 	<< stats( shark::Statistics::LowerQuartile() ) << std::endl;
	s << "Upper Quantile: " 	<< stats( shark::Statistics::UpperQuartile() ) << std::endl;

	return( s );
    }
}

#endif // SHARK_STATISTICS_H
