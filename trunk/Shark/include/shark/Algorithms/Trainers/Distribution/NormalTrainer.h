//===========================================================================
/*!
 * 
 *
 * \brief       Trainer for normal distribution
 * 
 * 
 * 
 *
 * \author      B. Li
 * \date        2012
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
//===========================================================================
#ifndef SHARK_ALGORITHMS_TRAINERS_DISTRIBUTION_NORMAL_H
#define SHARK_ALGORITHMS_TRAINERS_DISTRIBUTION_NORMAL_H

#include "shark/Rng/Normal.h"
#include "shark/Rng/Rng.h"

#include <boost/accumulators/framework/accumulator_set.hpp>
#include <boost/accumulators/statistics/count.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/bind/bind.hpp>
#include <boost/range/algorithm/for_each.hpp>

namespace shark {

/// Trainer for normal distribution
class NormalTrainer
{
public:

	/// The type of variance. The difference between them is:
	/// When you have "N" data values that are:
	/// By Population: divide by N when calculating Variance
	/// By Sample: divide by N-1 when calculating Variance
	enum VarianceType
	{
		VARIANCE_BY_POPULATION,
		VARIANCE_BY_SAMPLE
	};

	/// Constructor
	NormalTrainer(VarianceType varianceType = VARIANCE_BY_SAMPLE) : m_varianceType(varianceType) {}

	/// Internal implementation for trainer of normal distribution
	template <typename RngType>
	void train(Normal<RngType>& normal, const std::vector<double>& input) const
	{
		SIZE_CHECK(input.size() > 1u);
		namespace bae = boost::accumulators::extract;

		InternalAccumulatorType accu;
		boost::range::for_each(input, boost::bind(boost::ref(accu), _1));
		SIZE_CHECK(bae::count(accu) > 1u);

		normal.mean(bae::mean(accu));
		normal.variance(
			VARIANCE_BY_SAMPLE == m_varianceType
			? bae::variance(accu) * bae::count(accu) / (bae::count(accu) - 1)
			: bae::variance(accu));
	}

private:

	/// The covariance type this trainer will use
	VarianceType m_varianceType;

	/// Internal accumulator type
	typedef boost::accumulators::accumulator_set<
		double,
		boost::accumulators::stats<
			boost::accumulators::tag::count,
			boost::accumulators::tag::variance> > InternalAccumulatorType;
};

} // namespace shark {

#endif // SHARK_ALGORITHMS_TRAINERS_DISTRIBUTION_NORMAL_H
