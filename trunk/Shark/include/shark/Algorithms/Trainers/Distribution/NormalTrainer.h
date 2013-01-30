//===========================================================================
/*!
 *  \brief Trainer for normal distribution
 *
 *
 *  \author  B. Li
 *  \date    2012
 *
 *  \par Copyright (c) 2012:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *
 *
 *  <BR><HR>
 *  This file is part of Shark. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 3, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, see <http://www.gnu.org/licenses/>.
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
