/**
*
*  \brief Implements a univariate normal distribution.
*
*  \author  O. Krause
*  \date    2010-01-01
*  \par Copyright (c) 1998-2007:
*      Institut f&uuml;r Neuroinformatik<BR>
*      Ruhr-Universit&auml;t Bochum<BR>
*      D-44780 Bochum, Germany<BR>
*      Phone: +49-234-32-25558<BR>
*      Fax:   +49-234-32-14209<BR>
*      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
*      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
*      <BR>
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
*/
#ifndef SHARK_RNG_NORMAL_H
#define SHARK_RNG_NORMAL_H

#include "shark/Core/Math.h"
#include "shark/Rng/AbstractDistribution.h"
#include "shark/Rng/Rng.h"

#include <boost/math/special_functions.hpp>
#include <boost/random.hpp>

#include <cmath>

namespace shark {

/// \brief Implements a univariate normal (Gaussian) distribution.
///
/// For backwards compatibility with older shark versions
/// instead of the standard deviation sigma the
/// variance=sigma^2 is used as argument.
template<typename RngType = DefaultRngType>
class Normal
:
	public AbstractDistribution,
	public boost::variate_generator<RngType*, boost::normal_distribution<> >
{
private:
	typedef boost::variate_generator<RngType*, boost::normal_distribution<> > Base;

public:
	/// constructor
	/// \param rng: random number generator
	/// \param mean: mean of distribution
	/// \param variance: variance of distribution
	Normal( RngType & rng, double mean = 0., double variance =1. )
		:Base(&rng,boost::normal_distribution<>(mean,std::sqrt(variance)))
	{}

	using Base::operator();

	double operator()(double mean,double variance)
	{
		boost::normal_distribution<> dist(mean,std::sqrt(variance));
		return dist(Base::engine());
	}

	double mean()const
	{
		return Base::distribution().mean();
	}

	double variance()const
	{
		return Base::distribution().sigma() * Base::distribution().sigma();
	}

	void mean(double newMean)
	{
		Base::distribution() = boost::normal_distribution<>(newMean, Base::distribution().sigma());
	}

	void variance(double newVariance)
	{
		Base::distribution() = boost::normal_distribution<>(mean(), std::sqrt(newVariance));
	}

	double p(double x) const
	{
		const double standardDeviation = Base::distribution().sigma();
		return std::exp(-sqr((x - mean()) / standardDeviation) / 2.0) / (SQRT_2_PI * standardDeviation);
	}

	double logP(double x) const
	{
		const double standardDeviation = Base::distribution().sigma();
		return (-sqr((x - mean()) / standardDeviation) / 2.0) - safeLog(SQRT_2_PI * standardDeviation);
	}
};

template<typename RngType>
double entropy(const Normal< RngType > & normal);

} // namespace shark {

#endif // SHARK_RNG_NORMAL_H
