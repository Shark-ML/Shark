/*!
 * 
 * \file        Uniform.h
 *
 * \brief       Implements a uniform distribution.
 * 
 * 
 *
 * \author      O. Krause
 * \date        2010-01-01
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
#ifndef SHARK_RNG_UNIFORM_H
#define SHARK_RNG_UNIFORM_H

#include "shark/Rng/AbstractDistribution.h"
#include "shark/Rng/Rng.h"

#include <boost/random.hpp>

namespace shark{

/**
* \brief Implements a continuous uniform distribution.
*/
template<typename RngType = DefaultRngType>
class Uniform
:
	public AbstractDistribution,
	public boost::variate_generator<RngType*,boost::uniform_real<> >
{
private:

	typedef boost::variate_generator<RngType*,boost::uniform_real<> > Base;

public:

	/**
	* \brief Default c'tor. Initializes the sampling interval and associates
	* this distribution with the supplied RNG.
	* \param [in,out] rng The RNG to associate this distribution with.
	* \param [in] low_ The lower bound of the sampling interval.
	* \param [in] high_ The upper bound of the sampling interval.
	*/
	Uniform( RngType & rng, double low_ = 0., double high_ = 1. )
		:Base(&rng,boost::uniform_real<>(std::min(low_,high_),std::max(low_,high_)))
	{}

	/**
	* \brief Injects the default sampling operator.
	*/
	using Base::operator();

	/**
	* \brief Resets the distribution to the supplied interval and samples a random number.
	* \param [in] low_ The lower bound of the interval.
	* \param [in] high_ The upper bound of the interval.
	*/
	double operator()(double low_,double high_)
	{
		if(low_ == high_) return low_;
		boost::uniform_real<> dist( std::min(low_,high_), std::max( high_, low_ ) );
		return dist(Base::engine());
	}

	/**
	* \brief Accesses the lower bound of the interval.
	*/
	double low()const
	{
		return Base::distribution().min();
	}

	/**
	* \brief Accesses the upper bound of the interval.
	*/
	double high()const
	{
		return Base::distribution().max();
	}

	/**
	* \brief Adjusts the interval of the distribution.
	* \param [in] low_ The lower bound of the interval.
	* \param [in] high_ The upper bound of the interval.
	*/
	void setRange(double low_,double high_)
	{
		boost::uniform_real<> dist(std::min(low_,high_),std::max(low_,high_));
		Base::distribution()=dist;
	}

	/**
	* \brief Calculates the probability of x.
	* \param [in] x The observation.
	*/
	double p(double x) const {
		return x >= low() && x < high() ? 1 / (high() - low()) : 0;
	}
};

} // namespace shark {

#endif // SHARK_RNG_UNIFORM_H
