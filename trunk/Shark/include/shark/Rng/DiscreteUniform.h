/*!
 * 
 * \file        DiscreteUniform.h
 *
 * \brief       Discrete Uniform distribution
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
#ifndef SHARK_RNG_DISCRETEUNIFORM_H
#define SHARK_RNG_DISCRETEUNIFORM_H

#include <shark/Rng/Rng.h>

#include <boost/random.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace shark{
	/** 
	* \brief Implements the discrete uniform distribution.
	*/
	template<typename RngType = shark::DefaultRngType>
	class DiscreteUniform : public boost::variate_generator<RngType*,boost::uniform_int<long> > {
	private:
		/** \brief The base type this class inherits from. */
		typedef boost::variate_generator<RngType*,boost::uniform_int<long> > Base;
	public:
		/**
		* \brief C'tor, initializes the interval the random numbers are sampled from.
		* \param [in] low_ The lower bound of the interval, defaults to 0.
		* \param [in] high_ The upper bound of the interval, defaults to std::numeric_limits<long>::max().
		*/
		/*DiscreteUniform( long low_ = 0,long high_ = std::numeric_limits<long>::max() )
			:Base(&Rng::globalRng,boost::uniform_int<long>(std::min(low_,high_),std::max(low_,high_)))
		{}*/

		/**
		* \brief C'tor, initializes the interval the random numbers are sampled from and associates the distribution
		* with the supplied RNG.
		* \param [in,out] rng The RNG this distribution is associated with.
		* \param [in] low_ The lower bound of the interval, defaults to 0.
		* \param [in] high_ The upper bound of the interval, defaults to std::numeric_limits<long>::max().
		*/
		DiscreteUniform(RngType & rng, long low_ = 0,long high_ = std::numeric_limits<long>::max() )
			:Base(&rng,boost::uniform_int<long>(std::min(low_,high_),std::max(low_,high_)))
		{}

		/**
		* \brief Injects the default sampling operator.
		*/
		using Base::operator();

		/**
		* \brief Reinitializes the distribution for the supplied bounds and samples a new random number.
		* Default values are omitted to distinguish the operator from the default one.
		* 
		* \param [in] low_ The lower bound of the interval.
		* \param [in] high_ The upper bound of the interval.
		*/
		typename Base::result_type operator()(long low_,long high_)
		{
			boost::uniform_int<long> dist(std::min(low_,high_),std::max(low_,high_));
			return dist(Base::engine());
		}

		/** \brief Returns the lower bound of the interval. */
		long low()const
		{
			return Base::distribution().min();
		}

		/** \brief Adjusts the upper bound of the interval */
		long high()const
		{
			return Base::distribution().max();
		}

		/** \brief Adjusts the range of the interval. */
		void setRange(long low_,long high_)
		{
			boost::uniform_int<long> dist(std::min(low_,high_),std::max(low_,high_));
			Base::distribution()=dist;
		}

		/** \brief Calculates the probability of x. */
		double p( long x ) const {
			return 1.0/(high()-low()+1);
		}

	};

	template<typename RngType>
	double entropy(const DiscreteUniform<RngType> & uniform);

}
#endif
