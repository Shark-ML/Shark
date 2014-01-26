/*!
 * 
 *
 * \brief       Implements a poisson distribution.
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
#ifndef SHARK_RNG_POISSON_H
#define SHARK_RNG_POISSON_H

#include <shark/Rng/Rng.h>

#include <boost/random.hpp>
#include <boost/random/poisson_distribution.hpp>

namespace shark {

	/**
	* \brief Implements a Poisson distribution with parameter mean.
	*/
	template<typename RngType = shark::DefaultRngType>
	class Poisson : public boost::variate_generator< RngType*,boost::poisson_distribution<> > {
	private:
		typedef boost::variate_generator< RngType*,boost::poisson_distribution<> > Base;

	public:

		/**
		* \brief C'tor taking parameter mean as argument. Associates the
		* distribution with supplied RNG.
		* 
		*/
		Poisson( RngType & rng, double mean = 0.01 )
			:Base(&rng,boost::poisson_distribution<>(mean))
		{}

		/**
		* \brief Injects the default sampling operator.
		*/
		using Base::operator();

		/**
		* \brief Adjusts the parameter mean and samples the distribution.
		* \param [in] mean The new mean.
		*/
		double operator()(double mean)
		{
			boost::poisson_distribution<> dist(mean);
			return dist(Base::engine());
		}

		/**
		* \brief Accesses the parameter mean.
		*/
		double mean()const
		{
			return Base::distribution().mean();
		}

		/**
		* \brief Adjusts the parameter mean.
		*/
		void mean(double newMean)
		{
			Base::distribution()=boost::poisson_distribution<>(newMean);
		}

		/**
		* \brief Calculates the probability of x >= 0.
		*/
		double p(double x)const
		{
			if(x >= 0.0)
				return std::pow(mean(), x) * std::exp(-mean()) / boost::math::factorial<double>(std::size_t(x));
			else
				return 0;
		}

	};
}
#endif
