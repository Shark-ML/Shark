/*!
 * 
 *
 * \brief       Implements a binomial distribution.
 * 
 * 
 *
 * \author      O. Krause
 * \date        2010-01-01
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
#ifndef SHARK_RNG_BINOMIAL_H
#define SHARK_RNG_BINOMIAL_H

#include <shark/Rng/Rng.h>

#include <boost/random.hpp>
#include <boost/math/special_functions.hpp>
#include <boost/random/binomial_distribution.hpp>

#include <cmath>

namespace shark{

	/**
	* \brief Models a binomial distribution with parameters p and n.
	*/
	template<typename RngType = shark::DefaultRngType>
	class Binomial:public boost::variate_generator<RngType*,boost::binomial_distribution<> > {
	private:
		typedef boost::variate_generator<RngType*,boost::binomial_distribution<> > Base;
	public:

		/**
		* \brief C'tor, initializes parameters n and p, initializes for a custom RNG.
		* \param [in,out] rng The random number generator.
		* \param [in] n Parameter n.descibing the number of coin tosses
		* \param [in] prob Parameter p.
		*/
		Binomial(RngType & rng, unsigned int n=1,double prob=0.5 )
			:Base(&rng,boost::binomial_distribution<>(n,prob))
		{}

		/**
		* \brief Injects the default sampling operator.
		*/
		using Base::operator();

		/**
		* \brief Samples a random number from the distribution with parameter n and p.
		*/
		long operator()(unsigned int n,double prob) {
			boost::binomial_distribution<> dist(n,prob);
			return dist(Base::engine());
		}

		/**
		* \brief Accesses the parameter p of the distirbution.
		*/
		double prob()const {
			return Base::distribution().p();
		}

		/**
		* \brief Adjusts the parameter p of the distribution.
		*/
		void prob(double newProb) {
			Base::distribution()=boost::binomial_distribution<>(n(),newProb);
		}

		/**
		* \brief Accesses the parameter n of the distribution.
		*/
		unsigned int n() const {
			return Base::distribution().t();
		}

		/**
		* \brief Adjusts the parameter n of the distribution.
		*/
		void n(unsigned int newN) {
			Base::distribution()=boost::binomial_distribution<>(newN,prob());
		}

		/**
		* \brief Implements the pmf of the distribution.
		* \param [in] k Number of successful trials.
		* \returns The probability of k successful in n total trials.
		* \throws std::overflow_error if the result is too large to be represented in type double.
		*/
		double p(long k) const {
			return( 
				boost::math::binomial_coefficient<double>(n(),k) * 
				std::pow(prob(), static_cast<double>( k ) ) *
				std::pow(1.0-prob(),static_cast<double>( n()-k ) )
				);
		}

	};

	template<typename RngType>
	inline double entropy(const Binomial<RngType> & coin);
}
#endif




