/*!
 * 
 * \file        LogNormal.h
 *
 * \brief       Provides a log-normal distribution.
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
#ifndef SHARK_RNG_LOGNORMAL_H
#define SHARK_RNG_LOGNORMAL_H

#include <shark/Core/Math.h>

#include <boost/random/lognormal_distribution.hpp>

#include <cmath>

namespace shark {
	/**
	* \brief Implements a log-normal distribution with parameters location m and Scale s.
	*
	*  The propability distribution is
	* \f[ p(x)= \frac {1} {x s \sqrt{2 \pi}}e^{-\frac{(\ln x -m)^2}{2 s^2}}\f]
	*/
	//unfortunately, boost::math::lognormal_distribution uses scale/location as parameters as well as the next c++ standard, while
	//boost::lognormal_distribution uses mean and sigma. in boost 1.47 there exists boost::random::lognormal_distribution which uses
	//scale/location as well, but we use boost 1.44. For forward compliance, we use the new parameter set but transform them back to boost
	//parameters.
	template<typename RngType = shark::DefaultRngType>
	class LogNormal : public boost::variate_generator<RngType*,boost::lognormal_distribution<> >
	{
	private:
		typedef boost::variate_generator<RngType*,boost::lognormal_distribution<> > Base;
		double m_location;
		double m_scale;
	
		
		boost::lognormal_distribution<> createDistribution(double location,double scale){
			double mean = std::exp(location+scale*scale/2.0);
			double variance = (std::exp(scale*scale)-1.0)* std::exp(2*location+scale*scale);
			return boost::lognormal_distribution<>(mean,std::sqrt(variance));
		}
	public:
		/**
		* \brief C'tor, associates the distribution with a custom RNG.
		* \param [in,out] rng The custom rng.
		* \param [in] location The location of the distribution, default value is 0.
		* \param [in] scale The scale of the distribution, default value is 1.
		*/
		LogNormal( RngType & rng, 
			   double location = 0,
			   double scale= 1 )
			: Base(&rng,boost::lognormal_distribution<>(1,1)),m_location(location),m_scale(scale)
		{
			Base::distribution() = createDistribution(location,scale); 
		}

		/** \brief Injects the default sampling operator. */
		using Base::operator();

		/**
		* \brief Samples a random number.
		* \param [in] location The location of the distribution.
		* \param [in] scale The scale of the distribution.
		*/
		double operator()(double location, double scale) {
			return dist(createDistribution(location,scale) );
		}

		/**
		* \brief Accesses the location of the distribution.
		*/
		double location() const { 
			return m_location; 
		}

		/**
		* \brief Accesses the scale of the distribution.
		*/
		double scale() const { 
			return m_scale; 
		}

		/**
		* \brief Adjusts the location of the distribution.
		*/
		void location(double newLocation)
		{ 
			m_location = newLocation;
			
			Base::distribution() = createDistribution(location(),scale()); 
		}

		/**
		* \brief Adjusts the scale of the distribution.
		*/
		void scale(double newScale)
		{ 
			m_scale = newScale;
			Base::distribution() = createDistribution(location(),scale()); 
		}

		/**
		* \brief Calculates the probability of x > 0.
		*/
		double p(double x) const {

			double y = (std::log(x) - location()) / scale();
			return x > 0 ? std::exp(-y*y / 2.0) / (boost::math::constants::root_two_pi<double>() * scale() * x) : 0;
		}
	};


}
#endif
