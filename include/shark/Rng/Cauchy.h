/*!
 * 
 *
 * \brief       Standard Cauchy distribution
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
#ifndef SHARK_RNG_CAUCHY_H
#define SHARK_RNG_CAUCHY_H

#include <shark/Core/Math.h>
#include <shark/Rng/Rng.h>

#include <boost/random.hpp>
#include <boost/random/cauchy_distribution.hpp>
#include <cmath>

namespace shark{

	/*!
	*  \brief Cauchy distribution
	*
	*  This class is a thin wrapper for the boost::cauchy_distribution class.
	*  The %Cauchy distribution (aka "Lorentzian") is defined by:
	*
	*  \f$
	*      f(x) = \frac{1}{\pi \sigma (1 + \left[\frac {(x-x_0)} \sigma\right]^2 )}
	*  \f$
	*
	*  <br>
	*  The %Cauchy distribution is important as an example of a pathological
	*  case. The %Cauchy distribution looks similar to a Normal distribution,
	*  but has much heavier tails. When studying hypothesis tests that assume
	*  normality, seeing how the tests perform on data from a %Cauchy
	*  distribution is a good indicator of how sensitive the tests are to
	*  heavy-tail departures from normality. Likewise, it is a good check
	*  for robust techniques that are designed to work well under a wide
	*  variety of distributional assumptions.
	*/
	template<typename RngType = shark::DefaultRngType>
	class Cauchy:public boost::variate_generator<RngType*,boost::cauchy_distribution<> >
	{
	private:
		typedef boost::variate_generator<RngType*,boost::cauchy_distribution<> > Base;
	public:

		/*!
		*  \brief Creates a new %Cauchy random generator instance
		*
		*\param median the median of the distribution
		*\param sigma the width of the distribution
		*\param rng the used random number generator
		*/
		Cauchy(RngType& rng,double median=0,double sigma=1)
			:Base(&rng,boost::cauchy_distribution<>(median,sigma))
		{}

		//! creates a cauchy distributed number using the preset parameters
		using Base::operator();

		/*!
		*\brief creates a cauchy distributed number from parameters
		*
		*\param median the median of the distribution
		*\param sigma the width of the distribution
		*/
		double operator()(double median,double sigma)
		{
			boost::cauchy_distribution<> dist(median,sigma);
			return dist(Base::engine());
		}

		//! returns the current median of the distribution
		double median()const
		{
			return Base::distribution().median();
		}

		//! returns the width of the distribution
		double sigma()const
		{
			return Base::distribution().sigma();
		}
		//! sets the median of the distribution
		//! \param newMedian the new value for the Median
		void median(double newMedian)
		{
			Base::distribution()=boost::cauchy_distribution<>(newMedian,sigma());
		}
		//! sets the width of the distribution
		//! \param newSigma the new value for sigma
		void sigma(double newSigma)
		{
			Base::distribution()=boost::cauchy_distribution<>(median(),newSigma);
		}
		//! Returns the probability for the occurrence of random number "x".
		//! \param x the point for which to calculate the propability
		double p(double x)const {
			return 1.0/(sigma()*M_PI*(1+shark::sqr((x-median())/sigma())));
		}


	};
	//! Returns the entropy of the Cauchy distribution
	template<typename RngType>
	double entropy(const Cauchy<RngType> & distribution);
}
#endif
