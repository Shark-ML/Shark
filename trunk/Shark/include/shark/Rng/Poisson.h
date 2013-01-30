/**
*
*  \brief Implements a poisson distribution.
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
