/**
*
*  \brief Implements the negative exponential distribution
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
#ifndef SHARK_RNG_NEGEXPONENTIAL_H
#define SHARK_RNG_NEGEXPONENTIAL_H


#include <shark/SharkDefs.h>

#include <boost/random/exponential_distribution.hpp>

#include <cmath>

namespace shark{

	///\brief Implements the Negative exponential distribution.
	///
	/// It's propability distribution is defined as
	/// \f[ p(x) = e^{-\lambda x}\f]
	/// Instead of lambda, we define the exponential distribution using the mean 
	/// \f[ \mu = 1.0/\lambda \f]
	template<typename RngType = shark::DefaultRngType>
	class NegExponential:public boost::variate_generator<RngType*,boost::exponential_distribution<> >
	{
	private:
		typedef boost::variate_generator<RngType*,boost::exponential_distribution<> > Base;
	public:
	
		NegExponential( RngType & rng, double mean = 0 )
			:Base(&rng,boost::exponential_distribution<>(1.0/mean))
		{}

		using Base::operator();

		///\brief Draws a random number from the negative exponential distribution with the mean defined in the argument
		///
		///\param mean: the mean of the distribution from which the number is drawn
		double operator()(double mean)
		{
			boost::exponential_distribution<> dist(1.0/mean);
			return dist(Base::engine());
		}

		///\brief Returns the mean of the negative exponential distribution
		double mean()const
		{
			return 1.0/Base::distribution().lambda();
		}
		///\brief Sets the mean of the negative exponential distribution
		///
		///\param newMean the new Mean value
		void mean(double newMean)
		{
			Base::distribution()=boost::exponential_distribution<>(1.0/newMean);
		}

		double p(double x)
		{
			return x >= 0 ? Base::distribution().lambda() * exp(- Base::distribution().lambda() * x) : 0.;
		}

	};
}
#endif






