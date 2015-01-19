/*!
 * 
 *
 * \brief       Implements a Weibull distribution.
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
#ifndef SHARK_RNG_WEIBULL_H
#define SHARK_RNG_WEIBULL_H

#include <shark/Core/Exception.h>
#include <shark/Rng/Rng.h>


#include <boost/random.hpp>
#include <boost/random/uniform_01.hpp>
#include <cmath>
#include <istream>
namespace shark{

/// \brief Weibull distribution.
template<class RealType = double>
class Weibull_distribution
	{
	public:
		typedef RealType input_type;
		typedef RealType result_type;

		explicit Weibull_distribution(RealType alpha,RealType beta)
			:alpha_(alpha),beta_(beta)
		{}

		RealType alpha() const
		{
			return alpha_;
		}
		RealType beta()const
		{
			return beta_;
		}

		void reset() { }

		template<class Engine>
		result_type operator()(Engine& eng)
		{
			double uni = boost::uniform_01<RealType>()(eng);
			return std::pow(-beta_ * std::log(1. - uni), 1 / alpha_);
		}

		template<class CharT, class Traits>
		friend std::basic_ostream<CharT,Traits>&
			operator<<(std::basic_ostream<CharT,Traits>& os, const Weibull_distribution& d)
		{
			os << d.alpha_;
			os << d.beta_;
			return os;
		}

		template<class CharT, class Traits>
		friend std::basic_istream<CharT,Traits>&
			operator>>(std::basic_istream<CharT,Traits>& is, Weibull_distribution& d)
		{
			is >> d.alpha_;
			is >> d.beta_;
			return is;
		}
	private:
		RealType alpha_;
		RealType beta_;
	};

/// \brief Weibull distributed random variable.
template<typename RngType = shark::DefaultRngType>
class Weibull:public boost::variate_generator<RngType*,Weibull_distribution<> >
	{
	private:
		typedef boost::variate_generator<RngType*,Weibull_distribution<> > Base;
	public:

		Weibull( RngType & rng, double alpha = 1, double beta = 1 )
			:Base(&rng,Weibull_distribution<>(alpha,beta))
		{}

		using Base::operator();

		double operator()(double alpha,double beta)
		{
			Weibull_distribution<> dist(alpha,beta);
			return dist(Base::engine());
		}

		double alpha()const
		{
			return Base::distribution().alpha();
		}
		double beta()const
		{
			return Base::distribution().beta();
		}
		void alpha(double newAlpha)
		{
			Base::distribution()=Weibull_distribution<>(newAlpha,beta());
		}
		void vbeta(double newBeta)
		{
			Base::distribution()=Weibull_distribution<>(alpha(),newBeta);
		}

		double p(double x)const
		{
			if (x <= 0)
			{
				throw SHARKEXCEPTION("Weibull distribution not defined for x <= 0");
			}
			return alpha() / beta() * exp(-pow(x / beta(), alpha())) * pow(x / beta(), alpha() - 1.);

		}

	};
}
#endif



