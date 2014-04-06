/*!
 * 
 *
 * \brief       Hypergeometric distribution.
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
#ifndef SHARK_RNG_HYPERGEOMETRIC_H
#define SHARK_RNG_HYPERGEOMETRIC_H




#include <cmath>
#include <boost/random/uniform_01.hpp>

#ifndef BOOST_RANDOM_NO_STREAM_OPERATORS
#include <iostream>
#endif

namespace shark{
/// \brief Hypergeometric distribution.
	template<class IntType=int,class RealType = double>
	class HyperGeometric_distribution
	{
	public:
		typedef RealType input_type;
		typedef IntType result_type;

		explicit HyperGeometric_distribution(RealType mean,RealType variance)
			:mean_(mean),variance_(variance)
		{
			if (mean == 0)
			{
				p = 0.;
			}
			else
			{
				double z = variance / (mean * mean);
				p = (1 - std::sqrt((z - 1) / (z + 1))) / 2;

				if (p < 0.) p = 0.;
			}
		}

		RealType mean() const
		{
			return mean_;
		}
		RealType variance()const
		{
			return variance_;
		}

		void reset() { }

		template<class Engine>
		result_type operator()(Engine& eng)
		{
			double uni1 =  boost::uniform_01<RealType>(eng);
			double uni2 =  boost::uniform_01<RealType>(eng);
			return -mean_ * ::log( uni1) / (2 *( uni2 > p? 1 - p : p));
		}

#ifndef BOOST_RANDOM_NO_STREAM_OPERATORS
		template<class CharT, class Traits>
		friend std::basic_ostream<CharT,Traits>&
			operator<<(std::basic_ostream<CharT,Traits>& os, const HyperGeometric_distribution& d)
		{
			os << d.mean_;
			os << d.variance_;
			return os;
		}

		template<class CharT, class Traits>
		friend std::basic_istream<CharT,Traits>&
			operator>>(std::basic_istream<CharT,Traits>& is, HyperGeometric_distribution& d)
		{
			is >> d.mean_;
			is >> d.variance_;
			return is;
		}
#endif
	private:
		RealType mean_;
		RealType variance_;
		RealType p;
	};

	/**
	* \brief Random variable with a hypergeometric distribution.
	*/
	template<typename RngType = shark::DefaultRngType>
	class HyperGeometric:public boost::variate_generator<RngType*,HyperGeometric_distribution<> > {
	private:
		typedef boost::variate_generator<RngType*,HyperGeometric_distribution<> > Base;
	public:

		HyperGeometric( RngType & rng, double k=1, double theta=1 )
			:Base(&rng,HyperGeometric_distribution<>(k,theta))
		{}

		using Base::operator();

		double operator()(double k,double theta)
		{
			HyperGeometric_distribution<> dist(k,theta);
			return dist(Base::engine());
		}

		double mean()const
		{
			return Base::distribution().mean();
		}
		double variance()const
		{
			return Base::distribution().variance();
		}
		void mean(double newMean)
		{
			Base::distribution()=HyperGeometric_distribution<>(newMean,variance());
		}
		void variance(double newVariance)
		{
			Base::distribution()=HyperGeometric_distribution<>(mean(),newVariance);
		}

		double p(double x)const
		{
			return 0.0;///!!!
		}

	};
}
#endif

