/*!
 *
 *  \brief Implements a Gamma distribution.
 *
 *  \author  O. Krause
 *  \date    2010-01-01
 *
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
#ifndef SHARK_RNG_GAMMA_H
#define SHARK_RNG_GAMMA_H


#include <shark/SharkDefs.h>
#include <boost/random/uniform_01.hpp>
#include <cmath>

#ifndef BOOST_RANDOM_NO_STREAM_OPERATORS
#include <iostream>
#endif

namespace shark{
/// Gamma distribution.
template<class RealType = double>
class Gamma_distribution
	{
	public:
		typedef RealType input_type;
		typedef RealType result_type;

		explicit Gamma_distribution(RealType k,RealType theta)
			:k_(k),theta_(theta) {}

		RealType k() const
		{
			return k_;
		}
		RealType theta()const
		{
			return theta_;
		}

		void reset() { }

		template<class Engine>
		result_type operator()(Engine& eng)
		{
			unsigned i;
			unsigned n = unsigned(k_);
			RealType delta = k_ - RealType(n);
			RealType V_2, V_1, V;
			RealType v0 = M_E / (M_E + delta);
			RealType eta, xi;
			RealType Gn1 = 0; // Gamma(n, 1) distributed

			for(i=0; i<n; i++) Gn1 += -log(draw(eng));

			do {
				V_2 = draw(eng);
				V_1 = draw(eng);
				V   = draw(eng);
				if(V_2 <= v0)
				{
					xi = pow(V_1, 1./delta);
					eta = V * pow(xi, delta-1.);
				}
				else
				{
					xi = 1. - log(V_1);
					eta = V * exp(-xi);
				}
			} while(eta > (pow(xi, delta-1.) * exp(-xi)));

			return theta_ * (xi + Gn1);
		}

#ifndef BOOST_RANDOM_NO_STREAM_OPERATORS
		template<class CharT, class Traits>
		friend std::basic_ostream<CharT,Traits>&
			operator<<(std::basic_ostream<CharT,Traits>& os, const Gamma_distribution& gd)
		{
			os << gd.k_;
			os << gd.theta_;
			return os;
		}

		template<class CharT, class Traits>
		friend std::basic_istream<CharT,Traits>&
			operator>>(std::basic_istream<CharT,Traits>& is, Gamma_distribution& gd)
		{
			is >> gd.k_;
			is >> gd.theta_;
			return is;
		}
#endif
	private:
		template<class Engine>
		double draw(Engine& eng)
		{
			double res=0;
			do
			{
				res=boost::uniform_01<RealType>()(eng);
			}
			while(res==0);
			return res;
		}
		double k_;
		double theta_;
	};

/// Gamma distributed random variable.
template<typename RngType = shark::DefaultRngType>
class Gamma:public boost::variate_generator<RngType*,Gamma_distribution<> >
	{
	private:
		typedef boost::variate_generator<RngType*,Gamma_distribution<> > Base;
	public:

		explicit Gamma( RngType & rng, double k=1,double theta=1 )
			:Base(&rng,Gamma_distribution<>(k,theta))
		{}

		using Base::operator();

		double operator()(double k,double theta)
		{
			Gamma_distribution<> dist(k,theta);
			return dist(Base::engine());
		}

		double k()const
		{
			return Base::distribution().k();
		}
		double theta()const
		{
			return Base::distribution().theta();
		}
		void k(double newK)
		{
			Base::distribution()=Gamma_distribution<>(newK,theta());
		}
		void theta(double newTheta)
		{
			Base::distribution()=Gamma_distribution<>(k(),newTheta);
		}

		double p(double x)const
		{
			// return std::pow(x, k()-1) * std::exp(-x / theta()) / (shark::gamma(k()) * std::pow(theta(), k())); //  CI
			return std::pow(x, k()-1) * std::exp(-x / theta()) / (Gamma_distribution<>(k()) * std::pow(theta(), k()));
		}

	};
}
#endif
