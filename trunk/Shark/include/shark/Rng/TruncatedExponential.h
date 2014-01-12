/**
*
*  \brief Implements a truncated exponential.
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
#ifndef SHARK_RNG_TRUNCATED_EXPONENTIAL_H
#define SHARK_RNG_TRUNCATED_EXPONENTIAL_H

#include <shark/Rng/Rng.h>
#include <boost/random.hpp>
#include <boost/random/uniform_01.hpp>

#include <cmath>
#include <istream>

namespace shark{

    /**
     * \brief boost random suitable distribution for an truncated exponential. See TruncatedExponential for more details.
     */
    template<class RealType = double>
	class TruncatedExponential_distribution {
	public:
	typedef RealType input_type;
		typedef RealType result_type;

		explicit TruncatedExponential_distribution( RealType lambda, RealType max)
			:m_max(max),m_lambda(lambda),m_integral(1-std::exp(-lambda*max))
		{}
		explicit TruncatedExponential_distribution( RealType lambda, RealType max, RealType integral)
			:m_max(max),m_lambda(lambda),m_integral(integral)
		{}

		RealType max() const
		{
			return m_max;
		}
		RealType lambda()const
		{
			return m_lambda;
		}
		RealType integral()const
		{
			return m_integral;
		}
		void reset() { }

		template<class Engine>
		result_type operator()(Engine& eng)
		{
			if(m_lambda == 0){
				return boost::uniform_01<RealType>()(eng);
			}
			double y = m_max * boost::uniform_01<RealType>()(eng);
			return - std::log(1. - y*m_integral)/m_lambda;
		}

		template<class CharT, class Traits,class T>
		friend std::basic_ostream<CharT,Traits>&
		operator<<(std::basic_ostream<CharT,Traits>& os, const TruncatedExponential_distribution<T>& d){
			os << d.m_max;
			os << d.m_lambda;
			return os;
		}

		template<class CharT, class Traits,class T>
		friend std::basic_istream<CharT,Traits>&
		operator>>(std::basic_istream<CharT,Traits>& is, TruncatedExponential_distribution<T>& d){
			double max = 0;
			double lambda = 0;
			is >> max;
			is >> lambda;
			d = TruncatedExponential_distribution<T>(lambda,max);
			return is;
		}
	private:
		RealType m_max;
		RealType m_lambda;
		RealType m_integral;
	};

    /**
     * \brief Implements a generator for the truncated exponential function
     *
     * Often, not the full range of an exponential distribution is needed. instead only an interval between [0,b]
     * is required. In this case, the TruncatedExponential can be used. The propability function is
     * \f$ p(x)=\frac{\lambda e^{-\lambda x}}{1-e^{-\lambda b}} \f$
     * as default, the maximum value for x is 1
     */
     	template<typename RngType = DefaultRngType>
     	class TruncatedExponential:public boost::variate_generator<RngType*,TruncatedExponential_distribution<> > {
     	private:
     		typedef boost::variate_generator<RngType*,TruncatedExponential_distribution<> > Base;
     	public:
     	
     		TruncatedExponential(RngType& rng, double lambda = 1, double max = 1.0 )
     			:Base(&rng,TruncatedExponential_distribution<>(lambda,max))
     		{}
		
		///\brief special version, when the integral of the truncated exponential is allready known 
		TruncatedExponential(double integral, RngType& rng, double lambda = 1, double max = 1.0 )
     			:Base(&rng,TruncatedExponential_distribution<>(lambda,max, integral))
     		{}
     
     		using Base::operator();
     
     		double operator()(double lambda,double max = 1.0){
     			TruncatedExponential_distribution<> dist(lambda,max);
			return dist(Base::engine());
		}

		double lambda()const{
			return Base::distribution().lambda();
		}
		double max()const{
			return Base::distribution().max();
		}
		void setLambda(double newLambda){
			Base::distribution() = TruncatedExponential_distribution<>(newLambda, max());
		}
		void setMax(double newMax){
			Base::distribution() = TruncatedExponential_distribution<>(lambda(), newMax);
		}

		double p(double x)
		{
			if(x >= 0 && x<=max()) {
				return std::exp(-lambda()*x)/Base::distribution().integral();
			}
			return 0.;
		}

	};
}
#endif






