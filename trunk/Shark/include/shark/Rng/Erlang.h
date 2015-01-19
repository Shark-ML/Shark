/*!
 * 
 *
 * \brief       Implements an Erlang distribution.
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
#ifndef SHARK_RNG_ERLANG_H
#define SHARK_RNG_ERLANG_H



#include <shark/Rng/Gamma.h>
#include <boost/random/uniform_01.hpp>
#include <cmath>
#include <vector>

#ifndef BOOST_RANDOM_NO_STREAM_OPERATORS
#include <iostream>
#endif

namespace shark{

/// \brief Implements an Erlang distribution.
template<class RealType=double>
class Erlang_distribution
{
    public:
        typedef RealType input_type;
        typedef RealType result_type;

        explicit Erlang_distribution(RealType mean=0,RealType variance=1)
        :mean_(mean),variance_(variance)
        {
            k = static_cast<std::size_t>(std::ceil(mean * mean / variance));
            if (k == 0)
                k = 1;

            if (mean > 0)
                a = k / mean;
            else
                a = 0.5;
        }

        RealType mean() const
        {
            return mean_;
        }
        RealType variance() const
        {
            return variance_;
        }

        void reset() { }

        template<class Engine>
        result_type operator()(Engine& eng)
        {
		double prod = 1;

		if (k == 0 || a <= 0) return 0.;
		
		//double maxEng = eng.max();
		//double minEng = eng.min(); 

		unsigned int kPrime = k;
		while(kPrime--){
			double uni = boost::uniform_01<RealType>()(eng);
			prod *= uni;
		}
		return -std::log(prod) / a;
        }

#ifndef BOOST_RANDOM_NO_STREAM_OPERATORS
        template<class CharT, class Traits>
        friend std::basic_ostream<CharT,Traits>&
        operator<<(std::basic_ostream<CharT,Traits>& os, const Erlang_distribution& d)
        {
            os << d.alphas.size();
            for(int i=0;i!=d.alphas_.size();++i)
                os << d.alphas_[i];
            return os;
        }

        template<class CharT, class Traits>
        friend std::basic_istream<CharT,Traits>&
        operator>>(std::basic_istream<CharT,Traits>& is, Erlang_distribution& d)
        {
            size_t size;
            is >> size;
            for(int i=0;i!=size;++i)
            {
                double element;
                is >> element;
                d.alphas_.push_back(element);
            }
            return is;
        }
#endif
    private:
        double mean_;
        double variance_;
        unsigned k;
        RealType a;
};

/**
* \brief Erlang distributed random variable.
*/
template<typename RngType = shark::DefaultRngType>
class Erlang:public boost::variate_generator<RngType*,Erlang_distribution<> >
{
    private:
		/** \brief The base type this class inherits from. */
        typedef boost::variate_generator<RngType*,Erlang_distribution<> > Base;
    public:

		/**
		* \brief Default c'tor, associates the distribution with the supplied RNG.
		* \param [in,out] rng The RNG to associate the distribution with.
		* \param [in] mean The parameter mean, default value 0.
		* \param [in] variance The parameter variance, default value 1.
		*/
		explicit Erlang( RngType & rng, double mean=0,double variance=1 )
            :Base(&rng,Erlang_distribution<>(mean,variance))
        {}

		/**
		* \brief Injects the default sampling operator.
		*/
        using Base::operator();

		/**
		* \brief Reinitializes the distribution for the supplied parameters and samples a new random number.
		* Default values are omitted to distinguish the operator from the default one.
		* 
		* \param [in] mean The new mean.
		* \param [in] variance The new variance.
		*/
        double operator()(double mean,double variance)
        {
            Erlang_distribution<> dist(mean,variance);
            return dist(Base::engine());
        }

		/**
		* \brief Accesses the mean of the distribution.
		*/
        double mean()const
        {
            return Base::distribution().mean();
        }

		/**
		* \brief Accesses the variance of the distribution.
		*/
        double variance()const
        {
            return Base::distribution().variance();
        }

		/**
		* \brief Adjusts the mean of the distribution.
		*/
        void mean(double newMean)
        {
            Base::distribution()=Erlang_distribution<>(newMean,Base::distribution().variance());
        }

		/**
		* \brief Adjusts the variance of the distribution.
		*/
        void variance(double newVariance)
        {
            Base::distribution()=Erlang_distribution<>(mean(),newVariance);
        }

		/**
		* \brief Calculates the probability of x.
		*/
        double p(double&x)const
        {
            double k = mean() * mean() / variance() + 0.5;
            if (k == 0)
                k = 1;

            double a=0;
            if (mean() > 0)
                a = k / mean();
            else
                a = 0.5;

            if (k == 0 || a <= 0 || x < 0) return 0.;

            return std::pow(k * a, k) * std::pow(x, k - 1) * std::exp(- k * a * x);
        }

};
}
#endif

