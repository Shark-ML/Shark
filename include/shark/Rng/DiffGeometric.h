/*!
 * 
 *
 * \brief       Diff geometric distribution.
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
#ifndef SHARK_RNG_DIFFGEOMETRIC_H
#define SHARK_RNG_DIFFGEOMETRIC_H

#include <shark/Rng/Rng.h>

#include <boost/random.hpp>
#include <boost/random/geometric_distribution.hpp>

#ifndef BOOST_RANDOM_NO_STREAM_OPERATORS
#include <iostream>
#endif


#include <cmath>

namespace shark{

/**
*  \brief Implements a diff geometric distribution.
*/
template<class IntType = int, class RealType = double>
class DiffGeometric_distribution
{
    public:
        typedef RealType input_type;
        typedef IntType result_type;

        explicit DiffGeometric_distribution(const RealType& p = RealType(0.5))
        :geom(p) {}

        RealType p() const
        {
            return geom.p();
        }
        void reset() { }

        template<class Engine>
        result_type operator()(Engine& eng)
        {
            return geom(eng)-geom(eng);
        }

#ifndef BOOST_RANDOM_NO_STREAM_OPERATORS
        template<class CharT, class Traits>
        friend std::basic_ostream<CharT,Traits>&
        operator<<(std::basic_ostream<CharT,Traits>& os, const DiffGeometric_distribution& gd)
        {
            os << gd.geom;
            return os;
        }

        template<class CharT, class Traits>
        friend std::basic_istream<CharT,Traits>&
        operator>>(std::basic_istream<CharT,Traits>& is, DiffGeometric_distribution& gd)
        {
            is >> gd.geom;
            return is;
        }
#endif
    private:
        boost::geometric_distribution<IntType,RealType> geom;
};

/**
*  \brief Random variable with diff geometric distribution.
*/
template<typename RngType = shark::DefaultRngType>
class DiffGeometric: public boost::variate_generator<RngType*,DiffGeometric_distribution<> >
{
    private:
        typedef boost::variate_generator<RngType*,DiffGeometric_distribution<> > Base;
    public:
      
		DiffGeometric( RngType & rng, double mean = 0.5 )
            :Base(&rng,DiffGeometric_distribution<>(1.0-mean))
        {}

        using Base::operator();

        double operator()(double mean)
        {
            DiffGeometric_distribution<> dist(mean);
            return dist(Base::engine());
        }

        double mean()const
        {
            return 1-Base::distribution().p();
        }
        void mean(double newMean)
        {
            Base::distribution()=DiffGeometric_distribution<>(1-newMean);
        }

        double p(double x)const
        {
            return( mean() * std::pow( 1 - mean(), std::abs(x) ) / ( 2 - mean() ) );
        }

};

}
#endif
