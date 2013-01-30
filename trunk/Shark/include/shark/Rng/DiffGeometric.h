/**
*
*  \brief Diff geometric distribution.
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
