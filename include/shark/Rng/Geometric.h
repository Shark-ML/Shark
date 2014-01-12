/**
*
*  \brief Implements a geometric distribution.
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
#ifndef SHARK_RNG_GEOMETRIC_H
#define SHARK_RNG_GEOMETRIC_H



#include <cmath>

namespace shark{

	/**
	* \brief Implements the geometric distribution.
	*
	* Note that a support {1,2,3,...} is assumed here.
	*/
	template<typename RngType = shark::DefaultRngType>
	class Geometric:public boost::variate_generator<RngType*,boost::geometric_distribution<long> > {
	private:
		typedef boost::variate_generator<RngType*,boost::geometric_distribution<long> > Base;
	public:

		/**
		* \brief C'tor, initializes the parameter p modelling the probability of success. Associates
		* the distribution with a custom RNG.
		* \param [in,out] rng The RNG to associate the distribution with.
		* \param [in] p Parameter p, the probability of success.
		*/
		Geometric( RngType & rng, double p = 0.5 )
			:Base(&rng,boost::geometric_distribution<long>(1.0-p))
		{}

		/** \brief Injects the default sampling operator. */
		using Base::operator();

		/**
		* \brief Reinitializes the distribution with the supplied success probability and samples a random number.
		* \param [in] p The new success probability.
		*/
		long operator()(double p) {
			boost::geometric_distribution<long> dist( p );
			return dist(Base::engine());
		}

		/**
		* \brief Accesses the success probability.
		*/
		double prob() const {
			return 1-Base::distribution().p();
		}

		/**
		* \brief Adjusts the success probability.
		*/
		void prob(double newMean) {
			Base::distribution()=boost::geometric_distribution<long>(1-newMean);
		}

		/**
		* \brief Calculates the probability of x.
		*/
		double p( long x ) {
			return( x > 0 ? prob() * std::pow(1 - prob(), (double) x - 1) : 0 );
		}

	};
}
#endif
