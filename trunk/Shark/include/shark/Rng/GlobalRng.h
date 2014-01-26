/*!
 * 
 *
 * \brief       This class subsumes several often used random number generators.
 * 
 * This class offers convenience functions to generate numbers using a global random number generator from the following distributions:
 * 
 * <ul>
 * <li>Bernoulli with name \em coinToss
 * <li>DiscreteUniform with name \em discrete
 * <li>Uniform with name \em uni
 * <li>Normal with name \em gauss
 * <li>Cauchy with name \em cauchy
 * <li>Geometric with name \em geom
 * <li>DiffGeometric with name \em diffGeom
 * <li>Poisson with name \em poisson
 * <li>Gamma with name \em gam
 * <li>Dirichlet with name \em dir
 * </ul>
 * 
 * Additionally this class offers a global random number generator of Type #RngType. The default of this
 * is the Mersenne Twister with a cycle length of $2^19937$. This generator can be used to construct additional
 * distributions. The seed can be set via Rng::seed .
 * 
 * \par Example
 * \code
 * #include "shark/Rng/GlobalRng.h"
 * 
 * void main()
 * {
 * 
 * // Set seed for all subsumed random number generators:
 * Rng::seed( 1234 );
 * 
 * // Get random "numbers" for all subsumed random number generators:
 * bool   rn1 = Rng::coinToss( );
 * long   rn2 = Rng::discrete( );
 * double rn3 = Rng::uni( );
 * double rn4 = Rng::gauss( );
 * double rn5 = Rng::cauchy( );
 * long   rn6 = Rng::geom( );
 * long   rn7 = Rng::diffGeom( );
 * 
 * // Output of random numbers:
 * cout << "Bernoulli trial                              = " << rn1 << endl;
 * cout << "Discrete distribution number                 = " << rn2 << endl;
 * cout << "Uniform distribution number                  = " << rn3 << endl;
 * cout << "Normal distribution number                   = " << rn4 << endl;
 * cout << "Cauchy distribution number                   = " << rn5 << endl;
 * cout << "Geometric distribution number                = " << rn6 << endl;
 * cout << "Differential Geometric distribution number   = " << rn7 << endl;
 * }
 * \endcode
 * 
 * 
 *
 * \author      -
 * \date        -
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
#ifndef SHARK_RNG_GLOBALRNG_H
#define SHARK_RNG_GLOBALRNG_H

#include <shark/Rng/Rng.h>

#include <shark/Rng/Bernoulli.h>
#include <shark/Rng/Binomial.h>
#include <shark/Rng/Cauchy.h>
#include <shark/Rng/DiffGeometric.h>
#include <shark/Rng/Dirichlet.h>
#include <shark/Rng/DiscreteUniform.h>
#include <shark/Rng/Erlang.h>
#include <shark/Rng/Gamma.h>
#include <shark/Rng/Geometric.h>
#include <shark/Rng/HyperGeometric.h>
#include <shark/Rng/LogNormal.h>
#include <shark/Rng/NegExponential.h>
#include <shark/Rng/Normal.h>
#include <shark/Rng/Poisson.h>
#include <shark/Rng/Uniform.h>
#include <shark/Rng/Weibull.h>

#include <shark/Rng/Entropy.h>
#include <shark/Rng/KullbackLeiberDivergence.h>

#include <boost/random.hpp>
#include <vector>

namespace shark {

	/**
	* \brief Collection of different variate generators for different distributions.
	*
	* \tparam RNG The underlying random number generator, needs to model the boost rng concept.
	*/
	template<typename RNG>
	class BaseRng {
	public:

		typedef RNG rng_type;
		typedef BaseRng< rng_type> this_type;

		//! The global random number generator used by all distributions
		static rng_type globalRng;

		//! creates a bernoulli distributed number with propability "p"
		static inline bool coinToss( double p = 0.5 ) {
			Bernoulli< rng_type > coin(globalRng,p);
			return coin();
		}

		//! creates a discrete uniform distributed number in the range from "min" to "max"
		static int discrete(int min=0,int max=1) {
			if(min == max) return min;
			DiscreteUniform< rng_type > disc(globalRng,min,max);
			return disc( min, max );
		}


		//! creates a uniform distributed number in the range from "min" to "max"
		static double uni(double min=0.0,double max=1.0) {
			if(min == max) return min;
			Uniform< rng_type > uni( globalRng, min, max );
			return uni();
		}

                //! creates a log-normal distributed number with location "location" and scale "scale"
		static double logNormal(double location=0.0,double scale=1.0) {
			LogNormal< rng_type > logNormal(globalRng,location,scale);
			return logNormal();
		}

		//! creates a normal distributed number with mean "mean" and variance "sigma"
		static double gauss(double mean=0.0,double sigma=1.0) {
			Normal< rng_type > normal(globalRng,mean,sigma);
			return normal();
		}

		//! creates a cauchy distributed number
		static double cauchy(double median=0.0,double gamma=1.0) {
			Cauchy< rng_type > cauchy(globalRng,median,gamma);
			return cauchy();
		}

		//! creates a number using the geometric distribution and propability "p"
		static int geom(double p=0.0) {
			Geometric< rng_type > rng(globalRng,p);
			return rng();
		}

		//! creates a number using the diff-geometric distribution with mean "mean"
		static int diffGeom(double mean = 0.5) {
			DiffGeometric< rng_type > diff(globalRng,mean);
			return diff();
		}

		//! creates a poission distributed number with mean "mean"
		static double poisson(double mean=0.01) {
			Poisson< rng_type > poisson(globalRng,mean);
			return poisson();
		}

		//! creates a number using the gamma distribution
		static double gam(double k,double theta) {
			Gamma< rng_type > gamma(globalRng,k,theta);
			return cauchy();
		}

		//! creates a dirichlet distributed number
		static std::vector<double> dir(size_t n,double alpha) {
			Dirichlet< rng_type > dist(globalRng,n,alpha);
			return dist();
		}
		//! creates a dirichlet distributed number
		static std::vector<double> dir(const std::vector<double>& alphas) {
			Dirichlet< this_type > dist(globalRng,alphas);
			return dist();
		}

		//! Sets the seed for all random number generators to "s".
		static void seed( typename rng_type::result_type s ) {
			globalRng.seed( s );
		}
	};
	template<class Rng>
	typename BaseRng<Rng>::rng_type BaseRng<Rng>::globalRng = typename BaseRng<Rng>::rng_type();

	#define ANNOUNCE_SHARK_RNG( boost_rng_type, shark_rng_name )\
		typedef BaseRng< boost_rng_type > shark_rng_name; \

	ANNOUNCE_SHARK_RNG( shark::FastRngType,		FastRng );
	ANNOUNCE_SHARK_RNG( shark::DefaultRngType,	Rng		);
	/*
	typedef BaseRng< boost::rand48 > FastRng; FastRng::rng_type FastRng::globalRng = FastRng::rng_type();
		typedef BaseRng< boost::mt19937 > Rng; Rng::rng_type Rng::globalRng = Rng::rng_type();*/


}

#endif



