/*!
 * 
 *
 * \brief       Implements a Bernoulli distribution.
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
#ifndef SHARK_RNG_BERNOULLI_H
#define SHARK_RNG_BERNOULLI_H

#include <shark/Rng/Rng.h>

#include <boost/random.hpp>

#include <cmath>

namespace shark{

	/*!
	*  \brief This class simulates a "Bernoulli trial", which
	*         is like a coin toss.
	*
	*  This class is a thin wrapper for the boost::bernoulli_distribution class.
	*  A bernoulli distribution simulates a generalized coin toss.
	*  A probability for the occurrence of the event (coin side)
	*  is defined. When using the equal probability of "0.5" for the
	*  occurrence and non-occurrence of the event (coin side), then the
	*  event (coin) is named "normal", otherwise it is named "abnormal".
	*
	*  \author  O.Krause
	*  \date    2010-01-01
	*
	*  \par Changes:
	*      none
	*
	*  \par Status:
	*      testing
	*
	*/
	template<typename RngType = shark::DefaultRngType>
	class Bernoulli : public boost::variate_generator< RngType*,boost::bernoulli_distribution<> >
	{
	private:
		typedef boost::variate_generator<RngType*,boost::bernoulli_distribution<> > Base;
	public:
		//! Creates a new Bernoulli random generator using the global generator from instance and
		//! sets the probability for the occurrence of the event
		//! to "prob".
		/*
		Bernoulli(double prob=0.5)
		:Base(&Rng::globalRng,boost::bernoulli_distribution<>(prob))
		{}*/

		//! Creates a new Bernoulli random generator instance by
		//! using the pseudo random number generator "rng" for the determination
		//! of random values and sets the probability for the occurrence
		//! of the event to "prob".
		Bernoulli( RngType & rng, double prob = 0.5 )
			:Base(&rng,boost::bernoulli_distribution<>(prob))
		{}
		/*!
		*  \brief Returns a Bernoulli random number, i.e. a "true" or "false"
		*         marking the occurrence and non-occurrence of an event respectively,
		*         using the preset propability
		*
		*  \return a bernoulli distributed number
		*/
		using Base::operator();

		/*!
		*  \brief Returns a Bernoulli random number, i.e. a "true" or "false"
		*         marking the occurrence and non-occurrence of an event respectively,
		*         when the probability for the occurrence is "p".
		*
		*  \return a bernoulli distributed number
		*/
		bool operator()(double p)
		{
			boost::bernoulli_distribution<> dist(p);
			return dist(Base::engine());
		}
		/*!
		*  \brief Returns the probability for the occurrence of an event.
		*
		*  \return the probability for the occurrence of an event
		*/
		double prob()const
		{
			return Base::distribution().p();
		}
		/*!
		*  \brief Sets the probability for the occurrence of an event to "newP".
		*
		*  \param newP the new probability for the occurrence of an event
		*  \return none
		*/
		void prob(double newP)
		{
			Base::distribution()=boost::bernoulli_distribution<>(newP);
		}

		//! Returns the probability \f$p\f$ for the occurrence of an
		//! event ("x = true") or \f$1 - p\f$ for the non-occurrence
		//! ("x = false").
		double p(bool x) const
		{
			return x ? prob() : 1 - prob();
		}

	};
	//! Returns the entropy of the bernoulli distribution
	template<typename RngType>
	double entropy(const Bernoulli<RngType> & coin);
}
#endif
