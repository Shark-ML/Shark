/*!
 *
 *
 * \brief       Implements a multinomial distribution
 * 
 * 
 *
 * \author    O.Krause
 * \date        2014
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
#ifndef SHARK_STATISTICS_MULTINOMIALDISTRIBUTION_H
#define SHARK_STATISTICS_MULTINOMIALDISTRIBUTION_H

#include <shark/LinAlg/eigenvalues.h>
#include <shark/LinAlg/Cholesky.h>
#include <shark/Rng/GlobalRng.h>

namespace shark {

/// \brief Implements a multinomial distribution.
///
/// A multinomial distribution is a discrete distribution with states 0,...,N-1
/// and probabilities p_i for state i with sum_i p_i = 1. This implementation uses
/// the fast alias method (Kronmal and Peterson,1979) to draw the numbers in 
/// constant time. Setup is O(N) and also quite fast. It is advisable 
/// to use this method to draw many numbers in succession.
///
/// The idea of the alias method is to pair a state with high probability with a state with low
/// probability. A high probability state can in this case be included in several pairs. To draw,
/// first one of the states is selected and afterwards a coin toss decides which element of the pair 
/// is taken.
class MultiNomialDistribution{
public:
	typedef unsigned int result_type;

	MultiNomialDistribution(){}

	/// \brief Constructor
	/// \param [in] probabilities Probability vector
	MultiNomialDistribution( RealVector const& probabilities ) 
	: m_probabilities(probabilities){
		update();
	}
	
	/// \brief Stores/Restores the distribution from the supplied archive.
	/// \param [in,out] ar The archive to read from/write to.
	/// \param [in] version Currently unused.
	template<typename Archive>
	void serialize( Archive & ar, const unsigned int version ) {
		ar & BOOST_SERIALIZATION_NVP( m_probabilities );
		ar & BOOST_SERIALIZATION_NVP( m_q );
		ar & BOOST_SERIALIZATION_NVP( m_J );
	}

	/// \brief Accesses the probabilityvector defining the distribution.
	RealVector const& probabilities() const {
		return m_probabilities;
	}
	
	/// \brief Accesses a mutable reference to the probability vector
	/// defining the distribution. Allows for l-value semantics.
	/// 
	/// ATTENTION: If the reference is altered, update needs to be called manually.
	RealVector& probabilities() {
		return m_probabilities;
	}

	/// \brief Samples the distribution.
	result_type operator()() const {
		std::size_t numStates = m_probabilities.size();
 
		std::size_t index = Rng::discrete(0,numStates-1);
 
		if(Rng::coinToss(m_q[index]))
			return index;
		else
			return m_J[index];
	}	    


	void update() {
		std::size_t numStates = m_probabilities.size();
		m_q.resize(numStates);
		m_J.resize(numStates);
		m_probabilities/=sum(m_probabilities);

		// Sort the data into the outcomes with probabilities
		// that are larger and smaller than 1/K.
		std::deque<std::size_t> smaller;
		std::deque<std::size_t> larger;
		for(std::size_t i = 0;i != numStates; ++i){
			m_q(i) = numStates*m_probabilities(i);
			if(m_q(i) < 1.0)
				smaller.push_back(i);
			else
				larger.push_back(i);
		}
		// Loop though and create little binary mixtures that
		// appropriately allocate the larger outcomes over the
		// overall uniform mixture.
		while(!smaller.empty() && !larger.empty()){
			std::size_t small = smaller.front();
			std::size_t large = larger.front();
			smaller.pop_front();
			larger.pop_front();

			m_J[small] = large;
			m_q[large]  -= 1.0 - m_q[small];

			if(m_q[large] < 1.0)
				smaller.push_back(large);
			else
				larger.push_back(large);
		}
		for(std::size_t i = 0; i != larger.size(); ++i){
			m_q[larger[i]]=std::min(m_q[larger[i]],1.0);
		}
	}			

private:
	RealVector m_probabilities; ///< probability of every state.
	RealVector m_q; ///< probability of the pair (i,J[i]) to draw an.
	RealVector m_J; ///< defines the second element of the pair (i,J[i])
};
}

#endif
