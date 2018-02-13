/*!
 * 
 *
 * \brief       -
 *
 * \author      -
 * \date        -
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://shark-ml.org/>
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
#ifndef SHARK_UNSUPERVISED_RBM_STATESPACES_TWOSTATESPACE_H
#define SHARK_UNSUPERVISED_RBM_STATESPACES_TWOSTATESPACE_H

#include <limits>
#include <cmath>
#include <shark/Unsupervised/RBM/Tags.h>
#include <shark/Core/Exception.h>

namespace shark{
///\brief The TwoStateSpace is a discrete Space with only two values, for example {0,1} or {-1,1}.
template<int State1,int State2>
struct TwoStateSpace{

	///\brief Tag which tells an approximating function of the partition function, that this space can be enumerated. 
	typedef tags::DiscreteSpace EnumerationTag;

	///\brief Returns the number of states a vector of n random variables (neurons) with values in this state space may have.
	///
	///For example if {a,b} is the state space it returns the cardinality of the set \f$ {a,b}^n = 2^n \f$
	/// @param numberOfNeurons the size of the vector
	/// @return the number of States. 
	static std::size_t numberOfStates(std::size_t numberOfNeurons){
		long double result = std::pow( 2., static_cast< int >( numberOfNeurons ) );
		SHARK_RUNTIME_CHECK(result < std::numeric_limits<std::size_t>::max(), "number of neurons is too big for calculation");
		return static_cast<std::size_t>(result);
	  
	}

	///\brief Returns the i-th state vector for a matrix row
	///
	/// @param vec the vector the i-th state vector is stored in
	/// @param stateNumber the number of the state   
	template<class V>
	static void state(V&& vec,std::size_t stateNumber){
		for (std::size_t i = 0; i != vec.size(); i++) {
			bool secondState = stateNumber & (std::size_t(1)<<i);
			vec(i) = secondState? State2 : State1;
		}
	}
};

typedef TwoStateSpace<0,1> BinarySpace;
typedef TwoStateSpace<-1,1> SymmetricBinarySpace;
	
}
#endif
