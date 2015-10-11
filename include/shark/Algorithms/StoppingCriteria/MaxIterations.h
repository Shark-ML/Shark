/*!
 * 
 *
 * \brief       Stopping Criterion which stops after a fixed number of iterations
 * 
 * 
 *
 * \author      O. Krause
 * \date        2010
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

#ifndef SHARK_TRAINERS_STOPPINGCRITERIA_MAXITERATIONS_H
#define SHARK_TRAINERS_STOPPINGCRITERIA_MAXITERATIONS_H

#include "AbstractStoppingCriterion.h"
#include <shark/Core/ResultSets.h>

namespace shark{

/// This stopping criterion stops after a fixed number of iterations
template<class ResultSet = SingleObjectiveResultSet<RealVector> >
class MaxIterations: public AbstractStoppingCriterion< ResultSet >{
public:
	///constructs the MaxIterations stopping criterion
	///@param maxIterations maximum iterations before training should stop
	MaxIterations(unsigned int maxIterations){
		m_maxIterations = maxIterations;
		m_iteration = 0;
	}

	void setMaxIterations(unsigned int newIterations){
		m_maxIterations = newIterations;
	}
	/// returns true if training should stop
	bool stop(const ResultSet& set){
		++m_iteration;
		return m_iteration>=m_maxIterations;
	}
	///reset iteration counter
	void reset(){
		m_iteration = 0;
	}
protected:
	///current number of iterations
	unsigned int m_iteration;
	///maximum number of iteration allowed
	unsigned int m_maxIterations;
};
}


#endif
