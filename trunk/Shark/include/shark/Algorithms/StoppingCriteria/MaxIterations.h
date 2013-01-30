/*!
*  \file MaxIterations.h
*
*  \brief Stopping Criterion which stops after a fixed number of iterations
*
*  \author O. Krause
*  \date 2010
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
