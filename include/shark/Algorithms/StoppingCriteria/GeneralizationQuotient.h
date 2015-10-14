/*!
 * 
 *
 * \brief       Stopping criterion monitoring the quotient of generalization loss and training progress
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

#ifndef SHARK_TRAINERS_STOPPINGCRITERA_GENERALIZATION_QUOTIENT__H
#define SHARK_TRAINERS_STOPPINGCRITERA_GENERALIZATION_QUOTIENT__H

#include "AbstractStoppingCriterion.h"
#include <shark/Core/ResultSets.h>
#include <queue>
#include <numeric>
#include <algorithm>

namespace shark{


/// \brief SStopping criterion monitoring the quotient of generalization loss and training progress
///
/// The GeneralizationQuotient is, as the name suggests, a quotient of two other stopping criteria,
/// namely the generalization loss and the
///
///
///
///
///


/// This stopping criterion is based on the empirical fact that the generalization error does not have a smooth surface.
/// It is normal that during periods of fast learning the generalization loss might increase first and than decrease again.
/// This class calculates the quotient of training progress and generalization loss. It stops if it is bigger than
/// maxloss > 0.

///
/// Terminology for this and other stopping criteria is taken from (and also see):
///
/// Lutz Prechelt. Early Stopping - but when? In Genevieve B. Orr and
/// Klaus-Robert Müller: Neural Networks: Tricks of the Trade, volume
/// 1524 of LNCS, Springer, 1997.
///
template<class PointType = RealVector>
class GeneralizationQuotient: public AbstractStoppingCriterion< ValidatedSingleObjectiveResultSet<PointType> >{
private:
	typedef AbstractStoppingCriterion< ValidatedSingleObjectiveResultSet<PointType> > super;
public:
	typedef ValidatedSingleObjectiveResultSet<PointType> ResultSet;

	GeneralizationQuotient(std::size_t intervalSize,double maxLoss){
		SHARK_ASSERT( intervalSize > 0 );
		m_maxLoss = maxLoss;
		m_intervalSize = intervalSize;
		reset();
	}
	/// returns true if training should stop
	bool stop(ResultSet const& set){
		m_minTraining = std::min(m_minTraining, set.value);
		double gl = set.validation/m_minTraining -1;

		m_meanPerformance += set.value/m_intervalSize;
		m_interval.push(set.value/m_intervalSize);

		if(m_interval.size() > m_intervalSize){
			m_meanPerformance -= m_interval.front();
			m_interval.pop();
		}
		else
			return false;
		double progress = (m_meanPerformance/m_minTraining)-1;

		return gl/progress > m_maxLoss;
	}
	void reset(){
		m_interval = std::queue<double>();
		m_minTraining = std::numeric_limits<double>::max();
		m_meanPerformance = 0;
	}
protected:
	double m_minTraining;
	double m_maxLoss;
	double m_meanPerformance;

	std::queue<double> m_interval;
	std::size_t m_intervalSize;
};
}


#endif
