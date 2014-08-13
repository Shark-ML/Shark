/*!
 * 
 *
 * \brief       Stopping Criterion which stops, when the training error seems to converge
 * 
 * 
 *
 * \author      O. Krause
 * \date        2010
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

#ifndef SHARK_TRAINERS_STOPPINGCRITERA_TRAININGPROGRESS_H
#define SHARK_TRAINERS_STOPPINGCRITERA_TRAININGPROGRESS_H

#include "AbstractStoppingCriterion.h"
#include <shark/Core/ResultSets.h>
#include <queue>
#include <numeric>

namespace shark{


///\brief This stopping criterion tracks the improvement of the training error over an interval of iterations.
///
///If the mean performance over this strip divided by the minimum is too low, training is stopped. The difference to TrainingError
///is, that this class tests the relative improvement of the error compared to the minimum training error,
///while the TrainingError measures the absolute difference. This class is a bit better tuned to noisy error functions since it takes the
///mean of the interval as comparison.
///
/// Terminology for this and other stopping criteria is taken from (and also see):
///
/// Lutz Prechelt. Early Stopping - but when? In Genevieve B. Orr and
/// Klaus-Robert Müller: Neural Networks: Tricks of the Trade, volume
/// 1524 of LNCS, Springer, 1997.
///
template<class PointType = RealVector>
class TrainingProgress: public AbstractStoppingCriterion< SingleObjectiveResultSet<PointType> >{
public:
	typedef SingleObjectiveResultSet<PointType> ResultSet;
	///constructs the TrainingProgress
	///@param intervalSize the size of the interval which is checked
	///@param minImprovement minimum relative improvement of the interval to the minimum training error before training stops
	TrainingProgress(size_t intervalSize, double minImprovement){
		m_minImprovement = minImprovement;
		m_intervalSize = intervalSize;
		reset();
	}
	/// returns true if training should stop
	bool stop(const ResultSet& set){
		m_minTraining = std::min(m_minTraining, set.value);

		m_meanPerformance += set.value;
		m_interval.push(set.value);
		if(m_interval.size()>m_intervalSize){
			m_meanPerformance -= m_interval.front();
			m_interval.pop();
		}
		m_progress = (m_meanPerformance/(m_minTraining*m_interval.size()))-1;
		
		if(m_interval.size()<m_intervalSize){
			return false;
		}

		
		return m_progress < m_minImprovement;
	}
	///resets the internal state
	void reset(){
		m_interval = std::queue<double>();
		m_minTraining = 1.e10;
		m_meanPerformance = 0;
		m_progress = 0.0;
	}
	///returns current value of progress
	double value()const{
		return m_progress;
	}
protected:
	///minimum training error encountered
	double m_minTraining;
	///minimum improvement allowed before training stops
	double m_minImprovement;
	///mean performance over the last intervalSize timesteps
	double m_meanPerformance;
	///current progress measure. if it is below minTraining, stop() will return true
	double m_progress;

	///current interval
	std::queue<double> m_interval;
	///size of the interval
	size_t m_intervalSize;
};
}


#endif
