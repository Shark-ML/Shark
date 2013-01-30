/*!
*  \file TrainingError.h
*
*  \brief Stopping Criterion which stops, when the trainign error seems to converge
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

#ifndef SHARK_TRAINERS_STOPPINGCRITERA_TRAININGERROR_H
#define SHARK_TRAINERS_STOPPINGCRITERA_TRAININGERROR_H

#include "AbstractStoppingCriterion.h"
#include <shark/Core/ResultSets.h>
#include <queue>
#include <numeric>
namespace shark{

///  \brief This stopping criterion tracks the improvement of the error function of the training error over an interval of iterations.
///
/// If at one point, the difference between the error values of the beginning and the end of the interval are smaller
/// than a certain value, this stopping criterion assumes convergence and stops.
/// Of course, this may be misleading, when the algorithm temporarily gets stuck at a saddle point of the error surface.
/// The functions assumes that the algorithm is minimizing. For details, see:
///
/// Lutz Prechelt. Early Stopping - but when? In Genevieve B. Orr and
/// Klaus-Robert Müller: Neural Networks: Tricks of the Trade, volume
/// 1524 of LNCS, Springer, 1997.
///
template<class PointType = RealVector>
class TrainingError: public AbstractStoppingCriterion< SingleObjectiveResultSet<PointType> >{
public:
	/// constructs the TrainingError generalization loss
	/// @param intervalSize size of the interval over which the progress is monitored
	/// @param minDifference minimum difference between start and end of the interval allowed before training stops
	TrainingError(size_t intervalSize, double minDifference){
		m_minDifference = minDifference;
		m_intervalSize = intervalSize;
		reset();
	}
	/// returns true if training should stop
	bool stop(const SingleObjectiveResultSet<PointType>& set){

		m_interval.pop();
		m_interval.push(set.value);
		return (m_interval.front()-set.value) >= 0
		    && (m_interval.front()-set.value) < m_minDifference;
	}
	/// resets the internal state
	void reset(){
		m_interval = std::queue<double>();
		for(size_t i = 0; i != m_intervalSize;++i) {
			m_interval.push(std::numeric_limits<double>::max());
		}
	}
protected:
	/// monitored training interval
	std::queue<double> m_interval;
	/// minmum difference allowed
	double m_minDifference;
	/// size of the interval
	size_t m_intervalSize;
};
}


#endif
