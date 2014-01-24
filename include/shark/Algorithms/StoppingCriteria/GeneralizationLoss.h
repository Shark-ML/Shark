/*!
 * 
 * \file        GeneralizationLoss.h
 *
 * \brief       Stopping Criterion which stops, when the generalization of the solution gets worse
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

#ifndef SHARK_TRAINERS_STOPPINGCRITERA_GENERALIZATIONLOSS_H
#define SHARK_TRAINERS_STOPPINGCRITERA_GENERALIZATIONLOSS_H

#include "AbstractStoppingCriterion.h"
#include <shark/Core/ResultSets.h>
#include <queue>
#include <numeric>
#include <algorithm>

namespace shark{

/// \brief The generalization loss calculates the relative increase of the validation error compared to the minimum training error.
///
/// The generalization loss at iteration t is calculated as
/// \f$ GL(t) 100 \left( \frac {E_v(t)} {\min_{t'} E_l(t')}  -1 \right)  \f$
/// where \f$ E_v \f$ is the validation error and \f$ E_l \f$ the training error.
/// This is a good indicator for overfitting, since it measures directly the gap between the two values. This method
/// stops when the generalization error is bigger than some predefined value. The disadvantage is, that
/// when the training error is still changing much a big generalization loss might be repaired later on. So this method
/// might stop to soon.
///
/// Terminology for this and other stopping criteria is taken from (and also see):
///
/// Lutz Prechelt. Early Stopping - but when? In Genevieve B. Orr and
/// Klaus-Robert Müller: Neural Networks: Tricks of the Trade, volume
/// 1524 of LNCS, Springer, 1997.
///
template<class PointType = RealVector>
class GeneralizationLoss: public AbstractStoppingCriterion< ValidatedSingleObjectiveResultSet<PointType> >{
public:
	typedef ValidatedSingleObjectiveResultSet<PointType> ResultSet;
	///constructs a generaliazationLoss which stops, when the GL > maxLoss
	///@param maxLoss maximum loss allowed before stopping
	GeneralizationLoss(double maxLoss){
		m_maxLoss = maxLoss;
		reset();
	}
	/// returns true if the training should stop. The generalization
	/// loss orders the optimizer to stop as soon as the validation
	/// error grows larger than a certain factor of the minimum
	/// validation error encountered so far.
	bool stop(const ResultSet& set){
		m_minTraining = std::min(m_minTraining, set.value);
		m_gl = set.validation/m_minTraining - 1;

		return m_gl > m_maxLoss;
	}
	///resets the internal state
	void reset(){
		m_minTraining = std::numeric_limits<double>::max();
	}
	///returns the current generalization loss
	double value() const{
		return m_gl;
	}
protected:
	///minimum training error
	double m_minTraining;
	///minimum validation error
	double m_minValidation;

	///maximum loss allowed
	double m_maxLoss;
	///current generalization loss
	double m_gl;


};
}


#endif
