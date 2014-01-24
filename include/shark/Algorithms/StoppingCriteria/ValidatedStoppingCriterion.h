/*!
 * 
 * \file        ValidatedStoppingCriterion.h
 *
 * \brief       Stopping Criterion which evaluates the validation error and hands the result over to another stopping criterion.
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

#ifndef SHARK_TRAINERS_STOPPINGCRITERIONS_VALIDATEDSTOPPINGCRITERION_H
#define SHARK_TRAINERS_STOPPINGCRITERIONS_VALIDATEDSTOPPINGCRITERION_H

#include "AbstractStoppingCriterion.h"
#include <shark/Core/ResultSets.h>

namespace shark{


/// \brief Given the current Result set of the optimizer, calculates the validation error using a validation function and hands the results over to the underlying stopping criterion.
///
/// Currently only implemented for functions over VectorSpace<double>
class ValidatedStoppingCriterion: public AbstractStoppingCriterion< SingleObjectiveResultSet<RealVector> >{
private:
	typedef RealVector PointType;
	typedef AbstractStoppingCriterion< SingleObjectiveResultSet<PointType> > base_type;
public:
	//typedef typename base_type::ResultSet ResultSet;
	typedef ValidatedSingleObjectiveResultSet<PointType> ValidationResultSet;
	typedef AbstractStoppingCriterion< ValidationResultSet > StoppingCriterionType;
	typedef AbstractObjectiveFunction< VectorSpace<double>, double > ObjectiveFunctionType;


	ValidatedStoppingCriterion(ObjectiveFunctionType* validation, StoppingCriterionType* child)
	:mpe_validation(validation), mpe_child(child){
		reset();
	}
	/// returns true if training should stop
	bool stop(ResultSet const& set){
		double validationError = mpe_validation->eval(set.point);
		return mpe_child->stop(ValidationResultSet(set,validationError));
	}
	void reset(){
		mpe_child->reset();
	}
protected:
	ObjectiveFunctionType* mpe_validation;
	StoppingCriterionType* mpe_child;
};
}


#endif
