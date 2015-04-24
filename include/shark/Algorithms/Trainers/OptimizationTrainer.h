//===========================================================================
/*!
 * 
 *
 * \brief       Model training by means of a general purpose optimization procedure.
 * 
 * 
 *
 * \author      T. Glasmachers
 * \date        2011-2012
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
//===========================================================================

#ifndef SHARK_ALGORITHMS_TRAINERS_OPTIMIZATIONTRAINER_H
#define SHARK_ALGORITHMS_TRAINERS_OPTIMIZATIONTRAINER_H

#include <shark/Algorithms/AbstractSingleObjectiveOptimizer.h>
#include <shark/Core/ResultSets.h>
#include <shark/Models/AbstractModel.h>
#include <shark/ObjectiveFunctions/ErrorFunction.h>
#include <shark/Algorithms/Trainers/AbstractTrainer.h>
#include <shark/Algorithms/StoppingCriteria/AbstractStoppingCriterion.h>

namespace shark {


///
/// \brief Wrapper for training schemes based on (iterative) optimization.
///
/// \par
/// The OptimizationTrainer class is designed to allow for
/// model training via iterative minimization of a
/// loss function, such as in neural network
/// "backpropagation" training.
///
template <class Model, class LabelTypeT = typename Model::OutputType>
class OptimizationTrainer : public AbstractTrainer<Model,LabelTypeT>
{
	typedef AbstractTrainer<Model,LabelTypeT> base_type;

public:
	typedef typename base_type::InputType InputType;
	typedef typename base_type::LabelType LabelType;
	typedef typename base_type::ModelType ModelType;

	typedef AbstractSingleObjectiveOptimizer< RealVector > OptimizerType;
	typedef AbstractLoss< LabelType, InputType > LossType;
	typedef AbstractStoppingCriterion<SingleObjectiveResultSet<OptimizerType::SearchPointType> > StoppingCriterionType;

	OptimizationTrainer(
			LossType* loss,
			OptimizerType* optimizer,
			StoppingCriterionType* stoppingCriterion)
	: mep_loss(loss), mep_optimizer(optimizer), mep_stoppingCriterion(stoppingCriterion)
	{ }

	/// \brief From INameable: return the class name.
	std::string name() const
	{
		return "OptimizationTrainer<"
			+ mep_loss->name() + ","
			+ mep_optimizer->name() + ">";
	}

	void train(ModelType& model, LabeledData<InputType, LabelType> const& dataset) {
		ErrorFunction error(dataset, &model, mep_loss);
		mep_optimizer->init(error);
		mep_stoppingCriterion->reset();
		do {
			mep_optimizer->step(error);
		}
		while (! mep_stoppingCriterion->stop(mep_optimizer->solution()));
		model.setParameterVector(mep_optimizer->solution().point);
	}

	void read( InArchive & archive )
	{}

	void write( OutArchive & archive ) const
	{}

protected:
	LossType* mep_loss;
	OptimizerType* mep_optimizer;
	StoppingCriterionType* mep_stoppingCriterion;
};


}
#endif
