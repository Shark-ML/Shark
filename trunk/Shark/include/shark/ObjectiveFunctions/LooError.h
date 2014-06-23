/*!
 * 
 *
 * \brief       Leave-one-out error
 * 
 * 
 *
 * \author      T.Glasmachers
 * \date        2011
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_LOOERROR_H
#define SHARK_OBJECTIVEFUNCTIONS_LOOERROR_H


#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/Models/AbstractModel.h>
#include <shark/ObjectiveFunctions/Loss/AbstractLoss.h>
#include <shark/Algorithms/Trainers/AbstractTrainer.h>
#include <shark/Data/DataView.h>
#include <boost/range/algorithm_ext/iota.hpp>


namespace shark {


///
/// \brief Leave-one-out error objective function.
///
/// \par
/// The leave-one-out measure is the average prediction performance of
/// a learning machine on a dataset, where each sample is predicted by
/// a machine trained on all but the sample to be predicted. This is an
/// extreme form of cross-validation, with a fold size of one.
///
/// \par
/// In general the leave-one-out error is costly to compute, since it
/// requires training of a large number of learning machines. However,
/// certain machines allow for a more efficient implementation. Refer
/// to LooErrorCSvm for an example.
///
template<class ModelTypeT, class LabelType = typename ModelTypeT::OutputType>
class LooError : public SingleObjectiveFunction
{
public:
	typedef ModelTypeT ModelType;
	typedef typename ModelType::InputType InputType;
	typedef typename ModelType::OutputType OutputType;
	typedef LabeledData<InputType, LabelType> DatasetType;
	typedef AbstractTrainer<ModelType, LabelType> TrainerType;
	typedef AbstractLoss<LabelType, typename ModelType::OutputType> LossType;

	///
	/// \brief Constructor.
	///
	/// \param  dataset  Full data set for leave-one-out.
	/// \param  model    Model built on subsets of the data.
	/// \param  trainer  Trainer for learning on each subset.
	/// \param  loss     Loss function for judging the validation output.
	/// \param  meta     Meta object with parameters that influences the process, typically a trainer.
	///
	LooError(
		DatasetType const& dataset,
		ModelType* model,
		TrainerType* trainer,
		LossType* loss,
		IParameterizable* meta = NULL)
	: m_dataset(dataset)
	, mep_meta(meta)
	, mep_model(model)
	, mep_trainer(trainer)
	, mep_loss(loss)
	{
		m_features |= HAS_VALUE;
	}


	/// \brief From INameable: return the class name.
	std::string name() const
	{
		return "LooError<"
				+ mep_model->name() + ","
				+ mep_trainer->name() + ","
				+ mep_loss->name() + ">";
	}
	
	std::size_t numberOfVariables()const{
		return mep_meta->numberOfParameters();
	}

	/// Evaluate the leave-one-out error:
	/// train sub-models, evaluate objective,
	/// return the average.
	double eval() const {
		this->m_evaluationCounter++;

		std::size_t ell = m_dataset.size();
		Data<OutputType> output;
		double sum = 0.0;
		std::vector<std::size_t> indices(ell - 1);
		boost::iota(indices,0);
		for (std::size_t i=0; i<ell-1; i++) indices[i] = i+1;
		for (std::size_t i=0; i<ell; i++)
		{
			DatasetType train = toDataset(subset(m_dataset,indices));
			mep_trainer->train(*mep_model, train);
			OutputType validation = (*mep_model)(m_dataset[i].input);
			sum += mep_loss->eval(m_dataset[i].label, validation);
			if (i < ell - 1) indices[i] = i;
		}
		return sum / ell;
	}

	/// Evaluate the leave-one-out error for the given
	/// parameters passed to the meta object (typically
	/// these parameters need to be optimized in a model
	/// selection procedure).
	double eval(const RealVector& parameters) const {
		SHARK_ASSERT(mep_meta != NULL);
		mep_meta->setParameterVector(parameters);
		return eval();
	}
protected:
	DataView<DatasetType const> m_dataset;
	IParameterizable* mep_meta;
	ModelType* mep_model;
	TrainerType* mep_trainer;
	LossType* mep_loss;
};


}
#endif
