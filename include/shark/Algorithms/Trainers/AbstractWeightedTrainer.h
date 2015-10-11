//===========================================================================
/*!
 * 
 *
 * \brief       Abstract Trainer Interface for trainers that support weighting
 * 
 * 
 *
 * \author      O. Krause
 * \date        2014
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
#ifndef SHARK_ALGORITHMS_TRAINERS_ABSTRACT_WEIGHTED_TRAINER_H
#define SHARK_ALGORITHMS_TRAINERS_ABSTRACT_WEIGHTED_TRAINER_H

#include <shark/Data/WeightedDataset.h>
#include <shark/Algorithms/Trainers/AbstractTrainer.h>

namespace shark {


/// \brief Superclass of weighted supervised learning algorithms
///
/// \par
/// AbstractWeightedTrainer is the super class of all trainers
/// that support weighted datasets. Weights are interpreted here
/// as the importance of a sample. unweighted training assumes
/// that all samples have the same importance, or weight. 
/// The higher the weight, the more important a point. Weight
/// 0 is the same as if the point would not be part of the dataset.
/// Negative weights are not allowed.
///
/// When all weights are integral values there is a simple interpretation
/// of the weights as the multiplicity of a point. Thus training 
/// with a dataset with duplicate points is the same as counting the duplicates
/// and run the algorithm with a weighted dataset where all points are unique and
/// have their weight is the multiplicity.
template <class Model, class LabelTypeT = typename Model::OutputType>
class AbstractWeightedTrainer : public AbstractTrainer<Model,LabelTypeT>
{
private:
	typedef AbstractTrainer<Model,LabelTypeT> base_type;
public:
	typedef typename base_type::ModelType ModelType;
	typedef typename base_type::InputType InputType;
	typedef typename base_type::LabelType LabelType;
	typedef typename base_type::DatasetType DatasetType;
	typedef WeightedLabeledData<InputType, LabelType> WeightedDatasetType;

	/// \brief Executes the algorithm and trains a model on the given weighted data.
	virtual void train(ModelType& model, WeightedDatasetType const& dataset) = 0;

	/// \brief Executes the algorithm and trains a model on the given unweighted data.
	///
	/// This method behaves as using train with a weighted dataset where all weights are equal.
	/// The default implementation just creates such a dataset and executes the weighted
	/// version of the algorithm.
	virtual void train(ModelType& model, DatasetType const& dataset){
		train(model,WeightedDatasetType(dataset, 1.0));
	}
};


/// \brief Superclass of weighted unsupervised learning algorithms
///
/// \par
/// AbstractWeightedUnsupervisedTrainer is the super class of all trainers
/// that support weighted datasets. See AbstractWeightedTrainer for more information on
/// the weights.
/// \see AbstractWeightedTrainer
template <class Model>
class AbstractWeightedUnsupervisedTrainer : public AbstractUnsupervisedTrainer<Model>
{
private:
	typedef AbstractUnsupervisedTrainer<Model> base_type;
public:
	typedef typename base_type::ModelType ModelType;
	typedef typename base_type::InputType InputType;
	typedef typename base_type::DatasetType DatasetType;
	typedef WeightedUnlabeledData<InputType> WeightedDatasetType;

	/// \brief Excecutes the algorithm and trains a model on the given weighted data.
	virtual void train(ModelType& model, WeightedDatasetType const& dataset) = 0;

	/// \brief Excecutes the algorithm and trains a model on the given undata.
	///
	/// This method behaves as using train with a weighted dataset where all weights are equal.
	/// The default implementation just creates such a dataset and executes the weighted
	/// version of the algorithm.
	virtual void train(ModelType& model, DatasetType const& dataset){
		train(model, WeightedDatasetType(dataset, 1.0));
	}
};


}
#endif
