//===========================================================================
/*!
 *  \file CrossValidation.h
 *
 *  \brief Cross Validation
 *
 *
 *  The cross validation procedure is designed to adapt so-called
 *  hyperparameters of a model, which is based upon another model.
 *  In every hyperparameter evaluation step, the base model is
 *  trained. This inner training procedure depends on the
 *  hyperparameters. Thus, the hyperparameters can be evaluated by
 *  simply evaluating the trained base model. To avoid overfitting
 *  the cross validation procedure splits the available data into
 *  training and validation subsets, such that all data points
 *  appear in training and validation subsets equally often.
 *
 *
 *  \author  T. Glasmachers
 *  \date    2006
 *
 *  \par Copyright (c) 1999-2006:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *
 *
 *  <BR><HR>
 *  This file is part of Shark. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 2, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, write to the Free Software
 *  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */
//===========================================================================

#ifndef _CrossValidation_H_
#define _CrossValidation_H_


#include <ReClaM/Model.h>
#include <ReClaM/ErrorFunction.h>
#include <ReClaM/Optimizer.h>
#include <Rng/GlobalRng.h>
#include <vector>


//! The Partitioning class defined a partitioning of
//! a set of training points and labels as it is required
//! for a cross validation procedure. That is, the data
//! are split into N subsets, usually of comparable size.
//! Then, the data are combined into training and validation
//! sets in N different ways, called partitions. For every
//! partition, one of the N subsets is used for validation,
//! while the union of all other subsets it used for
//! training.
class Partitioning
{
public:
	//! Constructor
	Partitioning();

	//! Destructor
	~Partitioning();


	//!
	//! \brief Create a partition
	//!
	//! The subset each training examples belongs to
	//! is drawn independently and uniformly distributed.
	//! For every partition, all but one subset form the
	//! training set, while the remaining one is used for
	//! validation.
	//!
	//! \param numberOfPartitions  number of partitions to create
	//! \param input               input data to separate into subsets
	//! \param target              labels corresponding to the input data
	//!
	void CreateIID(int numberOfPartitions, const Array<double>& input, const Array<double>& target);

	//!
	//! \brief Create a partition
	//!
	//! Every subset contains (approximately) the same
	//! number of elements. For every partition, all
	//! but one subset form the training set, while the
	//! remaining one is used for validation.
	//!
	//! \param numberOfPartitions  number of partitions to create
	//! \param input               input data to separate into subsets
	//! \param target              labels corresponding to the input data
	//!
	void CreateSameSize(int numberOfPartitions, const Array<double>& input, const Array<double>& target);

	//!
	//! \brief Create a partition
	//!
	//! Every subset contains (approximately) the same
	//! number of elements. For every partition, all
	//! but one subset form the training set, while the
	//! remaining one is used for validation.
	//! The targets are assumed to be +1 and -1 and the
	//! subsets contain approximately the same fraction
	//! of positive and negative examples as the whole
	//! training set.
	//!
	//! \param numberOfPartitions  number of partitions to create
	//! \param input               input data to separate into subsets
	//! \param target              labels corresponding to the input data
	//!
	void CreateSameSizeBalanced(int numberOfPartitions, const Array<double>& input, const Array<double>& target);

	//!
	//! \brief Create a partition for multi-class problems in a stratified way
	//!
	//! Every subset contains (approximately) the same
	//! number of elements. For every partition, all
	//! but one subset form the training set, while the
	//! remaining one is used for validation.
	//! The targets are assumed to intergers and the
	//! subsets contain approximately the same fraction
	//! of examples from each class as the whole
	//! training set. The function is deterministic, so 
	//! consider calling ShuffleTraining() before.
	//!
	//! \param numberOfPartitions  number of partitions to create
	//! \param input               input data to split into subsets
	//! \param target              labels corresponding to the input data
	//!
	void CreateSameSizeMultiClassBalanced(int numberOfPartitions, const Array<double>& input, const Array<double>& target);

	//!
	//! \brief Create a partition
	//!
	//! Every subset contains (approximately) the same
	//! number of elements. For every partition, all
	//! but one subset form the training set, while the
	//! remaining one is used for validation.
	//!
	//! \param numberOfPartitions  number of partitions to create
	//! \param input               input data to separate into subsets
	//! \param target              labels corresponding to the input data
	//! \param index               partition indices of the examples in [0, ..., numberOfPartitions[.
	//!
	void CreateIndexed(int numberOfPartitions, const Array<double>& input, const Array<double>& target, const Array<int>& index);


	//! Return the number of partitions in the partitioning.
	inline int getNumberOfPartitions()
	{
		return partitions;
	}

	//! Return the part-th training point array
	inline const Array<double>& train_input(int part)
	{
		return *(part_train_input[part]);
	}

	//! Return the part-th training label array
	inline const Array<double>& train_target(int part)
	{
		return *(part_train_target[part]);
	}

	//! Return the part-th validation point array
	inline const Array<double>& validation_input(int part)
	{
		return *(part_validation_input[part]);
	}

	//! Return the part-th validation label array
	inline const Array<double>& validation_target(int part)
	{
		return *(part_validation_target[part]);
	}

	//! Return the validation set partition for a given example
	inline int getPartition(int example)
	{
		return part_index(example);
	}

	//! Return the example at a given index in a given partition
	inline int getExample(int part, int index)
	{
		return (*inverse_part_index[part])(index);
	}

protected:
	//! free memory
	void Clear();

	//! number of partitions of the data
	int partitions;

	//! index array: which example belongs to which validation set?
	Array<int> part_index;

	//! index array: which validation set consists of which examples?
	std::vector< Array<int>* > inverse_part_index;

	//! training points
	std::vector< Array<double>* > part_train_input;

	//! training labels
	std::vector< Array<double>* > part_train_target;

	//! validation points
	std::vector< Array<double>* > part_validation_input;

	//! validation labels
	std::vector< Array<double>* > part_validation_target;
};


//!
//! \brief Collection of sub-models for cross validation
//!
//! \par
//! The CVModel class is based upon a set of models. The idea is
//! to have one independent model for every partition of the data
//! during the cross validation procedure. The class simply
//! collects these base models and synchronizes its parameters,
//! as long as they are accessed via the CVModel class.
//!
//! \par
//! In principle, it is not clear which base model to use for
//! prediction on unseen data. However, it makes sence to use the
//! base models on the whole cross validation data set, that is
//! as well on the subsets used for training, as well as on the
//! subsets for validation.
class CVModel : public Model
{
public:
	//! \brief Constructor
	//!
	//! \par
	//! This is the default constructor of a cross-validation model.
	//! An array containing one base model per fold is provided as
	//! a parameter. CVModels constructed this way can be used to
	//! make predictions using setBaseModel, model, and modelDerivative.
	//!
	//! \param  models  Array of identical base models. There must be as many models in the array as there are cross validation partitions.
	CVModel(Array<Model*>& models);

	//! \brief Constructor
	//!
	//! \par
	//! This simplified constructor makes the CVModel use the same
	//! underlying base model for each fold. The model can be evaluated
	//! via CVError::error and CVError::errorDerivative, but the results
	//! of calls to setBaseModel, model, and modelDerivative are undefined.
	//!
	//! \param  models  base model used for all folds simultaneously.
	CVModel(unsigned int folds, Model* basemodel);

	//! Destructor
	~CVModel();


	//! Modifies a specific model parameter.
	void setParameter(unsigned int index, double value);

	//! Get the number of base models in use.
	inline int getBaseModels()
	{
		return baseModel.dim(0);
	}

	//! Set the currently used base model.
	void setBaseModel(int index);

	//! Return a reference to the current base model.
	inline Model& getBaseModel()
	{
		return *baseModel(baseModelIndex);
	}

	//! The base model is used for model computations.
	void model(const Array<double>& input, Array<double>& output);

	//! The base model is used for model computations.
	void modelDerivative(const Array<double>& input, Array<double>& derivative);

	//! The base model is used for model computations.
	void modelDerivative(const Array<double>& input, Array<double>& output, Array<double>& derivative);

	//! Check whether the parameters define a valid model
	bool isFeasible();

protected:
	//! Models used for the data splits.
	Array<Model*> baseModel;

	//! Index of the baseModel to use
	int baseModelIndex;
};


//!
//! \brief #ErrorFunction based on a cross validation procedure
//!
//! \par
//! The CVError class computes the mean error over partitions,
//! that is, the cross validation error. It trains the sub-models
//! defined by the #CVModel object with a given #Optimizer on a
//! given #ErrorFunction using the training parts of the partitions.
//! For the mean error computation, it uses the varidation part of
//! the partitions, which contain every example exactly once.
class CVError : public ErrorFunction
{
public:
	//! Constructor
	//!
	//! \param  part       #Partitioning defining subsets
	//! \param  error      #ErrorFunction used for the subset tasks
	//! \param  optimizer  #Optimizer used for the subset tasks
	//! \param  iter       number of optimization iterations for the subset tasks
	CVError(Partitioning& part, ErrorFunction& error, Optimizer& optimizer, int iter);

	//! Destructor
	~CVError();


	//! Compute the cross validation error defined as the mean
	//! error over the subsets.
	//!
	//! \param  model  The model parameter has to be a reference to a #CVModel object.
	double error(Model& model, const Array<double>& input, const Array<double>& target);

	//! Compute the cross validation error defined as the mean
	//! error over the subsets and its derivative. This makes
	//! rarely sense, as cross validation is usually used with
	//! non-differentiable error measures.
	double errorDerivative(Model& model, const Array<double>& input, const Array<double>& target, Array<double>& derivative);

	inline ErrorFunction& getBaseError()
	{
		return baseError;
	}

	inline Optimizer& getBaseOptimizer()
	{
		return baseOptimizer;
	}

	inline Partitioning& getPartitioning()
	{
		return partitioning;
	}

protected:
	//! Partitioning upon which the cross validation procedure is based
	Partitioning& partitioning;

	//! ErrorFunction to use for the single cross validation tasks
	ErrorFunction& baseError;

	//! Optimizer to use for the single cross validation tasks
	Optimizer& baseOptimizer;

	//! Number of iterations to perform for the optimization of the single cross validation tasks
	int iterations;
};


#endif

