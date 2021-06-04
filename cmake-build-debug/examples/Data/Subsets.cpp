//===========================================================================
/*!
 * 
 *
 * \brief       Data Subsets
 * 
 * This file is part of the tutorial "Creating and Using Subsets of Data".
 * By itself, it does not do anything particularly useful.
 *
 * \author      T. Glasmachers
 * \date        2014
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://shark-ml.org/>
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

#include <shark/Data/Dataset.h>
using namespace shark;

#include <shark/Data/CVDatasetTools.h>

#include <shark/Data/DataView.h>


int main()
{
	typedef RealVector I;
	typedef unsigned int L;

	std::size_t start = 0, end = 42;

{
	LabeledData<I,L> dataset;             // our dataset

	// create an indexed subset of batches
	std::vector<std::size_t> indices;     // indices of the batches to be contained in the subset
	LabeledData<I,L> subset = dataset.indexedSubset(indices);

	unsigned int k = 7;
	LabeledData<I,L> remaining_batches = dataset.splice(k);

	LabeledData<I,L> remaining_elements = splitAtElement(dataset, k);
}

	ClassificationDataset data;
	// ...
	repartitionByClass(data);

	std::size_t class0 = 0, class1 = 1;
	ClassificationDataset subproblem = binarySubProblem(data, class0, class1);

{
	DataView<ClassificationDataset> view(data);

	// creating a random subset from indices
	std::size_t k = 100;
	std::vector<std::size_t> indices(view.size());
	for (std::size_t i=0; i<view.size(); i++) indices[i] = i;
	for (std::size_t i=0; i<k; i++) std::swap(indices[i], indices[rand() % view.size()]);
	indices.resize(k);
	DataView<ClassificationDataset> subset1 = subset(view, indices);

	// same functionality in one line
	DataView<ClassificationDataset> subset2 = randomSubset(view, k);
}

{
	std::size_t numberOfPartitions = 5;
	std::vector<std::size_t> indices;
	// Creates partitions of approximately the same size.
	createCVSameSize(data, numberOfPartitions);

	// Creates IID drawn partitions of the data set (without replacement).
	createCVIID(data, numberOfPartitions);

	// Creates indexed cross-validation sets. For each element the
	// index describes the fold in which the data point acts as a
	// validation example. This function offers maximal control.
	createCVIndexed(data, numberOfPartitions, indices);

	createCVSameSizeBalanced(data, numberOfPartitions);

}

{
	std::size_t numberOfPartitions = 5;
	std::size_t numberOfFolds = 3;
	CVFolds<RegressionDataset> folds;

	for (std::size_t i=0; i<numberOfPartitions; i++)
	{
	// as created in the above example
	RegressionDataset training = folds.training(i);
	RegressionDataset validation = folds.validation(i);
	// explicit copy!
	training.makeIndependent();
	// creating a new fold
	CVFolds<RegressionDataset> innerFolds = createCVSameSize(training, numberOfFolds);
	}
}

}
