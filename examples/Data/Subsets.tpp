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
//===========================================================================

//###begin<include>
#include <shark/Data/Dataset.h>
using namespace shark;
//###end<include>

#include <shark/Data/CVDatasetTools.h>

//###begin<viewbased>
#include <shark/Data/DataView.h>
//###end<viewbased>


int main()
{
	typedef RealVector I;
	typedef unsigned int L;

	std::size_t start = 0, end = 42;

{
//###begin<subsets>
	LabeledData<I,L> dataset;             // our dataset

	// create an indexed subset of batches
	std::vector<std::size_t> indices;     // indices of the batches to be contained in the subset
	LabeledData<I,L> subset = indexedSubset(dataset, indices);

	// if also the complement of the set is needed, the call is:
	LabeledData<I,L> complement;
	dataset.indexedSubset(indices, subset, complement);

	// create subsets from ranges of batches
	LabeledData<I,L> range1 = rangeSubset(dataset, start, end);   // contains batches start,...,end-1
	LabeledData<I,L> range2 = rangeSubset(dataset, end);          // contains batches 0,...,end-1
//###end<subsets>

	unsigned int k = 7;
//###begin<splice>
	LabeledData<I,L> remaining_batches = dataset.splice(k);
//###end<splice>

//###begin<splitAtElement>
	LabeledData<I,L> remaining_elements = splitAtElement(dataset, k);
//###end<splitAtElement>
}

//###begin<repartitionByClass>
	ClassificationDataset data;
	// ...
	repartitionByClass(data);
//###end<repartitionByClass>

	std::size_t class0 = 0, class1 = 1;
//###begin<binarySubProblem>
	ClassificationDataset subproblem = binarySubProblem(data, class0, class1);
//###end<binarySubProblem>

{
//###begin<viewbased>
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
//###end<viewbased>
}

{
	std::size_t numberOfPartitions = 5;
	std::vector<std::size_t> indices;
//###begin<splitting>
	// Creates partitions of approximately the same size.
	createCVSameSize(data, numberOfPartitions);

	// Creates IID drawn partitions of the data set (without replacement).
	createCVIID(data, numberOfPartitions);

	// Creates indexed cross-validation sets. For each element the
	// index describes the fold in which the data point acts as a
	// validation example. This function offers maximal control.
	createCVIndexed(data, numberOfPartitions, indices);
//###end<splitting>

//###begin<balanced>
	createCVSameSizeBalanced(data, numberOfPartitions);
//###end<balanced>

}

{
	std::size_t numberOfPartitions = 5;
	std::size_t numberOfFolds = 3;
	CVFolds<RegressionDataset> folds;

	for (std::size_t i=0; i<numberOfPartitions; i++)
	{
//###begin<nested-cv>
	// as created in the above example
	RegressionDataset training = folds.training(i);
	RegressionDataset validation = folds.validation(i);
	// explicit copy!
	training.makeIndependent();
	// creating a new fold
	CVFolds<RegressionDataset> innerFolds = createCVSameSize(training, numberOfFolds);
//###end<nested-cv>
	}
}

}
