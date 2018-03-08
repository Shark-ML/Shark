//===========================================================================
/*!
 * 
 *
 * \brief   Support for importing and exporting data from and to sparse data (libSVM) formatted data files
 * 
 * 
 * \par
 * The most important application of the methods provided in this
 * file is the import of data from LIBSVM files to Shark Data containers.
 * 
 * 
 * 
 *
 * \author      M. Tuma, T. Glasmachers, C. Igel
 * \date        2010-2016
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

#ifndef SHARK_DATA_SPARSEDATA_H
#define SHARK_DATA_SPARSEDATA_H

#include <shark/Core/DLLSupport.h>
#include <shark/Core/utility/KeyValuePair.h>
#include <shark/Data/Dataset.h>
#include <fstream>

namespace shark {

/**
 * \ingroup shark_globals
 *
 * @{
 */



/// \brief Import classification data from a sparse data (libSVM) file.
///
/// \param  dataset       container storing the loaded data
/// \param  stream        stream to be read from
/// \param  highestIndex  highest feature index, or 0 for auto-detection
/// \param  batchSize     size of batch
SHARK_EXPORT_SYMBOL void importSparseData(
	LabeledData<RealVector, unsigned int>& dataset,
	std::istream& stream,
	unsigned int highestIndex = 0,
	std::size_t batchSize = LabeledData<RealVector, unsigned int>::DefaultBatchSize
);

SHARK_EXPORT_SYMBOL void importSparseData(
	LabeledData<FloatVector, unsigned int>& dataset,
	std::istream& stream,
	unsigned int highestIndex = 0,
	std::size_t batchSize = LabeledData<RealVector, unsigned int>::DefaultBatchSize
);

/// \brief Import regression data from a sparse data (libSVM) file.
///
/// \param  dataset       container storing the loaded data
/// \param  stream        stream to be read from
/// \param  highestIndex  highest feature index, or 0 for auto-detection
/// \param  batchSize     size of batch
SHARK_EXPORT_SYMBOL void importSparseData(
	LabeledData<RealVector, RealVector>& dataset,
	std::istream& stream,
	unsigned int highestIndex = 0,
	std::size_t batchSize = LabeledData<RealVector, RealVector>::DefaultBatchSize
);

SHARK_EXPORT_SYMBOL void importSparseData(
	LabeledData<FloatVector, FloatVector>& dataset,
	std::istream& stream,
	unsigned int highestIndex = 0,
	std::size_t batchSize = LabeledData<RealVector, RealVector>::DefaultBatchSize
);

/// \brief Import classification data from a sparse data (libSVM) file.
///
/// \param  dataset       container storing the loaded data
/// \param  stream        stream to be read from
/// \param  highestIndex  highest feature index, or 0 for auto-detection
/// \param  batchSize     size of batch
SHARK_EXPORT_SYMBOL void importSparseData(
	LabeledData<CompressedRealVector, unsigned int>& dataset,
	std::istream& stream,
	unsigned int highestIndex = 0,
	std::size_t batchSize = LabeledData<RealVector, unsigned int>::DefaultBatchSize
);
SHARK_EXPORT_SYMBOL void importSparseData(
	LabeledData<CompressedFloatVector, unsigned int>& dataset,
	std::istream& stream,
	unsigned int highestIndex = 0,
	std::size_t batchSize = LabeledData<RealVector, unsigned int>::DefaultBatchSize
);

/// \brief Import regression data from a sparse data (libSVM) file.
///
/// \param  dataset       container storing the loaded data
/// \param  stream        stream to be read from
/// \param  highestIndex  highest feature index, or 0 for auto-detection
/// \param  batchSize     size of batch
SHARK_EXPORT_SYMBOL void importSparseData(
	LabeledData<CompressedRealVector, RealVector>& dataset,
	std::istream& stream,
	unsigned int highestIndex = 0,
	std::size_t batchSize = LabeledData<RealVector, RealVector>::DefaultBatchSize
);
SHARK_EXPORT_SYMBOL void importSparseData(
	LabeledData<CompressedFloatVector, FloatVector>& dataset,
	std::istream& stream,
	unsigned int highestIndex = 0,
	std::size_t batchSize = LabeledData<RealVector, RealVector>::DefaultBatchSize
);


/// \brief Import classification data from a sparse data (libSVM) file.
///
/// \param  dataset       container storing the loaded data
/// \param  fn            the file to be read from
/// \param  highestIndex  highest feature index, or 0 for auto-detection
/// \param  batchSize     size of batch
SHARK_EXPORT_SYMBOL void importSparseData(
	LabeledData<RealVector, unsigned int>& dataset,
	std::string fn,
	unsigned int highestIndex = 0,
	std::size_t batchSize = LabeledData<RealVector, unsigned int>::DefaultBatchSize
);
SHARK_EXPORT_SYMBOL void importSparseData(
	LabeledData<FloatVector, unsigned int>& dataset,
	std::string fn,
	unsigned int highestIndex = 0,
	std::size_t batchSize = LabeledData<RealVector, unsigned int>::DefaultBatchSize
);

/// \brief Import regression data from a sparse data (libSVM) file.
///
/// \param  dataset       container storing the loaded data
/// \param  fn            the file to be read from
/// \param  highestIndex  highest feature index, or 0 for auto-detection
/// \param  batchSize     size of batch
SHARK_EXPORT_SYMBOL void importSparseData(
	LabeledData<RealVector, RealVector>& dataset,
	std::string fn,
	unsigned int highestIndex = 0,
	std::size_t batchSize = LabeledData<RealVector, RealVector>::DefaultBatchSize
);
SHARK_EXPORT_SYMBOL void importSparseData(
	LabeledData<FloatVector, FloatVector>& dataset,
	std::string fn,
	unsigned int highestIndex = 0,
	std::size_t batchSize = LabeledData<RealVector, RealVector>::DefaultBatchSize
);

/// \brief Import classification data from a sparse data (libSVM) file.
///
/// \param  dataset       container storing the loaded data
/// \param  fn            the file to be read from
/// \param  highestIndex  highest feature index, or 0 for auto-detection
/// \param  batchSize     size of batch
SHARK_EXPORT_SYMBOL void importSparseData(
	LabeledData<CompressedRealVector, unsigned int>& dataset,
	std::string fn,
	unsigned int highestIndex = 0,
	std::size_t batchSize = LabeledData<RealVector, unsigned int>::DefaultBatchSize
);
SHARK_EXPORT_SYMBOL void importSparseData(
	LabeledData<CompressedFloatVector, unsigned int>& dataset,
	std::string fn,
	unsigned int highestIndex = 0,
	std::size_t batchSize = LabeledData<RealVector, unsigned int>::DefaultBatchSize
);

/// \brief Import regression data from a sparse data (libSVM) file.
///
/// \param  dataset       container storing the loaded data
/// \param  fn            the file to be read from
/// \param  highestIndex  highest feature index, or 0 for auto-detection
/// \param  batchSize     size of batch
SHARK_EXPORT_SYMBOL void importSparseData(
	LabeledData<CompressedRealVector, RealVector>& dataset,
	std::string fn,
	unsigned int highestIndex = 0,
	std::size_t batchSize = LabeledData<RealVector, RealVector>::DefaultBatchSize
);
SHARK_EXPORT_SYMBOL void importSparseData(
	LabeledData<CompressedFloatVector, FloatVector>& dataset,
	std::string fn,
	unsigned int highestIndex = 0,
	std::size_t batchSize = LabeledData<RealVector, RealVector>::DefaultBatchSize
);


/// \brief Export classification data to sparse data (libSVM) format.
///
/// \param  dataset     Container storing the  data
/// \param  stream      Output stream
/// \param  oneMinusOne Flag for applying the transformation y<-2y-1 to binary labels
/// \param  sortLabels  Flag for sorting data points according to labels
template<typename InputType>
void exportSparseData(LabeledData<InputType, unsigned int> const& dataset, std::ostream& stream, bool oneMinusOne = true, bool sortLabels = false)
{
	if (numberOfClasses(dataset) != 2) oneMinusOne = false;

	std::vector< KeyValuePair<unsigned int, std::pair<std::size_t, std::size_t> > > order;
	for (std::size_t b=0; b<dataset.numberOfBatches(); b++)
	{
		auto batch = dataset.batch(b);
		for (std::size_t i=0; i<batchSize(batch); i++)
		{
			order.emplace_back(getBatchElement(batch, i).label, std::make_pair(b, i));
		}
	}
	if (sortLabels)
	{
		std::sort(order.begin(), order.end());
	}

	for (auto const& p : order)
	{
		auto element = getBatchElement(dataset.batch(p.value.first), p.value.second);
		// apply transformation to label and write it to file
		if (oneMinusOne) stream << 2*int(element.label)-1 << " ";
		//libsvm file format documentation is scarce, but by convention the first class seems to be 1..
		else stream << element.label+1 << " ";
		// write input data to file
		for (auto it = element.input.begin(); it != element.input.end(); ++it)
		{
			stream << " " << it.index()+1 << ":" << *it;
		}
		stream << std::endl;
	}
}

/// \brief Export classification data to sparse data (libSVM) format.
///
/// \param  dataset     Container storing the data
/// \param  fn          Output file name
/// \param  oneMinusOne Flag for applying the transformation y<-2y-1 to binary labels
/// \param  sortLabels  Flag for sorting data points according to labels
/// \param  append      Flag for appending to the output file instead of overwriting it
template<typename InputType>
void exportSparseData(LabeledData<InputType, unsigned int> const& dataset, const std::string &fn, bool oneMinusOne = true, bool sortLabels = false, bool append = false)
{
	std::ofstream ofs;

	// shall we append only or overwrite?
	if (append == true) {
	    ofs.open (fn.c_str(), std::fstream::out | std::fstream::app );
	} else {
	    ofs.open (fn.c_str());
	}
	SHARK_RUNTIME_CHECK(ofs, "File can not be opened for writing");

	exportSparseData(dataset, ofs, oneMinusOne, sortLabels);
}

/// \brief Export regression data to sparse data (libSVM) format.
///
/// \param  dataset     Container storing the data
/// \param  stream      Output stream
template<typename InputType>
void exportSparseData(LabeledData<InputType, RealVector> const& dataset, std::ostream& stream)
{
	for (std::size_t b=0; b<dataset.numberOfBatches(); b++)
	{
		auto batch = dataset.batch(b);
		for (std::size_t i=0; i<batchSize(batch); i++)
		{
			auto element = getBatchElement(batch, i);
			SHARK_ASSERT(element.label.size() == 1);
			stream << element.label(0);
			for (auto it = element.input.begin(); it != element.input.end(); ++it)
			{
				stream << " " << it.index()+1 << ":" << *it;
			}
			stream << std::endl;
		}
	}
}

/// \brief Export regression data to sparse data (libSVM) format.
///
/// \param  dataset     Container storing the  data
/// \param  fn          Output file
/// \param  append      Flag for appending to the output file instead of overwriting it
template<typename InputType>
void exportSparseData(LabeledData<InputType, RealVector> const& dataset, const std::string &fn, bool append = false)
{
	std::ofstream ofs;

	// shall we append only or overwrite?
	if (append == true) {
		ofs.open (fn.c_str(), std::fstream::out | std::fstream::app );
	} else {
		ofs.open (fn.c_str());
	}

	SHARK_RUNTIME_CHECK(ofs, "File can not be opened for writing");

	exportSparseData(dataset, ofs);
}

/** @}*/

}
#endif
