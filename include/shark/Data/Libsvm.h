//===========================================================================
/*!
 * 
 *
 * \brief   Deprecated import_libsvm and export_libsvm functions.
 * 
 * 
 * \par
 * This file is provided for backwards compatibility.
 * Its is deprecated, use SparseData.h for new projects.
 * 
 * 
 * 
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

#ifndef SHARK_DATA_LIBSVM_H
#define SHARK_DATA_LIBSVM_H
#include <shark/Data/SparseData.h>

namespace shark {

/**
 * \ingroup shark_globals
 *
 * @{
 */

/// \brief Import data from a LIBSVM file.
///
/// \deprecated { use importSparseData instead }
///
/// \param  dataset       container storing the loaded data
/// \param  stream        stream to be read from
/// \param  highestIndex  highest feature index, or 0 for auto-detection
/// \param  batchSize     size of batch
inline void import_libsvm(
	LabeledData<RealVector, unsigned int>& dataset,
	std::istream& stream,
	unsigned int highestIndex = 0,
	std::size_t batchSize = LabeledData<RealVector, unsigned int>::DefaultBatchSize
)
{ importSparseData(dataset, stream, highestIndex, batchSize); }

/// \brief Import data from a LIBSVM file.
///
/// \deprecated { use importSparseData instead }
///
/// \param  dataset       container storing the loaded data
/// \param  stream        stream to be read from
/// \param  highestIndex  highest feature index, or 0 for auto-detection
/// \param  batchSize     size of batch
inline void import_libsvm(
	LabeledData<CompressedRealVector, unsigned int>& dataset,
	std::istream& stream,
	unsigned int highestIndex = 0,
	std::size_t batchSize = LabeledData<RealVector, unsigned int>::DefaultBatchSize
)
{ importSparseData(dataset, stream, highestIndex, batchSize); }

/// \brief Import data from a LIBSVM file.
///
/// \deprecated { use importSparseData instead }
///
/// \param  dataset       container storing the loaded data
/// \param  fn            the file to be read from
/// \param  highestIndex  highest feature index, or 0 for auto-detection
/// \param  batchSize     size of batch
inline void import_libsvm(
	LabeledData<RealVector, unsigned int>& dataset,
	std::string fn,
	unsigned int highestIndex = 0,
	std::size_t batchSize = LabeledData<RealVector, unsigned int>::DefaultBatchSize
)
{ importSparseData(dataset, fn, highestIndex, batchSize); }

/// \brief Import data from a LIBSVM file.
///
/// \deprecated { use importSparseData instead }
///
/// \param  dataset       container storing the loaded data
/// \param  fn            the file to be read from
/// \param  highestIndex  highest feature index, or 0 for auto-detection
/// \param  batchSize     size of batch
inline void import_libsvm(
	LabeledData<CompressedRealVector, unsigned int>& dataset,
	std::string fn,
	unsigned int highestIndex = 0,
	std::size_t batchSize = LabeledData<RealVector, unsigned int>::DefaultBatchSize
)
{ importSparseData(dataset, fn, highestIndex, batchSize); }


/// \brief Export data to LIBSVM format.
///
/// \deprecated { use exportSparseData instead }
///
/// \param  dataset     Container storing the  data
/// \param  fn          Output file
/// \param  dense       Flag for using dense output format
/// \param  oneMinusOne Flag for applying the transformation y<-2y-1 to binary labels
/// \param  sortLabels  Flag for sorting data points according to labels
/// \param  append      Flag for appending to the output file instead of overwriting it
template<typename InputType>
inline void export_libsvm(LabeledData<InputType, unsigned int>& dataset, const std::string &fn, bool dense=false, bool oneMinusOne = true, bool sortLabels = false, bool append = false) {
	exportSparseData(dataset, fn, dense, oneMinusOne, sortLabels, append);
}

/** @}*/

}
#endif
