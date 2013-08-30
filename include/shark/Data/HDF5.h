//===========================================================================
/*!
 *  \brief Support for importing data from HDF5 file
 *
 *
 *  \par
 *  The most important application of the methods provided in this
 *  file is the import of data from HDF5 files into Shark data
 *  containers.
 *
 *
 *  \author  B. Li
 *  \date    2012
 *
 *  \par Copyright (c) 2012:
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
 *  Foundation; either version 3, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, see <http://www.gnu.org/licenses/>.
 *
 */
//===========================================================================

#ifndef SHARK_DATA_HDF5_H
#define SHARK_DATA_HDF5_H

#include "shark/Core/utility/ScopedHandle.h"
#include "shark/Data/Dataset.h"

#include <hdf5.h> // This must come before #include <hdf5_hl.h>
#include <hdf5_hl.h>

#include <boost/array.hpp>
#include <boost/foreach.hpp>
#include <boost/range/algorithm/fill.hpp>
#include <boost/range/algorithm/max_element.hpp>
#include <boost/smart_ptr/scoped_array.hpp>
#include <boost/type_traits.hpp>

namespace shark {

namespace detail {

/// Overload functions so that complier is able to automatically detect which function to call
/// @note
///     Basically there are two ways to add support for other data types:
///     (a) Use other corresponding API H5HTpublic.h if the type is supported(luckily)
///     (b) Use H5LTread_dataset() but need pass in the type_id which are listed at:
///         http://www.hdfgroup.org/HDF5/doc/RM/PredefDTypes.html
///         Need pay special attention to endian.
///@{
herr_t readHDF5Dataset( hid_t loc_id, const char *dset_name, int *buffer )
{
	return H5LTread_dataset_int( loc_id, dset_name, buffer );
}

herr_t readHDF5Dataset( hid_t loc_id, const char *dset_name, long *buffer )
{
	return H5LTread_dataset_long( loc_id, dset_name, buffer );
}

herr_t readHDF5Dataset( hid_t loc_id, const char *dset_name, float *buffer )
{
	return H5LTread_dataset_float( loc_id, dset_name, buffer );
}

herr_t readHDF5Dataset( hid_t loc_id, const char *dset_name, double *buffer )
{
	return H5LTread_dataset_double( loc_id, dset_name, buffer );
}
///@}

/// Check whether @typeClass and @typeSize are supported by current implementation
template<typename RawValueType>
bool isSupported(H5T_class_t typeClass, size_t typeSize)
{
	if (H5T_FLOAT == typeClass && 8 == typeSize && boost::is_floating_point < RawValueType > ::value
	    && sizeof(RawValueType) == 8) {
		// double
		return true;
	} else if (H5T_FLOAT == typeClass && 4 == typeSize && boost::is_floating_point < RawValueType > ::value
	    && sizeof(RawValueType) == 4) {
		// float
		return true;
	} else if (H5T_INTEGER == typeClass && 4 == typeSize && boost::is_integral < RawValueType > ::value
	    && sizeof(RawValueType) == 4) {
		// int
		return true;
	} else if (H5T_INTEGER == typeClass && 8 == typeSize && boost::is_integral < RawValueType > ::value
	    && sizeof(RawValueType) == 8) {
		// long
		return true;
	}

	return false;
}

/// @brief Load a dataset in a HDF5 file into a matrix
///
/// @param data
///     in vector of vector format which should support assignment operations
/// @param fileName
///     The name of HDF5 file to be read from
/// @param dataSetName
///     the HDF5 dataset name to access in the HDF5 file
///
/// @tparam MatrixType
///     The type of data container which will accept read-in data and should be a 2-dimension matrix
template<typename MatrixType>
void loadIntoMatrix(MatrixType& data, const std::string& fileName, const std::string& dataSetName)
{
	typedef typename MatrixType::value_type VectorType; // e.g., std::vector<double>
	typedef typename VectorType::value_type RawValueType; // e.g., double

	// Disable HDF5 diagnosis message which could be commented out in case of debugging HDF5 related issues
	H5Eset_auto1(0, 0);

	// 64 is big enough for HDF5, which supports no more than 32 dimensions presently
	const size_t MAX_DIMENSIONS = 64u;

	// Open the file, and then get dimension
	const ScopedHandle<hid_t> fileId(
		H5Fopen(fileName.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT),
		H5Fclose,
		(boost::format("[loadIntoMatrix] open file name: %1%") % fileName).str());

	boost::array<hsize_t, MAX_DIMENSIONS> dims;
	dims.assign(0);
	H5T_class_t typeClass;
	size_t typeSize;
	THROW_IF(
		H5LTget_dataset_info(*fileId, dataSetName.c_str(), dims.c_array(), &typeClass, &typeSize) < 0,
		(boost::format("[importHDF5] Get data set(%1%) info from file(%2%).") % dataSetName % fileName).str());

	if (0 == dims[0])
		return;

	// Support 1 or 2 dimensions only at the moment
	THROW_IF(
		0 != dims[2],
		(boost::format(
			"[loadIntoMatrix][%1%][%2%] Support 1 or 2 dimensions, but this dataset has at least 3 dimensions.") % fileName % dataSetName).str());

	const hsize_t dim0 = dims[0];
	const hsize_t dim1 = (0 == dims[1]) ? 1 : dims[1]; // treat one dimension as two-dimension of N x 1

	THROW_IF(
		!detail::isSupported<RawValueType>(typeClass, typeSize),
		(boost::format(
			"[loadIntoMatrix] DataType doesn't match. HDF5 data type in dataset(%3%::%4%): %1%, size: %2%")
			% typeClass
			% typeSize
			% fileName
			% dataSetName).str());

	// Read data into a buffer
	const boost::scoped_array<RawValueType> dataBuffer(new RawValueType[dim0 * dim1]);
	THROW_IF(detail::readHDF5Dataset(*fileId, dataSetName.c_str(), dataBuffer.get()) < 0, "[loadIntoMatrix] Read data set.");

	// dims[0] = M, dims[1] = N, means each basic vector has M elements, and there are N of them.
	for (size_t i = 0; i < dim1; ++i) {
		VectorType sample(dim0);
		for (size_t j = 0; j < dim0; ++j)
			sample[j] = dataBuffer[i + j * dim1]; // elements in memory are in row-major order
		data.push_back(sample);
	}
}

/// @brief load a matrix from HDF5 file in compressed sparse column format
///
/// @param data the container which will hold the output matrix
/// @param fileName the name of HDF5 file
/// @param cscDatasetName dataset names for describing the CSC
template<typename MatrixType>
void loadHDF5Csc(MatrixType& data, const std::string& fileName, const std::vector<std::string>& cscDatasetName)
{
	typedef typename MatrixType::value_type VectorType; // e.g., std::vector<double>

	THROW_IF(
		3 != cscDatasetName.size(),
		"[importHDF5] Must provide 3 dataset names for importing Compressed Sparse Column format.");

	std::vector<VectorType> valBuf;
	std::vector<std::vector<boost::int32_t> > indicesBuf;
	std::vector<std::vector<boost::int32_t> > indexPtrBuf;
	detail::loadIntoMatrix(valBuf, fileName, cscDatasetName[0]);
	detail::loadIntoMatrix(indicesBuf, fileName, cscDatasetName[1]);
	detail::loadIntoMatrix(indexPtrBuf, fileName, cscDatasetName[2]);
	THROW_IF(1u != valBuf.size() || 1u != indicesBuf.size() || 1u != indexPtrBuf.size(), "All datasets should be of one dimension.");

	const VectorType& val = valBuf.front();
	const std::vector<boost::int32_t>& indices = indicesBuf.front(); // WARNING: Not all indices are of int32 type
	const std::vector<boost::int32_t>& indexPtr = indexPtrBuf.front();
	THROW_IF(val.size() != indices.size(), "Size of value and indices should be the same.");
	THROW_IF(indexPtr.back() != (boost::int32_t)val.size(), "Last element of index pointer should equal to size of value.");

	// Figure out dimensions of dense matrix
	const boost::uint32_t columnCount = indexPtr.size() - 1; // the last one is place holder
	const boost::uint32_t rowCount = *boost::max_element(indices) + 1; // max index plus 1

	data.resize(columnCount);
	boost::fill(data, VectorType(rowCount, 0)); // pre-fill zero

	size_t valIdx = 0;
	for (size_t i = 0; i < columnCount; ++i) {
		for (boost::int32_t j = indexPtr[i]; j < indexPtr[i + 1]; ++j) {
			data[i][indices[j]] = val[valIdx++];
		}
	}
}

/// @brief Construct labeled data from passed in data and label
///
/// @param labeledData
///     Container storing the loaded data
/// @param dataBuffer
///     The data container will hold
/// @param labelBuffer
///     The label for data inside @a dataBuffer
template<typename VectorType, typename LabelType>
void constructLabeledData(
	LabeledData<VectorType, LabelType>& labeledData,
	const std::vector<VectorType>& dataBuffer,
	const std::vector<std::vector<LabelType> >& labelBuffer)
{
	THROW_IF(
		1 != labelBuffer.size(),
		(boost::format("[importHDF5] Expect only one label vector, but get %1%.") % labelBuffer.size()).str());
	THROW_IF(
		dataBuffer.size() != labelBuffer.front().size(),
		boost::format("[importHDF5] Dimensions of data and label don't match.").str());

	labeledData = createLabeledDataFromRange(dataBuffer, labelBuffer.front());
}

} // namespace details

/// @brief Import data from a HDF5 file.
///
/// @param data        Container storing the loaded data
/// @param fileName    The name of HDF5 file to be read from
/// @param datasetName the HDF5 dataset name to access in the HDF5 file
///
/// @tparam VectorType   Type of object stored in Shark data container
template<typename VectorType>
void importHDF5(
	Data<VectorType>& data,
	const std::string& fileName,
	const std::string& datasetName)
{
	std::vector<VectorType> readinBuffer;
	detail::loadIntoMatrix(readinBuffer, fileName, datasetName);
	data = createDataFromRange(readinBuffer);
}

/// @brief Import data to a LabeledData object from a HDF5 file.
///
/// @param labeledData
///     Container storing the loaded data
/// @param fileName
///     The name of HDF5 file to be read from
/// @param data
///     the HDF5 dataset name for data
/// @param label
///     the HDF5 dataset name for label
///
/// @tparam VectorType
///     Type of object stored in Shark data container
/// @tparam LableType
///     Type of label
template<typename VectorType, typename LabelType>
void importHDF5(
	LabeledData<VectorType, LabelType>& labeledData,
	const std::string& fileName,
	const std::string& data,
	const std::string& label)
{
	std::vector<VectorType> readinData;
	std::vector < std::vector<LabelType> > readinLabel;

	detail::loadIntoMatrix(readinData, fileName, data);
	detail::loadIntoMatrix(readinLabel, fileName, label);
	detail::constructLabeledData(labeledData, readinData, readinLabel);
}

/// @brief Import data from HDF5 dataset of compressed sparse column format.
///
/// @param data        Container storing the loaded data
/// @param fileName    The name of HDF5 file to be read from
/// @param cscDatasetName
///     the CSC dataset names used to construct a matrix
///
/// @tparam VectorType   Type of object stored in Shark data container
template<typename VectorType>
void importHDF5(
	Data<VectorType>& data,
	const std::string& fileName,
	const std::vector<std::string>& cscDatasetName)
{
	std::vector<VectorType> readinBuffer;
	detail::loadHDF5Csc(readinBuffer, fileName, cscDatasetName);
	data = createDataFromRange(readinBuffer);
}

/// @brief Import data from HDF5 dataset of compressed sparse column format.
///
/// @param labeledData
///     Container storing the loaded data
/// @param fileName
///     The name of HDF5 file to be read from
/// @param cscDatasetName
///     the CSC dataset names used to construct a matrix
/// @param label
///     the HDF5 dataset name for label
///
/// @tparam VectorType
///     Type of object stored in Shark data container
/// @tparam LabelType
///     Type of label
template<typename VectorType, typename LabelType>
void importHDF5(
	LabeledData<VectorType, LabelType>& labeledData,
	const std::string& fileName,
	const std::vector<std::string>& cscDatasetName,
	const std::string& label)
{
	std::vector<VectorType> readinData;
	std::vector < std::vector<LabelType> > readinLabel;

	detail::loadHDF5Csc(readinData, fileName, cscDatasetName);
	detail::loadIntoMatrix(readinLabel, fileName, label);
	detail::constructLabeledData(labeledData, readinData, readinLabel);
}

} // namespace shark {

#endif // SHARK_DATA_HDF5_H
