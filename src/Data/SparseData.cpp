//===========================================================================
/*!
 * 
 *
 * \brief       implementation of the sparse data (libsvm) import
 * 
 * 
 *
 * \author      O. Krause, T. Glasmachers
 * \date        2013-2016
 *
 *
 * \par Copyright 1995-2016 Shark Development Team
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
#define SHARK_COMPILE_DLL
#include <limits>
#include <boost/spirit/include/qi.hpp>
#include <boost/fusion/adapted/std_pair.hpp>
#include <shark/Data/SparseData.h>

using namespace shark;

namespace {

typedef std::pair<double, std::vector<std::pair<std::size_t, double> > > LibSVMPoint;
inline std::vector<LibSVMPoint> 
importSparseDataReader(std::istream& stream) {
	std::vector<LibSVMPoint> fileContents;
	while(stream) {
		std::string line;
		std::getline(stream, line);
		if (line.empty()) continue;

		using namespace boost::spirit::qi;
		std::string::const_iterator first = line.begin();
		std::string::const_iterator last = line.end();

		LibSVMPoint newPoint;
		bool r = phrase_parse(
			first, last, 
			double_ >> *(uint_ >> ':' >> double_),
			space, newPoint
		);
		if (!r || first != last) {
			throw SHARKEXCEPTION("[importSparseDataReader] failed to parse record: " + line);
		}

		fileContents.push_back(newPoint);
	}
	return fileContents;
}

template<class T>
void copySparsePoints(Data<blas::vector<T> >& dataset, std::vector<LibSVMPoint> const& points, bool hasZero){
	std::size_t delta = (hasZero ? 0 : 1);
	std::size_t i = 0;
	for(auto element: dataset.elements()){
		element.clear();

		auto const& inputs = points[i].second;
		for(std::size_t j = 0; j != inputs.size(); ++j)
			element(inputs[j].first - delta) = inputs[j].second;
		++i;
	}
}

//sparse vectors need some pre-allocation for fast insertion
template<class T>
void copySparsePoints(Data<blas::compressed_vector<T> >& dataset, std::vector<LibSVMPoint> const& points, bool hasZero){
	std::size_t delta = (hasZero ? 0 : 1);
	
	std::size_t start = 0;//first element of the batch in points
	for(auto& batch: dataset.batches()){
		//allocate nonzeros for every batch and every row in the batch
		std::size_t nnz = 0;
		for(std::size_t i = 0; i != batch.size1(); ++i){
			nnz += points[start+i].second.size();
		}
		batch.reserve(nnz);
		
		//copy data into the batch
		for(std::size_t i = 0; i != batch.size1(); ++i){
			//allocate nonzeros for the row
			batch.reserve_row(i,points[start+i].second.size());
			//copy elements
			auto const& inputs = points[start+i].second;
			auto pos = batch.row_end(i);
			for(std::size_t j = 0; j != inputs.size(); ++j,++pos){
				pos = batch.set_element(pos,inputs[j].first - delta,inputs[j].second);
			}
		}
		start += batch.size1();
	}
}
	
template<class T>//We assume T to be vectorial
shark::LabeledData<T, unsigned int> libsvm_importer_classification(
	std::istream& stream,
	unsigned int dimensions,
	std::size_t batchSize
){
	//read contents of stream
	std::vector<LibSVMPoint> contents = importSparseDataReader(stream);
	std::size_t numPoints = contents.size();
	
	//find data dimension by getting the maximum index
	std::size_t maxIndex = 0;
	for(std::size_t i = 0; i != numPoints; ++i){
		auto const& inputs = contents[i].second;
		if(!inputs.empty())
			maxIndex = std::max(maxIndex, inputs.back().first);
	}
	maxIndex = std::max<std::size_t>(maxIndex,dimensions);
	if(dimensions > 0 && maxIndex > dimensions){
		throw SHARKEXCEPTION("number of dimensions supplied is smaller than actual index data");
	}

	//check labels for conformity
	bool binaryLabels = false;
	int minPositiveLabel = std::numeric_limits<int>::max();
	{
		int maxPositiveLabel = -1;
		for(std::size_t i = 0; i != numPoints; ++i){
			int label = static_cast<int>(contents[i].first);
			if (label != contents[i].first)
				throw SHARKEXCEPTION("non-integer labels are only allows for regression");
			if(label < -1)
				throw SHARKEXCEPTION("negative labels are only allowed for classes -1/1");
			else if(label == -1)
				binaryLabels = true;
			else if(label < minPositiveLabel)
				minPositiveLabel = label;
			else if(label > maxPositiveLabel)
				maxPositiveLabel = label;
		}
		if(binaryLabels && (minPositiveLabel == 0||  maxPositiveLabel > 1))
			throw SHARKEXCEPTION("negative labels are only allowed for classes -1/1");
	}

	// check for feature index zero (non-standard, but it happens)
	bool hasZero = false;
	for (std::size_t i=0; i<numPoints; i++)
	{
		std::vector<std::pair<std::size_t, double> > const& input = contents[i].second;
		if (input.empty()) continue;
		if (input[0].first == 0)
		{
			hasZero = true;
			break;
		}
	}

	//copy contents into a new dataset
	typename shark::LabeledData<T, unsigned int>::element_type blueprint(T(maxIndex + (hasZero ? 1 : 0)),0);
	shark::LabeledData<T, unsigned int> data(numPoints,blueprint, batchSize);//create dataset with the right structure
	copySparsePoints(data.inputs(),contents, hasZero);
	{
		std::size_t i = 0;
		for(auto element: data.elements()){
			//todo: check label
			//we subtract minPositiveLabel to ensure that class indices starting from 0 and 1 are supported
			int label = static_cast<int>(contents[i].first);
			element.label = binaryLabels? 1 + (label-1)/2 : label-minPositiveLabel;
			++i;
		}
	}
	return data;
}

template<class T>//We assume T to be vectorial
shark::LabeledData<T, RealVector> libsvm_importer_regression(
	std::istream& stream,
	unsigned int dimensions,
	std::size_t batchSize
){
	//read contents of stream
	std::vector<LibSVMPoint> contents = importSparseDataReader(stream);
	std::size_t numPoints = contents.size();
	
	//find data dimension by getting the maximum index
	std::size_t maxIndex = 0;
	for(std::size_t i = 0; i != numPoints; ++i){
		auto const& inputs = contents[i].second;
		if(!inputs.empty())
			maxIndex = std::max(maxIndex, inputs.back().first);
	}
	maxIndex = std::max<std::size_t>(maxIndex,dimensions);
	if (dimensions > 0 && maxIndex > dimensions) {
		throw SHARKEXCEPTION("number of dimensions supplied is smaller than actual index data");
	}

	// check for feature index zero (non-standard, but it happens)
	bool hasZero = false;
	for (std::size_t i=0; i<numPoints; i++)
	{
		auto const& input = contents[i].second;
		if (input.empty()) continue;
		if (input[0].first == 0)
		{
			hasZero = true;
			break;
		}
	}

	//copy contents into a new dataset
	typename shark::LabeledData<T, RealVector>::element_type blueprint(T(maxIndex + (hasZero ? 1 : 0)), RealVector(1));
	shark::LabeledData<T, RealVector> data(numPoints, blueprint, batchSize);//create dataset with the right structure
	copySparsePoints(data.inputs(),contents, hasZero);
	{
		size_t delta = (hasZero ? 0 : 1);
		std::size_t i = 0;
		for(auto element: data.elements()) {
			element.label = RealVector(1, contents[i].first);
			++i;
		}
	}
	return data;
}

}

void shark::importSparseData(
	LabeledData<RealVector, unsigned int>& dataset,
	std::istream& stream,
	unsigned int highestIndex,
	std::size_t batchSize
){
	dataset =  libsvm_importer_classification<RealVector>(stream, highestIndex, batchSize);
}

void shark::importSparseData(
	LabeledData<RealVector, RealVector>& dataset,
	std::istream& stream,
	unsigned int highestIndex,
	std::size_t batchSize
){
	dataset =  libsvm_importer_regression<RealVector>(stream, highestIndex, batchSize);
}

void shark::importSparseData(
	LabeledData<CompressedRealVector, unsigned int>& dataset,
	std::istream& stream,
	unsigned int highestIndex,
	std::size_t batchSize
){
	dataset =  libsvm_importer_classification<CompressedRealVector>(stream, highestIndex, batchSize);
}

void shark::importSparseData(
	LabeledData<CompressedRealVector, RealVector>& dataset,
	std::istream& stream,
	unsigned int highestIndex,
	std::size_t batchSize
){
	dataset =  libsvm_importer_regression<CompressedRealVector>(stream, highestIndex, batchSize);
}

void shark::importSparseData(
	LabeledData<RealVector, unsigned int>& dataset,
	std::string fn,
	unsigned int highestIndex,
	std::size_t batchSize
){
	std::ifstream ifs(fn.c_str());
	if (! ifs.good()) throw SHARKEXCEPTION("[shark::importSparseData] failed to open file for input");
	dataset =  libsvm_importer_classification<RealVector>(ifs, highestIndex, batchSize);
}

void shark::importSparseData(
	LabeledData<RealVector, RealVector>& dataset,
	std::string fn,
	unsigned int highestIndex,
	std::size_t batchSize
){
	std::ifstream ifs(fn.c_str());
	if (! ifs.good()) throw SHARKEXCEPTION("[shark::importSparseData] failed to open file for input");
	dataset =  libsvm_importer_regression<RealVector>(ifs, highestIndex, batchSize);
}

void shark::importSparseData(
	LabeledData<CompressedRealVector, unsigned int>& dataset,
	std::string fn,
	unsigned int highestIndex,
	std::size_t batchSize
){
	std::ifstream ifs(fn.c_str());
	if (! ifs.good()) throw SHARKEXCEPTION("[shark::importSparseData] failed to open file for input");
	dataset =  libsvm_importer_classification<CompressedRealVector>(ifs, highestIndex, batchSize);
}

void shark::importSparseData(
	LabeledData<CompressedRealVector, RealVector>& dataset,
	std::string fn,
	unsigned int highestIndex,
	std::size_t batchSize
){
	std::ifstream ifs(fn.c_str());
	if (! ifs.good()) throw SHARKEXCEPTION("[shark::importSparseData] failed to open file for input");
	dataset =  libsvm_importer_regression<CompressedRealVector>(ifs, highestIndex, batchSize);
}
