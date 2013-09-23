//===========================================================================
/*!

 *  \brief implementation of the libsvm data import
 *
 *  \author O.Krause
 *  \date 2013
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
#include <limits>
#include <boost/spirit/include/qi.hpp>
#include <boost/fusion/adapted/std_pair.hpp>
#include <shark/Data/Libsvm.h>

namespace {

typedef std::pair<int, std::vector<std::pair<std::size_t, double> > > LibSVMPoint;
inline std::vector<LibSVMPoint> 
import_libsvm_reader(
	std::istream& stream
) {
	stream.unsetf(std::ios::skipws); // No white space skipping!
	std::istream_iterator<char> streamBegin(stream);
	std::string storage(// We will read the contents of the file here
		streamBegin,
		std::istream_iterator<char>()
	);
	
	using namespace boost::spirit::qi;
	std::string::const_iterator first = storage.begin();
	std::string::const_iterator last = storage.end();
	std::vector<LibSVMPoint>  fileContents;
	bool r = phrase_parse(
		first, last, 
		*(
			int_   >> -(lit('.')>>+lit('0'))//we also want to be able to parse 1.00000 as label 1
			>> *(uint_ >> ':' >> double_) >> eol
		),
		space-eol , fileContents
	);
	if(!r || first != last)
		throw SHARKEXCEPTION("[import_libsvm_reader] problems parsing file");
	return fileContents;
}

template<class T>//We assume T to be vectorial
shark::LabeledData<T, unsigned int> libsvm_importer(
	std::istream& stream,
	unsigned int dimensions
){
	//read contents of stream
	std::vector<LibSVMPoint> contents = import_libsvm_reader(stream);
	std::size_t numPoints = contents.size();
	
	//find data dimension by getting the maximum index
	std::size_t maxIndex = 0;
	for(std::size_t i = 0; i != numPoints; ++i){
		std::vector<std::pair<std::size_t, double> > const& inputs = contents[i].second;
		if(!inputs.empty())
			maxIndex = std::max(maxIndex, inputs.back().first);
	}
	if(dimensions == 0){
		dimensions = maxIndex;
	}
	else if (maxIndex > dimensions)//LibSVM is one-indexed by default
		throw SHARKEXCEPTION("number of dimensions supplied is smaller than actual index data");

	//check labels for conformity
	bool binaryLabels = false;
	int minPositiveLabel = std::numeric_limits<int>::max();
	{
		int maxPositiveLabel = -1;
		for(std::size_t i = 0; i != numPoints; ++i){
			int label = contents[i].first;
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
	bool haszero = false;
	for (std::size_t i=0; i<numPoints; i++)
	{
		std::vector<std::pair<std::size_t, double> > const& input = contents[i].second;
		if (input.empty()) continue;
		if (input[0].first == 0)
		{
			haszero = true;
			break;
		}
	}

	//copy contents into a new dataset
	typename shark::LabeledData<T, unsigned int>::element_type blueprint(T(maxIndex + (haszero ? 1 : 0)),0);
	shark::LabeledData<T, unsigned int> data(numPoints,blueprint);//create dataset with the right structure
	{
		size_t delta = (haszero ? 0 : 1);
		std::size_t i = 0;
		typedef typename shark::LabeledData<T, unsigned int>::element_reference ElemRef;
		BOOST_FOREACH(ElemRef element, data.elements()){
			zero(element.input);
			//todo: check label
			//we subtract minPositiveLabel to ensore that class indices starting from 0 and 1 are supported
			element.label = binaryLabels? 1 + (contents[i].first-1)/2 : contents[i].first-minPositiveLabel;

			std::vector<std::pair<std::size_t, double> > const& inputs = contents[i].second;
			for(std::size_t j = 0; j != inputs.size(); ++j)
				element.input(inputs[j].first - delta) = inputs[j].second;//LibSVM is one-indexed
			++i;
		}
	}
	return data;
}

}

void shark::import_libsvm(
	LabeledData<RealVector, unsigned int>& dataset,
	std::istream& stream,
	int highestIndex
){
	dataset =  libsvm_importer<RealVector>(stream, highestIndex);
}

void shark::import_libsvm(
	LabeledData<CompressedRealVector, unsigned int>& dataset,
	std::istream& stream,
	int highestIndex
){
	dataset =  libsvm_importer<CompressedRealVector>(stream, highestIndex);
}


void shark::import_libsvm(
	LabeledData<RealVector, unsigned int>& dataset,
	std::string fn,
	int highestIndex
){
	std::ifstream ifs(fn.c_str());
	if (! ifs.good()) throw SHARKEXCEPTION("[shark::import_libsvm] failed to open file for input");
	dataset =  libsvm_importer<RealVector>(ifs, highestIndex);
}

void shark::import_libsvm(
	LabeledData<CompressedRealVector, unsigned int>& dataset,
	std::string fn,
	int highestIndex
){
	std::ifstream ifs(fn.c_str());
	if (! ifs.good()) throw SHARKEXCEPTION("[shark::import_libsvm] failed to open file for input");
	dataset =  libsvm_importer<CompressedRealVector>(ifs, highestIndex);
}
