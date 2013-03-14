//===========================================================================
/*!
 *
 *  \brief Support for importing and exporting data from and to LIBSVM formatted data files
 *
 *
 *  \par
 *  The most important application of the methods provided in this
 *  file is the import of data from LIBSVM files to Shark Data containers.
 *
 *
 *  \author  M. Tuma, T. Glasmachers, C. Igel
 *  \date    2010
 *
 *  \par Copyright (c) 2010:
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

#ifndef SHARK_DATA_LIBSVM_H
#define SHARK_DATA_LIBSVM_H
#include <fstream>
#include <limits>
#include <boost/spirit/include/qi.hpp>
#include <boost/fusion/adapted/std_pair.hpp>

#include <shark/Data/Dataset.h>

namespace shark {

namespace detail {

typedef std::pair<int, std::vector<std::pair<std::size_t, double> > > LibSVMPoint;
inline std::vector<LibSVMPoint> 
import_libsvm_reader(
	std::istream& stream
) {
	;
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
LabeledData<T, unsigned int> import_libsvm(
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
	else if (maxIndex > dimensions)//LibSVM is one-indexed
		throw SHARKEXCEPTION("number of dimensions supplied is smaller than actual index data");
	
	//check labels for conformity
	bool binaryLabels = false;
	{
		int minPositiveLabel = std::numeric_limits<int>::max();
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
	
	//copy contents into a new dataset
	typename LabeledData<T, unsigned int>::element_type blueprint(T(maxIndex),0);
	LabeledData<T, unsigned int> data(numPoints,blueprint);//create dataset with the right structure
	{
		std::size_t i = 0;
		typedef typename LabeledData<T, unsigned int>::element_reference ElemRef;
		BOOST_FOREACH(ElemRef element, data.elements()){
			shark::zero(element.input);
			//todo: check label
			element.label = binaryLabels? 1 + (contents[i].first-1)/2 : contents[i].first-1;
			
			std::vector<std::pair<std::size_t, double> > const& inputs = contents[i].second;
			for(std::size_t j = 0; j != inputs.size(); ++j)
				element.input(inputs[j].first-1) = inputs[j].second;//LibSVM is one-indexed
			++i;
		}
	}
	return data;
}
typedef std::pair< unsigned int, size_t > LabelSortPair;
static bool cmpLabelSortPair(const  LabelSortPair& left, const LabelSortPair& right) {
	return left.first > right.first; // for sorting in decreasing order
}

} // namespace detail

/**
 * \ingroup shark_globals
 *
 * @{
 */

/// \brief Import data from a LIBSVM file.
///
/// \param  dataset       container storing the loaded data
/// \param  fn            the file to be read from
/// \param  highestIndex  highest feature index, or 0 for auto-detection
/// \param  allowMissingClasses set this flag to false if you accept datasets having classes without samples
/// \param  labelmap      explicit mapping from LIBSVM to Shark labels
/// \param  verbose       prints sparseness ratio for sparse data
template<typename InputType>
void import_libsvm(
	LabeledData<InputType, unsigned int>& dataset,
	std::string fn,
	int highestIndex = 0
){
	std::ifstream ifs(fn.c_str());
	dataset =  detail::import_libsvm<InputType>(ifs, highestIndex);
}

template<typename InputType>
void import_libsvm(
	LabeledData<InputType, unsigned int>& dataset,
	std::istream& stream,
	int highestIndex = 0
){
	dataset =  detail::import_libsvm<InputType>(stream, highestIndex);
}


/// \brief Export data to LIBSVM format.
///
/// \param  dataset     Container storing the  data
/// \param  fn          Output file
/// \param  dense       Flag for using dense output format
/// \param  oneMinusOne Flag for applying the transformation y<-2y-1 to binary labels
/// \param  sortLabels  Flag for sorting data points according to labels
template<typename InputType>
void export_libsvm(LabeledData<InputType, unsigned int>& dataset, const std::string &fn, bool dense=false, bool oneMinusOne = true, bool sortLabels = false) {
	std::size_t elements = dataset.numberOfElements();
	std::ofstream ofs(fn.c_str());
	if( !ofs ) {
		throw( SHARKEXCEPTION( "[export_libsvm] file can not be opened for reading" ) );
	}

	size_t dim = inputDimension(dataset);
	if(numberOfClasses(dataset)!=2) oneMinusOne = false;

	std::vector<detail::LabelSortPair> L;
	if(sortLabels) {
		for(std::size_t i = 0; i < elements; i++) 
			L.push_back(detail::LabelSortPair(dataset.element(i).label, i));
		std::sort (L.begin(), L.end(), detail::cmpLabelSortPair);
	}

	for(std::size_t ii = 0; ii < elements; ii++) {
		// apply mapping to sorted indices
		std::size_t i = 0;
		if(sortLabels) i = L[ii].second;
		else i = ii;
		// apply transformation to label and write it to file
		if(oneMinusOne) ofs << 2*int(dataset.element(i).label)-1 << " ";
		//libsvm file format documentation is scarce, but by convention the first class seems to be 1..
		else ofs << dataset.element(i).label+1 << " ";
		// write input data to file
		for(std::size_t j=0; j<dim; j++) {
			if(dense) 
				ofs << " " << j+1 << ":" <<dataset.element(i).input(j);
			else if(dataset.element(i).input(j) != 0) 
				ofs << " " << j+1 << ":" << dataset.element(i).input(j);
		}
		ofs << std::endl;
	}
}

/** @}*/

}
#endif
