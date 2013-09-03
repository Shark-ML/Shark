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
#include <shark/Data/Dataset.h>

namespace shark {

namespace detail {

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
/// \param  stream        stream to be read from
/// \param  highestIndex  highest feature index, or 0 for auto-detection
void import_libsvm(
	LabeledData<RealVector, unsigned int>& dataset,
	std::istream& stream,
	int highestIndex = 0
);

/// \brief Import data from a LIBSVM file.
///
/// \param  dataset       container storing the loaded data
/// \param  stream        stream to be read from
/// \param  highestIndex  highest feature index, or 0 for auto-detection
void import_libsvm(
	LabeledData<CompressedRealVector, unsigned int>& dataset,
	std::istream& stream,
	int highestIndex = 0
);

/// \brief Import data from a LIBSVM file.
///
/// \param  dataset       container storing the loaded data
/// \param  fn            the file to be read from
/// \param  highestIndex  highest feature index, or 0 for auto-detection
void import_libsvm(
	LabeledData<RealVector, unsigned int>& dataset,
	std::string fn,
	int highestIndex = 0
);

/// \brief Import data from a LIBSVM file.
///
/// \param  dataset       container storing the loaded data
/// \param  fn            the file to be read from
/// \param  highestIndex  highest feature index, or 0 for auto-detection
void import_libsvm(
	LabeledData<CompressedRealVector, unsigned int>& dataset,
	std::string fn,
	int highestIndex = 0
);


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
