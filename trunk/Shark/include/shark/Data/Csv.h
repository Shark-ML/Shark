//===========================================================================
/*!
 *  \brief Support for importing and exporting data from and to character separated value (CSV) files
 *
 *
 *  \par
 *  The most important application of the methods provided in this
 *  file is the import of data from CSV files into Shark data
 *  containers.
 *
 *
 *  \author  T. Voss, M. Tuma
 *  \date    2010
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

#ifndef SHARK_DATA_CSV_H
#define SHARK_DATA_CSV_H

#include <shark/Data/Dataset.h>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/format.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/newline.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/type_traits.hpp>

#include <exception>
#include <fstream>
#include <map>
#include <string>

namespace shark {

/**
 * \ingroup shark_globals
 *
 * @{
 */


/// \brief Position of the label in a CSV file
///
/// \par
/// This type describes the position of the label in a record of a CSV file.
/// The label can be positioned either in the first or the last column, or
/// there can be no label present at all.
enum LabelPosition {
    FIRST_COLUMN,
    LAST_COLUMN,
};

namespace detail {

// export function for unlabeled data
template<typename T, typename Stream>
void export_csv(const T &data,   // Container that holds the samples
        Stream &out,  // The file to be read from
        const std::string &separator,  // The separator between elements
        bool scientific = true, //scientific notation?
        unsigned int fieldwidth = 0
) {
	if (!out) {
		throw(std::invalid_argument("[export_csv (1)] Stream cannot be opened for writing."));
	}

	// set output format
	if (scientific)
		out.setf(std::ios_base::scientific);
	std::streamsize ss = out.precision();
	out.precision(10);

	// write out
	typename T::const_iterator it = data.begin();
	for (; it != data.end(); ++it) {
		SHARK_CHECK(it->begin() != it->end(), "[export_csv (1)] record must not be empty");
		for (std::size_t i=0; i<(*it).size()-1; i++) {
			out << std::setw(fieldwidth) << (*it)(i) << separator;
		}
		out << std::setw(fieldwidth) << (*it)((*it).size()-1) << std::endl;
	}

	// restore output format
	out.precision(ss);
}

// export function for labeled data

template<typename T, typename U, typename Stream>
void export_csv(const T &input,   // Container that holds the samples
        const U &labels,  // Container that holds the labels
        Stream &out,  // The file to be read from
        LabelPosition lp,  // The position of the label
        const std::string &separator,  // The separator between elements
        bool scientific = true, //scientific notation?
        unsigned int fieldwidth = 0, //column-align using this field width
	typename boost::enable_if<
		boost::is_arithmetic<typename boost::range_value<U>::type>
	>::type* dummy = 0//nabl this only for arithmetic types
) {

	if (!out) {
		throw(std::invalid_argument("[export_csv (2)] Stream cannot be opened for writing."));
	}


	if (scientific)
		out.setf(std::ios_base::scientific);
	std::streamsize ss = out.precision();
	out.precision(10);

	typename T::const_iterator iti = input.begin();
	typename U::const_iterator itl = labels.begin();


	for (; iti != input.end(); ++iti, ++itl) {
		SHARK_CHECK(iti->begin() != iti->end(), "[export_csv (2)] record must not be empty");
		if (lp == FIRST_COLUMN)
			out << *itl << separator;
		for (std::size_t i=0; i<(*iti).size()-1; i++) {
			out << std::setw(fieldwidth) << (*iti)(i) << separator;
		}
		if (lp == FIRST_COLUMN) {
			out << std::setw(fieldwidth) << (*iti)((*iti).size()-1) << std::endl;
		} else {
			out << std::setw(fieldwidth) << (*iti)((*iti).size()-1) << separator << *itl << std::endl;
		}
	}
	out.precision(ss);
}

// export function for data with vector labels
template<typename T, typename U, typename Stream>
void export_csv(
	const T &input,  // Container that holds the samples
	const U &labels,  // Container that holds the labels
	Stream &out,  // The file to be read from
	LabelPosition lp,  // The position of the label
	const std::string &separator,  // The separator between elements
	bool scientific = true, //scientific notation?
	unsigned int fieldwidth = 0, //column-align using this field width
	typename boost::disable_if<
		boost::is_arithmetic<typename boost::range_value<U>::type>
	>::type* dummy = 0//enable this only for complex types
) {

	if (!out) {
		throw(std::invalid_argument("[export_csv (2)] Stream cannot be opened for writing."));
	}


	if (scientific)
		out.setf(std::ios_base::scientific);
	std::streamsize ss = out.precision();
	out.precision(10);

	typename T::const_iterator iti = input.begin();
	typename U::const_iterator itl = labels.begin();

	for (; iti != input.end(); ++iti, ++itl) {
		SHARK_CHECK(iti->begin() != iti->end(), "[export_csv (2)] record must not be empty");
		if (lp == FIRST_COLUMN) {
			for (std::size_t j = 0; j < itl->size(); j++) out << std::setw(fieldwidth) << (*itl)(j) << separator;
		}
		for (std::size_t i=0; i<(*iti).size()-1; i++) {
			out << std::setw(fieldwidth) << (*iti)(i) << separator;
		}
		if (lp == FIRST_COLUMN) {
			out << std::setw(fieldwidth) << (*iti)((*iti).size()-1) << std::endl;
		} else {
			out << std::setw(fieldwidth) << (*iti)((*iti).size()-1);
			for (std::size_t j = 0; j < itl->size(); j++) out << std::setw(fieldwidth)  << separator << (*itl)(j);
			out << std::endl;
		}
	}
	out.precision(ss);
}
} // namespace detail



// ACTUAL READ IN ROUTINES BELOW

/// \brief Import unlabeled vectors from a read-in character-separated value file.
///
/// \param  data       Container storing the loaded data
/// \param  contents    The read in csv-file
/// \param  separator  Optional separator between entries, typically a comma, spaces ar automatically ignored
/// \param  comment    Trailing character indicating comment line. By dfault it is '#'
/// \param  maximumBatchSize   Size of batches in the dataset
void csvStringToData(
    Data<RealVector> &data,
    std::string const& contents,
    char separator = ',',
    char comment = '#',
    std::size_t maximumBatchSize = Data<RealVector>::DefaultBatchSize
);

/// \brief Import "csv" from string consisting only of a single unsigned int per row
///
/// \param  data               Container storing the loaded data
/// \param  contents    The read in csv-file
/// \param  separator          Optional separator between entries, typically a comma, spaces ar automatically ignored
/// \param  comment            Trailing characters indicating comment line. By default it is "#"
/// \param  maximumBatchSize   Size of batches in the dataset
void csvStringToData(
    Data<unsigned int> &data,
    std::string const& contents,
    char separator = ',',
    char comment = '#',
    std::size_t maximumBatchSize = Data<unsigned int>::DefaultBatchSize
);

/// \brief Import "csv" from string consisting only of a single  int per row
///
/// \param  data               Container storing the loaded data
/// \param  contents    The read in csv-file
/// \param  separator          Optional separator between entries, typically a comma, spaces ar automatically ignored
/// \param  comment            Trailing characters indicating comment line. By default it is "#"
/// \param  maximumBatchSize   Size of batches in the dataset
void csvStringToData(
    Data<int> &data,
    std::string const& contents,
    char separator = ',',
    char comment = '#',
    std::size_t maximumBatchSize = Data<int>::DefaultBatchSize
);

/// \brief Import "csv" from string consisting only of a single double per row
///
/// \param  data               Container storing the loaded data
/// \param  contents    The read in csv-file
/// \param  separator          Optional separator between entries, typically a comma, spaces ar automatically ignored
/// \param  comment            Trailing characters indicating comment line. By default it is "#"
/// \param  maximumBatchSize   Size of batches in the dataset
void csvStringToData(
    Data<double> &data,
    std::string const& contents,
    char separator = ',',
    char comment = '#',
    std::size_t maximumBatchSize = Data<double>::DefaultBatchSize
);

/// \brief Import labeled data from a character-separated value file.
///
/// \param  dataset    Container storing the loaded data
/// \param  contents the read-in file contents.
/// \param  lp         Position of the label in the record, either first or last column
/// \param  separator  Optional separator between entries, typically a comma, spaces ar automatically ignored
/// \param  comment    Character for indicating a comment, by default '#'
/// \param  maximumBatchSize  maximum size of a batch in the dataset after import
void csvStringToData(
    LabeledData<RealVector, unsigned int> &dataset,
    std::string const& contents,
    LabelPosition lp,
    char separator = ',',
    char comment = '#',
    std::size_t maximumBatchSize = LabeledData<RealVector, unsigned int>::DefaultBatchSize
);


/// \brief Import regression data from a read-in character-separated value file.
///
/// \param  dataset             Container storing the loaded data
/// \param  contents             The read in csv-file
/// \param  lp                  Position of the label in the record, either first or last column
/// \param  separator           Separator between entries, typically a comma or a space
/// \param  comment             Character for indicating a comment, by default empty
/// \param  numberOfOutputs     Dimensionality of label/output
/// \param  maximumBatchSize  maximum size of a batch in the dataset after import
void csvStringToData(
	LabeledData<RealVector, RealVector> &dataset,
	std::string const& contents,
	LabelPosition lp,
	std::size_t numberOfOutputs = 1,
	char separator = ',',
	char comment = '#',
	std::size_t maximumBatchSize = LabeledData<RealVector, RealVector>::DefaultBatchSize
);


/// \brief Import a Dataset from a csv file
///
/// \param  data       Container storing the loaded data
/// \param  fn         The file to be read from
/// \param  separator  Optional separator between entries, typically a comma, spaces ar automatically ignored
/// \param  comment    Trailing character indicating comment line. By dfault it is '#'
/// \param  maximumBatchSize   Size of batches in the dataset
template<class T>
void import_csv(
	Data<T>& data,
	std::string fn,
	char separator = ',',
	char comment = '#',
	std::size_t maximumBatchSize = Data<T>::DefaultBatchSize
){
	std::ifstream stream(fn.c_str());
	stream.unsetf(std::ios::skipws);
	std::istream_iterator<char> streamBegin(stream);
	std::string contents(//read contents of file in string
		streamBegin,
		std::istream_iterator<char>()
	);
	//call the actual parser
	csvStringToData(data,contents,separator,comment,maximumBatchSize);
}

/// \brief Import a labeled Dataset from a csv file
///
/// \param  data       Container storing the loaded data
/// \param  fn         The file to be read from
/// \param  lp                  Position of the label in the record, either first or last column
/// \param  separator  Optional separator between entries, typically a comma, spaces ar automatically ignored
/// \param  comment    Trailing character indicating comment line. By dfault it is '#'
/// \param  maximumBatchSize   Size of batches in the dataset
void import_csv(
	LabeledData<RealVector, unsigned int>& data,
	std::string fn,
	LabelPosition lp,
	char separator = ',',
	char comment = '#',
	std::size_t maximumBatchSize = LabeledData<RealVector, unsigned int>::DefaultBatchSize
);

/// \brief Import a labeled Dataset from a csv file
///
/// \param  data       Container storing the loaded data
/// \param  fn         The file to be read from
/// \param  lp                  Position of the label in the record, either first or last column
/// \param  numberOfOutputs dimensionality of the labels
/// \param  separator  Optional separator between entries, typically a comma, spaces ar automatically ignored
/// \param  comment    Trailing character indicating comment line. By dfault it is '#'
/// \param  maximumBatchSize   Size of batches in the dataset
void import_csv(
	LabeledData<RealVector, RealVector>& data,
	std::string fn,
	LabelPosition lp,
	std::size_t numberOfOutputs = 1,
	char separator = ',',
	char comment = '#',
	std::size_t maximumBatchSize = LabeledData<RealVector, RealVector>::DefaultBatchSize
);

/// \brief Format unlabeled data into a character-separated value file.
///
/// \param  set       Container to be exported
/// \param  fn         The file to be written to
/// \param  separator  Separator between entries, typically a comma or a space
/// \param  sci        should the output be in scientific notation?
/// \param  width      argument to std::setw when writing the output
template<typename Type>
void export_csv(
	Data<Type> const& set,
	std::string fn,
	std::string separator = ",",
	bool sci = true,
	unsigned int width = 0
) {
	std::ofstream ofs(fn.c_str());
	detail::export_csv(set.elements(), ofs, separator, sci, width);
}


/// \brief Format labeled data into a character-separated value file.
///
/// \param  dataset    Container to be exported
/// \param  fn         The file to be written to
/// \param  lp         Position of the label in the record, either first or last column
/// \param  separator  Separator between entries, typically a comma or a space
/// \param  sci        should the output be in scientific notation?
/// \param  width      argument to std::setw when writing the output
template<typename InputType, typename LabelType>
void export_csv(
    LabeledData<InputType, LabelType> const &dataset,
    std::string fn,
    LabelPosition lp,
    std::string separator = ",",
    bool sci = true,
    unsigned int width = 0
) {
	std::ofstream ofs(fn.c_str());
	detail::export_csv(dataset.inputs().elements(), dataset.labels().elements(), ofs, lp, separator, sci, width);
}

/** @}*/

} // namespace shark
#endif // SHARK_ML_CSV_H
