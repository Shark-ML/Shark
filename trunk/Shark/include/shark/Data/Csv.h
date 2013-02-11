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
#include <limits>
#include <map>
#include <string>
#include <vector>

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


// import function for unlabeled data

template<typename T, typename Stream>
void import_csv(T &data,   // Container that holds the samples
        Stream &pre_in,  // The file to be read from
        const std::string &separator,  // The separator between elements
        const std::string &comment  // Character indicating comment
               ) {
	typedef typename T::value_type SampleType;

	if (!pre_in) {
		throw(std::invalid_argument("[import_csv (1)] Stream cannot be opened for reading."));
	}

	// fix line ending encodings and final newline. see http://svn.boost.org/svn/boost/trunk/libs/iostreams/test/newline_test.cpp
	boost::iostreams::filtering_istream fin;
#if defined( _WIN32 )
	fin.push(boost::iostreams::newline_checker(boost::iostreams::newline::dos | boost::iostreams::newline::final_newline));
	fin.push(boost::iostreams::newline_filter(boost::iostreams::newline::dos));
	fin.push(pre_in);
#elif defined( __unix__ ) || defined ( __unix ) || defined ( __linux__ ) || defined ( __posix__ ) || defined ( __APPLE__ )
	fin.push(boost::iostreams::newline_checker(boost::iostreams::newline::posix | boost::iostreams::newline::final_newline));
	fin.push(boost::iostreams::newline_filter(boost::iostreams::newline::posix));
	fin.push(pre_in);
#elif defined( macintosh ) || defined( Macintosh )
	fin.push(boost::iostreams::newline_checker(boost::iostreams::newline::mac | boost::iostreams::newline::final_newline));
	fin.push(boost::iostreams::newline_filter(boost::iostreams::newline::mac));
	fin.push(pre_in);
#endif

	// pipe data through line-ending-repair-stream into regular stream
	std::stringstream in(std::stringstream::in | std::stringstream::out);
	try {
		boost::iostreams::copy(fin, in);
	} catch (...) {
		throw SHARKEXCEPTION("[import_csv (1)] Failed to process input file. Most likely, your file is not terminated by a newline.");
	}

	// check for final newline
	boost::iostreams::newline_checker *checker = 0;
	checker = BOOST_IOSTREAMS_COMPONENT(fin, 0, boost::iostreams::newline_checker);
	SHARK_ASSERT(checker->has_final_newline());

	// helper vars
	std::size_t lineCounter = 0;
	std::string line;
	std::vector<std::string> tokens;

	// actual read-in
	while (std::getline(in, line)) {

		lineCounter++;

		if (line.empty()) continue;
		if (!comment.empty()) {
			if (line.c_str()[0] == comment.c_str()[0]) {
				continue;
			}
		}

		boost::algorithm::split(tokens, line, boost::is_any_of(separator));
		for (std::vector<std::string>::iterator it=tokens.begin(); it != tokens.end(); ++it) boost::algorithm::trim(*it);
		std::vector< std::string >::iterator it = tokens.begin();
		// eliminate empty tokens resulting from multiple separators (i.e., take care of multiple spaces)
		while (it != tokens.end()) {
			if (*it == "")
				it = tokens.erase(it);
			else
				++it;
		}

		// fill tokens into sample (= read current example)
		typename SampleType::value_type cur_value;
		SampleType sample(tokens.size(), 0);   //for dense, the 0 sets the default value. for sparse, the number of non-zeros.
		for (std::size_t i=0; i<tokens.size(); i++) {
			try {
				cur_value = boost::lexical_cast< typename SampleType::value_type >(tokens[i]);
			} catch (boost::bad_lexical_cast const &blc) {
				throw(std::runtime_error((boost::format("[import_csv (1)] Problem casting data in line %d: %s") % lineCounter % blc.what()).str()));
			}
			if (cur_value != 0) {
				sample(i) = cur_value;
			}
		}
		data.push_back(sample);
	}
}


// import function for unlabeled data - spezialization when each sample is one uint

template<typename Stream>
void import_csv(std::vector<unsigned int> &data,  // Container that holds the samples
        Stream &pre_in,  // The file to be read from
        const std::string &separator,  // The separator between elements
        const std::string &comment  // Character indicating comment
               ) {
	if (!pre_in) {
		throw(std::invalid_argument("[import_csv (2)] Stream cannot be opened for reading."));
	}

	// fix line ending encodings and final newline. see http://svn.boost.org/svn/boost/trunk/libs/iostreams/test/newline_test.cpp
	boost::iostreams::filtering_istream fin;
#if defined( _WIN32 )
	fin.push(boost::iostreams::newline_checker(boost::iostreams::newline::dos | boost::iostreams::newline::final_newline));
	fin.push(boost::iostreams::newline_filter(boost::iostreams::newline::dos));
	fin.push(pre_in);
#elif defined( __unix__ ) || defined ( __unix ) || defined ( __linux__ ) || defined ( __posix__ ) || defined ( __APPLE__ )
	fin.push(boost::iostreams::newline_checker(boost::iostreams::newline::posix | boost::iostreams::newline::final_newline));
	fin.push(boost::iostreams::newline_filter(boost::iostreams::newline::posix));
	fin.push(pre_in);
#elif defined( macintosh ) || defined( Macintosh )
	fin.push(boost::iostreams::newline_checker(boost::iostreams::newline::mac | boost::iostreams::newline::final_newline));
	fin.push(boost::iostreams::newline_filter(boost::iostreams::newline::mac));
	fin.push(pre_in);
#endif

	// pipe data through line-ending-repair-stream into regular stream
	std::stringstream in(std::stringstream::in | std::stringstream::out);
	try {
		boost::iostreams::copy(fin, in);
	} catch (...) {
		throw SHARKEXCEPTION("[import_csv (2)] Failed to process input file. Most likely, your file is not terminated by a newline.");
	}

	// check for final newline
	boost::iostreams::newline_checker *checker = 0;
	checker = BOOST_IOSTREAMS_COMPONENT(fin, 0, boost::iostreams::newline_checker);
	SHARK_ASSERT(checker->has_final_newline());

	// helper vars
	std::size_t lineCounter = 0;
	std::string line;
	std::vector<std::string> tokens;

	// actual read-in
	while (std::getline(in, line)) {

		lineCounter++;

		if (line.empty()) continue;
		if (!comment.empty()) {
			if (line.c_str()[0] == comment.c_str()[0]) {
				continue;
			}
		}

		boost::algorithm::split(tokens, line, boost::is_any_of(separator));
		for (std::vector<std::string>::iterator it=tokens.begin(); it != tokens.end(); ++it) boost::algorithm::trim(*it);
		std::vector< std::string >::iterator it = tokens.begin();
		// eliminate empty tokens resulting from multiple separators (i.e., take care of multiple spaces)
		while (it != tokens.end()) {
			if (*it == "")
				it = tokens.erase(it);
			else
				++it;
		}
		unsigned int sample;
		it = tokens.begin();
		std::vector< std::string >::iterator itE = tokens.end();
		for (; it != itE; ++it) {
			try {
				sample = boost::lexical_cast< unsigned int >(*it);
			} catch (boost::bad_lexical_cast const &blc) {
				throw(std::runtime_error((boost::format("[import_csv (2)] Problem casting data in line %d: %s") % lineCounter % blc.what()).str()));
			}
			data.push_back(sample);
		}

	}
}


// import function for unlabeled data - spezialization when each sample is one double

template<typename Stream>
void import_csv(std::vector<double> &data,  // Container that holds the samples
        Stream &pre_in,  // The file to be read from
        const std::string &separator,  // The separator between elements
        const std::string &comment  // Character indicating comment
               ) {
	if (!pre_in) {
		throw(std::invalid_argument("[import_csv (3)] Stream cannot be opened for reading."));
	}

	// fix line ending encodings and final newline. see http://svn.boost.org/svn/boost/trunk/libs/iostreams/test/newline_test.cpp
	boost::iostreams::filtering_istream fin;
#if defined( _WIN32 )
	fin.push(boost::iostreams::newline_checker(boost::iostreams::newline::dos | boost::iostreams::newline::final_newline));
	fin.push(boost::iostreams::newline_filter(boost::iostreams::newline::dos));
	fin.push(pre_in);
#elif defined( __unix__ ) || defined ( __unix ) || defined ( __linux__ ) || defined ( __posix__ ) || defined ( __APPLE__ )
	fin.push(boost::iostreams::newline_checker(boost::iostreams::newline::posix | boost::iostreams::newline::final_newline));
	fin.push(boost::iostreams::newline_filter(boost::iostreams::newline::posix));
	fin.push(pre_in);
#elif defined( macintosh ) || defined( Macintosh )
	fin.push(boost::iostreams::newline_checker(boost::iostreams::newline::mac | boost::iostreams::newline::final_newline));
	fin.push(boost::iostreams::newline_filter(boost::iostreams::newline::mac));
	fin.push(pre_in);
#endif

	// pipe data through line-ending-repair-stream into regular stream
	std::stringstream in(std::stringstream::in | std::stringstream::out);
	try {
		boost::iostreams::copy(fin, in);
	} catch (...) {
		throw SHARKEXCEPTION("[import_csv (3)] Failed to process input file. Most likely, your file is not terminated by a newline.");
	}

	// check for final newline
	boost::iostreams::newline_checker *checker = 0;
	checker = BOOST_IOSTREAMS_COMPONENT(fin, 0, boost::iostreams::newline_checker);
	SHARK_ASSERT(checker->has_final_newline());

	// helper vars
	std::size_t lineCounter = 0;
	std::string line;
	std::vector<std::string> tokens;

	// actual read-in
	while (std::getline(in, line)) {

		lineCounter++;

		if (line.empty()) continue;
		if (!comment.empty()) {
			if (line.c_str()[0] == comment.c_str()[0]) {
				continue;
			}
		}

		boost::algorithm::split(tokens, line, boost::is_any_of(separator));
		for (std::vector<std::string>::iterator it=tokens.begin(); it != tokens.end(); ++it) boost::algorithm::trim(*it);
		std::vector< std::string >::iterator it = tokens.begin();
		// eliminate empty tokens resulting from multiple separators (i.e., take care of multiple spaces)
		while (it != tokens.end()) {
			if (*it == "")
				it = tokens.erase(it);
			else
				++it;
		}
		double sample;
		it = tokens.begin();
		std::vector< std::string >::iterator itE = tokens.end();
		for (; it != itE; ++it) {
			try {
				sample = boost::lexical_cast< double >(*it);
			} catch (boost::bad_lexical_cast const &blc) {
				throw(std::runtime_error((boost::format("[import_csv (3)] Problem casting data in line %d: %s") % lineCounter % blc.what()).str()));
			}
			data.push_back(sample);
		}

	}
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

// import function for labeled data

template<typename T, typename U, typename Stream>
void import_csv(T &input,   // Container that holds the samples
        U &labels,  // Container that holds the labels
        Stream &pre_in,  // The file to be read from
        LabelPosition lp,  // The position of the label
        const std::string &separator,  // The separator between elements
        const std::string &comment,  // Character for indicating a comment
        bool allowMissingFeatures = false, // if true, treat missing features as NaN
        bool allowMissingClasses = false, // if true, skip test if all classes occur
        std::map<typename U::value_type, typename U::value_type> const *labelmap = NULL // explicit mapping from input data labels to Shark labels
               ) {
	typedef typename T::value_type SampleType;
	typedef typename U::value_type LabelType;


	if (!pre_in) {
		throw(std::invalid_argument("[import_csv (4)] Stream cannot be opened for reading."));
	}

	// fix line ending encodings and final newline. see http://svn.boost.org/svn/boost/trunk/libs/iostreams/test/newline_test.cpp
	boost::iostreams::filtering_istream fin;
#if defined( _WIN32 )
	fin.push(boost::iostreams::newline_checker(boost::iostreams::newline::dos | boost::iostreams::newline::final_newline));
	fin.push(boost::iostreams::newline_filter(boost::iostreams::newline::dos));
	fin.push(pre_in);
#elif defined( __unix__ ) || defined ( __unix ) || defined ( __linux__ ) || defined ( __posix__ ) || defined ( __APPLE__ )
	fin.push(boost::iostreams::newline_checker(boost::iostreams::newline::posix | boost::iostreams::newline::final_newline));
	fin.push(boost::iostreams::newline_filter(boost::iostreams::newline::posix));
	fin.push(pre_in);
#elif defined( macintosh ) || defined( Macintosh )
	fin.push(boost::iostreams::newline_checker(boost::iostreams::newline::mac | boost::iostreams::newline::final_newline));
	fin.push(boost::iostreams::newline_filter(boost::iostreams::newline::mac));
	fin.push(pre_in);
#endif

	// pipe data through line-ending-repair-stream into regular stream
	std::stringstream in(std::stringstream::in | std::stringstream::out);
	try {
		boost::iostreams::copy(fin, in);
	} catch (...) {
		throw SHARKEXCEPTION("[import_csv (4)] Failed to process input file. Most likely, your file is not terminated by a newline.");
	}

	// check for final newline
	boost::iostreams::newline_checker *checker = 0;
	checker = BOOST_IOSTREAMS_COMPONENT(fin, 0, boost::iostreams::newline_checker);
	SHARK_ASSERT(checker->has_final_newline());

	// helper vars
	std::size_t lineCounter = 0;
	std::string line;
	std::vector<std::string> tokens;
	LabelType cur_label;

	// actual read-in
	while (std::getline(in, line)) {

		lineCounter++;

		// skip empty and comment lines
		if (line.empty()) continue;
		if (!comment.empty()) {
			if (line.c_str()[0] == comment.c_str()[0]) {
				continue;
			}
		}

		// split line into tokens
		boost::algorithm::split(tokens, line, boost::is_any_of(separator));
		for (std::vector<std::string>::iterator it=tokens.begin(); it != tokens.end(); ++it) boost::algorithm::trim(*it);
		std::vector< std::string >::iterator it = tokens.begin();

		// eliminate empty tokens resulting from multiple separators (i.e., take care of multiple spaces)
		while (it != tokens.end()) {
			if (*it == "")
				it = tokens.erase(it);
			else
				++it;
		}

		//OK: this happens on VC
		//if the sequence happens to be empty, just go on
		if (tokens.empty())
			continue;

		// read label
		try {
			cur_label = boost::lexical_cast< LabelType >(lp == FIRST_COLUMN ? tokens.front() : tokens.back());
		} catch (boost::bad_lexical_cast const &blc) {
			throw(std::runtime_error((boost::format("[import_csv (4)] Problem re-casting label in line %d: %s") % lineCounter % blc.what()).str()));
		}
		if (cur_label == std::numeric_limits<LabelType>::max())
			throw SHARKEXCEPTION("[import_csv (4)] Suspecting negative labels in file and unsigned label type in Shark."
			        "Please use the labelmap argument to automatically convert the labels at read-in.");

		// store label
		if (labelmap != NULL) {
			typename std::map<LabelType, LabelType>::const_iterator it = labelmap->find(cur_label);
			if (it == labelmap->end()) throw SHARKEXCEPTION((boost::format("[import_csv (4)] label %d not found in explicitly given map") %cur_label).str());
			labels.push_back(it->second);
		} else {
			labels.push_back(cur_label);
		}

		const std::string absentFeature("?");

		// fill tokens into sample (= read current example's features)
		typename SampleType::value_type cur_value;
		SampleType sample(tokens.size() - 1, 0);   //for dense, the 0 sets the default value. for sparse, the number of non-zeros.
		bool fc = (lp == FIRST_COLUMN);
		std::size_t start = (std::size_t)(fc);
		std::size_t end = tokens.size() - (std::size_t)(!fc);

		for (std::size_t i=start; i<end; i++) {
			try {
				if (allowMissingFeatures && absentFeature == tokens[i])
					cur_value = std::numeric_limits<double>::quiet_NaN();
				else
					cur_value = boost::lexical_cast< typename SampleType::value_type >(tokens[i]);
			} catch (boost::bad_lexical_cast const &blc) {
				throw(std::runtime_error((boost::format("[import_csv (4)] Problem re-casting data in line %d: %s") % lineCounter % blc.what()).str()));
			}
			if (cur_value != 0) {
				sample(i-fc) = cur_value;
			}
		}
		input.push_back(sample);

	}


	// SCAN LABELS: if a classification dataset and contiguous labels were requested,
	//              look through the newly-imported label vector once to ensure sanity.
	//              In other words, this check is triggered iff the user declared an
	//              integral label type (e.g., via ClassificationDataset) and did not
	//              bypass this check via passing allowMissingClasses as true in the ctor
	if (boost::is_integral<LabelType>::value) {
		if (!allowMissingClasses) {

			LabelType min_label = std::numeric_limits<LabelType>::max();
			LabelType max_label = std::numeric_limits<LabelType>::min();

			// get max and min label
			for (typename U::iterator it=labels.begin(); it != labels.end(); ++it) {
				LabelType label = *it;
				if (label > max_label) max_label = label;
				if (label < min_label) min_label = label;
			}
			if (min_label != 0) throw SHARKEXCEPTION("[import_csv (4)] Label error: first label must occur in dataset.");

			// count labels: create histogram array
			std::size_t *label_histogram;
			try {
				label_histogram = new std::size_t [static_cast<std::size_t>(max_label)+1];
			} catch (std::bad_alloc &ba) {
				throw(std::runtime_error((boost::format("[import_csv (4)] Error assigning memory. Probably the labels are too large. Exception: %s")
				        % ba.what()).str()));
			}
			// init histogram to zero
			for (std::size_t i=0; i<max_label; i++) {
				label_histogram[i] = 0;
			}
			// count
			for (typename U::iterator it=labels.begin(); it != labels.end(); ++it) {
				LabelType label = *it;
				label_histogram[(std::size_t) label ] ++;
			}
			// check that every label occured
			for (std::size_t i=0; i<max_label; i++) {
				if (label_histogram[i] == 0) {
					delete [] label_histogram; //free memory
					throw SHARKEXCEPTION("[import_csv (4)] Label error: every label in range must occur at least once.");
				}
			}
			delete [] label_histogram; //free memory

		} //if (!allowMissingClasses)

	} //if (is_integral)

}


// import function for regression data
template<typename T, typename U, typename Stream>
void import_csv_regression(T &input,   // Container that holds the samples
        U &labels,  // Container that holds the labels
        Stream &pre_in,  // The file to be read from
        LabelPosition lp,  // The position of the label
        const std::string &separator,  // The separator between elements
        const std::string &comment,  // Character for indicating a comment
        std::size_t noOutputs, // output/label dimension
        bool allowMissingFeatures = false // if true, treat missing features as NaN
                          ) {
	typedef typename T::value_type SampleType;
	typedef typename U::value_type LabelType;

	if (!pre_in) {
		throw(std::invalid_argument("[import_csv_regression] Stream cannot be opened for reading."));
	}

	// fix line ending encodings and final newline. see http://svn.boost.org/svn/boost/trunk/libs/iostreams/test/newline_test.cpp
	boost::iostreams::filtering_istream fin;
#if defined( _WIN32 )
	fin.push(boost::iostreams::newline_checker(boost::iostreams::newline::dos | boost::iostreams::newline::final_newline));
	fin.push(boost::iostreams::newline_filter(boost::iostreams::newline::dos));
	fin.push(pre_in);
#elif defined( __unix__ ) || defined ( __unix ) || defined ( __linux__ ) || defined ( __posix__ ) || defined ( __APPLE__ )
	fin.push(boost::iostreams::newline_checker(boost::iostreams::newline::posix | boost::iostreams::newline::final_newline));
	fin.push(boost::iostreams::newline_filter(boost::iostreams::newline::posix));
	fin.push(pre_in);
#elif defined( macintosh ) || defined( Macintosh )
	fin.push(boost::iostreams::newline_checker(boost::iostreams::newline::mac | boost::iostreams::newline::final_newline));
	fin.push(boost::iostreams::newline_filter(boost::iostreams::newline::mac));
	fin.push(pre_in);
#endif

	// pipe data through line-ending-repair-stream into regular stream
	std::stringstream in(std::stringstream::in | std::stringstream::out);
	try {
		boost::iostreams::copy(fin, in);
	} catch (...) {
		throw SHARKEXCEPTION("[import_csv_regression] Failed to process input file. Most likely, your file is not terminated by a newline.");
	}

	// check for final newline
	boost::iostreams::newline_checker *checker = 0;
	checker = BOOST_IOSTREAMS_COMPONENT(fin, 0, boost::iostreams::newline_checker);
	SHARK_ASSERT(checker->has_final_newline());

	// helper vars
	std::size_t start, end, lineCounter = 0;
	std::string line;
	std::vector<std::string> tokens;
	LabelType cur_label;

	// actual read-in
	while (std::getline(in, line)) {

		lineCounter++;

		// skip empty and comment lines
		if (line.empty()) continue;
		if (!comment.empty()) {
			if (line.c_str()[0] == comment.c_str()[0]) {
				continue;
			}
		}

		// split line into tokens
		boost::algorithm::split(tokens, line, boost::is_any_of(separator));
		for (std::vector<std::string>::iterator it=tokens.begin(); it != tokens.end(); ++it) boost::algorithm::trim(*it);
		std::vector< std::string >::iterator it = tokens.begin();

		// eliminate empty tokens resulting from multiple separators (i.e., take care of multiple spaces)
		while (it != tokens.end()) {
			if (*it == "")
				it = tokens.erase(it);
			else
				++it;
		}

		const std::string absentFeature("?");

		// fill tokens into label (= read current example's label vector)
		typename LabelType::value_type cur_label;
		LabelType sampleLabel(noOutputs, 0);   //for dense, the 0 sets the default value. for sparse, the number of non-zeros.
		bool fc = (lp == FIRST_COLUMN);

		if (fc) {
			start = (std::size_t)(0);
			end = noOutputs;
		} else {
			start = tokens.size() - noOutputs;
			end = tokens.size();
		}

		for (std::size_t i=start, j=0; i<end; i++) {
			try {
				if (allowMissingFeatures && absentFeature == tokens[i])
					cur_label = std::numeric_limits<double>::quiet_NaN();
				else
					cur_label = boost::lexical_cast< typename SampleType::value_type >(tokens[i]);
			} catch (boost::bad_lexical_cast const &blc) {
				throw(std::runtime_error((boost::format("[import_csv_regression] Problem re-casting outputs in line %d: %s") % lineCounter % blc.what()).str()));
			}
			if (cur_label != 0) {
				sampleLabel(j) = cur_label;
			}
			j++;
		}
		labels.push_back(sampleLabel);

		// fill tokens into sample (= read current example's features)
		typename SampleType::value_type cur_value;
		SampleType sample(tokens.size() - noOutputs, 0);   //for dense, the 0 sets the default value. for sparse, the number of non-zeros.

		if (!fc) {
			start = (std::size_t)(0);
			end = tokens.size() - noOutputs;
		} else {
			start = noOutputs;
			end = tokens.size();
		}

		for (std::size_t i=start, j=0; i<end; i++) {
			try {
				if (allowMissingFeatures && absentFeature == tokens[i])
					cur_value = std::numeric_limits<double>::quiet_NaN();
				else {
					cur_value = boost::lexical_cast< typename SampleType::value_type >(tokens[i]);
				}
			} catch (boost::bad_lexical_cast const &blc) {
				throw(std::runtime_error((boost::format("[import_csv_regression] Problem re-casting data in line %d: %s") % lineCounter % blc.what()).str()));
			}
			if (cur_value != 0) {
				sample(j) = cur_value;
			}
			j++;
		}
		input.push_back(sample);

	}
}

} // namespace detail



// FUNCTION WRAPPERS BELOW


/// \brief Import unlabeled data from a character-separated value file.
///
/// \param  data       Container storing the loaded data
/// \param  fn         The file to be read from
/// \param  separator  Separator between entries, typically a comma or a space
/// \param  comment    Trailing character indicating comment line
template<typename Type>
void import_csv(
    Data<Type> &data,
    std::string fn,
    std::string separator = ",",
    std::string comment = "",
    std::size_t batchSize = Data<Type>::DefaultBatchSize
) {
	std::ifstream ifs(fn.c_str());
	std::vector<Type> tmp;
	detail::import_csv(tmp, ifs, separator, comment);
	data = createDataFromRange(tmp,batchSize);
}


/// \brief Format unlabeled data into a character-separated value file.
///
/// \param  set       Container to be exported
/// \param  fn         The file to be written to
/// \param  separator  Separator between entries, typically a comma or a space
/// \param  sci        should the output be in scientific notation?
/// \param  width      argument to std::setw when writing the output
template<typename Type>
void export_csv(
    Data<Type> const &set,
    std::string fn,
    std::string separator = ",",
    bool sci = true,
    unsigned int width = 0
) {
	std::ofstream ofs(fn.c_str());
	detail::export_csv(set.elements(), ofs, separator, sci, width);
}


/// \brief Import labeled data from a character-separated value file.
///
/// \param  dataset                 Container storing the loaded data
/// \param  fn                      The file to be read from
/// \param  lp                      Position of the label in the record, either first or last column
/// \param  separator               Separator between entries, typically a comma or a space
/// \param  comment                 Character for indicating a comment, by default empty
/// \param  labelmap                explicit mapping from input data labels to Shark labels (optional)
template<typename InputType, typename LabelType>
void import_csv(
    LabeledData<InputType, LabelType> &dataset,
    std::string fn,
    LabelPosition lp,
    std::string separator = ",",
    std::string comment = "",

    bool allowMissingClasses = false, // if true, skip test if all classes occur
    std::map<LabelType, LabelType> const *labelmap = NULL, // explicit mapping from input data labels to Shark labels
    std::size_t batchSize = LabeledData<InputType, LabelType>::DefaultBatchSize
) {
	std::ifstream ifs(fn.c_str());
	std::vector<InputType> x;
	std::vector<LabelType> y;
	detail::import_csv(x, y, ifs, lp, separator, comment, false, allowMissingClasses, labelmap);
	dataset = createLabeledDataFromRange(x, y,batchSize);
}

/// \brief Import regression data from a character-separated value file.
///
/// \param  dataset                 Container storing the loaded data
/// \param  fn                      The file to be read from
/// \param  lp                      Position of the label in the record, either first or last column
/// \param  separator               Separator between entries, typically a comma or a space
/// \param  comment                 Character for indicating a comment, by default empty
/// \param  numberOfOutputs         Dimensionality of label/output
void import_csv(LabeledData<RealVector, RealVector> &dataset,
        std::string fn,
        LabelPosition lp,
        std::string separator = ",",
        std::string comment = "",
        std::size_t numberOfOutputs = 1
) {
	std::ifstream ifs(fn.c_str());
	std::vector<RealVector> x;
	std::vector<RealVector> y;
	detail::import_csv_regression(x, y, ifs, lp, separator, comment, numberOfOutputs, false);
	dataset = createLabeledDataFromRange(x, y);
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

/// \brief Construct Shark data from a string
///
/// \param data Container storing the constructed data
/// \param dataInString the string to construct data from
template<typename InputType>
void string2data(
    Data<InputType> &data,
    const std::string &dataInString,
    std::size_t batchSize = Data<InputType>::DefaultBatchSize
) {
	std::stringstream ss(dataInString);
	std::vector<InputType> tmp;
	detail::import_csv(tmp, ss, ",", "");
	data = createDataFromRange(tmp,batchSize);
}

/// \brief Construct Shark labeled data from a string
///
/// \param dataset Container storing the constructed labeled data
/// \param dataInString the string to construct data from
/// \param labelPosition the position of label. The default value is @a LAST_COLUMN
/// \param labelmap explicit mapping from input data labels to Shark labels (optional)
template<typename InputType, typename LabelType>
void string2data(
    LabeledData<InputType, LabelType> &dataset,
    const std::string &dataInString,
    LabelPosition labelPosition = LAST_COLUMN,
    bool allowMissingFeatures = false, // if true, treat missing features as NaN
    bool allowMissingClasses = false, // if true, skip test if all classes occur
    std::map<LabelType, LabelType> const *labelmap = NULL, // explicit mapping from input data labels to Shark labels
    std::size_t batchSize = LabeledData<InputType, LabelType>::DefaultBatchSize
) {
	std::stringstream ss(dataInString);
	std::vector<InputType> x;
	std::vector<LabelType> y;
	detail::import_csv(x, y, ss, labelPosition, ",", "", allowMissingFeatures, allowMissingClasses, labelmap);
	dataset = createLabeledDataFromRange(x, y,batchSize);
}

/** @}*/

} // namespace shark
#endif // SHARK_ML_CSV_H
