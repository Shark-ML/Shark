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

#include <exception>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <limits>

#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/minmax_element.hpp>
#include <boost/iostreams/filter/newline.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/copy.hpp>

#include <shark/Data/Dataset.h>

namespace shark {

namespace detail {


template<typename T, typename U, typename Stream>
void import_libsvm( T & input, // Container that holds the samples
		    U & labels, // Container that holds the labels
		    Stream & pre_in, // The file to be read from
		    int highestIndex = 0, // highest feature index, or 0 for auto-detect
		    bool allowMissingClasses = false, // if true, skip test if all classes occur
		    std::map<typename U::value_type, typename U::value_type> const* labelmap = NULL, // explicit mapping from LIBSVM to Shark labels
		    bool printSparsenessRatio = false
		    ) {

	typedef typename T::value_type SampleType;
	typedef typename U::value_type LabelType;

	if( !pre_in ) {
		throw( SHARKEXCEPTION( "[import_libsvm] Stream cannot be opened for reading." ) );
	}

	// fix line ending encodings and final newline. see http://svn.boost.org/svn/boost/trunk/libs/iostreams/test/newline_test.cpp
	boost::iostreams::filtering_istream fin;
	#if defined( _WIN32 )
		fin.push( boost::iostreams::newline_checker(boost::iostreams::newline::dos | boost::iostreams::newline::final_newline) );
		fin.push( boost::iostreams::newline_filter(boost::iostreams::newline::dos) );
		fin.push( pre_in );
	#elif defined( __unix__ ) || defined ( __unix ) || defined ( __linux__ ) || defined ( __posix__ ) || defined ( __APPLE__ )
		fin.push( boost::iostreams::newline_checker(boost::iostreams::newline::posix | boost::iostreams::newline::final_newline) );
		fin.push( boost::iostreams::newline_filter(boost::iostreams::newline::posix) );
		fin.push( pre_in );
	#elif defined( macintosh ) || defined( Macintosh )
		fin.push( boost::iostreams::newline_checker(boost::iostreams::newline::mac | boost::iostreams::newline::final_newline) );
		fin.push( boost::iostreams::newline_filter(boost::iostreams::newline::mac) );
		fin.push( pre_in );
	#endif

	// pipe data through line-ending-repair-stream into regular stream
	std::stringstream in( std::stringstream::in | std::stringstream::out );
	try {
		boost::iostreams::copy( fin, in );
	} catch (...) {
		throw SHARKEXCEPTION("[import_libsvm] Failed to process input file. Most likely, your file is not terminated by a newline.");
	}

	// check for final newline
	boost::iostreams::newline_checker* checker = 0;
	checker = BOOST_IOSTREAMS_COMPONENT( fin, 0, boost::iostreams::newline_checker );
	SHARK_ASSERT( checker->has_final_newline() );

	// helper vars
	LabelType storable_label;
	double cur_label;
	double min_label = std::numeric_limits<double>::max();
	double max_label = -std::numeric_limits<double>::max();
	int cur_index = 0;
	int last_index = 0;
	int max_index = 0;
	typename SampleType::value_type cur_value;

	// info vars
	bool binary = true;        // labels +1/-1
	bool regression = false;   // non-integer labels
	double sparsenessRatio;
	unsigned long noof_nonzeros = 0;

	// i/o helpers
	std::string line;
	unsigned int lineCounter = 0;
	std::vector<std::string> tokens, sub_tokens;

	// SCAN CARDINALITIES: only look through the file once. also some sanity checks.
	while( std::getline( in, line ) ) //one loop = one line
	{
		++lineCounter;
		if( line.empty() ) continue;
		boost::algorithm::split( tokens, line, boost::is_any_of( "\t " ) );
		std::vector< std::string >::iterator it = tokens.begin();
		while ( *it == "" ) //delete initial whitespace
			it = tokens.erase(it);
		try { //read and look at label
			cur_label = boost::lexical_cast< double >( *it );
		} catch( boost::bad_lexical_cast & blc ) {
			throw( SHARKEXCEPTION( ( boost::format( "[import_libsvm] Problem casting label in line %d: %s" ) % lineCounter % blc.what() ).str() ) );
		}
		if ( cur_label > max_label ) max_label = cur_label;
		if ( cur_label < min_label ) min_label = cur_label;

		if (cur_label != (int)cur_label ) regression = true;
		if (cur_label != +1.0 && cur_label != -1.0) binary = false;

		// look for highest index in all index-value-pairs
		it = tokens.begin() + 1;
		for( ; it != tokens.end(); ++it ) {
			//skip empty tokens (i.e., results of boosts convention that N separators must yield N+1 fields )
			if ( *it == "" )
				continue;
			//std::cout<<*it<<std::endl;
			boost::algorithm::split( sub_tokens, *it, boost::is_any_of( ":" ) );

			if(sub_tokens.empty())
				continue;
			//std::cout << std::setw(2) << std::setfill('0') << std::hex << std::uppercase;
			//std::copy(sub_tokens.front().begin(), sub_tokens.front().end(), std::ostream_iterator<unsigned int>(std::cout, "a b"));
			//std::cout<<std::endl;
			try {
				cur_index = boost::lexical_cast< int >( sub_tokens.front() );
			} catch( boost::bad_lexical_cast & blc ) {
				throw( SHARKEXCEPTION( ( boost::format( "[import_libsvm] Problem casting data in line %d: %s" ) % lineCounter % blc.what() ).str() ) );
			}
			SHARK_CHECK( cur_index > last_index, "[import_libsvm] expecting strictly increasing feature indices");
			SHARK_CHECK( cur_index > 0, "[import_libsvm] expecting only 1-based feature indices");
			if ( cur_index > max_index ) max_index = cur_index;
			last_index = cur_index;
		}
		last_index = 0;
	}

	// check for type consistency
	if (boost::is_integral<LabelType>::value)
		if ( regression )
			throw SHARKEXCEPTION("[import_libsvm] Cannot load regression dataset into dataset for labels of integral type.");
	// check for correct minimum label in multi-class classification case
	if ( !binary )
		if ( !regression )
			if ( min_label != 1 ) {
				if (labelmap == NULL) {
					throw SHARKEXCEPTION("[import_libsvm] Detected multi-class classification dataset, but with lowest label different from 1.");
				} else {
					typename std::map<LabelType, LabelType>::const_iterator it = labelmap->find(static_cast<LabelType>(min_label));
					if ( it == labelmap->end() ) throw SHARKEXCEPTION("[import_libsvm] min_label not found in explicitly given map");
					if ( it->second != 1 )
						throw SHARKEXCEPTION("[import_libsvm] Detected multi-class classification dataset, but with lowest label different from 1.");
				}
			}

	// allow possibility to specify a higher index for the feature vectors
	if (highestIndex) {
		if ( highestIndex < max_index ) throw SHARKEXCEPTION("[import_libsvm] highestIndex must be higher than those found in the dataset.");
		max_index = highestIndex;
	}

	// reset input stream
	in.clear(); //always clear first, then seek beginning
	in.seekg(std::ios::beg); //go back to beginning

	// ACTUAL READ-IN
	for (unsigned int i=0; i<lineCounter; i++) //one loop = one line
	{
		if ( !std::getline( in, line ) )
			throw SHARKEXCEPTION("[import_libsvm] cannot re-read all examples");
		if( line.empty() ) continue;
		boost::algorithm::split( tokens, line, boost::is_any_of( "\t " ) );
		std::vector< std::string >::iterator it = tokens.begin();
		while ( *it == "" ) //delete initial whitespace
			it = tokens.erase(it);
		try { //read and look at label
			cur_label = boost::lexical_cast< double >( *it );
		} catch( boost::bad_lexical_cast & blc ) {
			throw( SHARKEXCEPTION( ( boost::format( "[import_libsvm] Could not re-cast label in line %d: %s" ) % lineCounter % blc.what() ).str() ) );
		}
		if (labelmap != NULL)
		{
			typename std::map<LabelType, LabelType>::const_iterator it = labelmap->find(static_cast<LabelType>(cur_label));
			if (it == labelmap->end()) throw SHARKEXCEPTION( ( boost::format( "[import_libsvm] label %d not found in explicitly given map") %cur_label ).str() );
			storable_label = it->second;
		}
		else if (regression)
		{
			storable_label = static_cast<LabelType>(cur_label);   // todo: does this work on a RealVector? //mt_comment: why should it? do we expect anything else to do? (i don't see that it would..)
		}
		else if (binary)
		{
			storable_label = (cur_label <= 0.0) ? 0 : 1; //convert from pos/neg-encoding to 0/1-encoding
		}
		else
		{
			storable_label = (int)cur_label - 1; //libsvm by default labels the first class 1, shark uses 0 -> substract one.
		}
		labels.push_back(storable_label);

		// read and assign feature values
		SampleType sample( static_cast<std::size_t>(max_index), 0 ); //for dense, the 0 sets the default value. for sparse, the number of non-zeros.
		it = tokens.begin() + 1;
		for( ; it != tokens.end(); ++it ) {
			//skip empty tokens (i.e., results of boosts convention that N separators must yield N+1 fields )
			if ( *it == "" )
				continue;
			boost::algorithm::split( sub_tokens, *it, boost::is_any_of( ":" ) );
			try {
				cur_index = boost::lexical_cast< int >( sub_tokens.front() );
				cur_value = boost::lexical_cast< typename SampleType::value_type >( sub_tokens.back() );
			} catch( boost::bad_lexical_cast & blc ) {
				throw( SHARKEXCEPTION( ( boost::format( "[import_libsvm] Could not re-cast data in line %d: %s" ) % lineCounter % blc.what() ).str() ) );
			}
			SHARK_CHECK( cur_index > last_index, "[import_libsvm] problem re-reading data: cur_index <= last_index");
			SHARK_CHECK( cur_index > 0, "[import_libsvm] problem re-reading data: cur_index <= 0");
			SHARK_CHECK( cur_index <= max_index, "[import_libsvm] problem re-reading data: cur_index > max_index");
			if ( cur_value )
			{
				++ noof_nonzeros;
				sample(cur_index-1) = cur_value;
			}
		}
		input.push_back(sample);
	}

	// SCAN LABELS: if a classification dataset, look through the
	//              newly-imported label vector once to ensure sanity.
	//              For clarity: this check is triggered iff the user
	//              declared an integral label type (e.g., via ClassificationDataset)
	if (boost::is_integral<LabelType>::value)
	{
		if ( !allowMissingClasses ) { // only check if not turned off
			std::pair<typename U::iterator,typename U::iterator> minmax = boost::minmax_element(labels.begin(),labels.end());
			LabelType min_label = *minmax.first;
			LabelType max_label = *minmax.second;
			if ( min_label != 0 ) throw SHARKEXCEPTION("[import_libsvm] Label error: first label must occur in dataset.");

			std::vector<std::size_t> label_histogram(static_cast<std::size_t>(max_label)+1,0);
			// count
			for (typename U::iterator it=labels.begin(); it != labels.end(); ++it) {
				LabelType label = *it;
				label_histogram[ (std::size_t) label ] ++;
			}
			// check that every label occured
			for ( std::size_t i=0; i<max_label; i++ ) {
				if ( label_histogram[i] == 0 )
					throw SHARKEXCEPTION("[import_libsvm] Label error: every label in range must occur at least once.");
			}

		} //if (!allowMissingClasses)

	} //if (is_integral)


	// optional: sparseness-related computation/output
	unsigned long noofElements = input.size() * max_index;
	sparsenessRatio = noof_nonzeros / (double) noofElements;
	if ( printSparsenessRatio )
	{
		std::cout << "overall sparseness ratio = " << sparsenessRatio << std::endl;
	}

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
template<typename InputType, typename LabelType>
void import_libsvm(
		LabeledData<InputType, LabelType>& dataset,
		std::string fn,
		int highestIndex = 0,
		bool allowMissingClasses = false,
		std::map<LabelType, LabelType> const* labelmap = NULL,
		bool verbose = false)
{
	std::ifstream ifs(fn.c_str());
	std::vector<InputType> x;
	std::vector<LabelType> y;
	detail::import_libsvm(x, y, ifs, highestIndex, allowMissingClasses, labelmap, verbose);
	dataset = LabeledData<InputType, LabelType>(x, y);
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
	size_t i, ii, j;

	std::ofstream ofs(fn.c_str());
	if( !ofs ) {
		throw( SHARKEXCEPTION( "[export_libsvm] file can not be opened for reading" ) );
	}

	size_t dim = inputDimension(dataset);
	if(numberOfClasses(dataset)!=2) oneMinusOne = false;

	std::vector<detail::LabelSortPair> L;
	if(sortLabels) {
		for(i = 0; i < dataset.numberOfElements(); i++) L.push_back(detail::LabelSortPair(dataset.labels()(i), i));
		std::sort (L.begin(), L.end(), detail::cmpLabelSortPair);
	}

	for(ii = 0; ii < dataset.numberOfElements(); ii++) {
		// apply mapping to sorted indices
		if(sortLabels) i = L[ii].second;
		else i = ii;
		// apply transformation to label and write it to file
		if(oneMinusOne) ofs << 2*int(dataset.labels()(i))-1 << " ";
		else ofs << dataset.labels()(i)+1 << " "; //libsvm file format documentation is scarce, but by convention the first class seems to be 1..
		// write input data to file
		for(j=0; j<dim; j++) {
			if(dense) ofs << " " << j+1 << ":" << dataset.inputs()(i)(j);
			else if(dataset.inputs()(i)(j) != 0) ofs << " " << j+1 << ":" << dataset.inputs()(i)(j);
		}
		ofs << std::endl;
	}
	ofs.close();
}

/** @}*/

}
#endif
