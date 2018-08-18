//===========================================================================
/*!
 * 
 *
 * \brief       implementation of the csv data import
 * 
 * 
 *
 * \author      O.Krause
 * \date        2013
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
#define SHARK_COMPILE_DLL
#include <limits>
#include <boost/spirit/include/qi.hpp>
#include <boost/fusion/adapted/std_pair.hpp>
#include <shark/Data/Csv.h>
#include <vector>
#include <ctype.h>

using namespace shark;

namespace {

//csv input for multiple homogenous values in a row
inline std::vector<std::vector<double> > importCSVReaderSingleValues(
	std::string const& contents,
	char separator,
	char comment = '#'
) {
	std::string::const_iterator first = contents.begin();
	std::string::const_iterator last = contents.end();

	using namespace boost::spirit::qi;
	std::vector<std::vector<double> >  fileContents;

	double qnan = std::numeric_limits<double>::quiet_NaN();

	if(std::isspace(separator)){
		separator = 0;
	}

	bool r;
	if( separator == 0){
		r = phrase_parse(
			first, last,
			((
				+(double_ | ('?' >>  attr(qnan) ))
			) % eol) >> *eol,
			(space-eol) | (comment >> *(char_ - eol) >> (eol| eoi)), fileContents
		);
	}
	else{
		r = phrase_parse(
			first, last,
			(
				(double_ | ((lit('?')| &lit(separator)) >>  attr(qnan))) % separator
			) % eol >> *eol,
			(space-eol)| (comment >> *(char_ - eol) >> (eol| eoi)) , fileContents
		);
	}

	SHARK_RUNTIME_CHECK(r && first == last, "Failed to parse file");
	
	for(std::size_t i = 0; i != fileContents.size(); ++i){
		SHARK_RUNTIME_CHECK(
			fileContents[i].size() == fileContents[0].size(),
			"Detected different number of columns in a row of the file!"
		);
	}
	return fileContents;
}

//csv input for point-label pairs
typedef std::pair<int, std::vector<double > > CsvPoint;
inline std::vector<CsvPoint> import_csv_reader_points(
	std::string const& contents,
	LabelPosition position,
	char separator,
	char comment = '#'
) {
	typedef std::string::const_iterator Iterator;
	Iterator first = contents.begin();
	Iterator last = contents.end();
	std::vector<CsvPoint> fileContents;

	if(std::isspace(separator)){
		separator = 0;
	}

	using namespace boost::spirit::qi;
	using boost::spirit::_1;
	using namespace boost::phoenix;

	double qnan = std::numeric_limits<double>::quiet_NaN();

	bool r = false;
	if(separator == 0 && position == FIRST_COLUMN){
		r = phrase_parse(
			first, last,
			(
				lexeme[int_ >> -(lit('.')>>*lit('0'))]
				>> * (double_ | ('?' >>  attr(qnan) ))
			) % eol >> *eol,
			(space-eol)| (comment >> *(char_ - eol) >> (eol| eoi)), fileContents
		);
	}
	else if(separator != 0 && position == FIRST_COLUMN){
		r = phrase_parse(
			first, last,
			(
				lexeme[int_ >> -(lit('.')>>*lit('0'))]
				>> *(separator >> (double_ | (-lit('?') >>  attr(qnan) )))
			) % eol >> *eol,
			(space-eol)| (comment >> *(char_ - eol) >> (eol| eoi)) , fileContents
		);
	}
	else if(separator == 0 && position == LAST_COLUMN){
		do{
			std::pair<std::vector<double>, int > reversed_point;
			r = phrase_parse(
				first, last,
				*((double_ >> !(eol|eoi) ) | ('?' >>  attr(qnan)))
				>> lexeme[int_ >> -(lit('.')>>*lit('0'))]
				>> (*eol|eoi),
				(space-eol)| (comment >> *(char_ - eol) >> (eol| eoi)) , reversed_point
			);
			fileContents.push_back(CsvPoint(reversed_point.second,reversed_point.first));
		}while(r && first != last);
	}
	else{
		do{
			std::pair<std::vector<double>, int > reversed_point;
			r = phrase_parse(
				first, last,
				*((double_ | (-lit('?') >>  attr(qnan))) >> separator)
				>> lexeme[int_ >> -(lit('.')>>*lit('0'))]
				>> (*eol|eoi),
				(space-eol)| (comment >> *(char_ - eol) >> (eol| eoi)) , reversed_point
			);
			fileContents.push_back(CsvPoint(reversed_point.second,reversed_point.first));
		}while(r && first != last);
	}
	SHARK_RUNTIME_CHECK(r && first == last, "Failed to parse file");

	for(std::size_t i = 0; i != fileContents.size(); ++i){
		SHARK_RUNTIME_CHECK(
			fileContents[i].second.size() == fileContents[0].second.size(),
			"Detected different number of columns in a row of the file!"
		);
		SHARK_RUNTIME_CHECK(fileContents[i].first >= -1, "labels can not be smaller than -1" );
	}
	return fileContents;
}


//copy file with vectorial data into dataset
template<class T>
void readCSVData(
	Data<blas::vector<T> > &dataset,
	std::string const& contents,
	char separator,
	char comment,
	std::size_t maximumBatchSize
){
	std::vector<std::vector<double> > rows = importCSVReaderSingleValues(contents, separator,comment);
	if(rows.empty()){//empty file leads to empty data object.
		dataset = Data<blas::vector<T> >();
		return;
	}

	//copy rows of the file into the dataset
	std::size_t dimensions = rows[0].size();
	dataset = Data<blas::vector<T> >(rows.size(), dimensions, maximumBatchSize);
	std::size_t currentRow = 0;
	for(auto&& batch: dataset) {
		//copy the rows into the batch
		for(std::size_t i = 0; i != batch.size1(); ++i,++currentRow){
			for(std::size_t j = 0; j != dimensions; ++j){
				batch(i,j) = rows[currentRow][j];
			}
		}
	}
}

//copy file with input-class pair into dataset
template<class T>
void readCSVData(
	LabeledData<blas::vector<T>, unsigned int> &dataset,
	std::string const& contents,
	LabelPosition lp,
	char separator,
	char comment,
	std::size_t maximumBatchSize
){
	std::vector<CsvPoint> rows = import_csv_reader_points(contents, lp, separator,comment);
	if(rows.empty()){//empty file leads to empty data object.
		dataset = LabeledData<blas::vector<T>, unsigned int>();
		return;
	}

	//check labels for conformity
	bool binaryLabels = false;
	int minPositiveLabel = std::numeric_limits<int>::max();
	int maxPositiveLabel = -1;
	{
		for(std::size_t i = 0; i != rows.size(); ++i){
			if(rows[i].first == -1)
				binaryLabels = true;
			maxPositiveLabel = std::max(rows[i].first, maxPositiveLabel);
			minPositiveLabel = std::min(rows[i].first, minPositiveLabel);
		}
		SHARK_RUNTIME_CHECK(
			minPositiveLabel >= 0 || (minPositiveLabel == -1 && maxPositiveLabel == 1),
			"negative labels are only allowed for classes -1/1"
		);
	}
	//copy rows of the file into the dataset
	std::size_t dimensions = rows[0].second.size();
	dataset = LabeledData<blas::vector<T>, unsigned int>(rows.size(), {dimensions, maxPositiveLabel + 1}, maximumBatchSize);
	std::size_t currentRow = 0;
	for(auto&& batch: dataset){
		//copy the rows into the batch
		for(std::size_t i = 0; i != batch.input.size1(); ++i,++currentRow){
			for(std::size_t j = 0; j != dimensions; ++j){
				batch.input(i,j) = rows[currentRow].second[j];
			}
			int rawLabel = rows[currentRow].first;
			batch.label[i] = binaryLabels? 1 + (rawLabel-1)/2 : rawLabel;
		}
	}
}

//copy file with input-vector-label pair into dataset
template<class T>
void readCSVData(
	LabeledData<blas::vector<T>, blas::vector<T> > &dataset,
	std::string const& contents,
	LabelPosition lp,
	std::size_t numberOfOutputs,
	char separator,
	char comment,
	std::size_t maximumBatchSize
){
	std::vector<std::vector<double> > rows = importCSVReaderSingleValues(contents, separator,comment);
	if(rows.empty()){//empty file leads to empty data object.
		dataset = LabeledData<blas::vector<T>, blas::vector<T> >();
		return;
	}
	//copy rows of the file into the dataset
	SHARK_RUNTIME_CHECK(rows[0].size() > numberOfOutputs,"Files must have more columns than requested number of outputs");
	std::size_t numberOfInputs = rows[0].size() - numberOfOutputs;
	dataset = LabeledData<blas::vector<T>, blas::vector<T> >(rows.size(), {numberOfInputs, numberOfOutputs}, maximumBatchSize);
	std::size_t inputStart = (lp == FIRST_COLUMN)? numberOfOutputs : 0;
	std::size_t outputStart = (lp == FIRST_COLUMN)? 0: numberOfInputs;
	std::size_t currentRow = 0;
	for(auto&& batch: dataset){
		//copy the rows into the batch
		for(std::size_t i = 0; i != batch.input.size1(); ++i,++currentRow){
			for(std::size_t j = 0; j != numberOfInputs; ++j){
				batch.input(i,j) = rows[currentRow][j+inputStart];
			}
			for(std::size_t j = 0; j != numberOfOutputs; ++j){
				batch.label(i,j) = rows[currentRow][j+outputStart];
			}
		}
	}
}

}//end unnamed namespace

//start function implementations

void shark::csvStringToData(
	Data<unsigned int> &dataset,
	std::string const& contents,
	char separator,
	char comment,
	std::size_t maximumBatchSize
){
	//read file contents
	std::string::const_iterator first = contents.begin();
	std::string::const_iterator last = contents.end();

	using namespace boost::spirit::qi;
	std::vector<int>  rows;

	bool r = phrase_parse(
		first, last,
		*auto_,
		space| (comment >> *(char_ - eol) >> (eol| eoi)), rows
	);
	SHARK_RUNTIME_CHECK(r && first == last, "Failed to parse file");
	
	//empty file leads to empty data object.
	if(rows.empty()){
		dataset = Data<unsigned int>();
		return;
	}
	
	//check labels for conformity
	bool binaryLabels = false;
	int minPositiveLabel = std::numeric_limits<int>::max();
	int maxPositiveLabel = -1;
	{
		for(std::size_t i = 0; i != rows.size(); ++i){
			if(rows[i] == -1)
				binaryLabels = true;
			maxPositiveLabel = std::max(rows[i], maxPositiveLabel);
			minPositiveLabel = std::min(rows[i], minPositiveLabel);
		}
		SHARK_RUNTIME_CHECK(
			minPositiveLabel >= 0 || (minPositiveLabel == -1 && maxPositiveLabel == 1),
			"negative labels are only allowed for classes -1/1"
		);
	}
	
	//copy rows of the file into the dataset
	dataset = Data<unsigned int>(rows.size(), maxPositiveLabel + 1, maximumBatchSize);
	std::size_t currentRow = 0;
	for(std::size_t b = 0; b != dataset.size(); ++b) {
		auto& batch = dataset[b];
		for(std::size_t i = 0; i != batch.size(); ++i,++currentRow){
			int rawLabel = rows[currentRow];
			batch[i] = static_cast<unsigned int>(binaryLabels? 1 + (rawLabel-1)/2 : rawLabel);
		}
	}
}

void shark::csvStringToData(
    Data<RealVector> &dataset,
    std::string const& contents,
    char separator,
    char comment,
    std::size_t maximumBatchSize
){
	readCSVData(dataset,contents, separator, comment, maximumBatchSize);
}

void shark::csvStringToData(
    Data<FloatVector> &dataset,
    std::string const& contents,
    char separator,
    char comment,
    std::size_t maximumBatchSize
){
	readCSVData(dataset,contents, separator, comment, maximumBatchSize);
}

void shark::csvStringToData(
	LabeledData<RealVector, unsigned int> &dataset,
	std::string const& contents,
	LabelPosition lp,
	char separator,
	char comment,
	std::size_t maximumBatchSize
){
	readCSVData(dataset, contents, lp, separator, comment, maximumBatchSize);
}

void shark::csvStringToData(
	LabeledData<FloatVector, unsigned int> &dataset,
	std::string const& contents,
	LabelPosition lp,
	char separator,
	char comment,
	std::size_t maximumBatchSize
){
	readCSVData(dataset, contents, lp, separator, comment, maximumBatchSize);
}

void shark::csvStringToData(
	LabeledData<RealVector, RealVector> &dataset,
	std::string const& contents,
	LabelPosition lp,
	std::size_t numberOfOutputs,
	char separator,
	char comment,
	std::size_t maximumBatchSize
){
	readCSVData(dataset, contents, lp, numberOfOutputs, separator, comment, maximumBatchSize);
}

void shark::csvStringToData(
	LabeledData<FloatVector, FloatVector> &dataset,
	std::string const& contents,
	LabelPosition lp,
	std::size_t numberOfOutputs,
	char separator,
	char comment,
	std::size_t maximumBatchSize
){
	readCSVData(dataset, contents, lp, numberOfOutputs, separator, comment, maximumBatchSize);
}


///////////////IMPORT WRAPPERS


