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
 * <https://shark-ml.github.io/Shark/>
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

namespace {

template<class T>
inline std::vector<T> importCSVReaderSingleValue(
	std::string const& contents,
	char comment = '#'
) {
	std::string::const_iterator first = contents.begin();
	std::string::const_iterator last = contents.end();

	using namespace boost::spirit::qi;
	std::vector<T>  fileContents;

	bool r = phrase_parse(
		first, last,
		*auto_,
		space| (comment >> *(char_ - eol) >> (eol| eoi)), fileContents
	);
	SHARK_RUNTIME_CHECK(r && first == last, "Failed to parse file");
	return fileContents;
}

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
	return fileContents;
}

//csv input for point-label pairs
typedef std::pair<int, std::vector<double > > CsvPoint;
inline std::vector<CsvPoint> import_csv_reader_points(
	std::string const& contents,
	shark::LabelPosition position,
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
	if(separator == 0 && position == shark::FIRST_COLUMN){
		r = phrase_parse(
			first, last,
			(
				lexeme[int_ >> -(lit('.')>>*lit('0'))]
				>> * (double_ | ('?' >>  attr(qnan) ))
			) % eol >> *eol,
			(space-eol)| (comment >> *(char_ - eol) >> (eol| eoi)), fileContents
		);
	}
	else if(separator != 0 && position == shark::FIRST_COLUMN){
		r = phrase_parse(
			first, last,
			(
				lexeme[int_ >> -(lit('.')>>*lit('0'))]
				>> *(separator >> (double_ | (-lit('?') >>  attr(qnan) )))
			) % eol >> *eol,
			(space-eol)| (comment >> *(char_ - eol) >> (eol| eoi)) , fileContents
		);
	}
	else if(separator == 0 && position == shark::LAST_COLUMN){
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


	return fileContents;
}

template<class T>
void csvStringToDataImpl(
    shark::Data<T> &data,
    std::string const& contents,
    char separator,
    char comment,
    std::size_t maximumBatchSize
){
	std::vector<T> rows = importCSVReaderSingleValue<T>(contents, comment);
	if(rows.empty()){//empty file leads to empty data object.
		data = shark::Data<T>();
		return;
	}

	//copy rows of the file into the dataset
	std::vector<std::size_t> batchSizes = shark::detail::optimalBatchSizes(rows.size(),maximumBatchSize);
	data = shark::Data<T>(batchSizes.size());
	std::size_t currentRow = 0;
	for(std::size_t b = 0; b != batchSizes.size(); ++b) {
		typename shark::Data<T>::batch_type& batch = data.batch(b);
		batch.resize(batchSizes[b]);
		//copy the values into the batch
		for(std::size_t i = 0; i != batchSizes[b]; ++i,++currentRow){
			batch(i) = rows[currentRow];
		}
	}
	SIZE_CHECK(currentRow == rows.size());
}

}//end unnamed namespace

//start function implementations

void shark::csvStringToData(
    Data<RealVector> &data,
    std::string const& contents,
    char separator,
    char comment,
    std::size_t maximumBatchSize
){
	std::vector<std::vector<double> > rows = importCSVReaderSingleValues(contents, separator,comment);
	if(rows.empty()){//empty file leads to empty data object.
		data = Data<RealVector>();
		return;
	}

	//copy rows of the file into the dataset
	std::size_t dimensions = rows[0].size();
	std::vector<std::size_t> batchSizes = shark::detail::optimalBatchSizes(rows.size(),maximumBatchSize);
	data = Data<RealVector>(batchSizes.size());
	std::size_t currentRow = 0;
	for(std::size_t b = 0; b != batchSizes.size(); ++b) {
		RealMatrix& batch = data.batch(b);
		batch.resize(batchSizes[b],dimensions);
		//copy the rows into the batch
		for(std::size_t i = 0; i != batchSizes[b]; ++i,++currentRow){
			SHARK_RUNTIME_CHECK(rows[currentRow].size() == dimensions, "Vectors are required to have same size");

			for(std::size_t j = 0; j != dimensions; ++j){
				batch(i,j) = rows[currentRow][j];
			}
		}
	}
	data.shape() = {dimensions};
	SIZE_CHECK(currentRow == rows.size());
}

void shark::csvStringToData(
    Data<FloatVector> &data,
    std::string const& contents,
    char separator,
    char comment,
    std::size_t maximumBatchSize
){
	std::vector<std::vector<double> > rows = importCSVReaderSingleValues(contents, separator,comment);
	if(rows.empty()){//empty file leads to empty data object.
		data = Data<FloatVector>();
		return;
	}

	//copy rows of the file into the dataset
	std::size_t dimensions = rows[0].size();
	std::vector<std::size_t> batchSizes = shark::detail::optimalBatchSizes(rows.size(),maximumBatchSize);
	data = Data<FloatVector>(batchSizes.size());
	std::size_t currentRow = 0;
	for(std::size_t b = 0; b != batchSizes.size(); ++b) {
		FloatMatrix& batch = data.batch(b);
		batch.resize(batchSizes[b],dimensions);
		//copy the rows into the batch
		for(std::size_t i = 0; i != batchSizes[b]; ++i,++currentRow){
			SHARK_RUNTIME_CHECK(rows[currentRow].size() == dimensions, "Vectors are required to have same size");

			for(std::size_t j = 0; j != dimensions; ++j){
				batch(i,j) = static_cast<float>(rows[currentRow][j]);
			}
		}
	}
	data.shape() = {dimensions};
	SIZE_CHECK(currentRow == rows.size());
}

void shark::csvStringToData(
    Data<int> &data,
    std::string const& contents,
    char separator,
    char comment,
    std::size_t maximumBatchSize
){
	csvStringToDataImpl(data,contents,separator,comment,maximumBatchSize);
}

void shark::csvStringToData(
    Data<unsigned int> &data,
    std::string const& contents,
    char separator,
    char comment,
    std::size_t maximumBatchSize
){
	csvStringToDataImpl(data,contents,separator,comment,maximumBatchSize);
}

void shark::csvStringToData(
    Data<float> &data,
    std::string const& contents,
    char separator,
    char comment,
    std::size_t maximumBatchSize
){
	csvStringToDataImpl(data,contents,separator,comment,maximumBatchSize);
}

void shark::csvStringToData(
    Data<double> &data,
    std::string const& contents,
    char separator,
    char comment,
    std::size_t maximumBatchSize
){
	csvStringToDataImpl(data,contents,separator,comment,maximumBatchSize);
}

void shark::csvStringToData(
    LabeledData<RealVector, unsigned int> &dataset,
    std::string const& contents,
    LabelPosition lp,
    char separator,
    char comment,
    std::size_t maximumBatchSize
){
	std::vector<CsvPoint> rows = import_csv_reader_points(contents, lp, separator,comment);
	if(rows.empty()){//empty file leads to empty data object.
		dataset = LabeledData<RealVector, unsigned int>();
		return;
	}

	//check labels for conformity
	bool binaryLabels = false;
	int minPositiveLabel = std::numeric_limits<int>::max();
	{

		int maxPositiveLabel = -1;
		for(std::size_t i = 0; i != rows.size(); ++i){
			int label = rows[i].first;
			SHARK_RUNTIME_CHECK(label >= -1, "labels can not be smaller than -1" );
			if(label == -1)
				binaryLabels = true;
			else if(label < minPositiveLabel)
				minPositiveLabel = label;
			else if(label > maxPositiveLabel)
				maxPositiveLabel = label;
		}
		SHARK_RUNTIME_CHECK(
			minPositiveLabel >= 0 || (minPositiveLabel == -1 && maxPositiveLabel == 1),
			"negative labels are only allowed for classes -1/1"
		);
	}

	//copy rows of the file into the dataset
	std::size_t dimensions = rows[0].second.size();
	std::vector<std::size_t> batchSizes = shark::detail::optimalBatchSizes(rows.size(),maximumBatchSize);
	dataset = LabeledData<RealVector, unsigned int>(batchSizes.size());
	std::size_t currentRow = 0;
	for(std::size_t b = 0; b != batchSizes.size(); ++b) {
		RealMatrix& inputs = dataset.batch(b).input;
		UIntVector& labels = dataset.batch(b).label;
		inputs.resize(batchSizes[b],dimensions);
		labels.resize(batchSizes[b]);
		//copy the rows into the batch
		for(std::size_t i = 0; i != batchSizes[b]; ++i,++currentRow){
			SHARK_RUNTIME_CHECK(rows[currentRow].second.size() == dimensions, "Vectors are required to have same size");

			for(std::size_t j = 0; j != dimensions; ++j){
				inputs(i,j) = rows[currentRow].second[j];
			}
			int rawLabel = rows[currentRow].first;
			labels[i] = binaryLabels? 1 + (rawLabel-1)/2 : rawLabel -minPositiveLabel;
		}
	}
	dataset.inputs().shape() = {dimensions};
	SIZE_CHECK(currentRow == rows.size());
}

void shark::csvStringToData(
    LabeledData<FloatVector, unsigned int> &dataset,
    std::string const& contents,
    LabelPosition lp,
    char separator,
    char comment,
    std::size_t maximumBatchSize
){
	std::vector<CsvPoint> rows = import_csv_reader_points(contents, lp, separator,comment);
	if(rows.empty()){//empty file leads to empty data object.
		dataset = LabeledData<FloatVector, unsigned int>();
		return;
	}

	//check labels for conformity
	bool binaryLabels = false;
	int minPositiveLabel = std::numeric_limits<int>::max();
	{

		int maxPositiveLabel = -1;
		for(std::size_t i = 0; i != rows.size(); ++i){
			int label = rows[i].first;
			SHARK_RUNTIME_CHECK(label >= -1, "labels can not be smaller than -1" );
			if(label == -1)
				binaryLabels = true;
			else if(label < minPositiveLabel)
				minPositiveLabel = label;
			else if(label > maxPositiveLabel)
				maxPositiveLabel = label;
		}
		SHARK_RUNTIME_CHECK(
			minPositiveLabel >= 0 || (minPositiveLabel == -1 && maxPositiveLabel == 1),
			"negative labels are only allowed for classes -1/1"
		);
	}

	//copy rows of the file into the dataset
	std::size_t dimensions = rows[0].second.size();
	std::vector<std::size_t> batchSizes = shark::detail::optimalBatchSizes(rows.size(),maximumBatchSize);
	dataset = LabeledData<FloatVector, unsigned int>(batchSizes.size());
	std::size_t currentRow = 0;
	for(std::size_t b = 0; b != batchSizes.size(); ++b) {
		FloatMatrix& inputs = dataset.batch(b).input;
		UIntVector& labels = dataset.batch(b).label;
		inputs.resize(batchSizes[b],dimensions);
		labels.resize(batchSizes[b]);
		//copy the rows into the batch
		for(std::size_t i = 0; i != batchSizes[b]; ++i,++currentRow){
			SHARK_RUNTIME_CHECK(rows[currentRow].second.size() == dimensions, "Vectors are required to have same size");

			for(std::size_t j = 0; j != dimensions; ++j){
				inputs(i,j) = static_cast<float>(rows[currentRow].second[j]);
			}
			int rawLabel = rows[currentRow].first;
			labels[i] = binaryLabels? 1 + (rawLabel-1)/2 : rawLabel -minPositiveLabel;
		}
	}
	dataset.inputs().shape() = {dimensions};
	SIZE_CHECK(currentRow == rows.size());
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
	std::vector<std::vector<double> > rows = importCSVReaderSingleValues(contents, separator,comment);
	if(rows.empty()){//empty file leads to empty data object.
		dataset = LabeledData<RealVector, RealVector>();
		return;
	}

	//copy rows of the file into the dataset
	SHARK_RUNTIME_CHECK(rows[0].size() > numberOfOutputs,"Files must have more columns than requested number of outputs");
	std::size_t dimensions = rows[0].size();
	std::size_t numberOfInputs = dimensions-numberOfOutputs;
	std::vector<std::size_t> batchSizes = shark::detail::optimalBatchSizes(rows.size(),maximumBatchSize);
	dataset = LabeledData<RealVector, RealVector>(batchSizes.size());
	std::size_t inputStart = (lp == FIRST_COLUMN)? numberOfOutputs : 0;
	std::size_t outputStart = (lp == FIRST_COLUMN)? 0: numberOfInputs;
	std::size_t currentRow = 0;
	for(std::size_t b = 0; b != batchSizes.size(); ++b) {
		RealMatrix& inputs = dataset.batch(b).input;
		RealMatrix& labels = dataset.batch(b).label;
		inputs.resize(batchSizes[b],numberOfInputs);
		labels.resize(batchSizes[b],numberOfOutputs);
		//copy the rows into the batch
		for(std::size_t i = 0; i != batchSizes[b]; ++i,++currentRow){
			SHARK_RUNTIME_CHECK(rows[currentRow].size() == dimensions,"Detected different number of columns in a row of the file!");

			for(std::size_t j = 0; j != numberOfInputs; ++j){
				inputs(i,j) = rows[currentRow][j+inputStart];
			}
			for(std::size_t j = 0; j != numberOfOutputs; ++j){
				labels(i,j) = rows[currentRow][j+outputStart];
			}
		}
	}
	dataset.inputs().shape() = {numberOfInputs};
	dataset.labels().shape() = {numberOfOutputs};
	SIZE_CHECK(currentRow == rows.size());
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
	std::vector<std::vector<double> > rows = importCSVReaderSingleValues(contents, separator,comment);
	if(rows.empty()){//empty file leads to empty data object.
		dataset = LabeledData<FloatVector, FloatVector>();
		return;
	}

	//copy rows of the file into the dataset
	SHARK_RUNTIME_CHECK(rows[0].size() > numberOfOutputs,"Files must have more columns than requested number of outputs");
	std::size_t dimensions = rows[0].size();
	std::size_t numberOfInputs = dimensions-numberOfOutputs;
	std::vector<std::size_t> batchSizes = shark::detail::optimalBatchSizes(rows.size(),maximumBatchSize);
	dataset = LabeledData<FloatVector, FloatVector>(batchSizes.size());
	std::size_t inputStart = (lp == FIRST_COLUMN)? numberOfOutputs : 0;
	std::size_t outputStart = (lp == FIRST_COLUMN)? 0: numberOfInputs;
	std::size_t currentRow = 0;
	for(std::size_t b = 0; b != batchSizes.size(); ++b) {
		FloatMatrix& inputs = dataset.batch(b).input;
		FloatMatrix& labels = dataset.batch(b).label;
		inputs.resize(batchSizes[b],numberOfInputs);
		labels.resize(batchSizes[b],numberOfOutputs);
		//copy the rows into the batch
		for(std::size_t i = 0; i != batchSizes[b]; ++i,++currentRow){
			SHARK_RUNTIME_CHECK(rows[currentRow].size() == dimensions,"Detected different number of columns in a row of the file!");

			for(std::size_t j = 0; j != numberOfInputs; ++j){
				inputs(i,j) = static_cast<float>(rows[currentRow][j+inputStart]);
			}
			for(std::size_t j = 0; j != numberOfOutputs; ++j){
				labels(i,j) = static_cast<float>(rows[currentRow][j+outputStart]);
			}
		}
	}
	dataset.inputs().shape() = {numberOfInputs};
	dataset.labels().shape() = {numberOfOutputs};
	SIZE_CHECK(currentRow == rows.size());
}


///////////////IMPORT WRAPPERS


