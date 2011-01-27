//===========================================================================
/*!
*  \file Dataset.cpp
*
*  \brief Functions for loading ReClaM datasets
*
*  \author  T. Glasmachers
*  \date	2006
*
*  \par Copyright (c) 1999-2006:
*	  Institut f&uuml;r Neuroinformatik<BR>
*	  Ruhr-Universit&auml;t Bochum<BR>
*	  D-44780 Bochum, Germany<BR>
*	  Phone: +49-234-32-25558<BR>
*	  Fax:   +49-234-32-14209<BR>
*	  eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
*	  www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
*
*
*  <BR>
*
*  <BR><HR>
*  This file is part of Shark. This library is free software;
*  you can redistribute it and/or modify it under the terms of the
*  GNU General Public License as published by the Free Software
*  Foundation; either version 2, or (at your option) any later version.
*
*  This library is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with this library; if not, write to the Free Software
*  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
*/
//===========================================================================


#include <fstream>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <algorithm>
#include <SharkDefs.h>
#include <Rng/GlobalRng.h>
#include <ReClaM/Dataset.h>



#define DataFile_ReadType(T) \
	T value; \
	for (i = 0; i < dataDim; i++) { \
		if (fread(&value, sizeof(T), 1, file) != 1) return false; \
		data(number, i) = value; \
	} \
	for (i = 0; i < targetDim; i++) { \
		if (fread(&value, sizeof(T), 1, file) != 1) return false; \
		target(number, i) = value; \
	}


#define Dataset_WriteType(T) \
	T value; \
	if (training) \
	{ \
		for (t=0; t<ttr; t++) \
		{ \
			for (i=0; i<ic; i++) \
			{ \
				value = (T)trainingData(t, i); \
				if (fwrite(&value, sizeof(T), 1, file) != 1) { fclose(file); return false; } \
			} \
			for (o=0; o<oc; o++) \
			{ \
				value = (T)trainingTarget(t, o); \
				if (fwrite(&value, sizeof(T), 1, file) != 1) { fclose(file); return false; } \
			} \
		} \
	} \
	if (test) \
	{ \
		for (t=0; t<tte; t++) \
		{ \
			for (i=0; i<ic; i++) \
			{ \
				value = (T)testData(t, i); \
				if (fwrite(&value, sizeof(T), 1, file) != 1) { fclose(file); return false; } \
			} \
			for (o=0; o<oc; o++) \
			{ \
				value = (T)testTarget(t, o); \
				if (fwrite(&value, sizeof(T), 1, file) != 1) { fclose(file); return false; } \
			} \
		} \
	}


////////////////////////////////////////////////////////////


DataSource::DataSource()
{
}

DataSource::~DataSource()
{
}


////////////////////////////////////////////////////////////


DataFile::DataFile(const char* filename)
{
	file = fopen(filename, "r");
	if (file == NULL) throw SHARKEXCEPTION("[DataFile::DataFile] cannot open file");

	if (! ReadHeaderLine())
	{
		fclose(file);
		file = NULL;
		throw SHARKEXCEPTION("[DataFile::DataFile] error in file header");
	}
}

DataFile::~DataFile()
{
	if (file != NULL)
	{
		fclose(file);
		file = NULL;
	}
}


bool DataFile::GetData(Array<double>& data, Array<double>& target, int count)
{
	if (currentExample + count > numberOfExamples) return false;

	data.resize(count, dataDim, false);
	target.resize(count, targetDim, false);

	int i;
	for (i = 0; i < count; i++)
	{
		if (! ReadExample(data, target, i)) return false;
		currentExample++;
	}

	return true;
}

bool DataFile::GetData(Array<double>& training_data, Array<double>& training_target, int training,
		Array<double>& test_data, Array<double>& test_target, int test,
		bool shuffle)
{
	if (currentExample + training + test > numberOfExamples) return false;

	if (shuffle)
	{
		training_data.resize(training, dataDim, false);
		training_target.resize(training, targetDim, false);
		test_data.resize(test, dataDim, false);
		test_target.resize(test, targetDim, false);

		int i_train = 0;
		int i_test = 0;
		while (training + test > 0)
		{
			int r = Rng::discrete(0, training + test - 1);
			if (r < training)
			{
				if (! ReadExample(training_data, training_target, i_train)) return false;
				i_train++;
			}
			else
			{
				if (! ReadExample(test_data, test_target, i_test)) return false;
				i_test++;
			}
			currentExample++;
		}

		return true;
	}
	else
	{
		return (GetData(training_data, training_target, training)
				&& GetData(test_data, test_target, test));
	}
}

bool DataFile::ReadHeaderLine()
{
	int res;
	char buffer[256];

	if (fread(buffer, 1, 1, file) != 1) return false;
	if (buffer[0] != '#') return false;

	res = ReadToken(buffer, sizeof(buffer), " ");
	if (res != ' ') return false;
	numberOfExamples = atoi(buffer);

	res = ReadToken(buffer, sizeof(buffer), " ");
	if (res != ' ') return false;
	dataDim = atoi(buffer);

	res = ReadToken(buffer, sizeof(buffer), " ");
	if (res != ' ') return false;
	targetDim = atoi(buffer);

	res = ReadToken(buffer, sizeof(buffer), " \n");
	if (res > 1000) return false;
	if (res != '\n')
	{
		res = DiscardUntil("\n");
		if (res > 1000) return false;
	}

	if (strcmp(buffer, "ascii") == 0) format = 0;
	else if (strcmp(buffer, "sparse") == 0) format = 1;
	else if (strcmp(buffer, "float") == 0) format = 2;
	else if (strcmp(buffer, "double") == 0) format = 3;
	else if (strcmp(buffer, "int8") == 0) format = 4;
	else if (strcmp(buffer, "int16") == 0) format = 5;
	else if (strcmp(buffer, "int32") == 0) format = 6;
	else if (strcmp(buffer, "uint8") == 0) format = 7;
	else if (strcmp(buffer, "uint16") == 0) format = 8;
	else if (strcmp(buffer, "uint32") == 0) format = 9;
	else return false;

	currentExample = 0;

	return true;
}

bool DataFile::ReadExample(Array<double>& data, Array<double>& target, int number)
{
	int i, index;
	int res;
	char buffer[256];

	if (format == 0)
	{
		// "ascii" format
		for (i = 0; i < dataDim; i++)
		{
			res = ReadToken(buffer, sizeof(buffer), " ");
			if (res != ' ') return false;
			data(number, i) = atof(buffer);
		}
		for (i = 0; i < targetDim; i++)
		{
			res = ReadToken(buffer, sizeof(buffer), " \n");
			if (res > 1000) return false;
			if (i == targetDim - 1)
			{
				if (res == ' ')
				{
					if (DiscardUntil("\n") > 1000) return false;
				}
			}
			else if (res != ' ') return false;
			target(number, i) = atof(buffer);
		}
	}
	else if (format == 1)
	{
		// "sparse" format
		for (i = 0; i < dataDim; i++) data(number, i) = 0.0;
		while (true)
		{
			res = ReadToken(buffer, sizeof(buffer), " :\n");
			if (buffer[0] == ';') break;
			if (res == '\n')
			{
				printf("[number=%d --1--]", number); return false;
			}
			if (res > 1000)
			{
				printf("[2]"); return false;
			}
			index = atoi(buffer);
			if (res == ' ') res = DiscardUntil(":;\n");
			if (res != ':')
			{
				printf("[3]"); return false;
			}
			res = ReadToken(buffer, sizeof(buffer), " :;\n");
			if (res == ':')
			{
				printf("[4]"); return false;
			}
			if (res == '\n')
			{
				printf("[5]"); return false;
			}
			if (res > 1000)
			{
				printf("[6]"); return false;
			}
			data(number, index) = atof(buffer);
			if (res == ';') break;
		}
		for (i = 0; i < targetDim; i++)
		{
			res = ReadToken(buffer, sizeof(buffer), " \n");
			if (res > 1000) return false;
			if (i == targetDim - 1)
			{
				if (res == ' ')
				{
					if (DiscardUntil("\n") > 1000) return false;
				}
			}
			else if (res != ' ') return false;
			target(number, i) = atof(buffer);
		}
	}
	else if (format == 2)
	{
		DataFile_ReadType(float);
	}
	else if (format == 3)
	{
		DataFile_ReadType(double);
	}
	else if (format == 4)
	{
		DataFile_ReadType(char);
	}
	else if (format == 5)
	{
		DataFile_ReadType(short);
	}
	else if (format == 6)
	{
		DataFile_ReadType(int);
	}
	else if (format == 7)
	{
		DataFile_ReadType(unsigned char);
	}
	else if (format == 8)
	{
		DataFile_ReadType(unsigned short);
	}
	else if (format == 9)
	{
		DataFile_ReadType(unsigned int);
	}

	return true;
}

int DataFile::ReadToken(char* buffer, int maxlength, const char* separators)
{
	int i;
	int s, sc = strlen(separators);
	char c;
	bool start = true;
	for (i = 0; i < maxlength - 1; i++)
	{
		if (fread(&c, 1, 1, file) == 0) return 1001;
		for (s = 0; s < sc; s++)
		{
			if (separators[s] == c) break;
			if (separators[s] == '\n' && c == '\r')
			{
				// assume CR/LF end of line
				if (fread(&c, 1, 1, file) == 0) return 1001;
				break;
			}
		}
		if (s < sc)
		{
			if (start)
			{
				i--;
				continue;
			}
			else
			{
				buffer[i] = 0;
				return separators[s];
			}
		}
		buffer[i] = c;
		start = false;
	}
	buffer[i] = 0;
	return 1003;
}

int DataFile::DiscardUntil(const char* separators)
{
	int s, sc = strlen(separators);
	char c;
	while (true)
	{
		if (fread(&c, 1, 1, file) == 0) return 1001;
		for (s = 0; s < sc; s++)
		{
			if (separators[s] == c) return c;
			if (separators[s] == '\n' && c == '\r')
			{
				// assume CR/LF end of line
				if (fread(&c, 1, 1, file) == 0) return 1001;
				return c;
			}
		}
	}
}


////////////////////////////////////////////////////////////

Dataset::Dataset()
{
	this->trainingData.resize(0);
	this->trainingTarget.resize(0);
	this->testData.resize(0);
	this->testTarget.resize(0);
}

Dataset::Dataset(const Dataset& dataset)
{
	this->trainingData = dataset.getTrainingData();
	this->trainingTarget = dataset.getTrainingTarget();
	this->testData = dataset.getTestData();
	this->testTarget = dataset.getTestTarget();
}

Dataset::~Dataset()
{
}


void Dataset::CreateFromSource(DataSource& source, int train, int test)
{
	if (! source.GetData(trainingData, trainingTarget, train))
		throw SHARKEXCEPTION("[Dataset::CreateFromSource] error generating the dataset");
	if (! source.GetData(testData, testTarget, test))
		throw SHARKEXCEPTION("[Dataset::CreateFromSource] error generating the dataset");
}

void Dataset::CreateFromFile(const char* filename, int train, int test)
{
	DataFile file(filename);
	if (test < 0) test = file.getNumberOfExamples() - train;
	if (train + test > file.getNumberOfExamples() || train <= 0 || test < 0)
		throw SHARKEXCEPTION("[Dataset::CreateFromFile] invalid split into training and test set");
	if (! file.GetData(trainingData, trainingTarget, train))
		throw SHARKEXCEPTION("[Dataset::CreateFromFile] error loading the dataset)");
	if (! file.GetData(testData, testTarget, test))
		throw SHARKEXCEPTION("[Dataset::CreateFromFile] error loading the dataset)");
}

void Dataset::CreateFromFile(const char* filename, double train, double test)
{
	DataFile file(filename);
	int n_train = (int)(file.getNumberOfExamples() * train);
	int n_test;
	if (test < 0.0) n_test = file.getNumberOfExamples() - n_train;
	else n_test = (int)(file.getNumberOfExamples() * test);
	CreateFromFile(filename, n_train, n_test);
}

void Dataset::CreateFromPairOfFiles(const char* trainfile, const char* testfile)
{
	DataFile file1(trainfile);
	DataFile file2(testfile);
	if (! file1.GetData(trainingData, trainingTarget, file1.getNumberOfExamples()))
		throw SHARKEXCEPTION("[Dataset::CreateFromPairOfFiles] error loading the dataset");
	if (! file2.GetData(testData, testTarget, file2.getNumberOfExamples()))
		throw SHARKEXCEPTION("[Dataset::CreateFromPairOfFiles] error loading the dataset");
}

void Dataset::CreateFromPairOfFiles(const char* trainfile, const char* testfile, int train)
{
	Array<double> dataTrain;
	Array<double> targetTrain;
	Array<double> dataTest;
	Array<double> targetTest;
	DataFile file1(trainfile);
	DataFile file2(testfile);
	if (! file1.GetData(dataTrain, targetTrain, file1.getNumberOfExamples()))
		throw SHARKEXCEPTION("[Dataset::CreateFromPairOfFiles] error loading the dataset");
	if (! file2.GetData(dataTest, targetTest, file2.getNumberOfExamples()))
		throw SHARKEXCEPTION("[Dataset::CreateFromPairOfFiles] error loading the dataset");

	int tfs = dataTrain.dim(0);
	int all = tfs + dataTest.dim(0);
	int test = all - train;
	int dim = dataTrain.dim(1);
	int i, j, k;
	if (train <= 0 || test <= 0) throw SHARKEXCEPTION("[Dataset::CreateFromPairOfFiles] invalid split into training and test set");

	trainingData.resize(train, dim, false);
	trainingTarget.resize(train, 1, false);
	testData.resize(test, dim, false);
	testTarget.resize(test, 1, false);
	std::vector<int> entry(all);
	for (i = 0; i < all; i++) entry[i] = i;

	for (i = 0; i < train; i++)
	{
		j = Rng::discrete(0, entry.size() - 1);
		k = entry[j];
		entry.erase(entry.begin() + j);

		if (k < tfs)
		{
			trainingData[i] = dataTrain[k];
			trainingTarget(i, 0) = targetTrain(k, 0);
		}
		else
		{
			trainingData[i] = dataTest[k - tfs];
			trainingTarget(i, 0) = targetTest(k - tfs, 0);
		}
	}

	for (i = 0; i < test; i++)
	{
		k = entry[i];

		if (k < tfs)
		{
			testData[i] = dataTrain[k];
			testTarget(i, 0) = targetTrain(k, 0);
		}
		else
		{
			testData[i] = dataTest[k - tfs];
			testTarget(i, 0) = targetTest(k - tfs, 0);
		}
	}
}

void Dataset::CreateFromSplitFile(const char* datafile, const char* splitfile)
{
	DataFile data(datafile);

	std::vector<unsigned int> train;
	std::vector<unsigned int> test;
	if (! ReadSplitFile(splitfile, train, test)) throw SHARKEXCEPTION("[Dataset::CreateFromSplitFile] error reading the split file");

	int n_train = train.size();
	int n_test = test.size();
	if (data.getNumberOfExamples() != n_train + n_test) throw SHARKEXCEPTION("[Dataset::CreateFromSplitFile] data file and split file do not match");
	int dim_data = data.getDataDimension();
	int dim_target = data.getTargetDimension();

	trainingData.resize(n_train, dim_data, false);
	trainingTarget.resize(n_train, dim_target, false);
	testData.resize(n_test, dim_data, false);
	testTarget.resize(n_test, dim_target, false);

	std::sort(train.begin(), train.end());
	std::sort(test.begin(), test.end());

	int i;
	int tr = 0;
	int te = 0;
	for (i=0; i<n_train + n_test; i++)
	{
		Array<double> tmp_data;
		Array<double> tmp_target;
		data.GetData(tmp_data, tmp_target, 1);
		if (tr < n_train && (int)train[tr] == i)
		{
			trainingData[tr] = tmp_data[0];
			trainingTarget[tr] = tmp_target[0];
			tr++;
		}
		else if (te < n_test && (int)test[te] == i)
		{
			testData[te] = tmp_data[0];
			testTarget[te] = tmp_target[0];
			te++;
		}
		else throw SHARKEXCEPTION("[Dataset::CreateFromSplitFile] split file is inconsistent");
	}
}

void Dataset::CreateFromArrays(const Array<double>& trainingData, const Array<double>& trainingTarget)
{
	this->trainingData = trainingData;
	this->trainingTarget = trainingTarget;
	this->testData = Array<double>();
	this->testTarget = Array<double>();
}

void Dataset::CreateFromArrays(const Array<double>& trainingData, const Array<double>& trainingTarget, const Array<double>& testData, const Array<double>& testTarget)
{
	this->trainingData = trainingData;
	this->trainingTarget = trainingTarget;
	this->testData = testData;
	this->testTarget = testTarget;
}

void Dataset::CreateFromLibsvmFile(const char* filename, int train, int test)
{
	std::ifstream file( filename );
	
	if ( !file.is_open() ) 
		throw SHARKEXCEPTION("[Dataset::CreateFromLibsvmFile] cannot open file");
	else 
	{
		// cardinalities:
		int examples = 0;
		unsigned int features = 0;
		unsigned int classes = 0;
		unsigned int cur_index = 0;
		unsigned int last_index = 0;
		
		// other info vars:
		double cur_target = 0.0;
		double min_target = 1e100;
		double max_target = -1e100;
		double cur_value = 0.0;
		bool binary; //binary or multi-class problem?
		bool regression = false; //real-valued or integer target values?

		// i/o helpers:
		std::string line; 
		std::string cur_token; 
		std::string cur_subtoken;
		std::istringstream line_stream;
		std::istringstream token_stream;
		std::istringstream subtoken_stream;
		char c;
		
		ws(file); //skip leading whitespace lines
		// SCAN CARDINALITIES: only look through the file once. also some sanity checks.
		while( std::getline(file, line) ) //one loop = one line
		{
			// prepare the line for further processing
			line_stream.clear(); //reset..
			line_stream.str( line ); //..and assign
			std::ws( line_stream ); //skip leading whitespace in front of target
			
			// read and process target label
			if ( !std::getline( line_stream, cur_token, ' ' ) )
				throw SHARKEXCEPTION("[Dataset::CreateFromLibsvmFile] cannot retrieve target token");
			token_stream.clear(); //reset..
			token_stream.str( cur_token ); //..and assign
			if ( !(token_stream >> cur_target) || token_stream.get(c) ) //convert to double with safety checks
				throw SHARKEXCEPTION("[Dataset::CreateFromLibsvmFile] cannot read target token");
			if (cur_target < min_target) min_target = cur_target;
			if (cur_target > max_target) max_target = cur_target;
			if (cur_target != (int)cur_target ) regression = true;
			std::ws( line_stream ); //skip leading whitespace in front of next i-v-pair
			
			// read and process index-value-pairs
			while ( std::getline( line_stream, cur_token, ' ' ) ) //one loop = one index-value-pair
			{
				// prepare index-value-pair for further processing
				token_stream.clear(); //reset..
				token_stream.str( cur_token ); //..and assign
				
				// extract feature index
				if ( !getline(token_stream, cur_subtoken, ':') )
					throw SHARKEXCEPTION("[Dataset::CreateFromLibsvmFile] cannot retrieve feature index");
				subtoken_stream.clear(); //reset..
				subtoken_stream.str( cur_subtoken ); //..and assign
				if ( !(subtoken_stream >> cur_index) || subtoken_stream.get(c) ) //convert to unsigned int with safety checks
					throw SHARKEXCEPTION("[Dataset::CreateFromLibsvmFile] cannot read feature index");
				if ( cur_index <= last_index ) 
					throw SHARKEXCEPTION("[Dataset::CreateFromLibsvmFile] expecting strictly increasing feature indices");
				if ( cur_index == 0 )
					throw SHARKEXCEPTION("[Dataset::CreateFromLibsvmFile] expecting only 1-based feature indices");
				if ( cur_index > features ) features = cur_index;
				last_index = cur_index;
				std::ws( line_stream ); //skip leading whitespace in front of next i-v-pair
				// the remaining part in token_stream (i.e., the value) is ignored
			}
			last_index = 0; //reset
			++examples; //count lines
		}
			
		// set target format
		if ( !regression )
		{
			if ( (min_target == -1) && (max_target == +1) ) binary = true;
			else if (min_target == 1) 
			{
				binary = false;
				classes = (unsigned int)max_target;
			}
			else throw SHARKEXCEPTION("[Dataset::CreateFromLibsvmFile] inconsistent target values");
		}
		
		// set train/test split
		if (test < 0) test = examples-train;
		if (train + test > examples || train <= 0 || test < 0 )
			throw SHARKEXCEPTION("[Dataset::CreateFromLibsvmFile] invalid split into training and test set");
		else if ( train+test < examples ) 
			examples = train+test; //adjust size if not all examples are used
		
		// resize internal arrays
		trainingData.resize(train, features, false);
		trainingTarget.resize(train, 1, false);
		testData.resize(test, features, false);
		testTarget.resize(test, 1, false);
		
		// set all features to zero (libsvm is a sparse file format)
		trainingData = 0.0;
		testData = 0.0;
		
		// ACTUAL READ-IN
		// reset vars
		file.clear(); //always clear first, then seek beginning
		file.seekg(std::ios::beg); //go back to beginning
		std::ws(file); //skip leading whitespace lines
		for (int i=0; i<examples; i++) //one loop = one line
		{
			// prepare next line for further processing
			if ( !std::getline(file, line) )
				throw SHARKEXCEPTION("[Dataset::CreateFromLibsvmFile] cannot re-read all examples");
			line_stream.clear(); //reset..
			line_stream.str( line ); //..and assign
			std::ws( line_stream ); //skip leading whitespace in front of target
			
			// read and assign target label
			if ( !std::getline( line_stream, cur_token, ' ' ) )
				throw SHARKEXCEPTION("[Dataset::CreateFromLibsvmFile] cannot re-retrieve target token");
			token_stream.clear(); //reset..
			token_stream.str( cur_token ); //..and assign
			if ( !(token_stream >> cur_target) || token_stream.get(c) ) //convert to double with safety checks
				throw SHARKEXCEPTION("[Dataset::CreateFromLibsvmFile] cannot re-read target token");
			if ( !regression && !binary )
				cur_target -= 1; //multi-class dataset: convert from 1-based to 0-based class indices
			if ( i < train ) 
				trainingTarget(i, 0) = cur_target;
			else 
				testTarget(i-train, 0) = cur_target;
			std::ws( line_stream ); //skip leading whitespace in front of next index-value-pair
			
			// read and assign feature values
			while ( getline( line_stream, cur_token, ' ' ) ) //one loop = one i-v-pair
			{
				// prepare index-value-pair for further processing
				token_stream.clear(); //reset..
				token_stream.str( cur_token ); //..and assign
				
				// extract feature index
				if ( !std::getline(token_stream, cur_subtoken, ':') )
					throw SHARKEXCEPTION("[Dataset::CreateFromLibsvmFile] cannot re-retrieve feature index");
				subtoken_stream.clear(); //reset..
				subtoken_stream.str( cur_subtoken ); //..and assign
				if ( !(subtoken_stream >> cur_index) || subtoken_stream.get(c) ) //convert to unsigned int with safety checks
					throw SHARKEXCEPTION("[Dataset::CreateFromLibsvmFile] cannot re-read feature index");
				--cur_index; //convert from 1-based to 0-based feature indices
				
				// extract corresponding feature value
				if ( !std::getline(token_stream, cur_subtoken) )
					throw SHARKEXCEPTION("[Dataset::CreateFromLibsvmFile] cannot retrieve feature value");
				subtoken_stream.clear(); //reset..
				subtoken_stream.str( cur_subtoken ); //..and assign
				if ( !(subtoken_stream >> cur_value) || subtoken_stream.get(c) ) //convert to double with safety checks
					throw SHARKEXCEPTION("[Dataset::CreateFromLibsvmFile] cannot read feature value");
				
				// assign feature value to data
				if ( i < train ) 
					trainingData(i, cur_index) = cur_value;
				else 
					testData(i-train, cur_index) = cur_value;
					
				std::ws( line_stream ); //skip leading whitespace in front of next i-v-pair
			}
		}
		
		file.close();
	}
}

void Dataset::ShuffleTraining()
{
	Array<double> tmp1;
	Array<double> tmp2;
	unsigned int i, ic = trainingData.dim(0);
	for (i=1; i<ic; i++)
	{
		unsigned int j = Rng::discrete(0, i);
		if (i != j)
		{
			tmp1 = trainingData[i];
			trainingData[i] = trainingData[j];
			trainingData[j] = tmp1;
			tmp2 = trainingTarget[i];
			trainingTarget[i] = trainingTarget[j];
			trainingTarget[j] = tmp2;
		}
	}
}

void Dataset::ShuffleTest()
{
	Array<double> tmp1;
	Array<double> tmp2;
	unsigned int i, ic = testData.dim(0);
	for (i=1; i<ic; i++)
	{
		unsigned int j = Rng::discrete(0, i);
		if (i != j)
		{
			tmp1 = testData[i];
			testData[i] = testData[j];
			testData[j] = tmp1;
			tmp2 = testTarget[i];
			testTarget[i] = testTarget[j];
			testTarget[j] = tmp2;
		}
	}
}

void Dataset::ShuffleAll()
{
	Array<double> tmp1;
	Array<double> tmp2;
	unsigned int i, ic = trainingData.dim(0) + testData.dim(0);
	unsigned int c = trainingData.dim(0);
	for (i=1; i<ic; i++)
	{
		unsigned int j = Rng::discrete(0, i);
		if (i != j)
		{
			if (i < c)
			{
				if (j < c)
				{
					tmp1 = trainingData[i];   trainingData[i]   = trainingData[j];   trainingData[j]   = tmp1;
					tmp2 = trainingTarget[i]; trainingTarget[i] = trainingTarget[j]; trainingTarget[j] = tmp2;
				}
				else
				{
					tmp1 = trainingData[i];   trainingData[i]   = testData[j-c];   testData[j-c]   = tmp1;
					tmp2 = trainingTarget[i]; trainingTarget[i] = testTarget[j-c]; testTarget[j-c] = tmp2;
				}
			}
			else
			{
				if (j < c)
				{
					tmp1 = testData[i-c];   testData[i-c]   = trainingData[j];   trainingData[j]   = tmp1;
					tmp2 = testTarget[i-c]; testTarget[i-c] = trainingTarget[j]; trainingTarget[j] = tmp2;
				}
				else
				{
					tmp1 = testData[i-c];   testData[i-c]   = testData[j-c];   testData[j-c]   = tmp1;
					tmp2 = testTarget[i-c]; testTarget[i-c] = testTarget[j-c]; testTarget[j-c] = tmp2;
				}
			}
		}
	}
}

bool Dataset::Save(const char* filename, bool training, bool test, const char* format)
{
	FILE* file = fopen(filename, "w+");
	if (file == NULL) return false;

	int i, ic = 0;
	int o, oc = 0;
	int t, ttr = 0, tte = 0, total = 0;
	if (training)
	{
		ttr = trainingData.dim(0);
		ic = trainingData.dim(1);
		oc = trainingTarget.dim(1);
		total += ttr;
	}
	if (test)
	{
		tte = testData.dim(0);
		ic = testData.dim(1);
		oc = testTarget.dim(1);
		total += tte;
	}
	if (total == 0) return false;		// does not make any sense --> failure.

	fprintf(file, "# %d %d %d %s\n", total, ic, oc, format);

	if (strcmp(format, "ascii") == 0)
	{
		if (training)
		{
			for (t=0; t<ttr; t++)
			{
				for (i=0; i<ic; i++)
				{
					fprintf(file, "%g ", trainingData(t, i));
				}
				for (o=0; o<oc-1; o++)
				{
					fprintf(file, "%g ", trainingTarget(t, o));
				}
				fprintf(file, "%g\n", trainingTarget(t, oc-1));
			}
		}
		if (test)
		{
			for (t=0; t<tte; t++)
			{
				for (i=0; i<ic; i++)
				{
					fprintf(file, "%g ", testData(t, i));
				}
				for (o=0; o<oc-1; o++)
				{
					fprintf(file, "%g ", testTarget(t, o));
				}
				fprintf(file, "%g\n", testTarget(t, oc-1));
			}
		}
	}
	else if (strcmp(format, "sparse") == 0)
	{
		if (training)
		{
			for (t=0; t<ttr; t++)
			{
				for (i=0; i<ic; i++)
				{
					if (trainingData(t, i) != 0.0)
					{
						fprintf(file, "%d:%g ", i, trainingData(t, i));
					}
				}
				fprintf(file, "; ");
				for (o=0; o<oc-1; o++)
				{
					fprintf(file, "%g ", trainingTarget(t, o));
				}
				fprintf(file, "%g\n", trainingTarget(t, oc-1));
			}
		}
		if (test)
		{
			for (t=0; t<tte; t++)
			{
				for (i=0; i<ic; i++)
				{
					if (testData(t, i) != 0.0)
					{
						fprintf(file, "%d:%g ", i, testData(t, i));
					}
				}
				fprintf(file, "; ");
				for (o=0; o<oc-1; o++)
				{
					fprintf(file, "%g ", testTarget(t, o));
				}
				fprintf(file, "%g\n", testTarget(t, oc-1));
			}
		}
	}
	else if (strcmp(format, "float") == 0)
	{
		Dataset_WriteType(float);
	}
	else if (strcmp(format, "double") == 0)
	{
		Dataset_WriteType(double);
	}
	else if (strcmp(format, "int8") == 0)
	{
		Dataset_WriteType(char);
	}
	else if (strcmp(format, "int16") == 0)
	{
		Dataset_WriteType(short);
	}
	else if (strcmp(format, "int32") == 0)
	{
		Dataset_WriteType(int);
	}
	else if (strcmp(format, "uint8") == 0)
	{
		Dataset_WriteType(unsigned char);
	}
	else if (strcmp(format, "uint16") == 0)
	{
		Dataset_WriteType(unsigned short);
	}
	else if (strcmp(format, "uint32") == 0)
	{
		Dataset_WriteType(unsigned int);
	}
	else return false;

	fclose(file);
	return true;
}

bool Dataset::SaveLIBSVM(const char* filename, bool training, bool test)
{
	std::ofstream f(filename);
	if (! f.is_open()) return false;
	
	double min_target = 1e100;
	double max_target = -1e100;
	bool binary; //do we have a binary problem, that additionally is labeled in the -1/+1 style, rather than 0/1?
	bool regression = false; //regression task?

	// determine min/max class values
	if (training)
	{
		int i, ic = trainingData.dim(0);
		for (i=0; i<ic; i++)
		{
			double label = trainingTarget(i, 0);
			if ( label != (int)label ) regression = true;
			if (label < min_target) min_target = label;
			if (label > max_target) max_target = label;
		}
	}
	if (test)
	{
		int i, ic = testData.dim(0);
		for (i=0; i<ic; i++)
		{
			double label = testTarget(i, 0);
			if ( label != (int)label ) regression = true;
			if (label < min_target) min_target = label;
			if (label > max_target) max_target = label;
		}
	}

	// set target format
	if ( !regression )
	{
		if ( (min_target == -1) && (max_target == +1) ) binary = true;
		else if ( min_target == 0 ) binary = false;
		else throw SHARKEXCEPTION("[Dataset::SaveLIBSVM] inconsistent target values");
	}

	if (training)
	{
		int i, ic = trainingData.dim(0);
		int d, dim = trainingData.dim(1);
		SIZE_CHECK(trainingTarget.dim(1) == 1);

		for (i=0; i<ic; i++)
		{
			double label = trainingTarget(i, 0);
			if ( regression )
				f << label;
			else if ( binary )
			{
				if ( label > 0 )
					f << "+1";
				else 
					f << "-1";
			}
			else 
				f << label+1; //convert from 0-based to 1-based class indices
			for (d=0; d<dim; d++)
			{
				double value = trainingData(i, d);
				if (value != 0.0) f << " " << d+1 << ":" << value; //features are 1-based, too
			}
			f << "\n";
		}
	}

	if (test)
	{
		int i, ic = testData.dim(0);
		int d, dim = testData.dim(1);
		SIZE_CHECK(testTarget.dim(1) == 1);

		for (i=0; i<ic; i++)
		{
			double label = testTarget(i, 0);
			if ( regression )
				f << label;
			else if ( binary )
			{
				if ( label > 0 )
					f << "+1";
				else 
					f << "-1";
			}
			else 
				f << label+1; //convert from 0-based to 1-based class indices
			for (d=0; d<dim; d++)
			{
				double value = testData(i, d);
				if (value != 0.0) f << " " << d+1 << ":" << value; //features are 1-based, too
			}
			f << "\n";
		}
	}

	f.close();

	return true;
}

void Dataset::NormalizeComponents()
{
	int i, ic = trainingData.dim(0);
	int j, jc = testData.dim(0);
	int d, dim = trainingData.dim(1);
	for (d=0; d<dim; d++)
	{
		double sum = 0.0;
		for (i=0; i<ic; i++)
		{
			sum += trainingData(i, d);
		}
		double mean = sum / (double)ic;
		double var = 0.0;
		for (i=0; i<ic; i++)
		{
			double diff = trainingData(i, d) - mean;
			var += diff * diff;
		}
		var /= (double)ic;
		double stddev = sqrt(var);
		if (stddev == 0.0) continue;

		for (i=0; i<ic; i++)
		{
			trainingData(i, d) = (trainingData(i, d) - mean) / stddev;
		}
		for (j=0; j<jc; j++)
		{
			testData(j, d) = (testData(j, d) - mean) / stddev;
		}
	}
}

void Dataset::NormalizeComponent( int d )
{
	int i, ic = trainingData.dim(0);
	int j, jc = testData.dim(0);
	int dim = trainingData.dim(1);

	if( d < 0 || d >= dim )
		return;

	double sum = 0.0;
	for (i=0; i<ic; i++)
	{
		sum += trainingData(i, d);
	}
	double mean = sum / (double)ic;
	double var = 0.0;
	for (i=0; i<ic; i++)
	{
		double diff = trainingData(i, d) - mean;
		var += diff * diff;
	}
	var /= (double)ic;
	double stddev = sqrt(var);
	if (stddev == 0.0) return;

	for (i=0; i<ic; i++)
	{
		trainingData(i, d) = (trainingData(i, d) - mean) / stddev;
	}
	for (j=0; j<jc; j++)
	{
		testData(j, d) = (testData(j, d) - mean) / stddev;
	}

}

void Dataset::BinaryToMulticlass()
{
	int i, ic = trainingTarget.dim(0);
	int j, jc = testData.dim(0);
	
	// first, scan and assert binary (-/+1) labels
	for (i=0; i<ic; i++)
	{
		if ( (trainingTarget(i, 0) != -1) && (trainingTarget(i, 0) != 1) ) 
			throw SHARKEXCEPTION("[Dataset::BinaryToMulticlass] Not a -1/+1 labeled dataset.");
	}
	for (j=0; j<jc; j++)
	{
		if ( (testTarget(j, 0) != -1) && (testTarget(j, 0) != 1) ) 
			throw SHARKEXCEPTION("[Dataset::BinaryToMulticlass] Not a -1/+1 labeled dataset.");
	}
	
	// now convert
	for (i=0; i<ic; i++)
	{
		if ( trainingTarget(i, 0) == -1 ) 
			trainingTarget(i, 0) = 0;
	}
	for (j=0; j<jc; j++)
	{
		if ( testTarget(j, 0) == -1 ) 
			testTarget(j, 0) = 0;
	}
}

void Dataset::MulticlassToBinary()
{
	int i, ic = trainingTarget.dim(0);
	int j, jc = testData.dim(0);
	
	// first, scan and assert multiclass style (0/1) labels
	for (i=0; i<ic; i++)
	{
		if ( (trainingTarget(i, 0) != 0) && (trainingTarget(i, 0) != 1) ) 
			throw SHARKEXCEPTION("[Dataset::MulticlassToBinary] Not a 0/1 labeled dataset.");
	}
	for (j=0; j<jc; j++)
	{
		if ( (testTarget(j, 0) != 0) && (testTarget(j, 0) != 1) ) 
			throw SHARKEXCEPTION("[Dataset::MulticlassToBinary] Not a 0/1 labeled dataset.");
	}
	
	// now convert
	for (i=0; i<ic; i++)
	{
		if ( trainingTarget(i, 0) == 0 )
			trainingTarget(i, 0) = -1;
	}
	for (j=0; j<jc; j++)
	{
		if ( testTarget(j, 0) == 0 )
			testTarget(j, 0) = -1;
	}
}


void Dataset::MulticlassToOneHot()
{
	int i, ic = trainingTarget.dim(0);
	int j, jc = testTarget.dim(0);
	
	Array<double> tmpTrainTarget = trainingTarget;
	Array<double> tmpTestTarget  = testTarget;
	
	int c = 0; // counter for number of classes
	
	// first, scan and assert multiclass style (0/1) labels
	for (i=0; i<ic; i++)
	{
		if ( (trainingTarget(i, 0) < 0) ) 
			throw SHARKEXCEPTION("[Dataset::MulticlassToOneHot] Not an interger labeled dataset.");
		if ( trainingTarget(i, 0) > c) c = trainingTarget(i, 0);
	}
	for (j=0; j<jc; j++)
	{
		if ( (testTarget(j, 0) < 0) ) 
			throw SHARKEXCEPTION("[Dataset::MulticlassToOneHot] Not an integer labeled dataset.");
		if ( testTarget(j, 0) > c) c = testTarget(j, 0);
	}
	
	c++;
	
	trainingTarget.resize(ic, c, false);
	testTarget.resize(jc, c, false);
	
	// now convert
	for (i=0; i<ic; i++)
	{
		for(unsigned k=0; k<c; k++) {
		if ( tmpTrainTarget(i, 0) == k )
			trainingTarget(i, k) = 1;
		else
			trainingTarget(i, k) = 0;
		}
	}
	for (j=0; j<jc; j++)
	{
		for(unsigned k=0; k<c; k++) {
		if ( tmpTestTarget(j, 0) == k )
			testTarget(j, k) = 1;
		else
			testTarget(j, k) = 0;
		}
	}
}



bool Dataset::ReadLine(FILE* file, char* buffer, int bufferlength)
{
	int pos = 0;
	while (true)
	{
		if (pos == bufferlength) return false;
		if (fread(&buffer[pos], 1, 1, file) != 1) return false;
		if (buffer[pos] == '\n')
		{
			buffer[pos] = 0;
			return true;
		}
		pos++;
	}
}

bool Dataset::ReadSplitFile(const char* filename, std::vector<unsigned int>& train, std::vector<unsigned int>& test)
{
	FILE* file = fopen(filename, "r");
	if (file == NULL) return false;

	int i;
	char buffer[256];
	char* end;

	// read the header line
	if (! ReadLine(file, buffer, 256)) return false;
	if (buffer[0] != '#' || buffer[1] != ' ') return false;
	int n_train = strtol(buffer+2, &end, 10);
	if (n_train <= 0) return false;
	if (*end != ' ') return false;
	int n_test = strtol(end+1, &end, 10);
	if (n_test <= 0) return false;

	// read the split
	train.resize(n_train);
	test.resize(n_test);
	for (i=0; i<n_train; i++)
	{
		if (! ReadLine(file, buffer, 256)) return false;
		train[i] = atoi(buffer);
	}
	for (i=0; i<n_test; i++)
	{
		if (! ReadLine(file, buffer, 256)) return false;
		test[i] = atoi(buffer);
	}

	fclose(file);
	return true;
}
