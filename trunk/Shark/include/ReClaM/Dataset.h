//===========================================================================
/*!
*  \file Dataset.h
*
*  \brief Functions for loading ReClaM datasets
*
*  \author  T. Glasmachers
*  \date    2006
*
*  \par Copyright (c) 2006-2008:
*      Institut f&uuml;r Neuroinformatik<BR>
*      Ruhr-Universit&auml;t Bochum<BR>
*      D-44780 Bochum, Germany<BR>
*      Phone: +49-234-32-25558<BR>
*      Fax:   +49-234-32-14209<BR>
*      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
*      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
*
*
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

#ifndef _Dataset_H_
#define _Dataset_H_


#include <Array/Array.h>


//!
//! \brief Abstract description of a source of a dataset
//!
//! \par
//! There are two main sources of data:
//! <ul>
//!   <li> Static datasets loaded from files, often readily separated
//!        into test and training data </li>
//!   <li> Artificial distributions allowing for the generation
//!        of arbitrary amounts of data using a random number
//!        generator or a dynamical system </li>
//! </ul>
//!
//! \par
//! To allow for a similar handling of these data types ReClaM offers
//! the #DataSource interface.
//!
class DataSource
{
public:
	//! Constructor
	DataSource();

	//! Destructor
	virtual ~DataSource();


	//! Returns the dimension of the input data
	inline int getDataDimension()
	{
		return dataDim;
	}

	//! Returns the dimension of the output data or target
	inline int getTargetDimension()
	{
		return targetDim;
	}

	//!
	//! \brief Data generation interface
	//!
	//! \par
	//! This pure virtual function has to be overridden to
	//! provide labeled data, that is, pairs of input and target.
	//!
	//! \par
	//! Usually, both data and target will be two dimensional. The
	//! first dimension is either the time or simply the number of
	//! the example. The second dimension corresponds to the input
	//! or target dimension.
	//!
	//! \par
	//! Often the target is one dimensional. In this case the
	//! implementation may decide to provide a one dimensional
	//! array of target values.
	//!
	//! \param  data    array to fill in with input data
	//! \param  target  array to fill in with corresponding targets
	//! \param  count   number of examples to produce
	virtual bool GetData(Array<double>& data, Array<double>& target, int count) = 0;

protected:
	int dataDim;
	int targetDim;
};


//!
//! \brief The #DataFile class is a #DataSource based upon a file.
//!
//! \par
//! ReClaM defines a simple file format for real valued datasets.
//! The first line, starting with a doublecross ('#'), serves as
//! a file header. It contains exactly 4 tokens separated by
//! whitespace with the following meaning:
//! <ol>
//!   <li> number of examples </li>
//!   <li> input dimension </li>
//!   <li> output/target dimension </li>
//!   <li> data format: one of the keywords "ascii", "sparse", "float",
//!             "double", "int8", "int16", "int32", "uint8", "uint16"
//!             or "uint32" </li>
//! </ol>
//!
//! \par
//! Generally, the data are organized sample by sample, and for
//! each sample, the inputs are followed by the outputs.
//!
//! \par
//! In ascii and sparse format, every line defines one example
//! where a single space character serves as a separator.
//! All other formats are binary, that is, the numbers are
//! organized continuously without separators. For integers
//! and unsigned integers little endian encoding and for
//! floating point numbers IEEE float or double format is assumed.
//!
//! \par
//! In sparse format, all numbers are assumed to be zero.
//! Exceptions are indicated by pairs of the format "index:value",
//! interpreted as "data(index) = value". A semicolon is used to
//! separate the data from the targets. The targets are NOT sparse
//! encoded, that is, they are in standard ascii format.
//!
class DataFile : public DataSource
{
public:
	//! Constructor
	DataFile(const char* filename);

	//! Destructor
	~DataFile();


	//! Deliver data from the file
	bool GetData(Array<double>& data, Array<double>& target, int count);

	//! Deliver data from the file
	bool GetData(Array<double>& training_data, Array<double>& training_target, int training,
			Array<double>& test_data, Array<double>& test_target, int test,
			bool shuffle = false);

	//! Return the number of examples available
	inline int getNumberOfExamples()
	{
		return numberOfExamples;
	}

protected:
	bool ReadHeaderLine();
	bool ReadExample(Array<double>& data, Array<double>& target, int number);
	int ReadToken(char* buffer, int maxlength, const char* separators);
	int DiscardUntil(const char* separators);

	//! open file descriptor
	FILE* file;

	//! number of examples
	int numberOfExamples;

	//! 0=ascii, 1=sparse, 2=float, 3=double
	int format;

	//! number of examples already deliviered
	int currentExample;
};


//!
//! \brief The #Dataset class encapsulates a realization of data from a #DataSource.
//!
//! \par
//! A Dataset consists of separate training and test data.
//! It may be necessary to split the training data into
//! sub blocks, e.g. for cross validation. However, the
//! test data are assumed to be completely unknown during
//! the whole training process.
//!
class Dataset
{
public:
	//! Default constructor
	Dataset();

	//! Construction of a #Dataset from another #Dataset
	Dataset(const Dataset & dataset);

	virtual ~Dataset();

	//! Create a Dataset from a generic DataSource
	void CreateFromSource(DataSource& source, int train, int test);

	//! Load a Dataset from a single file,
	//! with a given absolute number of training and test patterns
	void CreateFromFile(const char* filename, int train, int test = -1);

	//! Load a Dataset from a single file,
	//! with a given fraction of training and test patterns
	void CreateFromFile(const char* filename, double train, double test = -1.0);

	//! Load a Dataset from a pair of files
	void CreateFromPairOfFiles(const char* trainfile, const char* testfile);

	//! Load a Dataset from a pair of files,
	//! but using a different data separation into training
	//! and test set.
	void CreateFromPairOfFiles(const char* trainfile, const char* testfile, int train);

	//! Load a Dataset from a data file and a split file.
	void CreateFromSplitFile(const char* datafile, const char* splitfile);

	//! Create a Dataset object from arrays
	void CreateFromArrays(const Array<double>& trainingData, const Array<double>& trainingTarget);
	void CreateFromArrays(const Array<double>& trainingData, const Array<double>& trainingTarget, const Array<double>& testData, const Array<double>& testTarget);

	//! \brief Generate a Dataset from a LIBSVM formatted file.
	//!
	//! \par
	//! You might consider testing your LIBSVM files with the checkdata.py 
	//! script provided within LIBSVM to ensure its correctness.
	//! LIBSVM precomputed kernel format is not yet supported.
	//! 
	//! \param  filename  name of the LIBSVM file to read
	//! \param  train     how many training samples to get from the file
	//! \param  test      how many training samples to get from the file. default: all remaining ones
	void CreateFromLibsvmFile(const char* filename, int train, int test = -1);
	

	//! shuffles the training examples
	void ShuffleTraining();

	//! shuffles the test examples
	void ShuffleTest();

	//! shuffles the union of training and test examples,
	//! such that the number of training and test examples
	//! remains unchanged
	void ShuffleAll();

	//! access to the training data as a constant array
	inline const Array<double> & getTrainingData() const
	{
		return trainingData;
	}

	//! access to the training targets as a constant array
	inline const Array<double> & getTrainingTarget() const
	{
		return trainingTarget;
	}

	//! access to the test data as a constant array
	inline const Array<double> & getTestData() const
	{
		return testData;
	}

	//! access to the test targets as a constant array
	inline const Array<double> & getTestTarget() const
	{
		return testTarget;
	}

	//!
	//! \brief Save the current Dataset to a file.
	//!
	//! \param  filename  name of the file, must not exist
	//! \param  training  include the training data?
	//! \param  test      include the test data?
	//! \param  format    see description
	//! \return The method returns true on success and false in case of failure.
	//!
	//! The data can be saved in one of the following formats:
	//! <ul>
	//!   <li>ascii: text file with one ascii encoded number per input and output</li>
	//!   <li>sparse: text file with sparse encoding, usefull for datasets containing many zeros as input</li>
	//!   <li>float: binary file with one IEEE float number per input and output</li>
	//!   <li>double: binary file with one IEEE double number per input and output</li>
	//!   <li>int8: binary file with a signed 8-bit-integer per input and output</li>
	//!   <li>int16: binary file with a signed 16-bit-integer per input and output</li>
	//!   <li>int32: binary file with a signed 32-bit-integer per input and output</li>
	//!   <li>uint8: binary file with an unsigned 8-bit-integer per input and output</li>
	//!   <li>uint16: binary file with an unsigned 16-bit-integer per input and output</li>
	//!   <li>uint32: binary file with an unsigned 32-bit-integer per input and output</li>
	//! </ul>
	//!
	bool Save(const char* filename, bool training = true, bool test = true, const char* format = "ascii");

	//!
	//! \brief Save the current Dataset in LIBSVM format.
	//!
	//! \param  filename  name of the file, must not exist
	//! \param  training  include the training data?
	//! \param  test      include the test data?
	//! \return The method returns true on success and false in case of failure.
	//!
	bool SaveLIBSVM(const char* filename, bool training = true, bool test = true);

	//!
	//! \brief component wise normalization of the dataset
	//!
	//! \par
	//! Normalize each component of the dataset by an affine
	//! linear transformation such that afterwards the
	//! training set has zero mean and unit variance in every
	//! component.
	//!
	void NormalizeComponents();

	//!
	//! \brief normalizes a single component
	//!
	//! \par
	//! Normalizes a component of the dataset by an affine
	//! linear transformation such that afterwards the
	//! component has zero mean and unit variance.
	//!
	void NormalizeComponent( int d );
	
	//! convert labels of -1 and +1 to 0 and 1, respectively
	void BinaryToMulticlass();
	
	//! convert labels of 0 and 1 to -1 and +1, respectively
	void MulticlassToBinary();
	
	//! convert labels of multiclass problem to one-hot (one-of-N) encoding
	void MulticlassToOneHot();
	
	
	

protected:
	bool ReadSplitFile(const char* filename, std::vector<unsigned int>& train, std::vector<unsigned int>& test);
	bool ReadLine(FILE* file, char* buffer, int bufferlength);

	Array<double> trainingData;
	Array<double> trainingTarget;
	Array<double> testData;
	Array<double> testTarget;
};


#endif

