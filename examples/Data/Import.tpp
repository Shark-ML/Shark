//===========================================================================
/*!
 * 
 *
 * \brief       Data Import
 * 
 * This file is part of the tutorial "Importing Data".
 * By itself, it does not do anything particularly useful.
 *
 * \author      T. Glasmachers
 * \date        2014, 2016
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

//###begin<includes>
#include <shark/Data/Dataset.h>
#include <shark/Data/DataDistribution.h>
#include <shark/Data/Csv.h>
#include <shark/Data/SparseData.h>
#include <shark/Data/Download.h>
#include <iostream>
using namespace shark;
//###end<includes>


//###begin<distribution>
class YourDistribution : public LabeledDataDistribution<RealVector, unsigned int>
{
public:
	void draw(RealVector& input, unsigned int& label) const
	{
		input.resize(2);
		label = random::coinToss(random::globalRng);
		input(0) = random::uni(random::globalRng, -1,1);
		input(1) = random::uni(random::globalRng, -1,1) + label;
	}
};
//###end<distribution>


int main(int argc, char** argv)
{
	std::cout <<
			"\n"
			"WARNING: This program loads several data sets from disk.\n"
			"         If the files are not found then it will terminate\n"
			"         with an exception.\n"
			"\n";

//###begin<datasets>
	Data<RealVector> points;
	ClassificationDataset dataset;
//###end<datasets>

//###begin<generate>
	YourDistribution distribution;
	unsigned int numberOfSamples = 1000;
	dataset = distribution.generateDataset(numberOfSamples);
//###end<generate>

//###begin<csv>
	importCSV(points, "inputs.csv", ',', '#');
	importCSV(dataset, "data.csv", LAST_COLUMN, ',', '#');
//###end<csv>

{
//###begin<csv-regression>
	Data<RealVector> inputs;
	Data<RealVector> labels;
	importCSV(inputs, "inputs.csv");
	importCSV(labels, "labels.csv");
	RegressionDataset dataset(inputs, labels);
//###end<csv-regression>
}

{
//###begin<libsvm-dense>
	importSparseData(dataset, "data.libsvm");
//###end<libsvm-dense>

//###begin<libsvm-sparse>
	LabeledData<CompressedRealVector, unsigned int> sparse_dataset;
	importSparseData(sparse_dataset, "data.libsvm");
//###end<libsvm-sparse>
}

{
	ClassificationDataset dataset;
//###begin<download-url>
	// download dense data
	downloadCsvData(dataset, "http://mldata.org/repository/data/download/csv/banana-ida/", FIRST_COLUMN);

	// download sparse data
	downloadSparseData(dataset, "http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/svmguide1");
//###end<download-url>
}
{
	ClassificationDataset dataset;
//###begin<download-mldata>
	// fetch data set by name from mldata.org
	downloadFromMLData(dataset, "iris");
//###end<download-mldata>
}
}
