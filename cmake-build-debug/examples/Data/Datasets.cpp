//===========================================================================
/*!
 * 
 *
 * \brief       Data Normalization
 * 
 * This file is part of the tutorial "Data Containers".
 * By itself, it does not do anything particularly useful.
 *
 * \author      T. Glasmachers
 * \date        2014
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

#include <shark/Data/Dataset.h>

#include <shark/Data/DataView.h>

#include <shark/Models/LinearModel.h>

using namespace shark;


class F
{
public:
	typedef RealVector result_type;
	RealVector operator () (RealVector x) const
	{ return (2.0 * x); }
};

class G
{
public:
	typedef unsigned int result_type;
	unsigned int operator () (unsigned int y) const
	{ return y + 1; }
};

	class Add
	{
	public:
		Add(RealVector offset) : m_offset(offset) {}
	
		typedef RealVector result_type;   // do not forget to specify the result type
	
		RealVector operator () (RealVector input) const { // const is important
			return (input + m_offset);
		}
	
	private:
		RealVector m_offset;
	};


int main()
{

{
	std::vector<RealVector> points;
	Data<RealVector> data = createDataFromRange(points);
}
{
	std::vector<RealVector> inputs;
	std::vector<unsigned int> labels;
	ClassificationDataset data = createLabeledDataFromRange(inputs, labels);
}
{
	Data<RealVector> data(1000, RealVector(5));
}
{
	Data<RealVector> data(1000, RealVector(5), 100);
}
{
	Data<RealVector> data;
	Data<RealVector> data2(data);
	data = data2;
	data.makeIndependent();
}
{
	Data<RealVector> data;
	typedef Data<RealVector>::batch_range Batches;
	Batches batches = data.batches();

	std::cout << batches.size() << std::endl;
	for (auto pos = batches.begin(); pos != batches.end(); ++pos) {
		std::cout << *pos << std::endl;
	}
}
{
	Data<RealVector> data;
	Data<RealVector>::const_batch_range batches = data.batches();
	for(auto const& batch: data.batches()) {
		std::cout << batch << std::endl;
	}
	for (std::size_t i = 0; i != data.numberOfBatches(); ++i) {
		std::cout << data.batch(i) << std::endl;
	}
	for(auto const& batch: data.batches()) {
		for(std::size_t i=0; i != batchSize(batch); ++i) {
			std::cout << getBatchElement(batch,i );   // prints element i of the batch
		}
	}
	typedef Data<RealVector>::element_range Elements;

	// 1: explicit iterator loop using the range over the elements
	Elements elements = data.elements();
	for (auto pos = elements.begin(); pos != elements.end(); ++pos) {
		std::cout << *pos << std::endl;
	}

	// 2: foreach
	//note pass by value, the range returns proxy elements instead of references
	for(auto element: data.elements()) {
		std::cout << element << std::endl;
	}
}
{
	Data<unsigned int> data;
	std::size_t classes = numberOfClasses(data);       // maximal class label minus one
	std::vector<std::size_t> sizes = classSizes(data); // number of occurrences of every class label

	Data<RealVector> dataVec;
	std::size_t dim = dataDimension(dataVec);          // dimensionality of the data points
}
{
	LabeledData<RealVector, unsigned int> data;
	std::size_t classes = numberOfClasses(data);       // maximal class label minus one
	std::vector<std::size_t> sizes = classSizes(data); // number of occurrences of every class label
	std::size_t dim = inputDimension(data);            // dimensionality of the data points
}
{
	F f;
	G g;
	Data<RealVector> data;                             // initial data set
	data = transform(data, f);                         // applies f to each element

	LabeledData<RealVector, unsigned int> labeledData; // initial labeled dataset
	labeledData = transformInputs(labeledData, f);     // applies f to each input
	labeledData = transformLabels(labeledData, g);     // applies g to each label

	// a linear model, for example for whitening
	LinearModel<> model;
	// application of the model to the data
	labeledData = transformInputs(labeledData, model);
	// or an alternate shortcut:
	data = model(data);
}
{
	Data<RealVector> data;
	RealVector v(3); v(0) = 1.0; v(1) = 3.0; v(2) = -0.5;
	data = transform(data, Add(v));
}
{
	Data<unsigned int> dataset;
	DataView<Data<unsigned int> > view(dataset);
	for (std::size_t i=0; i != view.size(); ++i) {
		std::cout << view[i] << std::endl;
	}
	std::vector<std::size_t> indices;
	// somehow choose a set of indices
	Data<unsigned int> subsetData = toDataset(subset(view, indices));
}
{
	Data<unsigned int> dataset;
	DataView<Data<unsigned int> > view(dataset);
	std::vector<std::size_t> indices;
	std::size_t maximumBatchSize = 100;
	Data<unsigned int> subsetData = toDataset(subset(view, indices), maximumBatchSize);
}
{
	LabeledData<RealVector, unsigned int> dataset;
	DataView<LabeledData<RealVector, unsigned int> > view(dataset);
	std::cout << numberOfClasses(view) << " " << inputDimension(view) << std::endl;
}

}
