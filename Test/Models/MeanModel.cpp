#define BOOST_TEST_MODULE Models_MeanModel
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include "derivativeTestHelper.h"

#include <shark/Models/MeanModel.h>
#include <shark/Models/LinearModel.h>

#include <sstream>
#include <boost/archive/polymorphic_text_iarchive.hpp>
#include <boost/archive/polymorphic_text_oarchive.hpp>
#include <shark/Rng/GlobalRng.h>

using namespace std;
using namespace boost::archive;
using namespace shark;

BOOST_AUTO_TEST_CASE( MeanModel_Test )
{
	MeanModel<LinearModel<> > model;
	
	RealMatrix weights(2,2,0.0);
	RealVector bias(2,0.0);
	
	double alphaSum = 0.0;
	
	for(std::size_t i = 0; i != 5; ++i){
		RealMatrix curWeights(2,2,0.0);
		RealVector curBias(2,0.0);
		curWeights(0,0) = Rng::gauss(0,1);
		curWeights(0,1) = Rng::gauss(0,1);
		curWeights(1,0) = Rng::gauss(0,1);
		curWeights(1,1) = Rng::gauss(0,1);
		curBias(0) = Rng::gauss(0,1);
		curBias(1) = Rng::gauss(0,1);
		double curAlpha = Rng::uni(0.1,1);
		alphaSum +=curAlpha;
		weights += curAlpha*curWeights;
		bias +=curAlpha*curBias;
		model.addModel(LinearModel<>(curWeights,curBias),curAlpha);
		BOOST_CHECK_EQUAL(model.weight(i), curAlpha);
		BOOST_CHECK_EQUAL(model.numberOfModels(), i+1);
	}
	weights/=alphaSum;
	bias/=alphaSum;
	LinearModel<> linear(weights,bias);

	for(std::size_t i = 0; i != 100; ++i){
		//the testpoint2
		RealMatrix point(2,2);
		point(0,0)=Rng::uni(-5,5);
		point(0,1)= Rng::uni(-5,5);
		point(1,0)=Rng::uni(-5,5);
		point(1,1)= Rng::uni(-5,5);
		
		//evaluate ground truth result
		RealMatrix truth = linear(point);
		RealMatrix test = model(point);
		
		RealMatrix dist=distanceSqr(truth,test);
		
		BOOST_CHECK_SMALL(dist(0,0),1.e-10);
		BOOST_CHECK_SMALL(dist(1,1),1.e-10);
	}
	
}
BOOST_AUTO_TEST_CASE( MeanModel_Serialize )
{
	//the target modelwork
	MeanModel<LinearModel<> > model;
	
	RealMatrix weights(2,2,0.0);
	RealVector bias(2,0.0);
	
	double alphaSum = 0.0;
	
	for(std::size_t i = 0; i != 5; ++i){
		RealMatrix curWeights(2,2,0.0);
		RealVector curBias(2,0.0);
		curWeights(0,0) = Rng::gauss(0,1);
		curWeights(0,1) = Rng::gauss(0,1);
		curWeights(1,0) = Rng::gauss(0,1);
		curWeights(1,1) = Rng::gauss(0,1);
		curBias(0) = Rng::gauss(0,1);
		curBias(1) = Rng::gauss(0,1);
		double curAlpha = Rng::uni(0.1,1);
		alphaSum +=curAlpha;
		weights += curAlpha*curWeights;
		bias +=curAlpha*curBias;
		model.addModel(LinearModel<>(curWeights,curBias),curAlpha);
	}

	//the test is, that after deserialization, the results must be identical
	//so we generate some data first
	std::vector<RealVector> data;
	std::vector<RealVector> target;
	RealVector input(2);
	for (size_t i=0; i<1000; i++)
	{
		for(size_t j=0;j!=2;++j)
		{
			input(j)=Rng::uni(-1,1);
		}
		data.push_back(input);
		target.push_back(model(input));
	}
	RegressionDataset dataset = createLabeledDataFromRange(data,target);

	//now we serialize the model
	ostringstream outputStream;
	{
		polymorphic_text_oarchive oa(outputStream);  
		oa << model;
	}
	//and create a new model from the serialization
	MeanModel<LinearModel<> > modelDeserialized;
	istringstream inputStream(outputStream.str());  
	polymorphic_text_iarchive ia(inputStream);
	ia >> modelDeserialized;
	
	for (size_t i=0; i<1000; i++)
	{
		RealVector output = modelDeserialized(dataset.element(i).input);
		BOOST_CHECK_SMALL(norm_2(output -dataset.element(i).label),1.e-2);
	}
}
