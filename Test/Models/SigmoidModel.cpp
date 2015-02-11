#define BOOST_TEST_MODULE ML_SigmoidModel
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include "derivativeTestHelper.h"

#include <shark/Models/SigmoidModel.h>
#include <shark/Core/Math.h>
#include <sstream>



using namespace std;
using namespace boost::archive;
using namespace shark;

double sigmoid(double a){
	return 1.0/(1.0+std::exp(-a));
}

BOOST_AUTO_TEST_SUITE (Models_SigmoidModel)

BOOST_AUTO_TEST_CASE( SigmoidModel_Value )
{
	SigmoidModel model( false );

	//initialize parameters
	RealVector parameters(2);
	parameters(0)=0.5;
	parameters(1)=1;
	model.setParameterVector(parameters);

	//the testpoint
	RealVector point(1);
	point(0)=10;
	double testResult=sigmoid<double>(5-1);

	//evaluate point
	RealVector result=model(point);
	BOOST_CHECK_CLOSE(testResult,result[0],1.e-13);
}
BOOST_AUTO_TEST_CASE( SigmoidModel_Value_Unconstrained )
{
	SigmoidModel model( true );

	//initialize parameters
	RealVector parameters(2);
	parameters(0)=0.5;
	parameters(1)=1;
	model.setParameterVector(parameters);

	//the testpoint
	RealVector point(1);
	point(0)=10;
	double testResult=sigmoid<double>(16.48721270-1);

	//evaluate point
	RealVector result=model(point);
	BOOST_CHECK_CLOSE(testResult,result[0],1.e-12);
}
BOOST_AUTO_TEST_CASE( SigmoidModel_Value_NoOffset )
{
	SigmoidModel model( false );

	//initialize parameters
	RealVector parameters(2);
	parameters(0)=0.5;
	parameters(1)=1;
	model.setParameterVector(parameters);
    model.setOffsetActivity(false);

	//the testpoint
	RealVector point(1);
	point(0)=10;
	double testResult=sigmoid<double>(5);

	//evaluate point
	RealVector result=model(point);
	BOOST_CHECK_CLOSE(testResult,result[0],1.e-13);
}


BOOST_AUTO_TEST_CASE( SigmoidModel_weightedParameterDerivative )
{
	SigmoidModel model( false );

	//initialize parameters
	RealVector parameters(2);
	parameters(0)=0.5;
	parameters(1)=1;
	model.setParameterVector(parameters);

	//the testpoint
	RealVector point(1);
	point(0)=10;
	model(point);

	RealVector coefficients(1);
	coefficients(0)=2;

	testWeightedDerivative(model,point,coefficients);
}
BOOST_AUTO_TEST_CASE( SigmoidModel_weightedParameterDerivativeUnconstrained )
{
	SigmoidModel model( true );

	//initialize parameters
	RealVector parameters(2);
	parameters(0)=0.5;
	parameters(1)=1;
	model.setParameterVector(parameters);

	//the testpoint
	RealVector point(1);
	point(0)=10;
	model(point);

	RealVector coefficients(1);
	coefficients(0)=2;

	testWeightedDerivative(model,point,coefficients);
}
BOOST_AUTO_TEST_CASE( SigmoidModel_weightedParameterDerivative_NoOffset )
{
	SigmoidModel model( false );

	//initialize parameters
	RealVector parameters(2);
	parameters(0)=0.5;
	parameters(1)=1;
	model.setParameterVector(parameters);
    model.setOffsetActivity(false);

	//the testpoint
	RealVector point(1);
	point(0)=10;
	model(point);

	RealVector coefficients(1);
	coefficients(0)=2;

	testWeightedDerivative(model,point,coefficients);
}

BOOST_AUTO_TEST_CASE( SigmoidModel_weightedInputDerivative )
{
	SigmoidModel model( false );

	//initialize parameters
	RealVector parameters(2);
	parameters(0)=0.5;
	parameters(1)=1;
	model.setParameterVector(parameters);

	//the testpoint
	RealVector point(1);
	point(0)=10;
	model(point);

	RealVector coefficients(1);
	coefficients(0)=2;

	testWeightedInputDerivative(model,point,coefficients);
}
BOOST_AUTO_TEST_CASE( SigmoidModel_weightedInputDerivativeUnconstrained )
{
	SigmoidModel model( true );

	//initialize parameters
	RealVector parameters(2);
	parameters(0)=0.5;
	parameters(1)=1;
	model.setParameterVector(parameters);

	//the testpoint
	RealVector point(1);
	point(0)=10;
	model(point);

	RealVector coefficients(1);
	coefficients(0)=2;

	testWeightedInputDerivative(model,point,coefficients);
}


BOOST_AUTO_TEST_CASE( SigmoidModel_Serialize )
{
	//the target modelwork
	SigmoidModel model( false );

	//create random parameters
	RealVector testParameters(model.numberOfParameters());
	for(size_t param=0;param!=model.numberOfParameters();++param)
	{
		 testParameters(param)=Rng::gauss(0,1);
	}
	model.setParameterVector( testParameters );

	//the test is, that after deserialization, the results must be identical
	//so we generate some data first
	std::vector<RealVector> data;
	std::vector<RealVector> target;
	RealVector input(1);
	RealVector output(1);

	for (size_t i=0; i<1000; i++)
	{
		input(0)=Rng::uni(-1,1);
		data.push_back(input);
		target.push_back(model(input));
	}
	RegressionDataset dataset = createLabeledDataFromRange(data,target);

	//now we serialize the model
	ostringstream outputStream;  
	TextOutArchive oa(outputStream);  
	oa << model;

	//and create a new model from the serialization
	SigmoidModel modelDeserialized( false );
	istringstream inputStream(outputStream.str());  
	TextInArchive ia(inputStream);
	ia >> modelDeserialized;
	
	//test whether serialization works
	BOOST_CHECK_SMALL(norm_2(modelDeserialized.parameterVector() - testParameters),1.e-50);
	for (size_t i=0; i<1000; i++)
	{
		RealVector output = modelDeserialized(dataset.element(i).input);
		BOOST_CHECK_SMALL(norm_2(output -dataset.element(i).label),1.e-50);
	}
}
BOOST_AUTO_TEST_CASE( SigmoidModel_Serialize_NoOffset )
{
	//the target modelwork
	SigmoidModel model( false );

	//create random parameters
	RealVector testParameters(model.numberOfParameters());
	for(size_t param=0;param!=model.numberOfParameters();++param)
	{
		 testParameters(param)=Rng::gauss(0,1);
	}
	model.setParameterVector( testParameters );
	model.setOffsetActivity(false);
	testParameters(1) = 0.0;

	//the test is, that after deserialization, the results must be identical
	//so we generate some data first
	std::vector<RealVector> data;
	std::vector<RealVector> target;
	RealVector input(1);
	RealVector output(1);

	for (size_t i=0; i<1000; i++)
	{
		input(0)=Rng::uni(-1,1);
		data.push_back(input);
		target.push_back(model(input));
	}
	RegressionDataset dataset = createLabeledDataFromRange(data,target);

	//now we serialize the model
	ostringstream outputStream;  
	TextOutArchive oa(outputStream);  
	oa << model;

	//and create a new model from the serialization
	SigmoidModel modelDeserialized( false );
	istringstream inputStream(outputStream.str());  
	TextInArchive ia(inputStream);
	ia >> modelDeserialized;
	
	//test whether serialization works
	BOOST_CHECK_SMALL(norm_2(modelDeserialized.parameterVector() - testParameters),1.e-50);
	for (size_t i=0; i<1000; i++)
	{
		RealVector output = modelDeserialized(dataset.element(i).input);
		BOOST_CHECK_SMALL(norm_2(output -dataset.element(i).label),1.e-50);
	}
}
BOOST_AUTO_TEST_CASE( SigmoidModel_Serialize_Unconstrained )
{
	//the target modelwork
	SigmoidModel model( true );

	//create random parameters
	RealVector testParameters(model.numberOfParameters());
	for(size_t param=0;param!=model.numberOfParameters();++param)
	{
		 testParameters(param)=Rng::gauss(0,1);
	}
	model.setParameterVector( testParameters );

	//the test is, that after deserialization, the results must be identical
	//so we generate some data first
	std::vector<RealVector> data;
	std::vector<RealVector> target;
	RealVector input(1);
	RealVector output(1);

	for (size_t i=0; i<1000; i++)
	{
		input(0)=Rng::uni(-1,1);
		data.push_back(input);
		target.push_back(model(input));
	}
	RegressionDataset dataset = createLabeledDataFromRange(data,target);

	//now we serialize the model
	ostringstream outputStream;  
	TextOutArchive oa(outputStream);  
	oa << model;

	//and create a new model from the serialization
	SigmoidModel modelDeserialized( true );
	istringstream inputStream(outputStream.str());  
	TextInArchive ia(inputStream);
	ia >> modelDeserialized;
	
	//test whether serialization works
	BOOST_CHECK_SMALL(norm_2(modelDeserialized.parameterVector() - testParameters),1.e-50);
	for (size_t i=0; i<1000; i++)
	{
		RealVector output = modelDeserialized(dataset.element(i).input);
		BOOST_CHECK_SMALL(norm_2(output -dataset.element(i).label),1.e-50);
	}
}

BOOST_AUTO_TEST_SUITE_END()
