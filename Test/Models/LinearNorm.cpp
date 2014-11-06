#define BOOST_TEST_MODULE ML_LinearNorm
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Models/LinearNorm.h>
#include "derivativeTestHelper.h"
#include <cmath>

#include <sstream>
#include <boost/archive/polymorphic_text_iarchive.hpp>
#include <boost/archive/polymorphic_text_oarchive.hpp>

using namespace std;
using namespace boost::archive;
using namespace shark;

BOOST_AUTO_TEST_SUITE (Models_LinearNorm)

BOOST_AUTO_TEST_CASE( LinearNorm_Value )
{
	LinearNorm model(2);

	//the testpoint
	RealVector point(2);
	point(0)=1;
	point(1)=3;

	RealVector testResult(2);
	testResult(0)=0.25;
	testResult(1)=0.75;

	//evaluate point
	RealVector result=model(point);
	double difference=norm_sqr(testResult-result);
	BOOST_CHECK_SMALL(difference,1.e-15);
}
BOOST_AUTO_TEST_CASE( LinearNorm_weightedInputDerivative )
{
	LinearNorm model(2);

	//the testpoint
	RealVector point(2);
	point(0)=1;
	point(1)=3;

	RealVector coefficients(2);
	coefficients(0)=2;
	coefficients(1)=-1;

	testWeightedInputDerivative(model,point,coefficients);
}

BOOST_AUTO_TEST_CASE( LinearNorm_SERIALIZE )
{
	//the target modelwork
	LinearNorm model(10);

	//now we serialize the model
	ostringstream outputStream;  
	polymorphic_text_oarchive oa(outputStream);  
	oa << const_cast<LinearNorm const&>(model);

	//and create a new model from the serialization
	LinearNorm modelDeserialized;
	istringstream inputStream(outputStream.str());  
	polymorphic_text_iarchive ia(inputStream);
	ia >> modelDeserialized;
	
	//test whether serialization works
	//topology check
	BOOST_REQUIRE_EQUAL(modelDeserialized.inputSize(),model.inputSize());
	BOOST_REQUIRE_EQUAL(modelDeserialized.outputSize(),model.outputSize());
}

BOOST_AUTO_TEST_SUITE_END()
