#include <shark/Algorithms/GradientDescent/Rprop.h>
#include <shark/ObjectiveFunctions/ErrorFunction.h>
#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h>
#include <shark/Models/CMAC.h>
#include <shark/Core/Random.h>

#define BOOST_TEST_MODULE Models_CMAC
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include "derivativeTestHelper.h"

#include <sstream>



using namespace shark;
using namespace std;
using namespace boost::archive;

BOOST_AUTO_TEST_SUITE (Models_CMAC)

BOOST_AUTO_TEST_CASE( CMAC_PARAMETERS ){
	CMACMap model;
	model.setStructure(3,5,2,4,-1,1,true);
	BOOST_REQUIRE_EQUAL(model.numberOfParameters(),640);
	//this test simply checks whether a CMAC can learn itself
	RealVector testParameters(model.numberOfParameters());
	for(size_t param=0;param!=model.numberOfParameters();++param)
	{
		 testParameters(param)=random::gauss(random::globalRng,0,1);
	}
	model.setParameterVector( testParameters);
	
	model.setParameterVector(testParameters);
	RealVector retrievedParameters = model.parameterVector();
	BOOST_REQUIRE_EQUAL(retrievedParameters.size(),model.numberOfParameters());
	for (size_t i=0; i!=model.numberOfParameters(); ++i)
		BOOST_CHECK_EQUAL(retrievedParameters(i), testParameters(i));
}

BOOST_AUTO_TEST_CASE( CMAC_DERIVATIVE )
{
	for(std::size_t i = 0; i != 10000; ++i){
		CMACMap model;
		model.setStructure(3,5,2,4,-1,1,true);

		//this test simply checks whether a CMAC can learn itself
		RealVector testParameters(model.numberOfParameters());
		for(size_t param=0;param!=model.numberOfParameters();++param)
		{
			testParameters(param)=random::gauss(random::globalRng,0,1);
		}
		model.setParameterVector( testParameters);

		// Test the general derivative
		RealVector coefficients(5);
		coefficients(0) = random::uni(random::globalRng,-1,1);
		coefficients(1) = random::uni(random::globalRng,-1,1);
		coefficients(2) = random::uni(random::globalRng,-1,1);
		coefficients(3) = random::uni(random::globalRng,-1,1);
		coefficients(4) = random::uni(random::globalRng,-1,1);
		
		RealVector testInput(3);
		testInput(0) = random::uni(random::globalRng,-1,1);
		testInput(1) = random::uni(random::globalRng,-1,1);
		testInput(2) = random::uni(random::globalRng,-1,1);
		
 		testWeightedDerivative(model,testInput,coefficients);
		//testWeightedDerivative(model);
	}
}

BOOST_AUTO_TEST_CASE( CMAC_COPY )
{
	//the target cmac
	CMACMap cmac;
	cmac.setStructure(2,1,2,2,-1,1,true);

	//this test simply checks whether a CMAC can learn itself
	RealVector testParameters(cmac.numberOfParameters());
	for(size_t param=0;param!=cmac.numberOfParameters();++param)
	{
		 testParameters(param)=random::gauss(random::globalRng,0,1);
	}
	cmac.setParameterVector( testParameters);

	//create dataset for training
	std::vector<RealVector> data;
	std::vector<RealVector> target;
	RealVector input(cmac.inputShape().numElements());

	for (size_t i=0; i<1000; i++)
	{
		for(size_t j=0;j!=input.size();++j)
		{
			input(j)=random::uni(random::globalRng,-1,1);
		}
		data.push_back(input);
		target.push_back(cmac(input));
	}

	RegressionDataset dataset = createLabeledDataFromRange(data,target);

	//the test cmac with the same tiling
	CMACMap cmacTest=cmac;
	RealVector parameters(cmac.numberOfParameters());
	for(size_t param=0;param!=cmacTest.numberOfParameters();++param)
	{
		parameters(param)=random::gauss(random::globalRng,0,1);
	}
	cmacTest.setParameterVector(parameters);

	Rprop<> optimizer;
	SquaredLoss<> loss;
	ErrorFunction<> mse(dataset,&cmacTest,&loss);
	optimizer.init(mse);
	// train the cmac
	double error=0;
	for(size_t iteration=0;iteration<500;++iteration)
	{
		optimizer.step(mse);
		error=optimizer.solution().value;
	}
	std::cout<<"Optimization done. Error:"<<error<<std::endl;
	
	
	//next check: test the distance between old and new parameter vector
	double squaredNorm = distanceSqr( testParameters,optimizer.solution().point);
	std::cout<<"Squared distance of parameters:"<<squaredNorm<<std::endl;
	for(size_t param=0;param!=cmacTest.numberOfParameters();++param)
	{
		std::cout<<testParameters(param)<<" "<<optimizer.solution().point(param)<<" "<<parameters(param)<<std::endl;
	}
	
	BOOST_CHECK_SMALL(error,1.e-15);
	//BOOST_CHECK_SMALL(squaredNorm,1.e-15);
}

BOOST_AUTO_TEST_CASE( CMAC_SERIALIZE )
{
	//the target cmac
	CMACMap cmac;
	cmac.setStructure(2,1,2,4,-1,1,true);

	//create random parameters
	RealVector testParameters(cmac.numberOfParameters());
	for(size_t param=0;param!=cmac.numberOfParameters();++param)
	{
		 testParameters(param)=random::gauss(random::globalRng,0,1);
	}
	cmac.setParameterVector( testParameters);

	//the test is, that after deserialization, the results must be identical
	//so we generate some data first

	std::vector<RealVector> data;
	std::vector<RealVector> target;
	RealVector input(cmac.inputShape().numElements());

	for (size_t i=0; i<1000; i++)
	{
		for(size_t j=0;j!= input.size();++j)
		{
			input(j)=random::uni(random::globalRng,-1,1);
		}
		data.push_back(input);
		target.push_back(cmac(input));
	}
	RegressionDataset dataset = createLabeledDataFromRange(data,target);

	//now we serialize the CMAC
	
	ostringstream outputStream;  
	TextOutArchive oa(outputStream);  
	oa << cmac;

	//and create a new CMAC from the serialization
	CMACMap cmacDeserialized;
	istringstream inputStream(outputStream.str());  
	TextInArchive ia(inputStream);
	ia >> cmacDeserialized;
	
	//test whether serialization works
	//first simple parameter and topology check
	BOOST_CHECK_SMALL(norm_2(cmacDeserialized.parameterVector() - testParameters),1.e-50);
	BOOST_REQUIRE_EQUAL(cmacDeserialized.inputShape(),cmac.inputShape());
	BOOST_REQUIRE_EQUAL(cmacDeserialized.outputShape(),cmac.outputShape());
	for (size_t i=0; i<1000; i++)
	{
		RealVector output = cmacDeserialized(dataset.element(i).input);
		BOOST_CHECK_SMALL(norm_2(output -dataset.element(i).label),1.e-50);
	}
}

BOOST_AUTO_TEST_SUITE_END()
