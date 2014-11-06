//===========================================================================
/*!
 * 
 *
 * \brief       unit test for the kernel expansion model
 * 
 * 
 * 
 *
 * \author      T. Glasmachers
 * \date        2011
 *
 *
 * \par Copyright 1995-2014 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://image.diku.dk/shark/>
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
#define BOOST_TEST_MODULE MODELS_KERNEL_EXPANSION
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Models/Kernels/KernelExpansion.h>
#include <shark/Rng/GlobalRng.h>

#include <sstream>
#include <boost/archive/polymorphic_text_iarchive.hpp>
#include <boost/archive/polymorphic_text_oarchive.hpp>

#include "../derivativeTestHelper.h" //for evalTest


using namespace shark;

BOOST_AUTO_TEST_SUITE (Models_Kernels_KernelExpansion)

BOOST_AUTO_TEST_CASE( KERNEL_EXPANSION_SIMPLE_VALUE )
{
	double k1 = exp(-0.5);
	RealVector test(2);
	test(0) = 3.0;
	test(1) = 4.0;
	std::stringstream ss;

	// define a set of two basis points
	std::vector<RealVector> points(2, RealVector(2));
	points[0](0) = 0.0;
	points[0](1) = 0.0;
	points[1](0) = 6.0;
	points[1](1) = 0.0;
	Data<RealVector> basis = createDataFromRange(points);

	// build a kernel expansion object
	DenseRbfKernel kernel(0.02);                               // sigma = 5
	KernelExpansion<RealVector> ex(&kernel, basis, true, 2);   // offset, two outputs
	RealVector param(6);
	param(0) = 1.0;
	param(1) = 1.0;
	param(2) = 0.0;
	param(3) = 1.0;
	param(4) = 2.0;
	param(5) = 3.0;
	ex.setParameterVector(param);

	// check its prediction
	RealVector prediction = ex(test);
	BOOST_CHECK_SMALL(prediction(0) - (1.0 * k1 + 2.0), 1e-10);
	BOOST_CHECK_SMALL(prediction(1) - (2.0 * k1 + 3.0), 1e-10);
}

BOOST_AUTO_TEST_CASE( KERNEL_EXPANSION_BATCH_VALUE )
{

	std::vector<RealVector> data(100,RealVector(3));
	for(std::size_t i = 0; i != 100; ++i){
		data[i](0) = Rng::uni(-1,1);
		data[i](1) = Rng::uni(-1,1);
		data[i](2) = Rng::uni(-1,1);
	}
	//BatchSize 10 to check whether complex structured bases also work
	Data<RealVector> basis = createDataFromRange(data,10);
	//BatchSize 100 results in a simple structured base
	Data<RealVector> simpleBasis = createDataFromRange(data,100);
	
	
	RealMatrix inputBatch(100,3);
	for(std::size_t i = 0; i != 100; ++i){
		inputBatch(i,0) = Rng::uni(-1,1);
		inputBatch(i,1) = Rng::uni(-1,1);
		inputBatch(i,2) = Rng::uni(-1,1);
	}
	
	//create expansions
	DenseRbfKernel kernel(0.02);
	KernelExpansion<RealVector> simpleExpansion(&kernel, simpleBasis, true, 2);
	KernelExpansion<RealVector> expansion(&kernel, basis, true, 2);
	
	RealVector parameters(simpleExpansion.numberOfParameters());
	for(std::size_t i = 0; i != parameters.size(); ++i){
		parameters(i) = Rng::uni(-1,1);
	}
	simpleExpansion.setParameterVector(parameters);
	expansion.setParameterVector(parameters);
	
	//test whether the choice of basis type changes the result
	//if this is the case, there is again something really wrong
	RealVector simpleOutput = simpleExpansion(row(inputBatch,0));
	RealVector output = expansion(row(inputBatch,0));
	double error = norm_sqr(simpleOutput-output);
	BOOST_REQUIRE_SMALL(error, 1.e-10);
	
	//if that worked, the next test checks, whether batch evaluation works at all
	testBatchEval(simpleExpansion,inputBatch);
	testBatchEval(expansion,inputBatch);

}

BOOST_AUTO_TEST_CASE( KERNEL_EXPANSION_SERIALIZATION )
{
	std::stringstream ss;

	{
		// define a set of two basis points
		std::vector<RealVector> points(2, RealVector(2));
		points[0](0) = 0.0;
		points[0](1) = 0.0;
		points[1](0) = 6.0;
		points[1](1) = 0.0;
		Data<RealVector> basis = createDataFromRange(points);

		// build a kernel expansion object
		DenseRbfKernel kernel(0.02);                             // sigma = 5
		KernelExpansion<RealVector> ex(&kernel, basis, true, 2); // offset, two outputs
		RealVector param(6);
		param(0) = 1.0;
		param(1) = 1.0;
		param(2) = 0.0;
		param(3) = 1.0;
		param(4) = 2.0;
		param(5) = 3.0;
		ex.setParameterVector(param);

		// serialize the model
		boost::archive::polymorphic_text_oarchive oa(ss);
		oa << const_cast<KernelExpansion<RealVector> const&>(ex);//prevent compilation warning
	}

	{
		// recover from the stream
		DenseRbfKernel kernel(1.0);
		KernelExpansion<RealVector> ex2(&kernel);
		boost::archive::polymorphic_text_iarchive ia(ss);
		ia >> ex2;

		// check whether the prediction still works
		double k1 = exp(-0.5);
		RealVector test(2);
		test(0) = 3.0;
		test(1) = 4.0;
		RealVector prediction2 = ex2(test);
		BOOST_CHECK_SMALL(prediction2(0) - (1.0 * k1 + 2.0), 1e-10);
		BOOST_CHECK_SMALL(prediction2(1) - (2.0 * k1 + 3.0), 1e-10);
	}
}

BOOST_AUTO_TEST_SUITE_END()
