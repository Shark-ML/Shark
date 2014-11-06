//===========================================================================
/*!
 * 
 *
 * \brief       Test cases for the functions defined in Models/Kernels/KernelHelpers.cpp
 * 
 * 
 *
 * \author      Oswin Krause
 * \date        2012
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

#define BOOST_TEST_MODULE Kernels_KernelHelpers
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Models/Kernels/KernelHelpers.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>

using namespace shark;
using namespace std;

struct KernelHelpersFixture {
	KernelHelpersFixture() : datasetSize(100),dimensions(5),batchSize(8),kernel(0.5){
		std::vector<RealVector> points(datasetSize, RealVector(dimensions));
		for(std::size_t i = 0; i != datasetSize; ++i){
			for(std::size_t j = 0; j != dimensions; ++j){
				points[i](j)=Rng::uni(-1,1);
			}
		}
		data = createDataFromRange(points,batchSize);
	}

	std::size_t datasetSize;
	std::size_t dimensions;
	std::size_t batchSize;
	DenseRbfKernel kernel;
	Data<RealVector> data;
	
};


BOOST_FIXTURE_TEST_SUITE (Models_Kernels_KernelHelpers, KernelHelpersFixture )


BOOST_AUTO_TEST_CASE( KernelHelpers_calculateRegularizedKernelMatrix ){
	//calculate kernelMatrix to test
	RealMatrix kernelMatrix = calculateRegularizedKernelMatrix(kernel,data,1.0);
	BOOST_REQUIRE_EQUAL(kernelMatrix.size1(),datasetSize);
	BOOST_REQUIRE_EQUAL(kernelMatrix.size2(),datasetSize);
	
	//now check the results
	for(std::size_t i = 0; i != datasetSize; ++i){
		for(std::size_t j = 0; j != datasetSize; ++j){
			double result = kernel(data.element(i),data.element(j))+double(i==j);
			BOOST_CHECK_SMALL(kernelMatrix(i,j)-result,1.e-12);
		}
	}
}
BOOST_AUTO_TEST_CASE( KernelHelpers_calculateKernelMatrixParameterDerivative ){
	for(std::size_t test = 0; test != 100; ++test){
		RealMatrix weights(datasetSize,datasetSize);
		for(std::size_t i = 0; i != datasetSize; ++i){
			for(std::size_t  j = 0; j <= i; ++j){
				weights(j,i) = weights(i,j)=Rng::uni(1,2);
			}
		};
		
		//calculate kernelMatrix to test
		RealVector kernelGradient = calculateKernelMatrixParameterDerivative(kernel,data,weights);
		BOOST_REQUIRE_EQUAL(kernelGradient.size(),1u);
		
		//now check the results y computing the kernel matrix from scratch
		boost::shared_ptr<State> state = kernel.createState();
		RealMatrix derivativeWeight(1,1);
		RealVector result(1);
		result(0) = 0;
		RealVector gradient(1);
		RealMatrix block;
		for(std::size_t i = 0; i != datasetSize; ++i){
			RealMatrix x1(1,dimensions);
			row(x1,0)=data.element(i);
			for(std::size_t j = 0; j != datasetSize; ++j){
				RealMatrix x2(1,dimensions);
				row(x2,0)=data.element(j);
				derivativeWeight(0,0) = weights(i,j);
				kernel.eval(x1,x2,block,*state);
				kernel.weightedParameterDerivative(x1,x2,derivativeWeight,*state,gradient);
				result +=gradient;
				
			}
		}
		BOOST_CHECK_CLOSE(kernelGradient(0),result(0),1.e-12);
	}
}
BOOST_AUTO_TEST_SUITE_END()