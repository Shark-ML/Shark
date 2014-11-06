//===========================================================================
/*!
 * 
 *
 * \brief       unit test for the (negative) kernel target alignment
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

#include <shark/ObjectiveFunctions/KernelTargetAlignment.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Models/Kernels/LinearKernel.h>
#include <shark/Models/Kernels/KernelHelpers.h>

#define BOOST_TEST_MODULE ObjectiveFunctions_KernelTargetAlignment
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include "TestObjectiveFunction.h"

using namespace shark;

/// Fixture for testing naive Bayes classifier
class KTAFixture
{
public:

	KTAFixture():numInputs(100),dims(10){
		std::vector<RealVector> inputs(numInputs, RealVector(dims));
		std::vector<unsigned int> labels(numInputs);
		for(std::size_t i = 0; i != numInputs; ++i){
			labels[i] = i%2;
			for(std::size_t j = 0; j != dims; ++j)
				inputs[i](j) = Rng::uni(j-1.0+3*labels[i],j+1.0+3*labels[i]);
		}
		data  = createLabeledDataFromRange(inputs,labels,10);
		//center data
		RealVector mean(dims,0.0);
		for(std::size_t i = 0; i != numInputs; ++i)
			mean += inputs[i];
		mean /= numInputs;
		for(std::size_t i = 0; i != numInputs; ++i)
			 inputs[i]-=mean;
		
		dataCentered  = createLabeledDataFromRange(inputs,labels,10);
		
		//calculate Y
		y.resize(numInputs);
		y.clear();
		Y.resize(numInputs,numInputs);
		for(std::size_t i = 0; i != numInputs; ++i){
			for(std::size_t j = 0; j != numInputs; ++j){
				Y(i,j) = 1;
				if(data.element(i).label != data.element(j).label)
					Y(i,j) = -1;
				y(i)+=Y(i,j);
			}
		}
		y/=numInputs;
		meanY=sum(y)/numInputs;
	}
	
	ClassificationDataset data;
	ClassificationDataset dataCentered;
	RealMatrix Y;
	RealVector y;
	double meanY;
	std::size_t numInputs;
	std::size_t dims;
};

BOOST_FIXTURE_TEST_SUITE (ObjectiveFunctions_KernelTargetAlignment, KTAFixture)

template<class Kernel, class Data>
RealMatrix calculateCenteredKernelMatrix(Kernel const& kernel, Data const& data){
	std::size_t numInputs = data.numberOfElements();
	RealMatrix K = calculateRegularizedKernelMatrix(kernel,data);
	RealVector k = sum_rows(K)/numInputs;
	double meanK = sum(k)/numInputs;
	K-= repeat(k,numInputs);
	K-= trans(repeat(k,numInputs));
	K+= blas::repeat(meanK,numInputs,numInputs);
	return K;
}

//just sanity check ensuring, that KTA on pre-centered data on a linear kernel is the same
//as on uncentered data (this is the only kernel where this equality holds!
BOOST_AUTO_TEST_CASE( ObjectiveFunctions_KernelTargetAlignment_eval_Linear_Centered )
{
	DenseLinearKernel kernel;
	KernelTargetAlignment<> kta(data,&kernel);
	KernelTargetAlignment<> ktaCentered(dataCentered,&kernel);
	
	
	//linear Kernel doesn't have any parameters...
	RealVector input;
	
	double evalCentered = ktaCentered.eval(input);
	double eval = kta.eval(input);
	BOOST_CHECK_CLOSE(eval,evalCentered,1.e-5);
}

//calculate the centered KTA and check against trivially calculated result 
BOOST_AUTO_TEST_CASE( ObjectiveFunctions_KernelTargetAlignment_eval_Linear )
{
	DenseLinearKernel kernel;
	KernelTargetAlignment<> kta(data, &kernel);
	
	//calculate analytic result from centered Kernel
	RealMatrix K = calculateRegularizedKernelMatrix(kernel,dataCentered.inputs());
	
	double KY=sum(element_prod(K,Y));
	double KK = sum(element_prod(K,K));
	double result = -KY/std::sqrt(KK);
	
	//linear Kernel doesn't have any parameters...
	RealVector input;
	double eval = kta.eval(input);
	BOOST_CHECK_CLOSE(eval,result,1.e-5);
	
}

//calculate centered KTA against "dumb" calculation
BOOST_AUTO_TEST_CASE( ObjectiveFunctions_KernelTargetAlignment_eval_GaussKernel ){
	GaussianRbfKernel<> kernel(1);
	KernelTargetAlignment<> kta(data, &kernel);
	
	//calculate analytic result from centered Kernel
	RealMatrix K = calculateCenteredKernelMatrix(kernel,data.inputs());
	
	double KY=sum(element_prod(K,Y));
	double KK = sum(element_prod(K,K));
	double result = -KY/std::sqrt(KK);
	
	//linear Kernel doesn't have any parameters...
	RealVector input(1);
	input(0) = 1;
	double eval = kta.eval(input);
	BOOST_CHECK_CLOSE(eval,result,1.e-5);
	
}

//testing the correctness of the calculated formulas
BOOST_AUTO_TEST_CASE( ObjectiveFunctions_KernelTargetAlignment_numerics){
	double epsilon  = 1.0e-8;
	
	//estimate derivatives
	double estimatedDerivativeKcKc = 0;
	double estimatedDerivativeYKc = 0;
	double estimatedDerivative = 0;
	{
		GaussianRbfKernel<> kernel(1.0+epsilon);
		RealMatrix K = calculateCenteredKernelMatrix(kernel,data.inputs());
		double KY=sum(element_prod(K,Y));
		double KK = sum(element_prod(K,K));
		estimatedDerivativeKcKc+=KK;
		estimatedDerivativeYKc+=KY;
		estimatedDerivative+=KY/std::sqrt(KK);
	}
	{
		GaussianRbfKernel<> kernel(1.0-epsilon);
		RealMatrix K = calculateCenteredKernelMatrix(kernel,data.inputs());
		double KY=sum(element_prod(K,Y));
		double KK = sum(element_prod(K,K));
		estimatedDerivativeKcKc-=KK;
		estimatedDerivativeYKc-=KY;
		estimatedDerivative-=KY/std::sqrt(KK);
	}
	estimatedDerivativeKcKc/=2*epsilon;
	estimatedDerivativeYKc/=2*epsilon;
	estimatedDerivative/=2*epsilon;
	
	//calculate derivatives
	GaussianRbfKernel<> kernel(1.0);
	RealMatrix K = calculateRegularizedKernelMatrix(kernel,data.inputs());
	RealMatrix Kc = calculateCenteredKernelMatrix(kernel,data.inputs());
	RealVector k = sum_rows(K)/numInputs;
	double meanK = sum(k)/numInputs;
	
	double YKc=sum(element_prod(Kc,Y));
	double KcKc = sum(element_prod(Kc,Kc));
	//(<Kc,Kc>)'
	RealMatrix WKcKc = 2*(K-repeat(k, numInputs) -trans(repeat(k, numInputs))+ blas::repeat(meanK,numInputs,numInputs));
	double derivativeKcKc = calculateKernelMatrixParameterDerivative(kernel,data.inputs(),WKcKc)(0);
	RealMatrix WYKc = Y-repeat(y, numInputs) -trans(repeat(y, numInputs))+ blas::repeat(meanY,numInputs,numInputs);
	double derivativeYKc = calculateKernelMatrixParameterDerivative(kernel,data.inputs(),WYKc)(0);
	BOOST_CHECK_CLOSE(derivativeKcKc,estimatedDerivativeKcKc,0.0001);
	BOOST_CHECK_CLOSE(derivativeYKc,estimatedDerivativeYKc,0.0001);
	
	double derivative =  KcKc*derivativeYKc - 0.5*YKc* derivativeKcKc;
	derivative /= KcKc*sqrt(KcKc);
	BOOST_CHECK_CLOSE(derivative,estimatedDerivative,0.0001);
}

BOOST_AUTO_TEST_CASE( ObjectiveFunctions_KernelTargetAlignment_evalDerivative_GaussKernel )
{
	GaussianRbfKernel<> kernel(1);
	KernelTargetAlignment<> kta(data,&kernel);
	
	for(std::size_t i = 0; i != 100; ++i){
		RealVector input(1);
		input(0) = Rng::uni(0.1,1);
		testDerivative(kta,input);
	}
	
}
BOOST_AUTO_TEST_SUITE_END()