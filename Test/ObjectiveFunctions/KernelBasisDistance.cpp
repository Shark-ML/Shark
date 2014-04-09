//===========================================================================
/*!
 * 
 *
 * \brief       unit test for the (negative) kernel target alignment
 * 
 * 
 * 
 *
 * \author      O. Krause
 * \date        2014
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

#include <shark/ObjectiveFunctions/KernelBasisDistance.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Models/Kernels/LinearKernel.h>
#include <shark/Data/DataDistribution.h>
#include <shark/Rng/GlobalRng.h>

#define BOOST_TEST_MODULE ObjectiveFunctions_KernelBasisDistance
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include "TestObjectiveFunction.h"

using namespace shark;

//Sanity check that checks that the error of the exact same basis is minimal and the derivative is small.
BOOST_AUTO_TEST_CASE( ObjectiveFunctions_Value_Derivative_Optimal )
{
	for(std::size_t trial = 0; trial != 10; ++trial){
		Chessboard problem;
		LabeledData<RealVector, unsigned int> dataset = problem.generateDataset(100,10);
		GaussianRbfKernel<> kernel(0.5);
		KernelExpansion<RealVector> expansion(&kernel,dataset.inputs(),false,3);
		for(std::size_t i = 0; i != 100; ++i){
			for(std::size_t j = 0; j != 3; ++j){
				expansion.alpha()(i,j) = Rng::gauss(0,1);
			}
		}
		KernelBasisDistance distance(&expansion,100);

		RealMatrix pointBatch(100,2);
		RealVector point(2*100);
		for(std::size_t i = 0; i != 100; ++i){
			point(2*i) = dataset.element(i).input(0);
			point(2*i+1) = dataset.element(i).input(1);
			row(pointBatch,i) = dataset.element(i).input;
		}
		RealMatrix K = kernel(pointBatch,pointBatch);
		//we omit the expensive to compute constant term in the distance, thus we compute it here to
		//assure that the corrected value is (very close to) zero.
		double correction = inner_prod(column(expansion.alpha(),0),prod(K,column(expansion.alpha(),0)));
		correction += inner_prod(column(expansion.alpha(),1),prod(K,column(expansion.alpha(),1)));
		correction += inner_prod(column(expansion.alpha(),2),prod(K,column(expansion.alpha(),2)));

		BOOST_CHECK_SMALL(2*distance(point) + correction, 1.e-10);
		RealVector derivative;
		BOOST_CHECK_SMALL(2*distance.evalDerivative(point,derivative) + correction, 1.e-10);
		BOOST_CHECK_SMALL(norm_2(derivative) / 200, 1.e-9);
	}
}

//test that checks that the result with respect to a linear kernel is correct
class NormalDistributedPoints:public DataDistribution<RealVector>
{
public:
	void draw(RealVector& input) const{
		input.resize(30);
		for(std::size_t i = 0; i != 30; ++i){
			input(i) = Rng::gauss(0,1);
		}
	}
};
BOOST_AUTO_TEST_CASE( ObjectiveFunctions_Value_Linear )
{
	for(std::size_t trial = 0; trial != 10; ++trial){
		NormalDistributedPoints problem;
		Data<RealVector> dataset = problem.generateDataset(100,10);
		LinearKernel<> kernel;
		KernelExpansion<RealVector> expansion(&kernel,dataset,false,1);
		RealVector alpha(100);
		for(std::size_t i = 0; i != 100; ++i){
			alpha(i) = expansion.alpha()(i,0) = Rng::gauss(0,1);
		}

		//construct the target vector in explicit form
		RealVector optimalPoint(30,0);
		for(std::size_t i = 0; i != 100; ++i){
			noalias(optimalPoint) += alpha(i) * dataset.element(i);
		}

		KernelBasisDistance distance(&expansion,20);

		for(std::size_t test = 0; test != 10; ++test){
			//create input point as well as batch version
			Data<RealVector> dataset = problem.generateDataset(20,20);
			RealMatrix& pointBatch = dataset.batch(0);
			RealVector point(30*20);
			for(std::size_t i = 0; i != 20; ++i){
				noalias(subrange(point,i*30,(i+1)*30)) = row(pointBatch,i);
			}

			//find optimal solution
			RealMatrix K = prod(pointBatch,trans(pointBatch));
			RealVector linear = prod(pointBatch,optimalPoint);
			RealVector beta;
			blas::solveSymmSystem<blas::SolveAXB>(K,beta,linear);
			RealVector optimalApproximation = prod(beta,pointBatch);

			double error = distanceSqr(optimalApproximation,optimalPoint)-norm_sqr(optimalPoint);
			error/=2;
			BOOST_CHECK_CLOSE(distance(point),error,1.e-10);
		}
	}
}

BOOST_AUTO_TEST_CASE( ObjectiveFunctions_KernelBasisDistance_Derivative_Linear)
{
	for(std::size_t trial = 0; trial != 10; ++trial){
		NormalDistributedPoints problem;
		Data<RealVector> dataset = problem.generateDataset(100,10);
		LinearKernel<> kernel;
		KernelExpansion<RealVector> expansion(&kernel,dataset,false,1);
		for(std::size_t i = 0; i != 100; ++i){
			for(std::size_t j = 0; j != 1; ++j){
				expansion.alpha()(i,j) = Rng::gauss(0,1);
			}
		}
		KernelBasisDistance distance(&expansion,10);

		for(std::size_t test = 0; test != 10; ++test){
			RealVector point(30*10);
			for(std::size_t i = 0; i != point.size(); ++i){
				point(i) = Rng::gauss(0,1);
			}

			testDerivative(distance,point,1.e-6,0,0.1);
		}
	}
}

BOOST_AUTO_TEST_CASE( ObjectiveFunctions_KernelBasisDistance_Derivative_Gaussian)
{
	for(std::size_t trial = 0; trial != 10; ++trial){
		NormalDistributedPoints problem;
		Data<RealVector> dataset = problem.generateDataset(100,10);
		GaussianRbfKernel<> kernel(0.5);
		KernelExpansion<RealVector> expansion(&kernel,dataset,false,1);
		for(std::size_t i = 0; i != 100; ++i){
			for(std::size_t j = 0; j != 1; ++j){
				expansion.alpha()(i,j) = Rng::gauss(0,1);
			}
		}
		KernelBasisDistance distance(&expansion,10);

		for(std::size_t test = 0; test != 10; ++test){
			RealVector point(30*10);
			for(std::size_t i = 0; i != point.size(); ++i){
				point(i) = Rng::gauss(0,1);
			}

			testDerivative(distance,point,1.e-6,0,0.1);
		}
	}
}
