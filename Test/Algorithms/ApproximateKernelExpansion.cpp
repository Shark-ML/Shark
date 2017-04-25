//===========================================================================
/*!
 * 
 *
 * \brief       unit test for the Kernel Basis Distance error function.
 * 
 *
 * \author      O. Krause
 * \date        2017
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

#include <shark/Algorithms/ApproximateKernelExpansion.h>

#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Models/Kernels/LinearKernel.h>
#include <shark/Models/Kernels/KernelHelpers.h>
#include <shark/Data/DataDistribution.h>

#define BOOST_TEST_MODULE Algorithms_ApproximateKernelExpansion
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include "../ObjectiveFunctions/TestObjectiveFunction.h"

using namespace shark;

//Sanity check that checks that the error of the exact same basis is minimal and the derivative is small.
BOOST_AUTO_TEST_SUITE (Algorithms_ApproximateKernelExpansion)

//in the linear case we know that a single basis vector per target is enough
//note that the solution is not unique in the kernel expansion as any 3-basis containing the three target
//vectors is feasible
//this the case on a 2 d problem as long as two vectors in the basis are linearly independent. therefore this
// test checks whether the computation of beta is correct.
BOOST_AUTO_TEST_CASE( Solution_Linear ){
	for(std::size_t trial = 0; trial != 10; ++trial){
		Chessboard problem;
		LabeledData<RealVector, unsigned int> dataset = problem.generateDataset(50,10);
		LinearKernel<> kernel;
		KernelExpansion<RealVector> expansion(&kernel,dataset.inputs(),false,3);
		RealMatrix WTrue(3,2,0.0);
		for(std::size_t i = 0; i != 50; ++i){
			for(std::size_t j = 0; j != 3; ++j){
				expansion.alpha()(i,j) = random::gauss(random::globalRng, 0,1);
			}
			noalias(WTrue) += outer_prod(row(expansion.alpha(),i), dataset.element(i).input);
		}
		//compute approximation
		KernelExpansion<RealVector> approx=approximateKernelExpansion(random::globalRng, expansion,2);
		
		BOOST_REQUIRE_EQUAL(approx.alpha().size1(), 2);
		BOOST_REQUIRE_EQUAL(approx.alpha().size2(), 3);
		//compute weight vectors of the approximation
		RealMatrix W = trans(approx.alpha()) % approx.basis().batch(0);
		
		BOOST_CHECK_SMALL(norm_inf(WTrue-W),1.e-4);
	}
}

//in the case of only a single basis vector with a linear kernel, the basis vector is congruent
//to the largest eigenvector of the covariance matrix formed by the true linear weights
BOOST_AUTO_TEST_CASE( Solution_Linear_Single ){
	for(std::size_t trial = 0; trial != 10; ++trial){
		Chessboard problem;
		LabeledData<RealVector, unsigned int> dataset = problem.generateDataset(50,10);
		LinearKernel<> kernel;
		KernelExpansion<RealVector> expansion(&kernel,dataset.inputs(),false,3);
		RealMatrix WTrue(3,2,0.0);
		for(std::size_t i = 0; i != 50; ++i){
			for(std::size_t j = 0; j != 3; ++j){
				expansion.alpha()(i,j) = random::gauss(random::globalRng, 0,1);
			}
			noalias(WTrue) += outer_prod(row(expansion.alpha(),i), dataset.element(i).input);
		}
		RealMatrix C = trans(WTrue) % WTrue;
		blas::symm_eigenvalue_decomposition<RealMatrix> eigen(C);
		//compute optimal basis and best W in that basis
		RealVector truthZ = column(eigen.Q(),0);
		RealMatrix truthApproxW = outer_prod(eval_block(WTrue % truthZ), truthZ); 
		//compute approximation
		KernelExpansion<RealVector> approx=approximateKernelExpansion(random::globalRng, expansion,1);
		
		BOOST_REQUIRE_EQUAL(approx.alpha().size1(), 1);
		BOOST_REQUIRE_EQUAL(approx.alpha().size2(), 3);
		RealVector Z = approx.basis().element(0)/norm_2(approx.basis().element(0));
		RealMatrix approxW = trans(approx.alpha()) % approx.basis().batch(0);
		
		BOOST_CHECK_SMALL(norm_inf(truthZ-Z),1.e-4);
		BOOST_CHECK_SMALL(norm_inf(truthApproxW-approxW),1.e-3);
	}
}


class DistanceTest : public SingleObjectiveFunction
{
public:
	DistanceTest(KernelExpansion<RealVector> const* kernelExpansion,std::size_t numApproximatingVectors)
	:mep_expansion(kernelExpansion),m_numApproximatingVectors(numApproximatingVectors){
		m_features|=HAS_FIRST_DERIVATIVE;
	}

	/// \brief Returns the name of the class
	std::string name() const
	{ return "DistanceTest"; }
	/// \brief Returns the number of variables of the function.
	std::size_t numberOfVariables()const{
		return m_numApproximatingVectors  * dataDimension(mep_expansion->basis());
	}
	
	double eval(RealVector const& input) const{
		Data<RealVector> const& expansionBasis = mep_expansion->basis();
		AbstractKernelFunction<RealVector> const& kernel = *mep_expansion->kernel();
		RealMatrix const& alpha = mep_expansion->alpha();
		std::size_t dim = dataDimension(expansionBasis);

		Data<RealVector> basis;
		basis.push_back(to_matrix(input, m_numApproximatingVectors,dim));
		RealMatrix Kz=calculateRegularizedKernelMatrix(kernel, basis);
		RealMatrix Kza=calculateMixedKernelMatrix(kernel,basis, expansionBasis);
		
		blas::symm_eigenvalue_decomposition<RealMatrix> eigen(Kz);
		RealMatrix t = trans(Kza) % inv(Kz,blas::symm_semi_pos_def()) % Kza % alpha;
		return -0.5 * frobenius_prod(t,alpha);
	}

private:
	KernelExpansion<RealVector> const* mep_expansion;     ///< kernel expansion to approximate
	std::size_t m_numApproximatingVectors; ///< number of vectors in the basis
};

BOOST_AUTO_TEST_CASE( Solution_Gaussian ){
	for(std::size_t trial = 0; trial != 10; ++trial){
		Chessboard problem;
		LabeledData<RealVector, unsigned int> dataset = problem.generateDataset(10,10);
		GaussianRbfKernel<> kernel(0.5);
		KernelExpansion<RealVector> expansion(&kernel,dataset.inputs(),false,3);
		for(std::size_t i = 0; i != 10; ++i){
			for(std::size_t j = 0; j != 3; ++j){
				expansion.alpha()(i,j) = random::gauss(random::globalRng, 0,1);
			}
		}
		//compute approximation
		KernelExpansion<RealVector> approx=approximateKernelExpansion(random::globalRng, expansion,4,1.e-4);
		BOOST_REQUIRE_EQUAL(approx.alpha().size1(), 4);
		BOOST_REQUIRE_EQUAL(approx.alpha().size2(), 3);
		
		RealMatrix Kz=calculateRegularizedKernelMatrix(kernel, approx.basis());
		blas::symm_eigenvalue_decomposition<RealMatrix> eigen(Kz);
		std::cout<<min(eigen.D())<<" "<<max(eigen.D())<<std::endl;
		
		
		DistanceTest test(&expansion,4);
		RealVector point = to_vector(approx.basis().batch(0));
		RealVector derivativeEst = estimateDerivative(test,point,1.e-5);
		
		//~ std::cout<<derivativeEst<<std::endl;
		std::cout<<norm_sqr(derivativeEst)<<std::endl;
		BOOST_CHECK_SMALL(norm_sqr(derivativeEst), 5.e-4);
	}
}

BOOST_AUTO_TEST_SUITE_END()
