//===========================================================================
/*!
 * 
 *
 * \brief       test case for the various multi-class SVM trainers
 * 
 * 
 * 
 *
 * \author      T. Glasmachers
 * \date        2011
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
#define BOOST_TEST_MODULE ALGORITHMS_TRAINERS_MCSVMTRAINER
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>


#define SHARK_COUNT_KERNEL_LOOKUPS //in this example, we want to count the kernel lookups


#include <shark/Algorithms/Trainers/CSvmTrainer.h>
#include <shark/Models/Kernels/LinearKernel.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Data/DataDistribution.h>

using namespace shark;


template<class T, class U>
void CHECK_ALPHAS(std::vector<RealVector> const& input, T const& is, U const& should ) {
	std::cout<<input.size()<<" "<<is.size()<<" "<<std::endl;
	double w1[3][2] = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
	double w2[3][2] = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
	for (unsigned int i=0; i<is.size(); i++) {
		w1[i % 3][0] += is(i) * input[i / 3](0);
		w1[i % 3][1] += is(i) * input[i / 3](1);
		w2[i % 3][0] += should[i] * input[i / 3](0);
		w2[i % 3][1] += should[i] * input[i / 3](1);
	}
	BOOST_CHECK_SMALL((w1[0][0] - w1[1][0]) - (w2[0][0] - w2[1][0]) , 1e-4);
	BOOST_CHECK_SMALL((w1[0][1] - w1[1][1]) - (w2[0][1] - w2[1][1]) , 1e-4);
	BOOST_CHECK_SMALL((w1[1][0] - w1[2][0]) - (w2[1][0] - w2[2][0]) , 1e-4);
	BOOST_CHECK_SMALL((w1[1][1] - w1[2][1]) - (w2[1][1] - w2[2][1]) , 1e-4);
}


// This test case checks the resulting model of
// training eight multi-class SVMs on a minimal
// test case.
BOOST_AUTO_TEST_SUITE (Algorithms_Trainers_McSvmTrainer)

BOOST_AUTO_TEST_CASE( MCSVM_TRAINER_TEST )
{
	// simple 5-point dataset, three classes
	std::vector<RealVector> input(5);
	std::vector<unsigned int> target(5);
	for (std::size_t i=0; i<5; i++) input[i].resize(2);
	input[0](0) = -1.0; input[0](1) = -1.0; target[0] = 0;
	input[1](0) =  1.0; input[1](1) = -1.0; target[1] = 1;
	input[2](0) =  0.0; input[2](1) =  1.0; target[2] = 2;
	input[3](0) =  0.0; input[3](1) =  2.0; target[3] = 2;
	input[4](0) =  0.0; input[4](1) = 99.0; target[4] = 2;
	ClassificationDataset dataset = createLabeledDataFromRange(input, target);

	LinearKernel<> kernel;

	// OVA-SVM
	{
		double alpha[15] = {0.0, -0.5, -0.5, -0.5, 0.0, -0.5, -0.5, -0.5, 0.0, -0.25, -0.25, 0.0, 0.0, 0.0, 0.0};
		KernelClassifier<RealVector> svm;
		CSvmTrainer<RealVector> trainer(&kernel, 0.5, false);
		trainer.setMcSvmType(McSvm::OVA);
		std::cout << "testing OVA" << std::endl;
		trainer.sparsify() = false;
		trainer.stoppingCondition().minAccuracy = 1e-8;
		trainer.train(svm, dataset);
		std::cout << "kernel computations: " << trainer.accessCount() << std::endl;
		CHECK_ALPHAS(input,svm.parameterVector(), alpha);
	}

	// MMR-SVM
	{
		double alpha[15] = {0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
		KernelClassifier<RealVector> svm;
		CSvmTrainer<RealVector> trainer(&kernel, 0.5, false);
		trainer.setMcSvmType(McSvm::MMR);
		std::cout << "testing MMR" << std::endl;
		trainer.sparsify() = false;
		trainer.stoppingCondition().minAccuracy = 1e-8;
		trainer.train(svm, dataset);
		std::cout << "kernel computations: " << trainer.accessCount() << std::endl;
		CHECK_ALPHAS(input,svm.parameterVector(), alpha);
	}

	// WW-SVM
	{
		double alpha[15] = {0.4375, -0.25, -0.1875, -0.25, 0.4375, -0.1875, -0.25, -0.25, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
		KernelClassifier<RealVector> svm;
		CSvmTrainer<RealVector> trainer(&kernel, 0.5, false);
		trainer.setMcSvmType(McSvm::WW);
		std::cout << "testing WW" << std::endl;
		trainer.sparsify() = false;
		trainer.shrinking() = false;
		trainer.stoppingCondition().minAccuracy = 1e-8;
		trainer.train(svm, dataset);
		
		std::cout << "kernel computations: " << trainer.accessCount() << std::endl;
		CHECK_ALPHAS(input,svm.parameterVector(), alpha);
	}

	// CS-SVM
	{
		double alpha[15] = {0.25, -0.25, 0.0, -0.25, 0.25, 0.0, -0.000163, -0.25, 0.250163, -0.1666, -0.04166, 0.2083, 0.0, 0.0, 0.0};
		KernelClassifier<RealVector> svm;
		CSvmTrainer<RealVector> trainer(&kernel, 0.5, false);
		trainer.setMcSvmType(McSvm::CS);
		std::cout << "testing CS" << std::endl;
		trainer.sparsify() = false;
		trainer.stoppingCondition().minAccuracy = 1e-8;
		trainer.train(svm, dataset);
		std::cout << "kernel computations: " << trainer.accessCount() << std::endl;
		CHECK_ALPHAS(input,svm.parameterVector(), alpha);
	}

	// LLW-SVM
	{
		double alpha[15] = {0.0, -0.5, -0.5, -0.5, 0.0, -0.5, -0.5, -0.5, 0.0, -0.25, -0.25, 0.0, 0.0, 0.0, 0.0};
		KernelClassifier<RealVector> svm;
		CSvmTrainer<RealVector> trainer(&kernel, 0.5, false);
		trainer.setMcSvmType(McSvm::LLW);
		std::cout << "testing LLW" << std::endl;
		trainer.sparsify() = false;
		trainer.stoppingCondition().minAccuracy = 1e-8;
		trainer.train(svm, dataset);
		std::cout << "kernel computations: " << trainer.accessCount() << std::endl;
		CHECK_ALPHAS(input,svm.parameterVector(), alpha);
	}

	// ADM-SVM
	{
		double alpha[15] = {0.0, -0.4375, -0.0625, -0.4375, 0.0, -0.0625, 0.0, -0.5, 0.0, -0.375, -0.125, 0.0, 0.0, 0.0, 0.0};
		KernelClassifier<RealVector> svm;
		CSvmTrainer<RealVector> trainer(&kernel, 0.5, false);
		trainer.setMcSvmType(McSvm::ADM);
		std::cout << "testing ADM" << std::endl;
		trainer.sparsify() = false;
		trainer.stoppingCondition().minAccuracy = 1e-8;
		trainer.train(svm, dataset);
		std::cout << "kernel computations: " << trainer.accessCount() << std::endl;
		CHECK_ALPHAS(input,svm.parameterVector(), alpha);
	}

	// ATS-SVM
	{
		double alpha[15] = {0.0, -0.5, 0.0, -0.5, 0.0, 0.0, -0.5, -0.5, 0.5, -0.5, -0.5, 0.0, 0.0, 0.0, 0.0};
		KernelClassifier<RealVector> svm;
		CSvmTrainer<RealVector> trainer(&kernel, 0.5, false);
		trainer.setMcSvmType(McSvm::ATS);
		std::cout << "testing ATS" << std::endl;
		trainer.sparsify() = false;
		trainer.stoppingCondition().minAccuracy = 1e-8;
		trainer.train(svm, dataset);
		std::cout << "kernel computations: " << trainer.accessCount() << std::endl;
		CHECK_ALPHAS(input,svm.parameterVector(), alpha);
	}

	// ATM-SVM
	{
		double alpha[15] = {0.0, -0.4375, -0.0625, -0.4375, 0.0, -0.0625, 0.0, -0.5, 0.0, -0.375, -0.125, 0.0, 0.0, 0.0, 0.0};
		KernelClassifier<RealVector> svm;
		CSvmTrainer<RealVector> trainer(&kernel, 0.5, false);
		trainer.setMcSvmType(McSvm::ATM);
		std::cout << "testing ATM" << std::endl;
		trainer.sparsify() = false;
		trainer.stoppingCondition().minAccuracy = 1e-8;
		trainer.train(svm, dataset);
		std::cout << "kernel computations: " << trainer.accessCount() << std::endl;
		CHECK_ALPHAS(input,svm.parameterVector(), alpha);
	}

	// Reinforced-SVM
	{
		double alpha[15] = {0.5, -0.5, 0.0, -0.5, 0.5, 0.0, -0.5, -0.5, 0.5, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0};
		KernelClassifier<RealVector> svm;
		CSvmTrainer<RealVector> trainer(&kernel, 0.5, false);
		trainer.setMcSvmType(McSvm::ReinforcedSvm);
		std::cout << "testing ReinforcedSVM" << std::endl;
		trainer.sparsify() = false;
		trainer.stoppingCondition().minAccuracy = 1e-8;
		trainer.train(svm, dataset);
		std::cout << "kernel computations: " << trainer.accessCount() << std::endl;
		CHECK_ALPHAS(input,svm.parameterVector(), alpha);
	}
}

BOOST_AUTO_TEST_SUITE_END()
