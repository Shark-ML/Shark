//===========================================================================
/*!
 *  \brief test case for the various multi-class SVM trainers
 *
 *
 *  \author T. Glasmachers
 *  \date 2011
 *
 *
 *  <BR><HR>
 *  This file is part of Shark. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 3, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, see <http://www.gnu.org/licenses/>.
 *
 */
//===========================================================================
#define BOOST_TEST_MODULE ALGORITHMS_TRAINERS_MCSVMTRAINER
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>


#define SHARK_COUNT_KERNEL_LOOKUPS //in this example, we want to count the kernel lookups


#include <shark/Algorithms/Trainers/McSvmOVATrainer.h>
#include <shark/Algorithms/Trainers/McSvmMMRTrainer.h>
#include <shark/Algorithms/Trainers/McSvmCSTrainer.h>
#include <shark/Algorithms/Trainers/McSvmWWTrainer.h>
#include <shark/Algorithms/Trainers/McSvmLLWTrainer.h>
#include <shark/Algorithms/Trainers/McSvmADMTrainer.h>
#include <shark/Algorithms/Trainers/McSvmATSTrainer.h>
#include <shark/Algorithms/Trainers/McSvmATMTrainer.h>
#include <shark/Models/Kernels/LinearKernel.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Data/DataDistribution.h>

using namespace shark;


/*
// verbose version:
#define CHECK_ALPHAS(input, is, should ) { \
	std::cout << "    # iterations: " << trainer.solutionProperties().iterations << std::endl; \
	for (unsigned int i=0; i<is.size(); i++) printf("alpha[%d]:  %g  %g\n", i, is(i), should[i]); \
	double w1[3][2] = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}; \
	double w2[3][2] = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}; \
	for (unsigned int i=0; i<is.size(); i++) { \
		w1[i % 3][0] += is(i) * input[i / 3](0); \
		w1[i % 3][1] += is(i) * input[i / 3](1); \
		w2[i % 3][0] += should[i] * input[i / 3](0); \
		w2[i % 3][1] += should[i] * input[i / 3](1); \
	} \
	printf("w[1] = (%g, %g)   (%g, %g)\n", w1[0][0], w1[0][1], w2[0][0], w2[0][1]); \
	printf("w[2] = (%g, %g)   (%g, %g)\n", w1[1][0], w1[1][1], w2[1][0], w2[1][1]); \
	printf("w[3] = (%g, %g)   (%g, %g)\n", w1[2][0], w1[2][1], w2[2][0], w2[2][1]); \
	BOOST_CHECK_SMALL((w1[0][0] - w1[1][0]) - (w2[0][0] - w2[1][0]) , 1e-4); \
	BOOST_CHECK_SMALL((w1[0][1] - w1[1][1]) - (w2[0][1] - w2[1][1]) , 1e-4); \
	BOOST_CHECK_SMALL((w1[1][0] - w1[2][0]) - (w2[1][0] - w2[2][0]) , 1e-4); \
	BOOST_CHECK_SMALL((w1[1][1] - w1[2][1]) - (w2[1][1] - w2[2][1]) , 1e-4); \
}
*/

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
		KernelExpansion<RealVector> svm(false, 3);
		McSvmOVATrainer<RealVector> trainer(&kernel, 0.5);
		std::cout << "testing " << trainer.name() << std::endl;
		trainer.sparsify() = false;
		trainer.stoppingCondition().minAccuracy = 1e-8;
		trainer.train(svm, dataset);
		std::cout << "kernel computations: " << trainer.accessCount() << std::endl;
		CHECK_ALPHAS(input,svm.parameterVector(), alpha);
	}

	// MMR-SVM
	{
		double alpha[15] = {0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
		KernelExpansion<RealVector> svm(false, 3);
		McSvmMMRTrainer<RealVector> trainer(&kernel, 0.5);
		std::cout << "testing " << trainer.name() << std::endl;
		trainer.sparsify() = false;
		trainer.stoppingCondition().minAccuracy = 1e-8;
		trainer.train(svm, dataset);
		std::cout << "kernel computations: " << trainer.accessCount() << std::endl;
		CHECK_ALPHAS(input,svm.parameterVector(), alpha);
	}

	// WW-SVM
	{
		double alpha[15] = {0.4375, -0.25, -0.1875, -0.25, 0.4375, -0.1875, -0.25, -0.25, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
		KernelExpansion<RealVector> svm(false, 3);
		McSvmWWTrainer<RealVector> trainer(&kernel, 0.5);
		std::cout << "testing " << trainer.name() << std::endl;
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
		KernelExpansion<RealVector> svm(false, 3);
		McSvmCSTrainer<RealVector> trainer(&kernel, 0.5);
		std::cout << "testing " << trainer.name() << std::endl;
		trainer.sparsify() = false;
		trainer.stoppingCondition().minAccuracy = 1e-8;
		trainer.train(svm, dataset);
		std::cout << "kernel computations: " << trainer.accessCount() << std::endl;
		CHECK_ALPHAS(input,svm.parameterVector(), alpha);
	}

	// LLW-SVM
	{
		double alpha[15] = {0.0, -0.5, -0.5, -0.5, 0.0, -0.5, -0.5, -0.5, 0.0, -0.25, -0.25, 0.0, 0.0, 0.0, 0.0};
		KernelExpansion<RealVector> svm(false, 3);
		McSvmLLWTrainer<RealVector> trainer(&kernel, 0.5);
		std::cout << "testing " << trainer.name() << std::endl;
		trainer.sparsify() = false;
		trainer.stoppingCondition().minAccuracy = 1e-8;
		trainer.train(svm, dataset);
		std::cout << "kernel computations: " << trainer.accessCount() << std::endl;
		CHECK_ALPHAS(input,svm.parameterVector(), alpha);
	}

	// ADM-SVM
	{
		double alpha[15] = {0.0, -0.4375, -0.0625, -0.4375, 0.0, -0.0625, 0.0, -0.5, 0.0, -0.375, -0.125, 0.0, 0.0, 0.0, 0.0};
		KernelExpansion<RealVector> svm(false, 3);
		McSvmADMTrainer<RealVector> trainer(&kernel, 0.5);
		std::cout << "testing " << trainer.name() << std::endl;
		trainer.sparsify() = false;
		trainer.stoppingCondition().minAccuracy = 1e-8;
		trainer.train(svm, dataset);
		std::cout << "kernel computations: " << trainer.accessCount() << std::endl;
		CHECK_ALPHAS(input,svm.parameterVector(), alpha);
	}

	// ATS-SVM
	{
		double alpha[15] = {0.0, -0.5, 0.0, -0.5, 0.0, 0.0, -0.5, -0.5, 0.5, -0.5, -0.5, 0.0, 0.0, 0.0, 0.0};
		KernelExpansion<RealVector> svm(false, 3);
		McSvmATSTrainer<RealVector> trainer(&kernel, 0.5);
		std::cout << "testing " << trainer.name() << std::endl;
		trainer.sparsify() = false;
		trainer.stoppingCondition().minAccuracy = 1e-8;
		trainer.train(svm, dataset);
		std::cout << "kernel computations: " << trainer.accessCount() << std::endl;
		CHECK_ALPHAS(input,svm.parameterVector(), alpha);
	}

	// ATM-SVM
	{
		double alpha[15] = {0.0, -0.4375, -0.0625, -0.4375, 0.0, -0.0625, 0.0, -0.5, 0.0, -0.375, -0.125, 0.0, 0.0, 0.0, 0.0};
		KernelExpansion<RealVector> svm(false, 3);
		McSvmATMTrainer<RealVector> trainer(&kernel, 0.5);
		std::cout << "testing " << trainer.name() << std::endl;
		trainer.sparsify() = false;
		trainer.stoppingCondition().minAccuracy = 1e-8;
		trainer.train(svm, dataset);
		std::cout << "kernel computations: " << trainer.accessCount() << std::endl;
		CHECK_ALPHAS(input,svm.parameterVector(), alpha);

	}
}
BOOST_AUTO_TEST_CASE( CSVM_TRAINER_ITERATIVE_BIAS_TEST )
{
	//Chessboard problem;
	CircleInSquare problem(2,0.0,true);
	ClassificationDataset dataset = problem.generateDataset(1000);
	

	//GaussianRbfKernel<> kernel(0.5);
	LinearKernel<> kernel;
	
	KernelExpansion<RealVector> svmTruth(false);
	KernelExpansion<RealVector> svmTruth2(true);
	KernelExpansion<RealVector> svmTruth3(true);
	KernelExpansion<RealVector> svmTest1(false,2);
	KernelExpansion<RealVector> svmTest2(true,2);
	
	double C = 3;

	{//train as a binary svm problem
		CSvmTrainer<RealVector> trainerTruth(&kernel, C);
		trainerTruth.sparsify() = false;
		trainerTruth.shrinking() = false;
		trainerTruth.stoppingCondition().minAccuracy = 1e-4;
		trainerTruth.train(svmTruth, dataset);
		std::cout<<"a"<<std::endl;
		trainerTruth.train(svmTruth2, dataset);
		trainerTruth.setUseIterativeBiasComputation(true);
		trainerTruth.train(svmTruth3, dataset);
		std::cout<<"b"<<std::endl;
	}
	
	//train as multiclass svm problem
	McSvmWWTrainer<RealVector> trainer(&kernel, 2*C);
	trainer.sparsify() = false;
	trainer.shrinking() = false;
	trainer.stoppingCondition().minAccuracy = 1e-4;
	trainer.train(svmTest1, dataset);
	std::cout<<"c"<<std::endl;
	trainer.train(svmTest2, dataset);
	std::cout<<"d"<<std::endl;
	
	
	//compare bias
	//BOOST_CHECK_CLOSE(svmTest.offset(0),svmTruth.offset(0),1.e-2);
	ZeroOneLoss<unsigned int, RealVector> loss;
	std::cout<<"CSVM: "<<loss.eval(dataset.labels(),svmTruth(dataset.inputs()))<<std::endl;
	std::cout<<"CSVM with bias : "<<loss.eval(dataset.labels(),svmTruth2(dataset.inputs()))<<" "<<svmTruth2.offset()<<std::endl;
	std::cout<<"CSVM with bias iterative : "<<loss.eval(dataset.labels(),svmTruth3(dataset.inputs()))<<" "<<svmTruth3.offset()<<std::endl;
	std::cout<<"WW: "<<loss.eval(dataset.labels(),svmTest1(dataset.inputs()))<<std::endl;
	std::cout<<"WW with bias: "<<loss.eval(dataset.labels(),svmTest2(dataset.inputs()))<<" "<<svmTest2.offset()<<std::endl;
}

