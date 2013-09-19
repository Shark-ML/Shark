//===========================================================================
/*!
 *
 *  \brief Unit test for the leave one out error for C-SVMs.
 *
 *
 *  \author  T. Glasmachers
 *  \date    2011
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

#include <shark/ObjectiveFunctions/LooErrorCSvm.h>
#include <shark/ObjectiveFunctions/LooError.h>
#include <shark/Models/Kernels/LinearKernel.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Algorithms/Trainers/CSvmTrainer.h>

#include <shark/Data/DataDistribution.h>


#define BOOST_TEST_MODULE ObjectiveFunctions_LooErrorCSvm
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

using namespace shark;




BOOST_AUTO_TEST_CASE( ObjectiveFunctions_LooErrorCSvm_Simple )
{
	std::cout<<"testing simple Test"<<std::endl;
	std::vector<RealVector> inputs(5, RealVector(2));
	inputs[0](0) = 0.0;
	inputs[0](1) = 0.0;
	inputs[1](0) = 1.0;
	inputs[1](1) = 0.0;
	inputs[2](0) = 0.0;
	inputs[2](1) = 1.0;
	inputs[3](0) = 1.0;
	inputs[3](1) = 1.0;
	inputs[4](0) = 0.5;
	inputs[4](1) = 0.4;
	std::vector<unsigned int> targets(5);
	targets[0] = 0;
	targets[1] = 0;
	targets[2] = 1;
	targets[3] = 1;
	targets[4] = 1;
	ClassificationDataset dataset = createLabeledDataFromRange(inputs, targets);

	// SVM setup
	LinearKernel<> kernel;
	double C = 1e100;  // hard margin
	CSvmTrainer<RealVector> trainer(&kernel, C,true);

	// efficiently computed loo error
	LooErrorCSvm<RealVector> loosvm(dataset, &kernel, true);
	double value = loosvm.eval(trainer.parameterVector());

	// brute force computation
	ZeroOneLoss<unsigned int> loss;
	KernelClassifier<RealVector> ke;
	LooError<KernelClassifier<RealVector>,unsigned int> loo(dataset, &ke, &trainer, &loss);
	double standardLoo = loo.eval();

	// compare to brute force computation
	// and to geometric intuition (3 out of
	// 5 are errors)
	BOOST_CHECK_SMALL(value - standardLoo, 1e-10);
	BOOST_CHECK_SMALL(value - 0.6, 1e-10);
}

BOOST_AUTO_TEST_CASE( ObjectiveFunctions_LooErrorCSvm_NoBias_Simple )
{
	std::cout<<"testing simple Test without bias"<<std::endl;
	std::vector<RealVector> inputs(5, RealVector(2));
	inputs[0](0) = 0.0;
	inputs[0](1) = 0.0;
	inputs[1](0) = 1.0;
	inputs[1](1) = 0.0;
	inputs[2](0) = 0.0;
	inputs[2](1) = 1.0;
	inputs[3](0) = 1.0;
	inputs[3](1) = 1.0;
	inputs[4](0) = 0.5;
	inputs[4](1) = 0.4;
	std::vector<unsigned int> targets(5);
	targets[0] = 0;
	targets[1] = 0;
	targets[2] = 1;
	targets[3] = 1;
	targets[4] = 1;
	ClassificationDataset dataset = createLabeledDataFromRange(inputs, targets);

	// SVM setup
	LinearKernel<> kernel;
	double C = 1e100;  // hard margin
	CSvmTrainer<RealVector> trainer(&kernel, C,false);

	// efficiently computed loo error
	LooErrorCSvm<RealVector> loosvm(dataset, &kernel, false);
	double value = loosvm.eval(trainer.parameterVector());

	// brute force computation
	ZeroOneLoss<unsigned int> loss;
	KernelClassifier<RealVector> ke;
	LooError<KernelClassifier<RealVector>,unsigned int> loo(dataset, &ke, &trainer, &loss);
	double standardLoo = loo.eval();

	// compare to brute force computation
	BOOST_CHECK_SMALL(value - standardLoo, 1e-10);
}

BOOST_AUTO_TEST_CASE( ObjectiveFunctions_LooErrorCSvm_Chessboard )
{
	std::cout<<"testing test on chessboard"<<std::endl;
	Chessboard problem;
	ClassificationDataset dataset = problem.generateDataset(100);

	// SVM setup
	GaussianRbfKernel<> kernel;
	double C = 10;
	CSvmTrainer<RealVector> trainer(&kernel, C,true);
	
	RealVector parameters = trainer.parameterVector();
	
	// brute force computation
	ZeroOneLoss<unsigned int> loss;
	KernelClassifier<RealVector> ke;
	LooError<KernelClassifier<RealVector>,unsigned int> loo(dataset, &ke, &trainer, &loss);
	double standardLoo = loo.eval();

	// efficiently computed loo error
	LooErrorCSvm<RealVector> loosvm(dataset, &kernel, true);
	double value = loosvm.eval(parameters);

	// compare to brute force computation
	// and to geometric intuition
	BOOST_CHECK_SMALL(value - standardLoo, 1e-10);
}

BOOST_AUTO_TEST_CASE( ObjectiveFunctions_LooErrorCSvm_Chessboard_NoBias )
{
	std::cout<<"testing test on chessboard without bias"<<std::endl;
	Chessboard problem;
	ClassificationDataset dataset = problem.generateDataset(100);

	// SVM setup
	GaussianRbfKernel<> kernel;
	double C = 10;
	CSvmTrainer<RealVector> trainer(&kernel, C,false);
	
	RealVector parameters = trainer.parameterVector();
	
	// efficiently computed loo error
	std::cout<<"efficient"<<std::endl;
	LooErrorCSvm<RealVector> loosvm(dataset, &kernel, false);
	double value = loosvm.eval(parameters);
	
	std::cout<<"\n\nbrute force"<<std::endl;
	// brute force computation
	ZeroOneLoss<unsigned int> loss;
	KernelClassifier<RealVector> ke;
	LooError<KernelClassifier<RealVector>,unsigned int> loo(dataset, &ke, &trainer, &loss);
	double standardLoo = loo.eval();

	// compare to brute force computation
	// and to geometric intuition
	BOOST_CHECK_SMALL(value - standardLoo, 1e-10);
}