//===========================================================================
/*!
 * 
 *
 * \brief       test case for the RankingSvmTrainer
 * 
 * 
 * 
 *
 * \author      T. Glasmachers
 * \date        -
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
#define BOOST_TEST_MODULE ALGORITHMS_TRAINERS_RANKINGSVMTRAINER
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/Trainers/RankingSvmTrainer.h>
#include <shark/Models/Kernels/LinearKernel.h>


using namespace shark;

// This test case consists of training SVMs with
// analytically computable solution. This known
// solution is used to validate the trainer.
BOOST_AUTO_TEST_SUITE (Algorithms_Trainers_RankingSvmTrainer)

BOOST_AUTO_TEST_CASE( RANKINGSVM_TRAINER_SIMPLE_TEST )
{
	// simple 5-point dataset
	std::vector<RealVector> input(5);
	std::vector<unsigned int> target(5);
	size_t i;
	for (i=0; i<5; i++) input[i].resize(2);
	input[0](0) =  0.0; input[0](1) =  0.0; target[0] = 1;
	input[1](0) =  1.0; input[1](1) =  2.0; target[1] = 2;
	input[2](0) = -1.0; input[2](1) = -8.0; target[2] = 0;
	input[3](0) = -1.0; input[3](1) =  8.0; target[3] = 0;
	input[4](0) =  2.0; input[4](1) = -3.0; target[4] = 3;
	ClassificationDataset dataset = createLabeledDataFromRange(input, target);

	// hard-margin training with linear kernel
	{
		std::cout << "Ranking-SVM hard margin" << std::endl;
		LinearKernel<> kernel;
		KernelExpansion<RealVector> svm;
		RankingSvmTrainer<RealVector> trainer(&kernel, 1e100);
		trainer.stoppingCondition().minAccuracy = 1e-8;
		trainer.train(svm, dataset);

		// check the predictions
		Data<RealVector> output = svm(dataset.inputs());
		BOOST_CHECK_SMALL(output.element(0)(0) - 0.0, 1e-6);
		BOOST_CHECK_SMALL(output.element(1)(0) - 1.0, 1e-6);
		BOOST_CHECK_SMALL(output.element(2)(0) + 1.0, 1e-6);
		BOOST_CHECK_SMALL(output.element(3)(0) + 1.0, 1e-6);
		BOOST_CHECK_SMALL(output.element(4)(0) - 2.0, 1e-6);
	}
}


BOOST_AUTO_TEST_SUITE_END()
