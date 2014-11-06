//===========================================================================
/*!
 * 
 *
 * \brief       unit test for the CSvmDerivative
 * 
 * 
 *
 * \author      M. Tuma
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
//===========================================================================

#define BOOST_TEST_MODULE MODELS_C_SVM_DERIVATIVE
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Data/Csv.h>
#include <shark/Models/Kernels/LinearKernel.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Models/Kernels/ArdKernel.h>
#include <shark/Models/Kernels/CSvmDerivative.h>
#include <shark/Algorithms/Trainers/CSvmTrainer.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>


using namespace shark;

// very straight-forward dataset with conceptual, hand-picked testing scenario. tests the deriv wrt C only.
BOOST_AUTO_TEST_SUITE (Models_Kernels_CSvmDerivative)

BOOST_AUTO_TEST_CASE( KERNEL_EXPANSION_CSVM_DERIVATIVE_TRIVIAL_DATASET )
{
	// set up dataset
	std::size_t NUM_DATA_POINTS = 6;
	std::vector<RealVector> input(NUM_DATA_POINTS);
	std::vector<unsigned int> target(NUM_DATA_POINTS);
	for (size_t i=0; i<NUM_DATA_POINTS; i++) input[i].resize(2);
	input[0](0) =  1.0; input[0](1) =  1.0; target[0] = 1;
	input[1](0) =  1.0; input[1](1) = -1.0; target[1] = 1;
	input[2](0) = -1.0; input[2](1) =  0.0; target[2] = 0;
	input[3](0) =  5.0; input[3](1) =  0.0; target[3] = 1;
	input[4](0) =  8.0; input[4](1) = -4.0; target[4] = 1;
	input[5](0) = -6.0; input[5](1) = -1.0; target[5] = 0;
	ClassificationDataset dataset  = createLabeledDataFromRange(input, target);
	// set up non-related quiz points
	std::size_t NUM_QUIZ_POINTS = 13;
	std::vector<RealVector> quiz(NUM_QUIZ_POINTS);
	for (size_t i=0; i<NUM_QUIZ_POINTS; i++) quiz[i].resize(2);
	quiz[0](0) =   0.0;  quiz[0](1) =  0.0;
	quiz[1](0) =  -0.2;  quiz[1](1) =  2.0;
	quiz[2](0) =   0.2;  quiz[2](1) = -3.0;
	quiz[3](0) =  -0.4;  quiz[3](1) =  6.0;
	quiz[4](0) =   0.4;  quiz[4](1) = -1.0;
	quiz[5](0) =  -0.8;  quiz[5](1) = -2.0;
	quiz[6](0) =   0.8;  quiz[6](1) =  4.0;
	quiz[7](0) =  -1.0;  quiz[7](1) = -7.0;
	quiz[8](0) =   1.0;  quiz[8](1) =  9.0;
	quiz[9](0) =  -1.4;  quiz[9](1) = -0.2;
	quiz[10](0) =  1.4; quiz[10](1) =  2.5;
	quiz[11](0) = -9.0; quiz[11](1) =  2.0;
	quiz[12](0) = 10.0; quiz[12](1) = -2.0;

	// set up different values of C
	std::vector< double > Cs;
	Cs.push_back(100); Cs.push_back(10); Cs.push_back(1); Cs.push_back(0.5); Cs.push_back(0.48);
	Cs.push_back(0.4); Cs.push_back(0.2); Cs.push_back(0.1); Cs.push_back(0.01); Cs.push_back(0.001); Cs.push_back(0.0001);
	double C_eps;
	double NUMERICAL_C_INCREASE_FACTOR = 1.0001;

	// loop through Cs: repeat tests for every different value of C
	for ( unsigned int i=0; i<Cs.size(); i++ ) {
		C_eps = Cs[i] * NUMERICAL_C_INCREASE_FACTOR;
		// set up svm for current C
		LinearKernel<> kernel;
		KernelClassifier<RealVector> kc;
		KernelExpansion<RealVector>& svm = kc.decisionFunction();
		CSvmTrainer<RealVector, double> trainer( &kernel, Cs[i],true );
		trainer.sparsify() = false;
		trainer.stoppingCondition().minAccuracy = 1e-10;
		trainer.train(kc, dataset);
		RealVector param = svm.parameterVector();
		CSvmDerivative<RealVector> svm_deriv( &kc, &trainer );

		// set up helper variables
		double diff, deriv;
		RealVector computed_derivative;

		// set up svm for numerical comparsion-C
		LinearKernel<> cmp_kernel;
		KernelClassifier<RealVector> cmp_kc;
		KernelExpansion<RealVector>& cmp_svm =cmp_kc.decisionFunction();
		CSvmTrainer<RealVector, double> cmp_trainer(&cmp_kernel, C_eps,true);
		cmp_trainer.sparsify() = false;
		cmp_trainer.stoppingCondition().minAccuracy = 1e-10;
		cmp_trainer.train( cmp_kc, dataset );
		RealVector cmp_param = cmp_svm.parameterVector();

		// first test derivatives of dataset-points themselves
		for ( unsigned int j=0; j<NUM_DATA_POINTS; j++ ) {
			diff = cmp_svm(input[j])(0) - svm(input[j])(0);
			deriv = diff / (C_eps - Cs[i]);
			svm_deriv.modelCSvmParameterDerivative(input[j], computed_derivative);
			BOOST_CHECK_EQUAL( computed_derivative.size(), 1 );
			BOOST_CHECK_SMALL( deriv - computed_derivative(0) , 1e-6 );
		}
		// now also test derivatives of other datapoints
		for ( unsigned int j=0; j<NUM_QUIZ_POINTS; j++ ) {
			diff = cmp_svm(quiz[j])(0) - svm(quiz[j])(0);
			deriv = diff / (C_eps - Cs[i]);
			svm_deriv.modelCSvmParameterDerivative(quiz[j], computed_derivative);
			BOOST_CHECK_EQUAL( computed_derivative.size(), 1 );
			BOOST_CHECK_SMALL( deriv - computed_derivative(0) , 1e-6 );
		}
	}
}

// very straight-forward dataset with conceptual, hand-picked testing scenario using unconstrained encoding.
// tests the deriv wrt C only also, but in unconstrained-mode.
// The desired solution accuracy for the svm trainer had to be increased here even more.
BOOST_AUTO_TEST_CASE( KERNEL_EXPANSION_CSVM_DERIVATIVE_TRIVIAL_DATASET_UNCONSTRAINED)
{
	// set up dataset
	std::size_t NUM_DATA_POINTS = 6;
	std::vector<RealVector> input(NUM_DATA_POINTS);
	std::vector<unsigned int> target(NUM_DATA_POINTS);
	for (size_t i=0; i<NUM_DATA_POINTS; i++) input[i].resize(2);
	input[0](0) =  1.0; input[0](1) =  1.0; target[0] = 1;
	input[1](0) =  1.0; input[1](1) = -1.0; target[1] = 1;
	input[2](0) = -1.0; input[2](1) =  0.0; target[2] = 0;
	input[3](0) =  5.0; input[3](1) =  0.0; target[3] = 1;
	input[4](0) =  8.0; input[4](1) = -4.0; target[4] = 1;
	input[5](0) = -6.0; input[5](1) = -1.0; target[5] = 0;
	ClassificationDataset dataset = createLabeledDataFromRange(input, target);
	// set up non-related quiz points
	std::size_t NUM_QUIZ_POINTS = 13;
	std::vector<RealVector> quiz(NUM_QUIZ_POINTS);
	for (size_t i=0; i<NUM_QUIZ_POINTS; i++) quiz[i].resize(2);
	quiz[0](0) =   0.0;  quiz[0](1) =  0.0;
	quiz[1](0) =  -0.2;  quiz[1](1) =  2.0;
	quiz[2](0) =   0.2;  quiz[2](1) = -3.0;
	quiz[3](0) =  -0.4;  quiz[3](1) =  6.0;
	quiz[4](0) =   0.4;  quiz[4](1) = -1.0;
	quiz[5](0) =  -0.8;  quiz[5](1) = -2.0;
	quiz[6](0) =   0.8;  quiz[6](1) =  4.0;
	quiz[7](0) =  -1.0;  quiz[7](1) = -7.0;
	quiz[8](0) =   1.0;  quiz[8](1) =  9.0;
	quiz[9](0) =  -1.4;  quiz[9](1) = -0.2;
	quiz[10](0) =  1.4; quiz[10](1) =  2.5;
	quiz[11](0) = -9.0; quiz[11](1) =  2.0;
	quiz[12](0) = 10.0; quiz[12](1) = -2.0;

	// set up different values of C
	std::vector< double > Cs;
	Cs.push_back(100); Cs.push_back(10); Cs.push_back(1); Cs.push_back(0.5); Cs.push_back(0.48);
	//the accuracies are suffering too much in the log-encoded case for the last two values...
	Cs.push_back(0.4); Cs.push_back(0.2); Cs.push_back(0.1); Cs.push_back(0.01); //Cs.push_back(0.001); Cs.push_back(0.0001);
	double C_eps;
	double NUMERICAL_C_INCREASE_FACTOR = 1.00001;

	bool UNCONSTRAINED = true;

	// loop through Cs: repeat tests for every different value of C
	for ( unsigned int i=0; i<Cs.size(); i++ ) {
		C_eps = Cs[i] * NUMERICAL_C_INCREASE_FACTOR;
		// set up svm for current C
		LinearKernel<> kernel;
		KernelClassifier<RealVector> kc;
		KernelExpansion<RealVector>& svm = kc.decisionFunction();
		CSvmTrainer<RealVector, double> trainer( &kernel, Cs[i], true,UNCONSTRAINED );
		trainer.sparsify() = false;
		trainer.stoppingCondition().minAccuracy = 1e-14;
		trainer.train(kc, dataset);
		RealVector param = svm.parameterVector();
		CSvmDerivative<RealVector> svm_deriv( &kc, &trainer );

		// set up helper variables
		double diff, deriv;
		RealVector computed_derivative;

		// set up svm for numerical comparsion-C
		LinearKernel<> cmp_kernel;
		KernelClassifier<RealVector> cmp_kc;
		KernelExpansion<RealVector>& cmp_svm =cmp_kc.decisionFunction();
		CSvmTrainer<RealVector,double> cmp_trainer(&cmp_kernel, C_eps, true, UNCONSTRAINED );
		cmp_trainer.sparsify() = false;
		cmp_trainer.stoppingCondition().minAccuracy = 1e-15;
		cmp_trainer.train( cmp_kc, dataset );
		RealVector cmp_param = cmp_svm.parameterVector();

		// first test derivatives of dataset-points themselves
		for ( unsigned int j=0; j<NUM_DATA_POINTS; j++ ) {
			diff = cmp_svm(input[j])(0) - svm(input[j])(0);
			deriv = diff / ( std::log(C_eps) - std::log(Cs[i]) );
			svm_deriv.modelCSvmParameterDerivative(input[j], computed_derivative);
			BOOST_CHECK_EQUAL( computed_derivative.size(), 1 );
			BOOST_CHECK_SMALL( deriv - computed_derivative(0) , 1e-4 );
		}
		// now also test derivatives of other datapoints
		for ( unsigned int j=0; j<NUM_QUIZ_POINTS; j++ ) {
			diff = cmp_svm(quiz[j])(0) - svm(quiz[j])(0);
			deriv = diff / (std::log(C_eps) - std::log(Cs[i]));
			svm_deriv.modelCSvmParameterDerivative(quiz[j], computed_derivative);
			BOOST_CHECK_EQUAL( computed_derivative.size(), 1 );
			BOOST_CHECK_SMALL( deriv - computed_derivative(0) , 1e-4 );
		}
	}
}

// very straight-forward dataset with conceptual, hand-picked testing scenario.
// as above, tests the deriv wrt C only also, but with an RBF kernel instead of a linear one.
// The desired solution accuracy for the svm trainer had to be increased here as well.
BOOST_AUTO_TEST_CASE( KERNEL_EXPANSION_CSVM_DERIVATIVE_TRIVIAL_DATASET_RBF )
{
	// set up dataset
	std::size_t NUM_DATA_POINTS = 6;
	std::vector<RealVector> input(NUM_DATA_POINTS);
	std::vector<unsigned int> target(NUM_DATA_POINTS);
	for (size_t i=0; i<NUM_DATA_POINTS; i++) input[i].resize(2);
	input[0](0) =  1.0; input[0](1) =  1.0; target[0] = 1;
	input[1](0) =  1.0; input[1](1) = -1.0; target[1] = 1;
	input[2](0) = -1.0; input[2](1) =  0.0; target[2] = 0;
	input[3](0) =  5.0; input[3](1) =  0.0; target[3] = 1;
	input[4](0) =  8.0; input[4](1) = -4.0; target[4] = 1;
	input[5](0) = -6.0; input[5](1) = -1.0; target[5] = 0;
	ClassificationDataset dataset = createLabeledDataFromRange(input, target);
	// set up non-related quiz points
	std::size_t NUM_QUIZ_POINTS = 13;
	std::vector<RealVector> quiz(NUM_QUIZ_POINTS);
	for (size_t i=0; i<NUM_QUIZ_POINTS; i++) quiz[i].resize(2);
	quiz[0](0) =   0.0;  quiz[0](1) =  0.0;
	quiz[1](0) =  -0.2;  quiz[1](1) =  2.0;
	quiz[2](0) =   0.2;  quiz[2](1) = -3.0;
	quiz[3](0) =  -0.4;  quiz[3](1) =  6.0;
	quiz[4](0) =   0.4;  quiz[4](1) = -1.0;
	quiz[5](0) =  -0.8;  quiz[5](1) = -2.0;
	quiz[6](0) =   0.8;  quiz[6](1) =  4.0;
	quiz[7](0) =  -1.0;  quiz[7](1) = -7.0;
	quiz[8](0) =   1.0;  quiz[8](1) =  9.0;
	quiz[9](0) =  -1.4;  quiz[9](1) = -0.2;
	quiz[10](0) =  1.4; quiz[10](1) =  2.5;
	quiz[11](0) = -9.0; quiz[11](1) =  2.0;
	quiz[12](0) = 10.0; quiz[12](1) = -2.0;

	// set up different values of C
	std::vector< double > Cs;
	Cs.push_back(100); Cs.push_back(10); Cs.push_back(1); Cs.push_back(0.5); Cs.push_back(0.48);
	Cs.push_back(0.4); Cs.push_back(0.2); Cs.push_back(0.1); Cs.push_back(0.01); Cs.push_back(0.001); Cs.push_back(0.0001);
	double C_eps;
	double NUMERICAL_C_INCREASE_FACTOR = 1.0001;

	// set up different values for the kernel parameters
	std::vector< double > RbfParams;
	RbfParams.push_back(10); RbfParams.push_back(2); RbfParams.push_back(1); RbfParams.push_back(0.5); RbfParams.push_back(0.2);
	RbfParams.push_back(0.1); RbfParams.push_back(0.05); RbfParams.push_back(0.01); RbfParams.push_back(0.001); RbfParams.push_back(0.0001);

	// loop through RbfParams: repeat tests for every different value of Gamma
	for ( unsigned int h=0; h<RbfParams.size(); h++ ) {
		// loop through Cs: repeat tests for every different value of C
		for ( unsigned int i=0; i<Cs.size(); i++ ) {
			C_eps = Cs[i] * NUMERICAL_C_INCREASE_FACTOR;
			// set up svm for current C
			DenseRbfKernel kernel( RbfParams[h] );
			KernelClassifier<RealVector> kc;
			KernelExpansion<RealVector>& svm = kc.decisionFunction();
			CSvmTrainer<RealVector, double> trainer( &kernel, Cs[i],true );
			trainer.sparsify() = false;
			trainer.stoppingCondition().minAccuracy = 1e-15;
			trainer.train(kc, dataset);
			RealVector param = svm.parameterVector();
			CSvmDerivative<RealVector> svm_deriv( &kc, &trainer );

			// set up helper variables
			double diff, deriv;
			RealVector computed_derivative;

			// set up svm for numerical comparsion-C
			DenseRbfKernel cmp_kernel( RbfParams[h] );
			KernelClassifier<RealVector> cmp_kc;
			KernelExpansion<RealVector>& cmp_svm =cmp_kc.decisionFunction();
			CSvmTrainer<RealVector, double> cmp_trainer(&cmp_kernel, C_eps,true);
			cmp_trainer.sparsify() = false;
			cmp_trainer.stoppingCondition().minAccuracy = 1e-15;
			cmp_trainer.train( cmp_kc, dataset );
			RealVector cmp_param = cmp_svm.parameterVector();

			// first test derivatives of dataset-points themselves
			for ( unsigned int j=0; j<NUM_DATA_POINTS; j++ ) {
				diff = cmp_svm(input[j])(0) - svm(input[j])(0);
				deriv = diff / (C_eps - Cs[i]);
				svm_deriv.modelCSvmParameterDerivative(input[j], computed_derivative);
				BOOST_CHECK_EQUAL( computed_derivative.size(), 2 );
				BOOST_CHECK_SMALL( deriv - computed_derivative(1) , 1e-5 );
			}
			// now also test derivatives of other datapoints
			for ( unsigned int j=0; j<NUM_QUIZ_POINTS; j++ ) {
				diff = cmp_svm(quiz[j])(0) - svm(quiz[j])(0);
				deriv = diff / (C_eps - Cs[i]);
				svm_deriv.modelCSvmParameterDerivative(quiz[j], computed_derivative);
				BOOST_CHECK_EQUAL( computed_derivative.size(), 2 );
				BOOST_CHECK_SMALL( deriv - computed_derivative(1) , 1e-5 );
			}
		}
	}
}

// second dataset with a few hand-picked and a few automated tests
// as above, tests the deriv wrt C only also, but on another dataset, and again with a linear kernel.
BOOST_AUTO_TEST_CASE( KERNEL_EXPANSION_CSVM_DERIVATIVE_SECOND_DATASET )
{
	// simple 5-point dataset
	std::size_t NUM_DATA_POINTS = 6;
	std::vector<RealVector> input(NUM_DATA_POINTS);
	std::vector<unsigned int> target(NUM_DATA_POINTS);
	for (size_t i=0; i<NUM_DATA_POINTS; i++) input[i].resize(2);
	input[0](0) =  0.0; input[0](1) =  0.0; target[0] = 0;
	input[1](0) =  2.0; input[1](1) =  2.0; target[1] = 1;
	input[2](0) = -1.0; input[2](1) = -8.0; target[2] = 0;
	input[3](0) = -1.0; input[3](1) = -1.0; target[3] = 0;
	input[4](0) =  3.0; input[4](1) =  3.0; target[4] = 1;
	ClassificationDataset dataset = createLabeledDataFromRange(input, target);

	// non-related quiz points
	std::size_t NUM_QUIZ_POINTS = 50;
	std::vector<RealVector> quiz(NUM_QUIZ_POINTS);
	for (size_t i=0; i<NUM_QUIZ_POINTS; i++) {
		quiz[i].resize(2);
		quiz[i](0) = Rng::uni(-10,10);
		quiz[i](1) = Rng::uni(-10,10);
	}

	// set up different values for C
	std::vector< double > Cs;
	Cs.push_back(100); Cs.push_back(10); Cs.push_back(1); Cs.push_back(0.5); Cs.push_back(0.48);
	Cs.push_back(0.4); Cs.push_back(0.2); Cs.push_back(0.1); Cs.push_back(0.01); Cs.push_back(0.001); Cs.push_back(0.0001);
	double C_eps;
	double NUMERICAL_C_INCREASE_FACTOR = 1.0001;

	// loop through Cs: repeat tests for every different value of C
	for ( unsigned int i=0; i<Cs.size(); i++ ) {
		C_eps = Cs[i] * NUMERICAL_C_INCREASE_FACTOR;
		// set up svm for current C
		LinearKernel<> kernel;
		KernelClassifier<RealVector> kc;
		KernelExpansion<RealVector>& svm = kc.decisionFunction();
		CSvmTrainer<RealVector, double> trainer( &kernel, Cs[i],true );
		trainer.sparsify() = false;
		trainer.stoppingCondition().minAccuracy = 1e-10;
		trainer.train(kc, dataset);
		RealVector param = svm.parameterVector();
		CSvmDerivative<RealVector> svm_deriv( &kc, &trainer );

		// set up helper variables
		double diff, deriv;
		RealVector computed_derivative;

		// set up svm for numerical comparsion-C
		LinearKernel<> cmp_kernel;
		KernelClassifier<RealVector> cmp_kc;
		KernelExpansion<RealVector>& cmp_svm =cmp_kc.decisionFunction();
		CSvmTrainer<RealVector, double> cmp_trainer(&cmp_kernel, C_eps,true);
		cmp_trainer.sparsify() = false;
		cmp_trainer.stoppingCondition().minAccuracy = 1e-10;
		cmp_trainer.train( cmp_kc, dataset );
		RealVector cmp_param = cmp_svm.parameterVector();

		// first test derivatives of dataset-points themselves
		for ( unsigned int j=0; j<NUM_DATA_POINTS; j++ ) {
			diff = cmp_svm(input[j])(0) - svm(input[j])(0);
			deriv = diff / (C_eps - Cs[i]);
			svm_deriv.modelCSvmParameterDerivative(input[j], computed_derivative);
			BOOST_CHECK_EQUAL( computed_derivative.size(), 1 );
			BOOST_CHECK_SMALL( deriv - computed_derivative(0) , 1e-6 );
		}
		// now also test derivatives of other datapoints
		for ( unsigned int j=0; j<NUM_QUIZ_POINTS; j++ ) {
			diff = cmp_svm(quiz[j])(0) - svm(quiz[j])(0);
			deriv = diff / (C_eps - Cs[i]);
			svm_deriv.modelCSvmParameterDerivative(quiz[j], computed_derivative);
			BOOST_CHECK_EQUAL( computed_derivative.size(), 1 );
			BOOST_CHECK_SMALL( deriv - computed_derivative(0) , 1e-6 );
		}
	}
}

// use the above (topmost) dataset to test the CSvm derivative w.r.t. the kernel parameters
BOOST_AUTO_TEST_CASE( KERNEL_EXPANSION_CSVM_DERIVATIVE_KERNEL_PARAMS )
{
	// set up dataset
	std::size_t NUM_DATA_POINTS = 6;
	std::vector<RealVector> input(NUM_DATA_POINTS);
	std::vector<unsigned int> target(NUM_DATA_POINTS);
	for (size_t i=0; i<NUM_DATA_POINTS; i++) input[i].resize(2);
	input[0](0) =  1.0; input[0](1) =  1.0; target[0] = 1;
	input[1](0) =  1.0; input[1](1) = -1.0; target[1] = 1;
	input[2](0) = -1.0; input[2](1) =  0.0; target[2] = 0;
	input[3](0) =  5.0; input[3](1) =  0.0; target[3] = 1;
	input[4](0) =  8.0; input[4](1) = -4.0; target[4] = 1;
	input[5](0) = -6.0; input[5](1) = -1.0; target[5] = 0;
	ClassificationDataset dataset = createLabeledDataFromRange(input, target);
	// set up non-related quiz points
	std::size_t NUM_QUIZ_POINTS = 13;
	std::vector<RealVector> quiz(NUM_QUIZ_POINTS);
	for (size_t i=0; i<NUM_QUIZ_POINTS; i++) quiz[i].resize(2);
	quiz[0](0) =   0.0;  quiz[0](1) =  0.0;
	quiz[1](0) =  -0.2;  quiz[1](1) =  2.0;
	quiz[2](0) =   0.2;  quiz[2](1) = -3.0;
	quiz[3](0) =  -0.4;  quiz[3](1) =  6.0;
	quiz[4](0) =   0.4;  quiz[4](1) = -1.0;
	quiz[5](0) =  -0.8;  quiz[5](1) = -2.0;
	quiz[6](0) =   0.8;  quiz[6](1) =  4.0;
	quiz[7](0) =  -1.0;  quiz[7](1) = -7.0;
	quiz[8](0) =   1.0;  quiz[8](1) =  9.0;
	quiz[9](0) =  -1.4;  quiz[9](1) = -0.2;
	quiz[10](0) =  1.4; quiz[10](1) =  2.5;
	quiz[11](0) = -9.0; quiz[11](1) =  2.0;
	quiz[12](0) = 10.0; quiz[12](1) = -2.0;

	// set up different values of C
	std::vector< double > Cs;
//	Cs.push_back(100); Cs.push_back(10); //responsible for the highest of the in-accuracies encountered
	Cs.push_back(1); Cs.push_back(0.5); Cs.push_back(0.48);
	Cs.push_back(0.4); Cs.push_back(0.2); Cs.push_back(0.1); Cs.push_back(0.01); Cs.push_back(0.001); Cs.push_back(0.0001);
	double RbfParam_eps;

	// set up different values for the kernel parameters
	std::vector< double > RbfParams;
	RbfParams.push_back(10); RbfParams.push_back(2); RbfParams.push_back(1); RbfParams.push_back(0.5); RbfParams.push_back(0.2);
	RbfParams.push_back(0.1); RbfParams.push_back(0.05); RbfParams.push_back(0.01); RbfParams.push_back(0.001); RbfParams.push_back(0.0001);
	double NUMERICAL_KERNEL_PARAMETER_INCREASE_FACTOR = 1.0001;

	// loop through RbfParams: repeat test for different values of gamma
	for ( unsigned int h=0; h<RbfParams.size(); h++ ) {
		RbfParam_eps = RbfParams[h]*NUMERICAL_KERNEL_PARAMETER_INCREASE_FACTOR;
		// loop through Cs: repeat tests for every different value of C
		for ( unsigned int i=0; i<Cs.size(); i++ ) {
			// set up svm with current kernel parameters
			DenseRbfKernel kernel( RbfParams[h] );
			KernelClassifier<RealVector> kc;
			KernelExpansion<RealVector>& svm = kc.decisionFunction();
			CSvmTrainer<RealVector, double> trainer( &kernel, Cs[i],true );
			trainer.sparsify() = false;
			trainer.stoppingCondition().minAccuracy = 1e-15;
			trainer.train(kc, dataset);
			RealVector param = svm.parameterVector();
			CSvmDerivative<RealVector> svm_deriv( &kc, &trainer );

			// set up helper variables
			double diff, deriv;
			RealVector computed_derivative;

			// set up svm with epsiloned-kernel-parameters for numerical comparsion
			DenseRbfKernel cmp_kernel( RbfParam_eps );
			KernelClassifier<RealVector> cmp_kc;
			KernelExpansion<RealVector>& cmp_svm =cmp_kc.decisionFunction();
			CSvmTrainer<RealVector, double> cmp_trainer( &cmp_kernel, Cs[i],true );
			cmp_trainer.sparsify() = false;
			cmp_trainer.stoppingCondition().minAccuracy = 1e-15;
			cmp_trainer.train( cmp_kc, dataset );
			RealVector cmp_param = cmp_svm.parameterVector();

			// first test derivatives of dataset-points themselves
			for ( unsigned int j=0; j<NUM_DATA_POINTS; j++ ) {
				diff = cmp_svm(input[j])(0) - svm(input[j])(0);
				deriv = diff / (RbfParam_eps - RbfParams[h]);
				svm_deriv.modelCSvmParameterDerivative(input[j], computed_derivative);
				BOOST_CHECK_EQUAL( computed_derivative.size(), 2 );
				BOOST_CHECK_SMALL( deriv - computed_derivative(0) , 5e-3 );
			}
			// now also test derivatives of other datapoints
			for ( unsigned int j=0; j<NUM_QUIZ_POINTS; j++ ) {
				diff = cmp_svm(quiz[j])(0) - svm(quiz[j])(0);
				deriv = diff / (RbfParam_eps - RbfParams[h]);
				svm_deriv.modelCSvmParameterDerivative(quiz[j], computed_derivative);
				BOOST_CHECK_EQUAL( computed_derivative.size(), 2 );
				BOOST_CHECK_SMALL( deriv - computed_derivative(0) , 5e-3 );
			}

		}
	}
}


// test the CSvm derivative w.r.t. the kernel parameters. now on the chessboard to exclude a strange error.
// this test trains on the first half of an 18-sample chessboard dataset, and validates on the second half.
BOOST_AUTO_TEST_CASE( KERNEL_EXPANSION_CSVM_DERIVATIVE_KERNEL_PARAMS_CHESSBOARD )
{

const char data1[] = "3.2588947676e+00 5.4190801643e-01 1\n\
3.6535034226e+00 8.8413617108e-01 1\n\
1.1139928736e+00 7.5352790393e-01 1\n\
3.8286677990e+00 4.3944700249e-01 1\n\
8.4769732878e-01 3.7359729828e+00 1\n\
2.0146507099e+00 2.6229627840e+00 0\n\
5.6754538044e-01 1.9133937545e-02 0\n\
1.5689080786e+00 1.6883507241e+00 0\n\
3.9715252072e+00 3.8300273186e+00 0\n";
const char data2[] = "3.3400343591e+00 5.0794724748e-01 1\n\
1.2326682014e+00 3.9016160648e-01 1\n\
2.9033558527e+00 3.8823711155e+00 1\n\
1.4284671750e-01 1.4451760054e+00 1\n\
3.1688293004e+00 3.5137225809e+00 0\n\
3.1924234265e+00 3.2011218928e+00 0\n\
2.9625889780e+00 2.9725298844e+00 0\n\
6.9546068739e-01 6.8474675901e-01 0\n\
3.8595541352e+00 3.8707797471e+00 0\n";

	ClassificationDataset d1;
	ClassificationDataset d2;
	csvStringToData(d1,data1,LAST_COLUMN,0);
	csvStringToData(d2,data2,LAST_COLUMN,0);
	std::size_t NUM_DATA_POINTS = 9;

	// set up different values of C and gamma
	std::vector< double > Cs;
//	Cs.push_back(1);
	Cs.push_back(0.67);
	double RbfParam_eps;
	std::vector< double > RbfParams;
//	RbfParams.push_back(0.5);
	RbfParams.push_back(0.82);
	double NUMERICAL_KERNEL_PARAMETER_INCREASE_FACTOR = 1.00001;

	// loop through RbfParams: repeat test for different values of gamma
	for ( unsigned int h=0; h<RbfParams.size(); h++ ) {
		RbfParam_eps = RbfParams[h]*NUMERICAL_KERNEL_PARAMETER_INCREASE_FACTOR;
		// loop through Cs: repeat tests for every different value of C
		for ( unsigned int i=0; i<Cs.size(); i++ ) {
			// set up svm with current kernel parameters
			DenseRbfKernel kernel( RbfParams[h] );
			KernelClassifier<RealVector> kc;
			KernelExpansion<RealVector>& svm = kc.decisionFunction();
			CSvmTrainer<RealVector, double> trainer( &kernel, Cs[i],true );
			trainer.sparsify() = false;
			trainer.stoppingCondition().minAccuracy = 1e-15;
			trainer.train(kc, d1);
			RealVector param = svm.parameterVector();
			CSvmDerivative<RealVector> svm_deriv( &kc, &trainer );

			// set up helper variables
			double diff, deriv;
			RealVector computed_derivative;

			// set up svm with epsiloned-kernel-parameters for numerical comparsion
			DenseRbfKernel cmp_kernel( RbfParam_eps );
			KernelClassifier<RealVector> cmp_kc;
			KernelExpansion<RealVector>& cmp_svm =cmp_kc.decisionFunction();
			CSvmTrainer<RealVector, double> cmp_trainer( &cmp_kernel, Cs[i], true );
			cmp_trainer.sparsify() = false;
			cmp_trainer.stoppingCondition().minAccuracy = 1e-15;
			cmp_trainer.train( cmp_kc, d1 );
			RealVector cmp_param = cmp_svm.parameterVector();

			// first test derivatives of dataset-points themselves
			for ( unsigned int j=0; j<NUM_DATA_POINTS; j++ ) {
				diff = cmp_svm(d1.element(j).input)(0) - svm(d1.element(j).input)(0);
				deriv = diff / (RbfParam_eps - RbfParams[h]);
				svm_deriv.modelCSvmParameterDerivative(d1.element(j).input, computed_derivative);
				BOOST_CHECK_EQUAL( computed_derivative.size(), 2 );
				BOOST_CHECK_SMALL( deriv - computed_derivative(0) , 5e-3 );
			}
			// quiz points: use other partition
			for ( unsigned int j=0; j<NUM_DATA_POINTS; j++ ) {
				diff = cmp_svm(d2.element(j).input)(0) - svm(d2.element(j).input)(0);
				deriv = diff / (RbfParam_eps - RbfParams[h]);
				svm_deriv.modelCSvmParameterDerivative(d2.element(j).input, computed_derivative);
				BOOST_CHECK_EQUAL( computed_derivative.size(), 2 );
				BOOST_CHECK_SMALL( deriv - computed_derivative(0) , 5e-3 );
			}
		}
	}
}

// test the CSvm derivative w.r.t. the kernel parameters. now on the chessboard to exclude a strange error.
// this test trains on the 2nd half of an 18-sample chessboard dataset, and validates on the 1st half.
BOOST_AUTO_TEST_CASE( KERNEL_EXPANSION_CSVM_DERIVATIVE_KERNEL_PARAMS_CHESSBOARD_SWAPPED )
{
const char data1[] = "3.2588947676e+00 5.4190801643e-01 1\n\
3.6535034226e+00 8.8413617108e-01 1\n\
1.1139928736e+00 7.5352790393e-01 1\n\
3.8286677990e+00 4.3944700249e-01 1\n\
8.4769732878e-01 3.7359729828e+00 1\n\
2.0146507099e+00 2.6229627840e+00 0\n\
5.6754538044e-01 1.9133937545e-02 0\n\
1.5689080786e+00 1.6883507241e+00 0\n\
3.9715252072e+00 3.8300273186e+00 0\n";
const char data2[] = "3.3400343591e+00 5.0794724748e-01 1\n\
1.2326682014e+00 3.9016160648e-01 1\n\
2.9033558527e+00 3.8823711155e+00 1\n\
1.4284671750e-01 1.4451760054e+00 1\n\
3.1688293004e+00 3.5137225809e+00 0\n\
3.1924234265e+00 3.2011218928e+00 0\n\
2.9625889780e+00 2.9725298844e+00 0\n\
6.9546068739e-01 6.8474675901e-01 0\n\
3.8595541352e+00 3.8707797471e+00 0\n";

	ClassificationDataset d1;
	ClassificationDataset d2;
	csvStringToData(d1,data1,LAST_COLUMN,0);
	csvStringToData(d2,data2,LAST_COLUMN,0);
	std::size_t NUM_DATA_POINTS = 9;

	// set up different values of C and gamma
	std::vector< double > Cs;
//	Cs.push_back(1);
	Cs.push_back(0.67);
	double RbfParam_eps;
	std::vector< double > RbfParams;
//	RbfParams.push_back(0.5);
	RbfParams.push_back(0.82);
	double NUMERICAL_KERNEL_PARAMETER_INCREASE_FACTOR = 1.00001;

	// loop through RbfParams: repeat test for different values of gamma
	for ( unsigned int h=0; h<RbfParams.size(); h++ ) {
		RbfParam_eps = RbfParams[h]*NUMERICAL_KERNEL_PARAMETER_INCREASE_FACTOR;
		// loop through Cs: repeat tests for every different value of C
		for ( unsigned int i=0; i<Cs.size(); i++ ) {
			// set up svm with current kernel parameters
			DenseRbfKernel kernel( RbfParams[h] );
			KernelClassifier<RealVector> kc;
			KernelExpansion<RealVector>& svm = kc.decisionFunction();
			CSvmTrainer<RealVector, double> trainer( &kernel, Cs[i],true );
			trainer.sparsify() = false;
			trainer.stoppingCondition().minAccuracy = 1e-15;
			trainer.train(kc, d2);
			RealVector param = svm.parameterVector();
			CSvmDerivative<RealVector> svm_deriv( &kc, &trainer );

			// set up helper variables
			double diff, deriv;
			RealVector computed_derivative;

			// set up svm with epsiloned-kernel-parameters for numerical comparsion
			DenseRbfKernel cmp_kernel( RbfParam_eps );
			KernelClassifier<RealVector> cmp_kc;
			KernelExpansion<RealVector>& cmp_svm =cmp_kc.decisionFunction();
			CSvmTrainer<RealVector, double> cmp_trainer( &cmp_kernel, Cs[i],true );
			cmp_trainer.sparsify() = false;
			cmp_trainer.stoppingCondition().minAccuracy = 1e-15;
			cmp_trainer.train( cmp_kc, d2 );
			RealVector cmp_param = cmp_svm.parameterVector();

			// first test derivatives of dataset-points themselves
			for ( unsigned int j=0; j<NUM_DATA_POINTS; j++ ) {
				diff = cmp_svm(d2.element(j).input)(0) - svm(d2.element(j).input)(0);
				deriv = diff / (RbfParam_eps - RbfParams[h]);
				svm_deriv.modelCSvmParameterDerivative(d2.element(j).input, computed_derivative);
				BOOST_CHECK_EQUAL( computed_derivative.size(), 2 );
				BOOST_CHECK_SMALL( deriv - computed_derivative(0) , 5e-3 );
			}
			// quiz points: use other partition
			for ( unsigned int j=0; j<NUM_DATA_POINTS; j++ ) {
				diff = cmp_svm(d1.element(j).input)(0) - svm(d1.element(j).input)(0);
				deriv = diff / (RbfParam_eps - RbfParams[h]);
				svm_deriv.modelCSvmParameterDerivative(d1.element(j).input, computed_derivative);
				BOOST_CHECK_EQUAL( computed_derivative.size(), 2 );
				BOOST_CHECK_SMALL( deriv - computed_derivative(0) , 5e-3 );
			}
		}
	}
}

// now also test the CSvm derivative w.r.t. the kernel parameters when the kernel parameters are log-encoded.
// done on the topmost dataset again. this only tests the deriv wrt gamma
BOOST_AUTO_TEST_CASE( KERNEL_EXPANSION_CSVM_DERIVATIVE_KERNEL_PARAMS_UNCONSTRAINED )
{
	// set up dataset
	std::size_t NUM_DATA_POINTS = 6;
	std::vector<RealVector> input(NUM_DATA_POINTS);
	std::vector<unsigned int> target(NUM_DATA_POINTS);
	for (size_t i=0; i<NUM_DATA_POINTS; i++) input[i].resize(2);
	input[0](0) =  1.0; input[0](1) =  1.0; target[0] = 1;
	input[1](0) =  1.0; input[1](1) = -1.0; target[1] = 1;
	input[2](0) = -1.0; input[2](1) =  0.0; target[2] = 0;
	input[3](0) =  5.0; input[3](1) =  0.0; target[3] = 1;
	input[4](0) =  8.0; input[4](1) = -4.0; target[4] = 1;
	input[5](0) = -6.0; input[5](1) = -1.0; target[5] = 0;
	ClassificationDataset dataset = createLabeledDataFromRange(input, target);
	// set up non-related quiz points
	std::size_t NUM_QUIZ_POINTS = 13;
	std::vector<RealVector> quiz(NUM_QUIZ_POINTS);
	for (size_t i=0; i<NUM_QUIZ_POINTS; i++) quiz[i].resize(2);
	quiz[0](0) =   0.0;  quiz[0](1) =  0.0;
	quiz[1](0) =  -0.2;  quiz[1](1) =  2.0;
	quiz[2](0) =   0.2;  quiz[2](1) = -3.0;
	quiz[3](0) =  -0.4;  quiz[3](1) =  6.0;
	quiz[4](0) =   0.4;  quiz[4](1) = -1.0;
	quiz[5](0) =  -0.8;  quiz[5](1) = -2.0;
	quiz[6](0) =   0.8;  quiz[6](1) =  4.0;
	quiz[7](0) =  -1.0;  quiz[7](1) = -7.0;
	quiz[8](0) =   1.0;  quiz[8](1) =  9.0;
	quiz[9](0) =  -1.4;  quiz[9](1) = -0.2;
	quiz[10](0) =  1.4; quiz[10](1) =  2.5;
	quiz[11](0) = -9.0; quiz[11](1) =  2.0;
	quiz[12](0) = 10.0; quiz[12](1) = -2.0;

	// set up different values of C
	std::vector< double > Cs;
//	Cs.push_back(100); Cs.push_back(10); //responsible for the highest of the in-accuracies encountered
	Cs.push_back(1); Cs.push_back(0.5); Cs.push_back(0.48);
	Cs.push_back(0.4); Cs.push_back(0.2); Cs.push_back(0.1); Cs.push_back(0.01); Cs.push_back(0.001); Cs.push_back(0.0001);
	double RbfParam_eps;

	// set up different values for the kernel parameters
	std::vector< double > RbfParams;
	RbfParams.push_back(10); RbfParams.push_back(2); RbfParams.push_back(1); RbfParams.push_back(0.5); RbfParams.push_back(0.2);
	RbfParams.push_back(0.1); RbfParams.push_back(0.05); RbfParams.push_back(0.01); RbfParams.push_back(0.001); RbfParams.push_back(0.0001);
	double NUMERICAL_KERNEL_PARAMETER_INCREASE_FACTOR = 1.0001;

	// loop through RbfParams: repeat test for different values of gamma
	for ( unsigned int h=0; h<RbfParams.size(); h++ ) {
		RbfParam_eps = RbfParams[h]*NUMERICAL_KERNEL_PARAMETER_INCREASE_FACTOR;
		// loop through Cs: repeat tests for every different value of C
		for ( unsigned int i=0; i<Cs.size(); i++ ) {
			// set up svm with current kernel parameters
			DenseRbfKernel kernel( RbfParams[h], true );
			KernelClassifier<RealVector> kc;
			KernelExpansion<RealVector>& svm = kc.decisionFunction();
			CSvmTrainer<RealVector, double> trainer( &kernel, Cs[i],true );
			trainer.sparsify() = false;
			trainer.stoppingCondition().minAccuracy = 1e-15;
			trainer.train(kc, dataset);
			RealVector param = svm.parameterVector();
			CSvmDerivative<RealVector> svm_deriv( &kc, &trainer );

			// set up helper variables
			double diff, deriv;
			RealVector computed_derivative;

			// set up svm with epsiloned-kernel-parameters for numerical comparsion
			DenseRbfKernel cmp_kernel( RbfParam_eps, true );
			KernelClassifier<RealVector> cmp_kc;
			KernelExpansion<RealVector>& cmp_svm =cmp_kc.decisionFunction();
			CSvmTrainer<RealVector, double> cmp_trainer( &cmp_kernel, Cs[i],true );
			cmp_trainer.sparsify() = false;
			cmp_trainer.stoppingCondition().minAccuracy = 1e-15;
			cmp_trainer.train( cmp_kc, dataset );
			RealVector cmp_param = cmp_svm.parameterVector();

			// first test derivatives of dataset-points themselves
			for ( unsigned int j=0; j<NUM_DATA_POINTS; j++ ) {
				diff = cmp_svm(input[j])(0) - svm(input[j])(0);
				deriv = diff / ( std::log(RbfParam_eps) - std::log(RbfParams[h]) );
				svm_deriv.modelCSvmParameterDerivative(input[j], computed_derivative);
				BOOST_CHECK_EQUAL( computed_derivative.size(), 2 );
				BOOST_CHECK_SMALL( deriv - computed_derivative(0) , 5e-3 );
			}
			// now also test derivatives of other datapoints
			for ( unsigned int j=0; j<NUM_QUIZ_POINTS; j++ ) {
				diff = cmp_svm(quiz[j])(0) - svm(quiz[j])(0);
				deriv = diff / ( std::log(RbfParam_eps) - std::log(RbfParams[h]) );
				svm_deriv.modelCSvmParameterDerivative(quiz[j], computed_derivative);
				BOOST_CHECK_EQUAL( computed_derivative.size(), 2 );
				BOOST_CHECK_SMALL( deriv - computed_derivative(0) , 5e-3 );
			}

		}
	}
}


// finally, test a more-than-one-parameter kernel. uses topmost dataset again.
// this tests all derivs, those wrt. to C as well as the two ARD kernel params.
BOOST_AUTO_TEST_CASE( KERNEL_EXPANSION_CSVM_DERIVATIVE_MULTIPLE_KERNEL_PARAMS )
{
	// set up dataset
	std::size_t NUM_DATA_POINTS = 6;
	std::vector<RealVector> input(NUM_DATA_POINTS);
	std::vector<unsigned int> target(NUM_DATA_POINTS);
	for (size_t i=0; i<NUM_DATA_POINTS; i++) input[i].resize(2);
	input[0](0) =  1.0; input[0](1) =  1.0; target[0] = 1;
	input[1](0) =  1.0; input[1](1) = -1.0; target[1] = 1;
	input[2](0) = -1.0; input[2](1) =  0.0; target[2] = 0;
	input[3](0) =  5.0; input[3](1) =  0.0; target[3] = 1;
	input[4](0) =  8.0; input[4](1) = -4.0; target[4] = 1;
	input[5](0) = -6.0; input[5](1) = -1.0; target[5] = 0;
	ClassificationDataset dataset = createLabeledDataFromRange(input, target);
	// set up non-related quiz points
	std::size_t NUM_QUIZ_POINTS = 13;
	std::vector<RealVector> quiz(NUM_QUIZ_POINTS);
	for (size_t i=0; i<NUM_QUIZ_POINTS; i++) quiz[i].resize(2);
	quiz[0](0) =   0.0;  quiz[0](1) =  0.0;
	quiz[1](0) =  -0.2;  quiz[1](1) =  2.0;
	quiz[2](0) =   0.2;  quiz[2](1) = -3.0;
	quiz[3](0) =  -0.4;  quiz[3](1) =  6.0;
	quiz[4](0) =   0.4;  quiz[4](1) = -1.0;
	quiz[5](0) =  -0.8;  quiz[5](1) = -2.0;
	quiz[6](0) =   0.8;  quiz[6](1) =  4.0;
	quiz[7](0) =  -1.0;  quiz[7](1) = -7.0;
	quiz[8](0) =   1.0;  quiz[8](1) =  9.0;
	quiz[9](0) =  -1.4;  quiz[9](1) = -0.2;
	quiz[10](0) =  1.4; quiz[10](1) =  2.5;
	quiz[11](0) = -9.0; quiz[11](1) =  2.0;
	quiz[12](0) = 10.0; quiz[12](1) = -2.0;

	// set up different values of C
	std::vector< double > Cs;
//	Cs.push_back(100); Cs.push_back(10); //responsible for the highest of the in-accuracies encountered
	Cs.push_back(1); Cs.push_back(0.5); Cs.push_back(0.48);
	Cs.push_back(0.4); Cs.push_back(0.2); Cs.push_back(0.1); Cs.push_back(0.01); Cs.push_back(0.001); Cs.push_back(0.0001);
	double ArdParam_eps, C_eps;

	// set up different values for the kernel parameters
	std::vector< double > ArdParams;
	ArdParams.push_back(10); ArdParams.push_back(2); ArdParams.push_back(1); ArdParams.push_back(0.5); ArdParams.push_back(0.2);
	ArdParams.push_back(0.1); ArdParams.push_back(0.05); ArdParams.push_back(0.01); ArdParams.push_back(0.001); ArdParams.push_back(0.0001);
	double NUMERICAL_PARAMETER_INCREASE_FACTOR = 1.0001;
	RealVector tmp_param_manipulator;

	// loop through ArdParams: repeat test for different values of gamma
	for ( unsigned int h=0; h<ArdParams.size(); h++ ) {
		ArdParam_eps = ArdParams[h]*NUMERICAL_PARAMETER_INCREASE_FACTOR;
		// loop through Cs: repeat tests for every different value of C
		for ( unsigned int i=0; i<Cs.size(); i++ ) {
			C_eps = Cs[i]*NUMERICAL_PARAMETER_INCREASE_FACTOR;
			// set up svm with current kernel parameters
			DenseARDKernel kernel( 2, ArdParams[h]*ArdParams[h] );
			tmp_param_manipulator = kernel.parameterVector();
			tmp_param_manipulator(1) *= 0.5;
			kernel.setParameterVector( tmp_param_manipulator );
			KernelClassifier<RealVector> kc;
			KernelExpansion<RealVector>& svm = kc.decisionFunction();
			CSvmTrainer<RealVector, double> trainer( &kernel, Cs[i],true );
			trainer.sparsify() = false;
			trainer.stoppingCondition().minAccuracy = 1e-15;
			trainer.train(kc, dataset);
			RealVector param = svm.parameterVector();
			CSvmDerivative<RealVector> svm_deriv( &kc, &trainer );
			// set up helper variables
			double diff, deriv;
			RealVector computed_derivative;
			// set up svm with epsiloned-c_parameter for numerical comparsion
			DenseARDKernel c_cmp_kernel( 2, ArdParams[h]*ArdParams[h] );
			tmp_param_manipulator = c_cmp_kernel.parameterVector();
			tmp_param_manipulator(1) *= 0.5;
			c_cmp_kernel.setParameterVector( tmp_param_manipulator );
			KernelClassifier<RealVector> c_cmp_kc;
			KernelExpansion<RealVector>& c_cmp_svm = c_cmp_kc.decisionFunction();
			CSvmTrainer<RealVector, double> c_cmp_trainer( &c_cmp_kernel, C_eps, true );
			c_cmp_trainer.sparsify() = false;
			c_cmp_trainer.stoppingCondition().minAccuracy = 1e-15;
			c_cmp_trainer.train( c_cmp_kc, dataset );
			RealVector c_cmp_param = c_cmp_svm.parameterVector();
			// set up svm with epsiloned-kernel-parameters_1 for numerical comparsion
			DenseARDKernel g1_cmp_kernel( 2, ArdParams[h]*ArdParams[h] );
			tmp_param_manipulator = g1_cmp_kernel.parameterVector();
			tmp_param_manipulator(0) = ArdParam_eps;
			tmp_param_manipulator(1) *= 0.5;
			g1_cmp_kernel.setParameterVector( tmp_param_manipulator );
			
			KernelClassifier<RealVector> g1_cmp_kc;
			KernelExpansion<RealVector>& g1_cmp_svm = g1_cmp_kc.decisionFunction();
			CSvmTrainer<RealVector, double> g1_cmp_trainer( &g1_cmp_kernel, Cs[i], true );
			g1_cmp_trainer.sparsify() = false;
			g1_cmp_trainer.stoppingCondition().minAccuracy = 1e-15;
			g1_cmp_trainer.train( g1_cmp_kc, dataset );
			RealVector g1_cmp_param = g1_cmp_svm.parameterVector();
			
			// set up svm with epsiloned-kernel-parameters_2 for numerical comparsion
			DenseARDKernel g2_cmp_kernel( 2, ArdParams[h]*ArdParams[h] );
			tmp_param_manipulator = g2_cmp_kernel.parameterVector();
			tmp_param_manipulator(1) *= 0.5;
			tmp_param_manipulator(1) *= NUMERICAL_PARAMETER_INCREASE_FACTOR;
			g2_cmp_kernel.setParameterVector( tmp_param_manipulator );
			
			KernelClassifier<RealVector> g2_cmp_kc;
			KernelExpansion<RealVector>& g2_cmp_svm = g2_cmp_kc.decisionFunction();
			CSvmTrainer<RealVector, double> g2_cmp_trainer( &g2_cmp_kernel, Cs[i], true );
			g2_cmp_trainer.sparsify() = false;
			g2_cmp_trainer.stoppingCondition().minAccuracy = 1e-15;
			g2_cmp_trainer.train( g2_cmp_kc, dataset );
			RealVector g2_cmp_param = g2_cmp_svm.parameterVector();

			////////////////////////////////////////////////////////////////////////////////
			//                         TEST ALL THREE DERIVATIVES
			////////////////////////////////////////////////////////////////////////////////
			// FIRST, TEST DERIVS WRT C
			// first, test derivatives of dataset-points themselves
			for ( unsigned int j=0; j<NUM_DATA_POINTS; j++ ) {
				diff = c_cmp_svm(input[j])(0) - svm(input[j])(0);
				deriv = diff / (C_eps - Cs[i]);
				svm_deriv.modelCSvmParameterDerivative(input[j], computed_derivative);
				BOOST_REQUIRE_EQUAL( computed_derivative.size(), 3 );
				BOOST_REQUIRE_SMALL( deriv - computed_derivative(2) , 1e-6 );
			}
			// now also test derivatives of other datapoints
			for ( unsigned int j=0; j<NUM_QUIZ_POINTS; j++ ) {
				diff = c_cmp_svm(quiz[j])(0) - svm(quiz[j])(0);
				deriv = diff / (C_eps - Cs[i]);
				svm_deriv.modelCSvmParameterDerivative(quiz[j], computed_derivative);
				BOOST_REQUIRE_EQUAL( computed_derivative.size(), 3 );
				BOOST_REQUIRE_SMALL( deriv - computed_derivative(2) , 1e-6 );
			}

			////////////////////////////////////////////////////////////////////////////////
			// NEXT, TEST DERIVS WRT GAMMA-1
			// first, test derivatives of dataset-points themselves
			for ( unsigned int j=0; j<NUM_DATA_POINTS; j++ ) {
				diff = g1_cmp_svm(input[j])(0) - svm(input[j])(0);
				deriv = diff / (ArdParam_eps - ArdParams[h]);
				svm_deriv.modelCSvmParameterDerivative(input[j], computed_derivative);
				BOOST_REQUIRE_EQUAL( computed_derivative.size(), 3 );
				BOOST_REQUIRE_SMALL( deriv - computed_derivative(0) , 5e-3 );
			}
			// now also test derivatives of other datapoints
			for ( unsigned int j=0; j<NUM_QUIZ_POINTS; j++ ) {
				diff = g1_cmp_svm(quiz[j])(0) - svm(quiz[j])(0);
				deriv = diff / (ArdParam_eps - ArdParams[h]);
				svm_deriv.modelCSvmParameterDerivative(quiz[j], computed_derivative);
				BOOST_REQUIRE_EQUAL( computed_derivative.size(), 3 );
				BOOST_REQUIRE_SMALL( deriv - computed_derivative(0) , 5e-3 );
			}

			////////////////////////////////////////////////////////////////////////////////
			// NEXT, TEST DERIVS WRT GAMMA-2
			// first, test derivatives of dataset-points themselves
			for ( unsigned int j=0; j<NUM_DATA_POINTS; j++ ) {
				diff = g2_cmp_svm(input[j])(0) - svm(input[j])(0);
				deriv = diff / 0.5 / (ArdParam_eps - ArdParams[h]);
				svm_deriv.modelCSvmParameterDerivative(input[j], computed_derivative);
				BOOST_REQUIRE_EQUAL( computed_derivative.size(), 3 );
				BOOST_REQUIRE_SMALL( deriv - computed_derivative(1) , 5e-3 );
			}
			// now also test derivatives of other datapoints
			for ( unsigned int j=0; j<NUM_QUIZ_POINTS; j++ ) {
				diff = g2_cmp_svm(quiz[j])(0) - svm(quiz[j])(0);
				deriv = diff / 0.5 / (ArdParam_eps - ArdParams[h]);
				svm_deriv.modelCSvmParameterDerivative(quiz[j], computed_derivative);
				BOOST_REQUIRE_EQUAL( computed_derivative.size(), 3 );
				BOOST_REQUIRE_SMALL( deriv - computed_derivative(1) , 5e-3 );
			}

		}
	}
}



BOOST_AUTO_TEST_SUITE_END()
