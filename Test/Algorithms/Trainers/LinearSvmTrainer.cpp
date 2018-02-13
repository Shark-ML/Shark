//===========================================================================
/*!
 * 
 *
 * \brief       Test case for various linear SVM trainers.
 * 
 * \par
 * This unit test trains a number of multi-class SVMs with two
 * different trainers, namely with a specialized trainer for
 * linear SVMs and a general purpose SVM trainer with linear
 * kernel function. It compares the weight vectors obtained
 * with both approaches. (Approximate) equality of the weight
 * vectors indicates correctness of both types of trainers.
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
#define BOOST_TEST_MODULE ALGORITHMS_TRAINERS_MCSVMTRAINER
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>


#include <shark/LinAlg/Base.h>
#include <shark/Data/Dataset.h>
#include <shark/Models/Kernels/LinearKernel.h>

#include <shark/Algorithms/Trainers/CSvmTrainer.h>


using namespace shark;
using namespace std;


#define RELATIVE_ACCURACY 0.01
#define MAX_KKT_VIOLATION 1e-5


// subtract the mean from each row
void ZeroSum(RealMatrix& mat)
{
	RealVector sum(mat.size2(), 0.0);
	for (size_t j=0; j<mat.size1(); j++) sum += row(mat, j);
	RealVector mean = (1.0 / mat.size1()) * sum;
	for (size_t j=0; j<mat.size1(); j++) row(mat, j) -= mean;
}


// This test case checks the resulting model of
// training eight multi-class SVMs on a minimal
// test case.
BOOST_AUTO_TEST_SUITE (Algorithms_Trainers_LinearSvmTrainer)


BOOST_AUTO_TEST_CASE( Binary_CSVM_TRAINER_TEST )
{
	size_t dim = 5;
	size_t ell = 200;

	double C = 1.0;
	LinearKernel<CompressedRealVector> kernel;

	LinearCSvmTrainer<CompressedRealVector> linNoBias(C,false);
	LinearCSvmTrainer<CompressedRealVector> linBias(C,true);
	CSvmTrainer<CompressedRealVector> kerNoBias(&kernel, C,false);
	CSvmTrainer<CompressedRealVector> kerBias(&kernel, C,true);
	
	linNoBias.stoppingCondition().minAccuracy = MAX_KKT_VIOLATION;
	linBias.stoppingCondition().minAccuracy = MAX_KKT_VIOLATION;
	kerNoBias.stoppingCondition().minAccuracy = MAX_KKT_VIOLATION;
	kerBias.stoppingCondition().minAccuracy = MAX_KKT_VIOLATION;

	for (unsigned int run=0; run<10; run++)
	{
		// generate random training set
		random::globalRng.seed(run);
		cout << endl << "generating test problem " << (run+1) << " out of 10" << endl;
		vector<CompressedRealVector> input(ell, CompressedRealVector(dim));
		vector<unsigned int> target(ell);
		for (size_t i=0; i<ell; i++)
		{
			unsigned int label = random::coinToss(random::globalRng);
			for (unsigned int d=0; d<dim; d++)
			{
				input[i].set_element(input[i].end(), d, 0.2 * random::gauss(random::globalRng) + 2*label-1);
			}
			target[i] = label;
		}
		LabeledData<CompressedRealVector, unsigned int> dataset = createLabeledDataFromRange(input, target);
		// train machines
		LinearClassifier<CompressedRealVector> linearNoBias;
		LinearClassifier<CompressedRealVector> linearBias;
		linNoBias.train(linearNoBias,dataset);
		linBias.train(linearBias,dataset);
		
		KernelClassifier<CompressedRealVector> nonlinearNoBias;
		KernelClassifier<CompressedRealVector> nonlinearBias;
		kerNoBias.train(nonlinearNoBias,dataset);
		kerBias.train(nonlinearBias,dataset);

		// extract weight matrices
		RealMatrix linear_w_noBias = linearNoBias.decisionFunction().matrix();
		RealMatrix linear_w_Bias = linearBias.decisionFunction().matrix();
		RealMatrix nonlinear_w_noBias(1, dim);
		RealMatrix nonlinear_w_Bias(1, dim);
		for (size_t j=0; j<dim; j++)
		{
			CompressedRealVector v(dim);
			v.set_element(v.end(),j,1.0);
			column(nonlinear_w_noBias, j) = nonlinearNoBias.decisionFunction()(v);
			column(nonlinear_w_Bias, j) = nonlinearBias.decisionFunction()(v)-nonlinearBias.decisionFunction().offset();
		}
		
		std::cout<<linearBias.decisionFunction().offset()<<" "<<nonlinearBias.decisionFunction().offset()<<std::endl;

		// output weight vectors for manual inspection
		cout << "      linear trainer weight vectors: " << endl;
		cout << "        " << row(linear_w_noBias, 0) << endl;
		cout << "        " << row(linear_w_Bias, 0) << endl;
		cout << "      nonlinear trainer weight vectors: " << endl;
		cout << "        " << row(nonlinear_w_noBias, 0) << endl;
		cout << "        " << row(nonlinear_w_Bias, 0) << endl;

		// compare weight vectors
		double n_noBias = norm_2(row(linear_w_noBias, 0));
		double n_Bias = norm_2(row(linear_w_Bias, 0));
		double d_noBias = norm_2(row(linear_w_noBias, 0) - row(nonlinear_w_noBias, 0));
		double d_Bias = norm_2(row(linear_w_Bias, 0) - row(nonlinear_w_Bias, 0));
		BOOST_CHECK_SMALL(d_noBias, RELATIVE_ACCURACY * n_noBias);
		BOOST_CHECK_SMALL(d_Bias, RELATIVE_ACCURACY * n_Bias);
	}
}

BOOST_AUTO_TEST_CASE( MCSVM_TRAINER_TEST )
{
	size_t classes = 5;
	size_t dim = 5;
	size_t ell = 100;

	const size_t var_per_class = dim / classes;

	double C = 1.0;
	LinearKernel<CompressedRealVector> kernel;

	// There are 9 trainers for multi-class SVMs in Shark which can train with or without bias:
	std::pair<std::string,McSvm> machines[9] ={
		{"OVA", McSvm::OVA},
		{"CS", McSvm::CS},
		{"WW",McSvm::WW},
		{"LLW",McSvm::LLW},
		{"ADM",McSvm::ADM},
		{"ATS",McSvm::ATS},
		{"ATM",McSvm::ATM},
		{"MMR",McSvm::MMR},
		{"Reinforced",McSvm::ReinforcedSvm},
	};

	for (unsigned int run=0; run<10; run++)
	{
		// generate random training set
		random::globalRng.seed(42+run);
		cout << endl << "generating test problem " << (run+1) << " out of 10" << endl;
		vector<CompressedRealVector> input(ell, CompressedRealVector(dim));
		vector<unsigned int> target(ell);
		for (size_t i=0; i<ell; i++)
		{
			unsigned int label = (unsigned int)random::discrete(random::globalRng, std::size_t(0), classes - 1);
			for (unsigned int d=0; d<dim; d++)
			{
				double y = ((d / var_per_class) == label) ? 1.0: -1.0;
				input[i].set_element(input[i].end(), d, 0.3 * random::gauss(random::globalRng) + y);
			}
			target[i] = label;
		}
		LabeledData<CompressedRealVector, unsigned int> dataset = createLabeledDataFromRange(input, target);

		for (size_t i=0; i<9; i++)
		{
			cout << "  testing linear vs non-linear " << machines[i].first << " -machine"<< endl;
			LinearCSvmTrainer<CompressedRealVector> linearTrainer(C, false);
			CSvmTrainer<CompressedRealVector> nonlinearTrainer(&kernel, C,false);
			linearTrainer.setMcSvmType(machines[i].second);
			nonlinearTrainer.setMcSvmType(machines[i].second);
			

			// train machine with two trainers
			LinearClassifier<CompressedRealVector> linear;
			linearTrainer.stoppingCondition().minAccuracy = MAX_KKT_VIOLATION;
			linearTrainer.train(linear, dataset);
			
			//Extract linear weights
			RealMatrix linear_w = linear.decisionFunction().matrix();
			ZeroSum(linear_w);
			cout << "      linear trainer weight vectors: " << endl;
			for (size_t j=0; j<classes; j++) cout << "        " << row(linear_w, j) << endl;
			
			KernelClassifier<CompressedRealVector> nonlinear;
			nonlinearTrainer.stoppingCondition().minAccuracy = MAX_KKT_VIOLATION;
			nonlinearTrainer.train(nonlinear, dataset);

			// extract nonlinear weight matrices
			
			RealMatrix nonlinear_w(classes, dim);
			for (size_t j=0; j<dim; j++)
			{
				CompressedRealVector v(dim);
				v.set_element(v.end(), j, 1.0);
				column(nonlinear_w, j) = nonlinear.decisionFunction()(v);
			}
			ZeroSum(nonlinear_w);
			cout << "      nonlinear trainer weight vectors: " << endl;
			for (size_t j=0; j<classes; j++) cout << "        " << row(nonlinear_w, j) << endl;

			// compare weight vectors
			double n = 0.0;
			for (size_t j=0; j<classes; j++) n += norm_2(row(linear_w, j));
			double d = 0.0;
			for (size_t j=0; j<classes; j++) d += norm_2(row(linear_w, j) - row(nonlinear_w, j));
			BOOST_CHECK_SMALL(d, RELATIVE_ACCURACY * n);
		}
	}
}

BOOST_AUTO_TEST_SUITE_END()
