//===========================================================================
/*!
 *  \brief Test case for various linear SVM trainers.
 *
 *  \par
 *  This unit test trains a number of multi-class SVMs with two
 *  different trainers, namely with a specialized trainer for
 *  linear SVMs and a general purpose SVM trainer with linear
 *  kernel function. It compares the weight vectors obtained
 *  with both approaches. (Approximate) equality of the weight
 *  vectors indicates correctness of both types of trainers.
 *
 *  \author T. Glasmachers
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


#include <shark/LinAlg/Base.h>
#include <shark/Data/Dataset.h>
#include <shark/Models/Kernels/LinearKernel.h>

#include <shark/Algorithms/Trainers/McSvmOVATrainer.h>
#include <shark/Algorithms/Trainers/McSvmMMRTrainer.h>
#include <shark/Algorithms/Trainers/McSvmCSTrainer.h>
#include <shark/Algorithms/Trainers/McSvmWWTrainer.h>
#include <shark/Algorithms/Trainers/McSvmLLWTrainer.h>
#include <shark/Algorithms/Trainers/McSvmADMTrainer.h>
#include <shark/Algorithms/Trainers/McSvmATSTrainer.h>
#include <shark/Algorithms/Trainers/McSvmATMTrainer.h>


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
BOOST_AUTO_TEST_CASE( MCSVM_TRAINER_TEST )
{
	size_t classes = 5;
	size_t dim = 5;
	size_t ell = 100;

	const size_t var_per_class = dim / classes;

	double C = 1.0;
	LinearKernel<CompressedRealVector> kernel;

	AbstractLinearSvmTrainer<CompressedRealVector>* linearTrainer[8];
	AbstractSvmTrainer<CompressedRealVector, unsigned int>* nonlinearTrainer[8];

#define TRAINER(index, kind) \
	linearTrainer[index] = new LinearMcSvm##kind##Trainer<CompressedRealVector>(C); \
	nonlinearTrainer[index] = new McSvm##kind##Trainer<CompressedRealVector>(&kernel, C,false);

	TRAINER(0, MMR);
	TRAINER(1, OVA);
	TRAINER(2, WW);
	TRAINER(3, CS);
	TRAINER(4, LLW);
	TRAINER(5, ADM);
	TRAINER(6, ATS);
	TRAINER(7, ATM);

	for (size_t run=0; run<10; run++)
	{
		// generate random training set
		Rng::seed(run);
		cout << endl << "generating test problem " << (run+1) << " out of 10" << endl;
		vector<CompressedRealVector> input(ell, CompressedRealVector(dim));
		vector<unsigned int> target(ell);
		for (size_t i=0; i<ell; i++)
		{
			unsigned int label = Rng::discrete(0, classes - 1);
			for (unsigned int d=0; d<dim; d++)
			{
				if ((d / var_per_class) == label) input[i](d) = 0.3 * Rng::gauss() + 1.0;
				else input[i](d) = 0.3 * Rng::gauss() - 1.0;
			}
			target[i] = label;
		}
		LabeledData<CompressedRealVector, unsigned int> dataset = createLabeledDataFromRange(input, target);

		for (size_t i=0; i<8; i++)
		{
			cout << "  testing " << linearTrainer[i]->name() << " vs. " << nonlinearTrainer[i]->name() << endl;

			// train machine with two trainers
			LinearClassifier<CompressedRealVector> linear;
			linearTrainer[i]->stoppingCondition().minAccuracy = MAX_KKT_VIOLATION;
			linearTrainer[i]->train(linear, dataset);
			KernelClassifier<CompressedRealVector> nonlinear;
			nonlinearTrainer[i]->stoppingCondition().minAccuracy = MAX_KKT_VIOLATION;
			nonlinearTrainer[i]->train(nonlinear, dataset);

			// extract weight matrices
			RealMatrix linear_w = linear.decisionFunction().matrix();
			RealMatrix nonlinear_w(classes, dim);
			for (size_t j=0; j<dim; j++)
			{
				CompressedRealVector v(dim);
				v(j) = 1.0;
				column(nonlinear_w, j) = nonlinear.decisionFunction()(v);
			}
			ZeroSum(linear_w);
			ZeroSum(nonlinear_w);

			// output weight vectors for manual inspection
			cout << "      linear trainer weight vectors: " << endl;
			for (size_t j=0; j<classes; j++) cout << "        " << row(linear_w, j) << endl;
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
