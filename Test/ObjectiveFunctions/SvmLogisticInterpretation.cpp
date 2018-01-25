//===========================================================================
/*!
 * 
 *
 * \brief       unit test for maximum-likelihood model selection for support vector machines.
 * 
 * 
 * 
 *
 * \author      M. Tuma, T. Glasmachers
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

#include <shark/ObjectiveFunctions/SvmLogisticInterpretation.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Algorithms/GradientDescent/Rprop.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
#include <shark/Models/Kernels/ArdKernel.h>
#include <shark/Data/Statistics.h>
#include <shark/Data/Csv.h>
#include <shark/Data/DataDistribution.h>

#define BOOST_TEST_MODULE ObjectiveFunctions_SvmLogisticInterpretation
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include "TestObjectiveFunction.h"
using namespace shark;

const char test[] = "3.2588947676e+00 5.4190801643e-01 1\n\
3.3400343591e+00 5.0794724748e-01 1\n\
3.6535034226e+00 8.8413617108e-01 1\n\
1.2326682014e+00 3.9016160648e-01 1\n\
1.1139928736e+00 7.5352790393e-01 1\n\
2.9033558527e+00 3.8823711155e+00 1\n\
3.8286677990e+00 4.3944700249e-01 1\n\
1.4284671750e-01 1.4451760054e+00 1\n\
8.4769732878e-01 3.7359729828e+00 1\n\
3.1688293004e+00 3.5137225809e+00 0\n\
2.0146507099e+00 2.6229627840e+00 0\n\
3.1924234265e+00 3.2011218928e+00 0\n\
5.6754538044e-01 1.9133937545e-02 0\n\
2.9625889780e+00 2.9725298844e+00 0\n\
1.5689080786e+00 1.6883507241e+00 0\n\
6.9546068739e-01 6.8474675901e-01 0\n\
3.9715252072e+00 3.8300273186e+00 0\n\
3.8595541352e+00 3.8707797471e+00 0\n";

// on the above mini chessboard problem, optimize CSVM parameters using rprop on the SvmLogisticInterpretation.
// after every rprop step, assert that the numerical and analytical derivative coincide.
BOOST_AUTO_TEST_SUITE (ObjectiveFunctions_SvmLogisticInterpretation)

BOOST_AUTO_TEST_CASE( ObjectiveFunctions_SvmLogisticInterpretation_Small_Chessboard ){
	
	ClassificationDataset training_dataset;
	csvStringToData(training_dataset,test,LAST_COLUMN,0);
	std::size_t num_eles = training_dataset.numberOfElements();
	std::size_t num_folds = 2;
	std::vector< std::size_t > indices(num_eles);
	for(std::size_t i = 0; i<num_eles; i++) {
		indices[i] = (i+num_folds-1) % num_folds;
	}
	CVFolds<ClassificationDataset> cv_folds = createCVIndexed( training_dataset, num_folds, indices );
	GaussianRbfKernel<> kernel(0.5);
	QpStoppingCondition stop(1e-10);
	SvmLogisticInterpretation<> mlms_score( cv_folds, &kernel, false, &stop );

	// optimize NCLL using rprop
	Rprop<> rprop;
	RealVector start(2);
	start(0) = 1.0; start(1) = 0.5;
	rprop.init( mlms_score, start );
	unsigned int its = 20;
	for (unsigned int i=0; i<its; i++) {
		rprop.step( mlms_score );
		testDerivative(mlms_score, rprop.solution().point, 1.e-6,0,0.1);

	}
}

// on the above mini chessboard problem, optimize CSVM parameters using rprop on the SvmLogisticInterpretation.
// after every rprop step, assert that the numerical and analytical derivative coincide.
// this test uses the unconstrained formulation for the C-SVMs.
BOOST_AUTO_TEST_CASE( ObjectiveFunctions_SvmLogisticInterpretation_Small_Chessboard_C_unconstrained )
{
	ClassificationDataset training_dataset;
	csvStringToData(training_dataset,test,LAST_COLUMN,0);
	std::size_t num_eles = training_dataset.numberOfElements();
	std::size_t num_folds = 2;
	std::vector< size_t > indices( num_eles );
	for(std::size_t i = 0; i<num_eles; i++) {
		indices[i] = (i+num_folds-1) % num_folds;
	}
	CVFolds<ClassificationDataset> cv_folds = createCVIndexed( training_dataset, num_folds, indices );
	GaussianRbfKernel<> kernel(0.5);
	QpStoppingCondition stop(1e-10);
	SvmLogisticInterpretation<> mlms_score( cv_folds, &kernel, false, &stop );

	// optimize NCLL using rprop
	Rprop<> rprop;
	RealVector start(2);
	start(0) = 1.0; start(1) = 0.5;
	rprop.init( mlms_score, start );
	unsigned int its = 20;
	for (unsigned int i=0; i<its; i++) {
		rprop.step( mlms_score );
		testDerivative(mlms_score, rprop.solution().point, 1.e-6,0,0.1);
	}
}


// also test for the ARD kernel on the toy example problem from the pami paper.
// automatically test derivatives for all components along an entire optimization path
BOOST_AUTO_TEST_CASE( ObjectiveFunctions_SvmLogisticInterpretation_Pami_Toy )
{
	// create dataset
	unsigned int ud, nd, td, ntrain = 500;
	unsigned int ntest = 4000;
	ud = 5; nd = 5; td = ud+nd;
	PamiToy prob( ud, nd ); // artificial benchmark data, 2 useful and 2 noise dimensions
	ClassificationDataset train = prob.generateDataset( ntrain );
	ClassificationDataset test = prob.generateDataset( ntest );
	unsigned int num_folds = 5;
	CVFolds<ClassificationDataset> cv_folds = createCVIID( train, num_folds );
	DenseARDKernel kernel( td, 0.1 ); //dummy gamma, set later for real.
	QpStoppingCondition stop(1e-10);
	SvmLogisticInterpretation<> mlms( cv_folds, &kernel, true, &stop );

	RealVector start( td+1 );
	start(0) = 0.1;
	for ( unsigned int i=0; i<td; i++ ) {
		start(i+1) = 0.5/td;
	}
	// original params
	testDerivative(mlms, start, 1.e-6,1.e-10,0.01);

	// optimize NCLL using rprop
	Rprop<> rprop;
	rprop.init( mlms, start );
	unsigned int its = 30;
	for (unsigned int i=0; i<its; i++) {
		rprop.step( mlms );
	}

	//construct and evaluate the final machine
	KernelClassifier<RealVector> svm;
	CSvmTrainer<RealVector> trainer( &kernel, exp(rprop.solution().point(0)), true );
	trainer.train( svm, train );
	ZeroOneLoss<unsigned int> loss; // 0-1 loss
	Data<unsigned int> output = svm( train.inputs() ); // evaluate on training set
	double train_error = loss.eval(train.labels(), output);
	std::cout << "train error " << train_error << std::endl;
	output = svm( test.inputs() ); // evaluate on test set
	double test_error = loss.eval(test.labels(), output);
	std::cout << "test error " << test_error << std::endl;
	BOOST_CHECK( test_error < 0.18 ); //should be enough, hopefully..

}



BOOST_AUTO_TEST_SUITE_END()
