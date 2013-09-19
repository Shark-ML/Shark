//===========================================================================
/*!
 *
 *  \brief unit test for maximum-likelihood model selection for support vector machines.
 *
 *
 *  \author  M. Tuma, T. Glasmachers
 *  \date    2011
 *
 *  \par Copyright (c) 2011:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
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



// mtq: TODO: add some trivial tests when the kernel-feasibility issue is settled, along the lines of:
//////	mlms_score.proposeStartingPoint( params );
//////	BOOST_CHECK( mlms_score.isFeasible( params) );




// on the above mini chessboard problem, optimize CSVM parameters using rprop on the SvmLogisticInterpretation.
// after every rprop step, assert that the numerical and analytical derivative coincide.
BOOST_AUTO_TEST_CASE( ObjectiveFunctions_SvmLogisticInterpretation_Small_Chessboard )
{
	double NUMERICAL_INCREASE_FACTOR = 1.00001;
	//~ std::stringstream ss(test);
	//~ std::vector<RealVector> x;
	//~ std::vector<unsigned int> y;
	// create dataset
	//~ detail::import_csv( x, y, ss, LAST_COLUMN, " ", "");
	
	ClassificationDataset training_dataset;
	csvStringToData(training_dataset,test,LAST_COLUMN,0);
	//~ ClassificationDataset training_dataset = createLabeledDataFromRange(x,y);
	unsigned int num_eles = training_dataset.numberOfElements();
	unsigned int num_folds = 2;
	std::vector< size_t > indices( num_eles );
	for ( unsigned int i=0; i<num_eles; i++ ) {
		indices[i] = (i+num_folds-1) % num_folds;
	}
	CVFolds<ClassificationDataset> cv_folds = createCVIndexed( training_dataset, num_folds, indices );
	GaussianRbfKernel<> kernel(0.5);
	QpStoppingCondition stop(1e-10);
	SvmLogisticInterpretation<> mlms_score( cv_folds, &kernel, false, &stop );

	// optimize NCLL using rprop
	IRpropPlus rprop;
	RealVector start(2);
	start(0) = 1.0; start(1) = 0.5;
	rprop.init( mlms_score, start );
	unsigned int its = 50;
	for (unsigned int i=0; i<its; i++) {
		rprop.step( mlms_score );
		// compare analytical and numerical derivative
		RealVector params(2);
		params(0) = rprop.solution().point(0); params(1) = rprop.solution().point(1);
		double result = mlms_score.eval( params );
		RealVector deriv;
		double der_result = mlms_score.evalDerivative( params, deriv );
		RealVector cmp_C_params(2);
		cmp_C_params(0) = params(0)*NUMERICAL_INCREASE_FACTOR; cmp_C_params(1) = params(1);
		double cmp_C_result = mlms_score.eval( cmp_C_params );
		RealVector cmp_gamma_params(2);
		cmp_gamma_params(0) = params(0); cmp_gamma_params(1) = params(1)*NUMERICAL_INCREASE_FACTOR;
		double cmp_gamma_result = mlms_score.eval( cmp_gamma_params );
		double diff_C = cmp_C_result - result;
		double diff_gamma = cmp_gamma_result - result;
		double denominator_C = cmp_C_params(0) - params(0);
		double denominator_gamma = cmp_gamma_params(1) - params(1);
		double ds_dC = diff_C / denominator_C;
		double ds_dgamma = diff_gamma / denominator_gamma;

		BOOST_CHECK_SMALL( ds_dC - deriv(0), 5e-4 );
		BOOST_CHECK_SMALL( ds_dgamma - deriv(1), 5e-4 );
		BOOST_CHECK_SMALL( result - der_result, 1e-12 );

	}
}

// on the above mini chessboard problem, optimize CSVM parameters using rprop on the SvmLogisticInterpretation.
// after every rprop step, assert that the numerical and analytical derivative coincide.
// this test uses the unconstrained formulation for the C-SVMs.
BOOST_AUTO_TEST_CASE( ObjectiveFunctions_SvmLogisticInterpretation_Small_Chessboard_C_unconstrained )
{
	double NUMERICAL_INCREASE_FACTOR = 1.00001;
	//~ std::stringstream ss(test);
	//~ std::vector<RealVector> x;
	//~ std::vector<unsigned int> y;
	//~ detail::import_csv( x, y, ss, LAST_COLUMN, " ", "");
	//~ ClassificationDataset training_dataset = createLabeledDataFromRange(x,y);
	ClassificationDataset training_dataset;
	csvStringToData(training_dataset,test,LAST_COLUMN,0);
	unsigned int num_eles = training_dataset.numberOfElements();
	unsigned int num_folds = 2;
	std::vector< size_t > indices( num_eles );
	for ( unsigned int i=0; i<num_eles; i++ ) {
		indices[i] = (i+num_folds-1) % num_folds;
	}
	CVFolds<ClassificationDataset> cv_folds = createCVIndexed( training_dataset, num_folds, indices );
	GaussianRbfKernel<> kernel(0.5);
	QpStoppingCondition stop(1e-10);
	SvmLogisticInterpretation<> mlms_score( cv_folds, &kernel, false, &stop );

	// optimize NCLL using rprop
	IRpropPlus rprop;
	RealVector start(2);
	start(0) = 1.0; start(1) = 0.5;
	rprop.init( mlms_score, start );
	unsigned int its = 50;
	for (unsigned int i=0; i<its; i++) {
		rprop.step( mlms_score );

		// compare analytical and numerical derivative
		RealVector params(2);
		params(0) = rprop.solution().point(0); params(1) = rprop.solution().point(1);
		double result = mlms_score.eval( params );
		RealVector deriv;
		double der_result = mlms_score.evalDerivative( params, deriv );
		RealVector cmp_C_params(2);
		cmp_C_params(0) = params(0)*NUMERICAL_INCREASE_FACTOR; cmp_C_params(1) = params(1);
		double cmp_C_result = mlms_score.eval( cmp_C_params );
		RealVector cmp_gamma_params(2);
		cmp_gamma_params(0) = params(0); cmp_gamma_params(1) = params(1)*NUMERICAL_INCREASE_FACTOR;
		double cmp_gamma_result = mlms_score.eval( cmp_gamma_params );
		double diff_C = cmp_C_result - result;
		double diff_gamma = cmp_gamma_result - result;
		double denominator_C = cmp_C_params(0) - params(0);
		double denominator_gamma = cmp_gamma_params(1) - params(1);
		double ds_dC = diff_C / denominator_C;
		double ds_dgamma = diff_gamma / denominator_gamma;

		BOOST_CHECK_SMALL( ds_dC - deriv(0), 5e-4 );
		BOOST_CHECK_SMALL( ds_dgamma - deriv(1), 5e-4 );
		BOOST_CHECK_SMALL( result - der_result, 1e-12 );

	}
}


// also test for the ARD kernel on the toy example problem from the pami paper.
// automatically test derivatives for all components along an entire optimization path
BOOST_AUTO_TEST_CASE( ObjectiveFunctions_SvmLogisticInterpretation_Pami_Toy )
{
	double NUMERICAL_INCREASE_FACTOR = 1.00001;
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
	double result, eps_result, der_result, eval_diff, param_diff, num_deriv;
	// original params
	result = mlms.eval( start );
	RealVector der;
	der_result = mlms.evalDerivative( start, der );
	BOOST_CHECK_SMALL( result - der_result, 1e-12 );

	// c-deriv
	{
		RealVector params = start;
		params(0) *= NUMERICAL_INCREASE_FACTOR;
		eps_result = mlms.eval( params );
		RealVector eps_deriv;
		mlms.evalDerivative( params, eps_deriv );
		eval_diff = eps_result - result;
		param_diff = params(0) - start(0);
		num_deriv = eval_diff / param_diff;
		BOOST_CHECK_SMALL( num_deriv - der(0), 5e-4 );
	}
	// kernel derivs
	for ( unsigned int i=1; i<td+1; i++ ) {
		RealVector params = start;
		params(i) *= NUMERICAL_INCREASE_FACTOR;
		eps_result = mlms.eval( params );
		RealVector eps_deriv;
		mlms.evalDerivative( params, eps_deriv );
		eval_diff = eps_result - result;
		param_diff = params(i) - start(i);
		num_deriv = eval_diff / param_diff;
		BOOST_CHECK_SMALL( num_deriv - der(i), 5e-4 );
	}

	// optimize NCLL using rprop
	IRpropPlus rprop;
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
	BOOST_CHECK( test_error < 0.155 ); //should be enough, hopefully..

}


