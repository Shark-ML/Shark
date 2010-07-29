//===========================================================================
/*!
 *  \file CrossValidation.cpp
 *
 *  \brief Cross Validation example
 *
 *  \author  T. Glasmachers
 *  \date    2006
 *
 *  \par Copyright (c) 1999-2006:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *
 *
 *
 *  <BR><HR>
 *  This file is part of Shark. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 2, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, write to the Free Software
 *  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

#include <ReClaM/Svm.h>
#include <ReClaM/ClassificationError.h>
#include <ReClaM/CrossValidation.h>
#include <ReClaM/GridSearch.h>
#include <ReClaM/ArtificialDistributions.h>


// number of cross validation folds
#define CV 5


//!
//! \par
//! The cross validation model selection example
//! uses cross validation on the classification
//! error to select the parameters gamma of the
//! #RBFKernel and C of the #C_SVM.
//!
//! \par
//! There is an outer optimization loop defined
//! by the parameter grid. For each grid point
//! the cross validation procedure trains a
//! number of SVMs using an inner optimization
//! loop. In this example, the inner loop is only
//! one iteration long because the #SVM_Optimizer
//! converges to the optimal solution within one
//! iteration. This situation is different e.g.
//! for neural network training.
//!
int main(int argc, char** argv)
{
	printf("\nSVM model selection using grid search and %d-fold cross validation\n", CV);

	ClassificationError err;
	SVM_Optimizer svmopt;

	// initial parameters do not matter as they will be optimized
	double gamma = 0.5;
	double C = 1000.0;

	// create 100 training and 5000 test examples
	// from the chessboard distribution
	printf("\ngenerating 500 training and 10000 test chess board examples ..."); fflush(stdout);
	Chessboard chess;
	Dataset dataset;
	dataset.CreateFromSource(chess, 500, 10000);
	printf(" done.\n");

	// kernel function
	RBFKernel kernel(gamma);

	// First, we have to define a partitioning of our data into folds:
	Partitioning part;
	part.CreateSameSize(CV, dataset.getTrainingData(), dataset.getTrainingTarget());

	// Then, we need models for the SVM and the C-SVM as usual:
	SVM cv_svm(&kernel);
	C_SVM cv_csvm(&cv_svm, C, C);

	// Last but not least, we need a model and an
	// error function for cross-validation:
	CVModel cv_model(CV, &cv_csvm);
	CVError cv_error(part, err, svmopt, 1);

	// Grid search over gamma from 0.5 to 5.0
	// and C from 50 to 500 with an equidistant grid:
	GridSearch grid;
	Array<double> minval(2);	// smallest value
	Array<double> maxval(2);	// largest value
	Array<int> numval(2);		// number of grid points
	minval(0) = 50.0;
	maxval(0) = 500.0;
	numval(0) = 10;
	minval(1) = 0.5;
	maxval(1) = 5.0;
	numval(1) = 10;
	grid.init(2, minval, maxval, numval);

	// Perform a grid search over the given parameter grid.
	// For each grid point, an inner optimization, that is,
	// 5-fold cross validation including support vector
	// machine training, takes place.
	// Note that the CVModel and CVError objects are passed
	// to the grid search optimizer.
	printf("10 x 10 grid search (may take some time) ..."); fflush(stdout);
	double cv_err = grid.optimize(cv_model, cv_error, dataset.getTrainingData(), dataset.getTrainingTarget());
	printf(" done.\n");
	printf("CV-error: %g\n", cv_err);

	// Get the best parameters from the CVModel.
	// Alternatively these parameters can be obtained
	// from the underlying C_SVM and KernelFunction objects.
	C = cv_model.getParameter(0);
	gamma = cv_model.getParameter(1);
	// C = cv_csvm.getParameter(0);			// alternative
	// gamma = kernel.getParameter(0);		// alternative
	printf("best parameters: C=%g, gamma=%g\n", C, gamma);

	// Use the best parameters for SVM training on the
	// whole training set. We construct new SVM and C_SVM
	// objects for this task for clarity.
	SVM svm(&kernel);
	C_SVM csvm(&svm, C, C);
	svmopt.init(csvm);
	printf("training final SVM classifier ..."); fflush(stdout);
	svmopt.optimize(svm, dataset.getTrainingData(), dataset.getTrainingTarget());
	printf(" done.\n");

	// test the model
	double train_err = err.error(svm, dataset.getTrainingData(), dataset.getTrainingTarget());
	double test_err = err.error(svm, dataset.getTestData(), dataset.getTestTarget());
	printf("training error: %g\ntest error: %g\n\n", train_err, test_err);

	// lines below are for self-testing this example, please ignore
	if (train_err <= 0.008) exit(EXIT_SUCCESS);
	else exit(EXIT_FAILURE);
}
