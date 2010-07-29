//===========================================================================
/*!
 *  \file SvmApproximationExample.cpp
 *
 *  \brief SVM approximation example
 *
 *  \author  T. Suttorp
 *  \date    2007
 *
 *  \par Copyright (c) 1999-2007:
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
 *  This file is part of ReClaM. This library is free software;
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
//===========================================================================


#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>

#include <Rng/GlobalRng.h>
#include <ReClaM/Svm.h>
#include <ReClaM/SvmApproximation.h>
#include <ReClaM/ClassificationError.h>
#include <ReClaM/ArtificialDistributions.h>
#include <ReClaM/Dataset.h>


using namespace std;
using std::cout;
using std::endl;


int main(int argc, char** argv)
{
	double C     = 10.0;
	double gamma = 1.0;
	RBFKernel k(gamma);
	SVM svm(&k, false);
	C_SVM Csvm(&svm, C, C);
	SVM_Optimizer SVMopt;
	SVMopt.init(Csvm);
	ClassificationError ce;

	//----------------------------------------
	// generate feature vectors
	//
	Chessboard chess;
	Dataset dataset;
	dataset.CreateFromSource(chess, 500, 10000);
	const Array<double>& trainData       = dataset.getTrainingData();
	const Array<double>& trainDataLabels = dataset.getTrainingTarget();
	const Array<double>& testData        = dataset.getTestData();
	const Array<double>& testDataLabels  = dataset.getTestTarget();

	cout << "SVM training\n";
	SVMopt.optimize(svm, trainData, trainDataLabels);

	cout << "classify test data" << endl;
	cout << "accuracy: " << 100.0 *(1.0 - ce.error(svm, testData, testDataLabels)) << "%\n";

	SvmApproximation* approximateSVM = new SvmApproximation(&svm, true);

	// configure approximation algorithm
	approximateSVM->setApproximationAlgorithm(SvmApproximation::iRpropPlus);
	approximateSVM->setVectorGenerationMode(SvmApproximation::randomSelection);

	approximateSVM->setApproximationTarget(SvmApproximation::noSVs);
	approximateSVM->setTargetNoVecsForApproximatedSVM(50);
	approximateSVM->setNoGradientDescentIterations(100);

	// approximate
	SVM* approxSvm = approximateSVM->getApproximatedSVM();
	float svmApproximationError = approximateSVM->approximate();
	double error=ce.error(*approxSvm, testData, testDataLabels);
	cout << "accuracy: " << 100.0 *(1.0 - error) << "%\n";

	if (svmApproximationError >= 0.0)
	{
		// final gradient descent on approximated model
		svmApproximationError = approximateSVM->gradientDescent();
		cout << "accuracy: " << 100.0 *(1.0 - ce.error(*approxSvm, testData, testDataLabels)) <<"%\n";
	}

	delete approximateSVM;

	// lines below are for self-testing this example, please ignore
	if (error <= 0.07595) exit(EXIT_SUCCESS);
	else exit(EXIT_FAILURE);
}
