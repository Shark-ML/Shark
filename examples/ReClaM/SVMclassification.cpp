/*!
*  \file SVMclassification.cpp
*
*  \author Tobias Glasmachers
*
*  \par
*      Institut f&uuml;r Neuroinformatik<BR>
*      Ruhr-Universit&auml;t Bochum<BR>
*      D-44780 Bochum, Germany<BR>
*      Phone: +49-234-32-25558<BR>
*      Fax:   +49-234-32-14209<BR>
*      eMail: shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
*      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
*      <BR> 
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


#include <Rng/GlobalRng.h>
#include <ReClaM/ArtificialDistributions.h>
#include <ReClaM/Dataset.h>
#include <ReClaM/Svm.h>
#include <ReClaM/ClassificationError.h>
#include <iostream>


using namespace std;


int main()
{
	Rng::seed(10);

	float C = 10.0;
	float gamma = 0.5;

	cout << endl;
	cout << "*** Support Vector Machine example program ***" << endl << endl;

	// create the xor problem with uniformly distributed examples
	cout << "Generating 100 training and 10000 test examples ..." << flush;
	Chessboard chess(2, 2);
	Dataset dataset;
	dataset.CreateFromSource(chess, 100, 10000);
	const Array<double>& x = dataset.getTrainingData();
	const Array<double>& y = dataset.getTrainingTarget();
	cout << " done." << endl;

	// create the SVM for prediction
	RBFKernel k(gamma);
	SVM svm(&k, false);

	// create a training scheme and an optimizer for learning
	C_SVM Csvm(&svm, C, C);
	SVM_Optimizer SVMopt;
	SVMopt.init(Csvm);

	// train the SVM
	cout << "Support Vector Machine training ..." << flush;
	SVMopt.optimize(svm, x, y);
	cout << " done." << endl;

// 	cout << "coefficients:" << endl;
// 	int i;
// 	for (i=0; i<100; i++) printf("alpha[%d] = %g\n", i, svm.getAlpha(i));

	// estimate the accuracy on the test set
	cout << "Testing ..." << flush;
	ClassificationError ce;
	double acc = 1.0 - ce.error(svm, dataset.getTestData(), dataset.getTestTarget());
	cout << " done." << endl;
	cout << "Estimated accuracy: " << 100.0 * acc << "%" << endl << endl;

	// lines below are for self-testing this example, please ignore
	if (acc >= 0.907) exit(EXIT_SUCCESS);
	else exit(EXIT_FAILURE);
}
