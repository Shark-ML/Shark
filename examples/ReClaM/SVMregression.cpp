/*!
*  \file SVMregression.cpp
*
*  \author T. Glasmachers
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
#include <ReClaM/Svm.h>
#include <ReClaM/MeanSquaredError.h>
#include <Array/ArrayIo.h>
#include <iostream>


using namespace std;


double sinc(double x)
{
	if (x == 0.0) return 1.0;
	else return sin(x) / x;
}


int main()
{
	cout << "*** Support Vector Machine example program ***" << endl << endl;
	cout << "The regression training data are sampled from a sinc function" << endl;
	cout << "with additive noise." << endl;
	cout << endl;

	unsigned int e;
	Rng::seed(42);

	double C = 1e10;
	double epsilon = 0.01;
	double sigma = 2.0;
	unsigned int examples = 100;

	// create the sinc problem
	Array<double> x(examples, 1);
	Array<double> t(examples, 1);
	Array<double> y(examples, 1);
	for (e = 0; e < examples; e++)
	{
		x(e, 0) = Rng::uni(-12.0, 12.0);						// point
		y(e, 0) = sinc(x(e, 0)) + Rng::uni(-epsilon, epsilon);	// target
	}

	// create the SVM for prediction
	double gamma = 0.5 / (sigma * sigma);
	RBFKernel k(gamma);
	SVM svm(&k, false);

	// create a training scheme and an optimizer for learning
	Epsilon_SVM esvm(&svm, C, epsilon);
	SVM_Optimizer SVMopt;
	SVMopt.init(esvm);

	// train the SVM
	cout << "Support Vector Machine training ..." << flush;
	SVMopt.optimize(svm, x, y);
	cout << " done." << endl << endl;

	// compute the mean squared error on the training data:
	MeanSquaredError mse;
	double err = mse.error(svm, x, y);
	cout << "mean squared error on the training data: " << err << endl << endl;

	// lines below are for self-testing this example, please ignore
	if (err <= epsilon*epsilon) exit(EXIT_SUCCESS);
	else exit(EXIT_FAILURE);
}
