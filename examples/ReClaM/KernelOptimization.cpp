//===========================================================================
/*!
*  \file KernelOptimization.cpp
*
*  \brief An example demonstrating kernel optimization.
*
*  \author  T. Glasmachers
*  \date    2006, 2009
*
*  \par
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
#include <ReClaM/Dataset.h>
#include <ReClaM/ArtificialDistributions.h>
#include <ReClaM/Svm.h>
#include <ReClaM/KTA.h>
#include <ReClaM/RadiusMargin.h>
#include <ReClaM/LOO.h>
#include <ReClaM/Rprop.h>
#include <ReClaM/CMAOptimizer.h>
#include <stdio.h>
#include <iostream>


using namespace std;


int main()
{
	// create the chess board example
	unsigned int examples = 100;
	double C = 1000.0;
	double gamma = 1.0;

	Chessboard chess;
	Dataset ds;
	ds.CreateFromSource(chess, examples, 0);
	const Array<double>& x = ds.getTrainingData();
	const Array<double>& y = ds.getTrainingTarget();

	// create svm models based on kernel models
	RBFKernel* k[3];
	k[0] = new RBFKernel(gamma);
	k[1] = new RBFKernel(gamma);
	k[2] = new RBFKernel(gamma);
	Model* model[3];
	SVM svm0(k[0], x);
	model[0] = new C_SVM(&svm0, C, C);
	model[1] = k[1];
	SVM svm2(k[2], x);
	model[2] = new C_SVM(&svm2, C, C);

	// create error functions
	ErrorFunction* err[3];
	err[0] = new RadiusMargin();
	err[1] = new NegativeKTA();
	err[2] = new LOO();

	// create optimizer
	Optimizer* optimizer[3];
	optimizer[0] = new IRpropPlus();
	optimizer[1] = new IRpropPlus();
	optimizer[2] = new CMAOptimizer();

	// define names
	char name[3][30] =
		{"Radius Margin Quotient",
		 "Kernel Target Alignment",
		 "Leave One Out Error",
		};

	// do independent optimization runs
	int j, i;
	for (j = 0; j < 3; j++)
	{
		cout << endl << "Optimization run " << (j + 1) << ":" << endl;
		cout << "Kernel optimization using the " << name[j] << "." << endl;
		cout << "initial kernel parameter: gamma = " << k[j]->getParameter(0) << endl;
		cout << "  Initializing the optimizer ..." << flush;
		optimizer[j]->init(*model[j]);
		cout << " done." << endl;

		for (i = 0; i < 40; i++)
		{
			cout << "  optimization iteration " << (i + 1) << flush;
			double f = optimizer[j]->optimize(*model[j], *err[j], x, y);
			cout << "   f = " << f << endl;
		}
		cout << "final kernel parameter: gamma = " << k[j]->getParameter(0) << endl;
	}


	// next three lines are need for self-testing, please ignore
	double err0 = err[0]->error(*model[0], x, y);
	double err1 = err[1]->error(*model[1], x, y);
	double err2 = err[2]->error(*model[2], x, y);
	// end self-testing block

	// clean up
	delete k[0];
	delete k[1];
	delete k[2];
	delete err[0];
	delete err[1];
	delete err[2];
	delete model[0];
	delete model[2];
	delete optimizer[0];
	delete optimizer[1];
	delete optimizer[2];

	// lines below are for self-testing this example, please ignore
	if (err0<=100 && err1<=-0.143 && err2<=0.09) exit(EXIT_SUCCESS);
	else exit(EXIT_FAILURE);
}
