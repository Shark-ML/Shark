/*!
*  \file SVMclassification-gnuplot.cpp
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


#include <Rng/GlobalRng.h>
#include <ReClaM/ArtificialDistributions.h>
#include <ReClaM/Dataset.h>
#include <ReClaM/Svm.h>
#include <ReClaM/ClassificationError.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/time.h>
#ifndef _WIN32
#include <sys/resource.h>
#include <sys/wait.h>
#endif
#include <iostream>


using namespace std;


int main()
{
	unsigned int e;
	double rad;
	Rng::seed(10);

	float C = 5.0;
	float sigma = 1.0;
	unsigned int examples = 100;

	cout << endl;
	cout << "*** Support Vector Machine example program ***" << endl << endl;
	cout << "Data generating distribution: XOR on a 5 x 3.5 rectangle\n";
	cout << "Gaussian RBF kernel width sigma (0.1 to 10): " << flush;
	if( scanf("%g", &sigma) != 1 )
		return( EXIT_FAILURE );
	getchar();				// read the '\n' character
	if (sigma < 0.1) sigma = 0.1;
	if (sigma > 10.0) sigma = 10.0;
	cout << "Complexity parameter C (0.001 to 100): " << flush;
	if( scanf("%g", &C) != 1 )
		return( EXIT_FAILURE );
	getchar();				// read the '\n' character
	if (C < 0.001) C = 0.001;
	if (C > 100.0) C = 100.0;
	cout << "Number of training examples (20 to 200): " << flush;
	if( scanf("%u", &examples) != 1 )
		return( EXIT_FAILURE );
	getchar();		// read the '\n' character
	if (examples < 20) examples = 20;
	if (examples > 200) examples = 200;
	cout << "        sigma = " << sigma << endl;
	cout << "            C = " << C << endl;
	cout << "   # examples = " << examples << endl;
	cout << endl;

	// create the xor problem with uniformly distributed examples
	cout << "Generating " << examples << " examples from the XOR distribution ..." << flush;
	Chessboard chess(2, 2);
	Dataset dataset;
	dataset.CreateFromSource(chess, examples, 10000);
	const Array<double>& x = dataset.getTrainingData();
	const Array<double>& y = dataset.getTrainingTarget();
	int l_plus = 0;
	int l_minus = 0;
	for (e = 0; e < examples; e++)
	{
		if (y(e, 0) > 0.0) l_plus++; else l_minus++;
	}
	cout << " done.\n";
	cout << "   " << l_plus << " positive examples\n";
	cout << "   " << l_minus << " negative examples" << endl;

	// create the SVM for prediction
	double gamma = 0.5 / (sigma * sigma);
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

	// output the non-trivial components of alpha
	cout << "\nThe support vector indices are given in square brackets, each\n";
	cout << "followed by the value of the corresponding alpha coefficient:" << endl;
	unsigned int u, uc = x.dim(0);
	unsigned int SV = 0;
	unsigned int BSV = 0;
	double maxalpha = 0.0;
	for (u = 0; u < uc; u++)
	{
		double v = svm.getAlpha(u);
		if (v != 0.0)
		{
			SV++;
			printf("[%3.3d] ->%7.5g ", u, v);
			if (v >= C) BSV++;
			if (v >= maxalpha) maxalpha = v;
		}
	}
	printf("\nSolution offset: b = %8.6g\n", svm.getOffset());
	cout << "Number of support vectors: " << SV << endl;
	cout << "Number of bounded support vectors: " << BSV << endl;
	cout << endl;

	// estimate the accuracy on the test set
	cout << "Testing ..." << flush;
	ClassificationError ce;
	double acc = 1.0 - ce.error(svm, dataset.getTestData(), dataset.getTestTarget());
	cout << " done." << endl;
	cout << "Estimated accuracy: " << 100.0 * acc << "%" << endl << endl;

	// output the solution to gnuplot
	char gp[65536]; gp[0] = 0;
	strcat(gp, "set title \"SVM example\"\n");
	strcat(gp, "unset key\n");
	strcat(gp, "unset label\n");
	strcat(gp, "unset clabel\n");
	strcat(gp, "set rmargin 0\n");
	strcat(gp, "set lmargin 0\n");
	strcat(gp, "set tmargin 0\n");
	strcat(gp, "set bmargin 0\n");
	strcat(gp, "set multiplot\n");
	strcat(gp, "set origin 0.0, 0.0\n");
	strcat(gp, "set size 1.0, 1.0\n");
	strcat(gp, "set palette gray\n");
	strcat(gp, "set pm3d map\n");
	strcat(gp, "set xrange [0:2]\n");
	strcat(gp, "set yrange [0:2]\n");
	strcat(gp, "set zrange [-10:10]\n");
	strcat(gp, "set samples 100\n");
	strcat(gp, "set isosamples 100\n");
	strcat(gp, "f(x,y)=");
	char part[256];
	bool first = true;
	for (u = 0; u < uc; u++)
	{
		double v = svm.getAlpha(u);
		if (v != 0.0)
		{
			if (! first && v > 0.0) strcat(gp, "+");
			first = false;
			sprintf(part, "%g*exp(-%g*((x-%g)**2+(y-%g)**2))", v, gamma, x(u, 0), x(u, 1));
			strcat(gp, part);
		}
	}
	if (svm.getOffset() >= 0.0) strcat(gp, "+");
	sprintf(part, "%g\n", svm.getOffset());
	strcat(gp, part);
	strcat(gp, "splot f(x,y)\n");
	strcat(gp, "unset pm3d\n");
	strcat(gp, "set contour\n");
	strcat(gp, "unset surface\n");
	strcat(gp, "set cntrparam levels incr 0,1,0\n");
	strcat(gp, "splot f(x,y)\n");
	strcat(gp, "set cntrparam levels incr -1,1,-1\n");
	strcat(gp, "splot f(x,y) with dots\n");
	strcat(gp, "set cntrparam levels incr 1,1,1\n");
	strcat(gp, "splot f(x,y) with dots\n");
	strcat(gp, "unset contour\n");
	strcat(gp, "set surface\n");
	strcat(gp, "set parametric\n");
	strcat(gp, "set isosamples 25\n");
	for (u = 0; u < uc; u++)
	{
		double v = svm.getAlpha(u);
		if (v != 0.0)
		{
			rad = 0.02 * sqrt(fabs(v) / maxalpha);
			sprintf(part, "x(u,v)=%g+%g*cos(u)*cos(v)\ny(u,v)=%g+%g*sin(u)*cos(v)\nz(u,v)=%g*sin(v)\nsplot x(u,v),y(u,v),z(u,v)\n", x(u, 0), rad, x(u, 1), rad, rad);
			strcat(gp, part);
		}
	}
	strcat(gp, "splot u,0,v\n");
	strcat(gp, "splot 0,u,v\n");
	strcat(gp, "unset parametric\n");

	FILE* pipe = popen("gnuplot", "w");
	if (pipe != NULL)
	{
		if( fwrite(gp, 1, strlen(gp), pipe ) != strlen(gp) )
			return( EXIT_FAILURE );
		fflush(pipe);

		cout << "The gnuplot window shows the SVM solution as a gray-coded\n";
		cout << "function on the input space. This function is a non-linear\n";
		cout << "pull-back of the linear SVM solution in a reproducing\n";
		cout << "kernel Hilbert space implicitly defined by the kernel.\n";
		cout << "The cross shows the XOR problem class boundaries, while\n";
		cout << "the solid curved line shows the SVM decision boundary. The\n";
		cout << "dotted curves are the +1 and the -1 niveaus on which all\n";
		cout << "unbounded support vectors are located.\n";
		cout << "The dots represent the support vectors. Their surface\n";
		cout << "areas represent the sizes of the corresponding solution\n";
		cout << "coefficients.\n\n";
		cout << "*** press enter to quit ***" << flush;
		getchar();

#ifndef _WIN32
		wait4(-2, NULL, 0, NULL);
#endif
		pclose(pipe);
	}
	else
	{
		cout << "*** unable to call gnuplot ***" << endl;
	}

	return 0;
}

