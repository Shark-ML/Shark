/*!
*  \file SVMregression-gnuplot.cpp
*
*  \author C. Igel
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
#include <ReClaM/Svm.h>
#include <stdio.h>
#ifndef _WIN32
#include <sys/wait.h>
#endif
#include <math.h>
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
	cout << "with additive Gaussian white noise." << endl;
	cout << endl;

	unsigned int e;
	double v;
	Rng::seed(42);

	double C = 1e10;
	double epsilon = 0.01;
	double sigma = 2.0;
	unsigned int examples = 100;

	// create the sinc problem
	Array<double> x(examples, 1);
	Array<double> y(examples, 1);
	for (e = 0; e < examples; e++)
	{
		x(e, 0) = Rng::uni(-12.0, 12.0);
		y(e, 0) = sinc(x(e, 0)) + Rng::uni(-epsilon, epsilon);
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
	cout << " done." << endl;

	// output the non-trivial components of alpha
	cout << endl << "The support vector indices are given in square brackets, each" << endl;
	cout << "followed by the value of the corresponding alpha coefficient:" << endl;
	unsigned int u, uc = x.dim(0);
	unsigned int SV = 0;
	unsigned int BSV = 0;
	double maxalpha = 0.0;
	for (u = 0; u < uc; u++)
	{
		v = svm.getAlpha(u);
		if (v != 0.0)
		{
			SV++;
			printf("[%3.3d] ->%7.5g ", u, v);
			v = fabs(v);
			if (v >= C) BSV++;
			if (v >= maxalpha) maxalpha = v;
		}
	}
	printf("\nSolution offset: b = %8.6g\n", svm.getOffset());
	cout << "Number of support vectors: " << SV << endl;
	cout << "Number of bounded support vectors: " << BSV << endl;
	cout << endl;

	// output the solution to gnuplot
	char gp[65536]; gp[0] = 0;
	strcat(gp, "set title \"SVM regression example\"\n");
	strcat(gp, "unset key\n");
	strcat(gp, "unset label\n");
	strcat(gp, "unset clabel\n");
	strcat(gp, "set rmargin 0\n");
	strcat(gp, "set lmargin 0\n");
	strcat(gp, "set tmargin 0\n");
	strcat(gp, "set bmargin 0\n");
	strcat(gp, "set multiplot\n");
	strcat(gp, "set xrange [-12:12]\n");
	strcat(gp, "set yrange [-0.7:1.5]\n");
	strcat(gp, "set samples 100\n");
	strcat(gp, "set isosamples 100\n");
	strcat(gp, "f(x,y)=");
	char part[256];
	bool first = true;
	for (u = 0; u < uc; u++)
	{
		v = svm.getAlpha(u);
		if (v != 0.0)
		{
			if (! first && v > 0.0) strcat(gp, "+");
			first = false;
			sprintf(part, "%g*exp(-%g*((x-%g)**2))", v, gamma, x(u, 0));
			strcat(gp, part);
		}
	}
	if (svm.getOffset() >= 0.0) strcat(gp, "+");
	sprintf(part, "%g\n", svm.getOffset());
	strcat(gp, part);
	sprintf(part, "plot sin(x)/x, f(x), f(x)-%g, f(x)+%g\n", epsilon, epsilon);
	strcat(gp, part);
	strcat(gp, "set parametric\n");
	strcat(gp, "set isosamples 25\n");
	for (u = 0; u < uc; u++)
	{
		v = fabs(svm.getAlpha(u));
		if (v > 0.0)
		{
			sprintf(part, "x(t)=%g+0.06*cos(t)\ny(t)=%g+0.01*sin(t)\nplot x(t),y(t)\n", x(u, 0), y(u, 0));
			strcat(gp, part);
		}
		else
		{
			sprintf(part, "x(t)=%g+0.03*cos(t)\ny(t)=%g+0.005*sin(t)\nplot x(t),y(t)\n", x(u, 0), y(u, 0));
			strcat(gp, part);
		}
	}
	strcat(gp, "plot 10*t,0\n");
	strcat(gp, "plot 0,t\n");
	strcat(gp, "unset parametric\n");

	FILE* pipe = popen("gnuplot", "w");
	if (pipe != NULL)
	{
		if( fwrite( gp, 1, strlen(gp), pipe ) != strlen(gp) )
			return( EXIT_FAILURE );
		fflush(pipe);

		cout << "The gnuplot window shows the target function as well as\n";
		cout << "the SVM solution. This function is a non-linear pull-back\n";
		cout << "of the linear SVM solution in a reproducing kernel\n";
		cout << "Hilbert space implicitly defined by the kernel.\n";
		cout << "The epsilon-tube around the solution function is drawn.\n";
		cout << "All support vectors (large circles) are outside the tube\n";
		cout << "or on its boundary, while the other vectors\n";
		cout << "(small circles) are inside the tube. Only the support\n";
		cout << "vectors contribute to the solution resulting in a sparse\n";
		cout << "representation.\n\n";
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

	return EXIT_SUCCESS;
}

