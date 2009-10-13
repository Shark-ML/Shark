/*!
*  \file    :  SchwefelEllipsoidCMSA.cpp
*
*  \brief example for CMSA-ES
*
*  The algorithm is described in:
*
*  Covariance Matrix Adaptation Revisited - the CMSA Evolution
*  Strategy - by Hans-Georg Beyer and Bernhard Senhoff, PPSN X, LNCS,
*  Springer-Verlag, 2008
*
*  \author Copyright (c) 2008 Christian Igel
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
*  This file is part of the Shark. This library is free software;
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


#include <EALib/CMSA.h>
#include <EALib/CMA.h>
#include <EALib/ObjectiveFunctions.h>


using namespace std;

int main(int argc, char **argv)
{
	unsigned seed = 0;
	if(argc > 1) seed = atoi(argv[1]);
	Rng::seed(seed);
	unsigned int i;

	//
	// fitness function
	//
	const unsigned Dimension  = 10;
	SchwefelEllipsoidRotated f(Dimension);

	//
	// EA parameters
	//
	const double   GlobalStepInit = 1.;

	// start point
	Array<double> start(Dimension);
	double* p;
	p = &start(0);	
	f.ProposeStartingPoint(p);

	//for(i = 0; i<Dimension; i++) cout << p[i] << " " << start(i) << endl;
	//f.transform(start);
	//for(i = 0; i<Dimension; i++) cout << start(i) << endl;


	// search algorithm
	CMSASearch cma;
	cma.init(f, start, GlobalStepInit, 8);

	// optimization loop
	i = 0;
	do {
		cma.run();
		i++;
		cout << i << "\t" << f.timesCalled() << " "  << cma.bestSolutionFitness() << endl; 
	} while (cma.bestSolutionFitness() > 10E-10);
}
