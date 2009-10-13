/*!
*  \file saveStateCMA.cpp
*
*  \author Oswin Krause
*
*  \brief Example of saving the cma state. This example is based on the paraboloidCMA test.
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
* Example of the CMA-ES as use, for example, in <BR>
* N. Hansen and S. Kern. Evaluating the CMA Evolution Strategy on <BR>
* Multimodal Test Functions. In X. Yao et al., eds.,  <BR>
* Parallel Problem Solving from Nature (PPSN VIII), pp. 282-291, <BR>
* Springer-Verlag <BR>
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


#include <EALib/CMA.h>
#include <EALib/ObjectiveFunctions.h>
#include <fstream>


using namespace std;

int main()
{
	Rng::seed(42);

	//
	// fitness function
	//
	const unsigned Dimension  = 8;
	const unsigned a          = 1000;  // determines (square root of) problem condition
	Paraboloid f(Dimension, a);

	//
	// EA parameters
	//
	const unsigned Iterations     = 600;
	const double   MinInit        = .1;
	const double   MaxInit        = .3;
	const double   GlobalStepInit = 1.;

	// start point
	Array<double> start(Dimension);
	start = Rng::uni(MinInit, MaxInit);

	// search algorithm
	CMASearch cma;
	cma.init(f, start, GlobalStepInit);

	// optimization loop
	unsigned int i;
	for (i=0; i<Iterations; i++)
	{
		cma.run();
		cout << f.timesCalled() << " "  << cma.bestSolutionFitness() << endl;
		if(i%10==0)
		{
		    ofstream ostr("test.wgt");
		    ostr<<cma;
		    ostr.close();
		    ifstream istr("test.wgt");
		    istr>>cma;
		}
	}

	// lines below are for self-testing this example, please ignore
	if (cma.bestSolutionFitness() < 10E-10) exit(EXIT_SUCCESS);
	else exit(EXIT_FAILURE);
}
