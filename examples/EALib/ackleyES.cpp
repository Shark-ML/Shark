/*!
*  \file ackleyES.cpp
*
*  \author Martin Kreutz
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
*  \par Project:
*      EALib
*
*
*  <BR>
*
*
*  <BR><HR>
*  This file is part of the EALib. This library is free software;
*  you can redistribute it and/or modify it under the terms of the
*  GNU General Public License as published by the Free Software
*  Foundation; either version 2, or (at your option) any later version.
*
*  This library is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
*  GNU General Public License for more details.
*  You should have received a copy of the GNU General Public License
*  along with this library; if not, write to the Free Software
*  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
*/


#include <SharkDefs.h>
#include <EALib/PopulationT.h>


//=======================================================================
//
// fitness function: Ackley's function
//
double ackley(const std::vector< double >& x)
{
	const double A = 20.;
	const double B = 0.2;
	const double C = M_2PI;

	unsigned i, n;
	double   a, b;

	for (a = b = 0., i = 0, n = x.size(); i < n; ++i) {
		a += x[ i ] * x[ i ];
		b += cos(C * x[ i ]);
	}

	return -A * exp(-B * sqrt(a / n)) - exp(b / n) + A + M_E;
}

//=======================================================================
//
// main program
//
int main(int argc, char **argv)
{
	//
	// constants
	//
	const unsigned Mu           = 15;
	const unsigned Lambda       = 100;
	const unsigned Dimension    = 30;
	const unsigned Iterations   = 500;
	const unsigned Interval     = 10;
	const unsigned NSigma       = 1;

	const double   MinInit      = -3;
	const double   MaxInit      = + 15;
	const double   SigmaInit    = 3;

	const bool     PlusStrategy = false;

	unsigned       i, t;

	//
	// initialize random number generator
	//
	Rng::seed(argc > 1 ? atoi(argv[ 1 ]) : 1234);

	//
	// define populations
	//
	PopulationT<double> parents(Mu,     ChromosomeT< double >(Dimension),
					   ChromosomeT< double >(NSigma));
	PopulationT<double> offsprings(Lambda, ChromosomeT< double >(Dimension),
						  ChromosomeT< double >(NSigma));

	//
	// minimization task
	//
	parents   .setMinimize();
	offsprings.setMinimize();

	//
	// initialize parent population
	//
	for (i = 0; i < parents.size(); ++i) {
		parents[ i ][ 0 ].initialize(MinInit,   MaxInit);
		parents[ i ][ 1 ].initialize(SigmaInit, SigmaInit);
	}

	//
	// selection parameters (number of elitists)
	//
	unsigned numElitists = PlusStrategy ? Mu : 0;

	//
	// standard deviations for mutation of sigma
	//
	double     tau0 = 1. / sqrt(2. * Dimension);
	double     tau1 = 1. / sqrt(2. * sqrt((double)Dimension));

	//
	// evaluate parents (only needed for elitist strategy)
	//
	if (PlusStrategy)
		for (i = 0; i < parents.size(); ++i)
			parents[ i ].setFitness(ackley(parents[ i ][ 0 ]));

	//
	// iterate
	//
	for (t = 0; t < Iterations; ++t) {
		//
		// generate new offsprings
		//
		for (i = 0; i < offsprings.size(); ++i) {
			//
			// select two random parents
			//
			Individual& mom = parents.random();
			Individual& dad = parents.random();

			//
			// recombine object variables discrete, step sizes intermediate
			//
			offsprings[ i ][ 0 ].recombineDiscrete(mom[ 0 ], dad[ 0 ]);
			offsprings[ i ][ 1 ].recombineGenIntermediate(mom[ 1 ], dad[ 1 ]);

			//
			// mutate object variables normal distributed,
			// step sizes log normal distributed
			//
			offsprings[ i ][ 1 ].mutateLogNormal(tau0,  tau1);
			offsprings[ i ][ 0 ].mutateNormal(offsprings[ i ][ 1 ], true);
		}

		//
		// evaluate objective function (parameters in chromosome #0)
		//
		for (i = 0; i < offsprings.size(); ++i)
			offsprings[ i ].setFitness(ackley(offsprings[ i ][ 0 ]));

		//
		// select (mu,lambda) or (mu+lambda)
		//
		parents.selectMuLambda(offsprings, numElitists);

		//
		// print out best value found so far
		//
		if (t % Interval == 0)
			std::cout << t << "\tbest value = "
			<< parents.best().fitnessValue() << std::endl;
	}

	// lines below are for self-testing this example, please ignore
	if(parents.best().fitnessValue()<1.e-14) exit(EXIT_SUCCESS);
	else exit(EXIT_FAILURE);
}

