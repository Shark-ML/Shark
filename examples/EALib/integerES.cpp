/*!
*  \file integerES.cpp
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
*
*  You should have received a copy of the GNU General Public License
*  along with this library; if not, write to the Free Software
*  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
*/

//
// Evolution strategy for solving nonlinear integer problems according to
//     Rudolph, G. (1994). An evolutionary algorithm for integer
//     programming. In PPSN III, LNCS 866 Springer.
//
#include <SharkDefs.h>
#include <EALib/Population.h>

//=======================================================================
//
// fitness function: x' x -> min according to
//     Schwefel, H.P. (1977). Numerische Optimierung von
//     Computer-Modellen mittels der Evolutionsstrategie.
//     Birkhaeuser.
//
double problem_1_1(const std::vector< int >& x)
{
	unsigned i;
	double   sum = 0;

	for (i = 0; i < x.size(); ++i)
		sum += Shark::sqr(x[ i ]);

	return sum;
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
	const unsigned Mu           = 30;
	const unsigned Lambda       = 100;
	const unsigned Dimension    = 30;
	const unsigned Iterations   = 150;
	const unsigned Interval     = 5;
	const unsigned NStepSizes   = 1;

	const int      MinInit      = -1000;
	const int      MaxInit      = + 1000;
	const double   StepSizeInit = 1000 / 3.;

	const bool     PlusStrategy = false;

	unsigned       i, t;

	//
	// initialize random number generator
	//
	Rng::seed(argc > 1 ? atoi(argv[ 1 ]) : 1234);

	//
	// define populations
	//
	Population parents(Mu,     ChromosomeT< int    >(Dimension),
					   ChromosomeT< double >(NStepSizes));
	Population offsprings(Lambda, ChromosomeT< int    >(Dimension),
						  ChromosomeT< double >(NStepSizes));

	//
	// minimization task
	//
	parents   .setMinimize();
	offsprings.setMinimize();

	//
	// initialize parent population
	//
	for (i = 0; i < parents.size(); ++i) {
		dynamic_cast< ChromosomeT< int >& >
		(parents[ i ][ 0 ]).initialize(MinInit, MaxInit);
		dynamic_cast< ChromosomeT< double >& >
		(parents[ i ][ 1 ]).initialize(StepSizeInit, StepSizeInit);
	}

	//
	// selection parameters (number of elitists)
	//
	unsigned numElitists = PlusStrategy ? Mu : 0;

	//
	// standard deviations for mutation of the step size
	//
	double   tau = 1. / sqrt(double(Dimension));

	//
	// evaluate parents (only needed for elitist strategy)
	//
	if (PlusStrategy)
		for (i = 0; i < parents.size(); ++i)
			parents[ i ].setFitness(problem_1_1(dynamic_cast< std::vector< int >& >(parents[ i ][ 0 ])));

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
			// define temporary references for convenience
			//
			ChromosomeT< int >& objvar =
				dynamic_cast< ChromosomeT< int >& >(offsprings[ i ][ 0 ]);
			ChromosomeT< double >& stepSize =
				dynamic_cast< ChromosomeT< double >& >(offsprings[ i ][ 1 ]);

			//
			// recombine object variables discrete, step sizes intermediate
			//
			objvar  .recombineDiscrete(mom[ 0 ], dad[ 0 ]);
			stepSize.recombineIntermediate(mom[ 1 ], dad[ 1 ]);

			//
			// mutate object variables by adding the difference of two
			// independent geometrically distributed random numbers
			// to each component,
			// mutate step sizes lognormally distributed
			//
			stepSize.mutateLogNormal(tau,  0);
			// step Size less than 1 is not sensible
			stepSize.cutOff(1, MaxInit);
			objvar  .mutateDiffGeom(stepSize, true);
		}

		//
		// evaluate objective function (parameters in chromosome #0)
		//
		for (i = 0; i < offsprings.size(); ++i)
			offsprings[ i ].setFitness(problem_1_1(dynamic_cast< std::vector< int >& >(offsprings[ i ][ 0 ])));

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
	if(parents.best().fitnessValue()==0) exit(EXIT_SUCCESS);
	else exit(EXIT_FAILURE);
}

