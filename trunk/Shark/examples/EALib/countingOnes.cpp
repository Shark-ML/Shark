/*!
*  \file countingOnes.cpp
*
*  \author  M. Kreutz
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

#include "EALib/PopulationT.h"

//=======================================================================
//
// fitness function: counting ones problem
//
double ones(const std::vector< bool >& x)
{
	unsigned i;
	double   sum;
	for (sum = 0., i = 0; i < x.size(); sum += x[ i++ ]);
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
	const unsigned PopSize     = 20;
	const unsigned Dimension   = 200;
	const unsigned Iterations  = 1000;
	const unsigned Interval    = 10;
	const unsigned NElitists   = 1;
	const unsigned Omega       = 5;
	const unsigned CrossPoints = 2;
	const double   CrossProb   = 0.6;
	const double   FlipProb    = 1. / Dimension;

	unsigned i, t;

	//
	// initialize random number generator
	//
	Rng::seed(argc > 1 ? atoi(argv[ 1 ]) : 1234);

	//
	// define populations
	//
	PopulationT<bool> parents(PopSize, ChromosomeT< bool >(Dimension));
	PopulationT<bool> offsprings(PopSize, ChromosomeT< bool >(Dimension));

	//
	// scaling window
	//
	std::vector< double > window(Omega);

	//
	// maximization task
	//
	parents   .setMaximize();
	offsprings.setMaximize();

	//
	// initialize all chromosomes of parent population
	//
	for (i = 0; i < parents.size(); ++i)
		parents[ i ][ 0 ].initialize();

	//
	// evaluate parents (only needed for elitist strategy)
	//
	if (NElitists > 0)
		for (i = 0; i < parents.size(); ++i)
			parents[ i ].setFitness(ones(parents[ i ][ 0 ]));

	//
	// iterate
	//
	for (t = 0; t < Iterations; ++t) {
		//
		// recombine by crossing over two parents
		//
		offsprings = parents;
		for (i = 0; i < offsprings.size() - 1; i += 2)
			if (Rng::coinToss(CrossProb))
				offsprings[ i ][ 0 ].crossover(offsprings[ i+1 ][ 0 ],
											   CrossPoints);

		//
		// mutate by flipping bits
		//
		for (i = 0; i < offsprings.size(); ++i)
			offsprings[ i ][ 0 ].flip(FlipProb);

		//
		// evaluate objective function
		//
		for (i = 0; i < offsprings.size(); ++i)
			offsprings[ i ].setFitness(ones(offsprings[ i ][ 0 ]));

		//
		// scale fitness values and use proportional selection
		//
		offsprings.linearDynamicScaling(window, t);
		parents.selectProportional(offsprings, NElitists);

		//
		// print out best value found so far
		//
		if (t % Interval == 0)
			std::cout << t << "\tbest value = "
			<< parents.best().fitnessValue() << "\n";
	}

	// lines below are for self-testing this example, please ignore
	if(parents.best().fitnessValue()>=200) exit(EXIT_SUCCESS);
	else exit(EXIT_FAILURE);
}

