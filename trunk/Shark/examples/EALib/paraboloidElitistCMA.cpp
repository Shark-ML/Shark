/*!
*  \file paraboloidElitistCMA.cpp
*
*  \author Stefan Roth
*
*  \brief simple example for the elitist CMA evolution strategy
* 
*  simple example for the elitist CMA evolution strategy as described
*  in 
*  Christian Igel, Thorsten Suttorp, and Nikolaus Hansen. A
*  Computational Efficient Covariance Matrix Update and a (1+1)-CMA
*  for Evolution Strategies. Proceedings of the Genetic and
*  Evolutionary Computation Conference (GECCO 2006), ACM Press, 2006
*
*  see http://www.neuroinformatik.ruhr-uni-bochum.de/PEOPLE/igel/solutions.html
* for further examples
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

#include <EALib/PopulationT.h>
#include <EALib/ChromosomeCMA.h>
#include <ReClaM/Paraboloid.h>


using namespace std;


int main(int argc, char* argv[])
{
	int lambdaSucc;
	unsigned i, t;

	//
	// fitness function
	//
	const unsigned Dimension  = 10;
	const unsigned a          = 1000;  // determines problem condition
	const bool     rotate     = true;
	Paraboloid f(Dimension, a, rotate);

	//
	// EA parameters
	//
	const unsigned Iterations     = 10000;
	const double   MinInit        = -3.;
	const double   MaxInit        = 7.;
	const double   GlobalStepInit = 1.;
	const unsigned Lambda = 1;

	PopulationCT<ChromosomeCMA>   parent(1, 1);
	PopulationCT<ChromosomeCMA>   offspring(Lambda, 1);

	parent.setMinimize();
	offspring.setMinimize();

	// init single parent
	parent[0][0].init(Dimension, GlobalStepInit, MinInit, MaxInit, Lambda);
	parent[0].setFitness(f.error(parent[0][0]));

	// loop over generations
	for (t = 0; t < Iterations; t++) {
		lambdaSucc = 0;
		IndividualCT<ChromosomeCMA> parentIndiv = parent[0];
		// generate Lambda offspring
		for (i = 0; i < Lambda; i++) {
			offspring[i] = parentIndiv;
			offspring[i][0].mutate();
			offspring[i].setFitness(f.error(offspring[i][0]));
			if (offspring[i].getFitness() <= parentIndiv.getFitness()) {
				lambdaSucc++;
				if (offspring[i].getFitness() <= parent[0].getFitness()) parent[0] = offspring[i];
			}
		}

		if ((t % 100) == 0)
			cout << t << " " 	<< parent[0].getFitness() << endl;

		// update strategy parameters
		parent[0][0].updateGlobalStepsize(lambdaSucc);
		if (lambdaSucc) parent[0][0].updateCovariance(parentIndiv);
	}

	cout << t << " " 	<< parent[0].getFitness() << endl;

	// lines below are for self-testing this example, please ignore
	if (parent[0].getFitness() < 1e-50) exit(EXIT_SUCCESS);
	else exit(EXIT_FAILURE);
}
