//===========================================================================
/*!
 *  \file fonSteaddStateMO-CMA.cpp
 *
 *  \brief basic example for the steady-state multi-objective
 *         covariance matrix adaptation evolution strategy, see
 *
 *  Christian Igel, Thorsten Suttorp, and Nikolaus
 *  Hansen. Steady-state Selection and Efficient Covariance Matrix
 *  Update in the Multi-objective CMA-ES. In S. Obayashi et al., eds.,
 *  Proceedings of the Fourth International Conference on Evolutionary
 *  Multi-Criterion Optimization (EMO 2007), pp. 171-185, LNCS 4403,
 *  Springer-Verlag, 2007
 *
 *  and for background information on the MO-CMA-ES
 *
 *  Christian Igel, Nikolaus Hansen, and Stefan Roth. Covariance
 *  Matrix Adaptation for Multi-objective Optimization. Evolutionary
 *  Computation 15(1), pp. 1-28, 2007
 *
 *  \author  Christian Igel
 *  \date    2007
 *
 *  \par Copyright (c) 2005:
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
//===========================================================================

#include <MOO-EALib/PopulationMOO.h>
#include <MOO-EALib/MOOTestFunctions.h>
#include <EALib/ChromosomeCMA.h>
#include <Array/Array.h>
#include <Array/ArraySort.h>

#include <iostream> 
#include <iomanip>



using namespace std;

//
// Carlos Fonseca's  concave benchmark function
//
void computeFitness(IndividualMOO &i) {
	double f1 = FonsecaConcaveF1(dynamic_cast< ChromosomeT< double >& >( i[ 0 ] ));
	double f2 = FonsecaConcaveF2(dynamic_cast< ChromosomeT< double >& >( i[ 0 ] ));

	i.setMOOFitnessValues( f1, f2 );
}


//
// main program
//
int main( int argc, char* argv[] )
{
	unsigned i, t;

	const unsigned Dimension  = 3;   // dimensionality of the problem
	const unsigned Mu         = 100; // size of parent population = size of archive
	const unsigned Iterations = 100 * Mu; // number of generations = number of evaluations

	const double   InitialGlobalStepSize = 1.; // initial value of the sigmas
	const double   MinInit   = -4.;
	const double   MaxInit   = 4.;

	ChromosomeCMA chrom(Dimension);
	IndividualMOO prot (chrom);
	PopulationMOO pop  (Mu + 1, prot);
    
	// set minimization 
	pop.setMinimize();
	
	// set # of objectives
	pop.setNoOfObj( 2 );
    
	// create vector of parameter bounds for for initialization and contraint handling
	vector<double>      stdv(Dimension);
	ChromosomeT<double> lower(Dimension), upper(Dimension);

	//
	// start of MO-CMA algorithm
	//

	// initialize all chromosomes of parent population
	for(i = 0; i < Mu; ++i )
		(dynamic_cast<ChromosomeCMA&>(pop[i][0])).init(Dimension, InitialGlobalStepSize, MinInit, MaxInit);
	
	// evaluate parents
	for(i = 0; i < Mu; ++i ) computeFitness(pop[i]);

	// iterate
	for (t = 1; t <= Iterations; ++t) {
		// reproduce, modify, and evaluate
		unsigned index = Rng::discrete(0, Mu-1); // select a parent
		pop[Mu] = pop[index];                  // reproduce parent

		static_cast< ChromosomeCMA& >(pop[Mu][0]).mutate(); // mutate offspring
		computeFitness(pop[Mu]); // evaluate offspring

		// compute rank and second level sorting criterion
		pop.SMeasure(); // use hypervolume for sorting
		//pop.crowdedDistance(); // use crowding distance for sorting

		// check success of offspring
		(static_cast<ChromosomeCMA&>(pop[Mu][0])).
			updateLambdaSucc(PopulationMOO::compareRankShare(&pop[Mu], &pop[index]) );
		(static_cast<ChromosomeCMA&>(pop[index][0])).
			updateLambdaSucc(PopulationMOO::compareRankShare(&pop[Mu], &pop[index]) );

		// update step sizes
		(static_cast<ChromosomeCMA&>(pop[Mu][0])).updateGlobalStepsize();
		(static_cast<ChromosomeCMA&>(pop[index][0])).updateGlobalStepsize();


		// environmental selection
		std::sort(pop.begin(), pop.end(),  PopulationMOO::compareRankShare);

		// update strategy parameters
		for(i=0; i<Mu; i++) {
			ChromosomeCMA& c = static_cast<ChromosomeCMA&>(pop[i][0]);
			if(c.covarianceUpdateNeeded()) c.updateCovariance();
		}
  }

	// write final objective function values to stdout
	for(i=0; i<Mu; i++) 
		cout << pop[i].getMOOFitness(0) << " " <<  pop[i].getMOOFitness(1) << endl;


  return(EXIT_SUCCESS);
}


