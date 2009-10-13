
/* ======================================================================
 *
 *  \file NSGA2-SCAT.cpp
 *
 *  \brief Sample Program for MOO-ES
 *  \author Tatsuya Okabe <tatsuya.okabe@honda-ri.de>
 *
 *  \par 
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>

 *  \par Project:
 *      MOO-EALib
 *
 *
 *  <BR>
 *
 *
 *  <BR><HR>
 *  This file is part of the MOO-EALib. This library is free software;
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
// 	Authors message
//======================================================================
/*	Thank you very much for your interest to MOO-EALib.

Since our company's name was changed on 1st, January, 2003,
my E-mail address in the source codes were also changed.
The current E-mail address (6th,Feb.,2004) is as follows:

tatsuya.okabe@honda-ri.de.

If you cannot contact me with the above E-mail address,
you can also use the following E-mail address:

t_okabe_de@hotmail.com.

If you have any questions, please don't hesitate to
ask me. It's my pleasure.

Best Regards,
Tatsuya Okabe

*********************************************************
Tatsuya Okabe
Honda Research Institute Europe GmbH
Carl-Legien-Strasse 30, 63073 Offenbach/Main, Germany
Tel: +49-69-89011-745
Fax: +49-69-89011-749
**********************************************************/


#include "MOO-EALib/PopulationMOO.h"
#include "MOO-EALib/ArchiveMOO.h"
#include "MOO-EALib/TestFunction.h"
#include "Array/Array.h"
#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#include "fstream"
#include "SpreadCAT/SpreadCAT.h"


// main program
int main(void)
{

	// constants
	VarT< unsigned > Seed("Seed"       , 1234);
	VarT< bool     > MOOF1("MOOF1"      , true);
	VarT< bool     > MOOF2("MOOF2"      , false);
	VarT< bool     > MOOF3("MOOF3"      , false);
	VarT< bool     > MOOF4("MOOF4"      , false);
	VarT< bool     > MOOF5("MOOF5"      , false);
	VarT< unsigned > PopSize("PopSize"    , 100);
	VarT< unsigned > Dimension("Dimension"  , 2);
	VarT< unsigned > NumOfBits("NumOfBits"  , 10);
	VarT< unsigned > Iterations("Iterations" , 1000);
	VarT< bool     > STOPPING("STOPPING"   , false);
	VarT< unsigned > DspInterval("DspInterval", 10);
	VarT< unsigned > CrossPoints("CrossPoints", 2);
	VarT< double   > CrossProb("CrossProb"  , 0.6);
	VarT< double   > FlipProb("FlipProb"   , 0.002);
	VarT< bool     > UseGrayCode("UseGrayCode", true);
	VarT< double   > Start("Start"      , -2);
	VarT< double   > End("End"        , 2);
	VarT< unsigned > FileNumber("FileNumber" , 1);

	//SpreadCAT
	SpreadCAT mySpread("NSGA2.conf");

	//Other Variable
	const Interval RangeOfValues(Start, End);
	const unsigned fileno = FileNumber;
	char filename[500];
	double f1 = 0, f2 = 0;
	double PF1[(unsigned)PopSize ], PF2[(unsigned)PopSize ];
	double OF1[(unsigned)PopSize ], OF2[(unsigned)PopSize ];

	// initialize random number generator
	Rng::seed(Seed);
	// define populations
	PopulationMOO parents(PopSize, ChromosomeT< bool >(Dimension * NumOfBits));
	PopulationMOO offsprings(PopSize, ChromosomeT< bool >(Dimension * NumOfBits));
	// Set Minimization Task
	parents   .setMinimize();
	offsprings.setMinimize();
	// Set No of Objective funstions
	parents   .setNoOfObj(2);
	offsprings.setNoOfObj(2);
	// Temporary chromosomes for decoding
	ChromosomeT< double > dblchrom;
	// initialize all chromosomes of parent population
	for (unsigned i = 0; i < parents.size(); ++i) {
		dynamic_cast< ChromosomeT< bool >& >(parents[ i ][ 0 ]).initialize();
	}
	// evaluate parents
	for (unsigned i = 0; i < parents.size(); ++i) {
		dblchrom.decodeBinary(parents[ i ][ 0 ], RangeOfValues, NumOfBits, UseGrayCode);
		if (MOOF1) {
			f1 = SphereF1(dblchrom);
			f2 = SphereF2(dblchrom);
		}
		else if (MOOF2) {
			f1 = DebConvexF1(dblchrom);
			f2 = DebConvexF2(dblchrom);
		}
		else if (MOOF3) {
			f1 = DebConcaveF1(dblchrom);
			f2 = DebConcaveF2(dblchrom);
		}
		else if (MOOF4) {
			f1 = DebDiscreteF1(dblchrom);
			f2 = DebDiscreteF2(dblchrom);
		}
		else if (MOOF5) {
			f1 = FonsecaConcaveF1(dblchrom);
			f2 = FonsecaConcaveF2(dblchrom);
		}
		parents[ i ].setMOOFitnessValues(f1, f2);
		PF1[ i ] = f1;
		PF2[ i ] = f2;
	}

	// iterate
	for (unsigned t = 1; t < Iterations + 1; ++t) {
		std::cout << "Generations : " << t << std::endl;
		// copy parents to offsprings
		offsprings = parents;
		// recombine by crossing over two parents
		for (unsigned i = 0; i < offsprings.size() - 1; i += 2) {
			if (Rng::coinToss(CrossProb)) {
				offsprings[ i ][ 0 ].crossover(offsprings[ i+1 ][ 0 ], CrossPoints);
			}
		}
		// mutate by flipping bits
		for (unsigned i = 0; i < offsprings.size(); ++i) {
			dynamic_cast< ChromosomeT< bool >& >(offsprings[ i ][ 0 ]).flip(FlipProb);
		}
		// evaluate objective function
		for (unsigned i = 0; i < offsprings.size(); ++i) {
			dblchrom.decodeBinary(offsprings[ i ][ 0 ], RangeOfValues, NumOfBits, UseGrayCode);
			if (MOOF1) {
				f1 = SphereF1(dblchrom);
				f2 = SphereF2(dblchrom);
			}
			else if (MOOF2) {
				f1 = DebConvexF1(dblchrom);
				f2 = DebConvexF2(dblchrom);
			}
			else if (MOOF3) {
				f1 = DebConcaveF1(dblchrom);
				f2 = DebConcaveF2(dblchrom);
			}
			else if (MOOF4) {
				f1 = DebDiscreteF1(dblchrom);
				f2 = DebDiscreteF2(dblchrom);
			}
			else if (MOOF5) {
				f1 = FonsecaConcaveF1(dblchrom);
				f2 = FonsecaConcaveF2(dblchrom);
			}
			offsprings[ i ].setMOOFitnessValues(f1, f2);
			OF1[ i ] = f1;
			OF2[ i ] = f2;
		}

		// Selection
		parents.crowdedTournamentSelection(offsprings);
		for (unsigned k = 0; k < parents.size(); k++) {
			PF1[ k ] = parents[ k ].getMOOFitness(0);
			PF2[ k ] = parents[ k ].getMOOFitness(1);
		}

		if (STOPPING) {
			if (t % DspInterval == 0) {
				std::cout << "\nHit return key" << std::endl;
				fgetc(stdin);
			}
		}
	} // Iteration

	// Data Output
	ArchiveMOO archive(PopSize);
	archive.minimize();
	for (unsigned ii = 0; ii < PopSize; ii++) {
		archive.addArchive(parents[ ii ]);
	}
	archive.nonDominatedSolutions();
	sprintf(filename, "Archive%i.txt", fileno);
	archive.saveArchive(filename);
	std::cout << "\nThe result is stored in " << filename << "\n" << std::endl;
	std::cout << "Number of Solutions" << std::endl;
	std::cout << "Number of Objectives" << std::endl;
	std::cout << "Solution 1 (f1,f2,...)" << std::endl;
	std::cout << "Solution 2 (f1,f2,...)\n" << std::endl;

	std::cout << "Hit return key" << std::endl;

	fgetc(stdin);

	return 0;

}

