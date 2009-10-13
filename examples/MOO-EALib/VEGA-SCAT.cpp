
/* ======================================================================
 *
 *  \file VEGA-SCAT.cpp
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


#include <MOO-EALib/PopulationMOO.h>
#include <MOO-EALib/ArchiveMOO.h>
#include <MOO-EALib/TestFunction.h>
#include <Array/Array.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <fstream>
#include <SpreadCAT/SpreadCAT.h>

int main(void)
{
	//Variable
	VarT< bool     > MOOF1("MOOF1"      , true);
	VarT< bool     > MOOF2("MOOF2"      , false);
	VarT< bool     > MOOF3("MOOF3"      , false);
	VarT< bool     > MOOF4("MOOF4"      , false);
	VarT< bool     > MOOF5("MOOF5"      , false);
	VarT< unsigned > PopSize("PopSize"    , 100);
	VarT< unsigned > Dimension("Dimension"  , 2);
	VarT< unsigned > NumOfBits("NumOfBits"  , 10);
	VarT< unsigned > Iterations("Iterations" , 500);
	VarT< unsigned > DspInterval("DspInterval", 100);
	VarT< bool     > Stopping("Stopping"   , false);
	VarT< unsigned > Omega("Omega"      , 5);
	VarT< unsigned > CrossPoints("CrossPoints", 2);
	VarT< double   > CrossProb("CrossProb"  , 0.6);
	VarT< double   > FlipProb("FlipProb"   , 0.005);
	VarT< bool     > UseGrayCode("UseGrayCode", true);
	VarT< unsigned > Seed("Seed"       , 1234);
	VarT< double   > Start("Start"      , 0);
	VarT< double   > End("End"        , 1);
	VarT< unsigned > FileNumber("FileNumber" , 1);
	VarT< bool     > VEGA("VEGA"    , true);
	VarT< bool     > DOUBLEGA("DOUBLEGA", false);

	const unsigned NElitists = 0;

	//SpreadCAT
	SpreadCAT mySpread("VEGA.conf");

	//Other Variable
	const Interval RangeOfValues(Start, End);
	const unsigned ChromLen    = Dimension * NumOfBits;
	const unsigned SubPopSize  = PopSize / 2;
	double X1[(unsigned)PopSize ], Y1[(unsigned)PopSize ];
	double X2[(unsigned)PopSize ], Y2[(unsigned)PopSize ];

	//Random Seed
	Rng::seed(Seed);

	//Define Populations
	Population parents(PopSize   , ChromosomeT< bool >(ChromLen));
	Population parents1(SubPopSize, ChromosomeT< bool >(ChromLen));
	Population parents2(SubPopSize, ChromosomeT< bool >(ChromLen));
	Population offsprings(PopSize   , ChromosomeT< bool >(ChromLen));
	Population offsprings1(SubPopSize, ChromosomeT< bool >(ChromLen));
	Population offsprings2(SubPopSize, ChromosomeT< bool >(ChromLen));

	// Archive for output
	ArchiveMOO archive(PopSize);
	archive.minimize();

	//Scaling Window
	std::vector< double > window1(Omega);
	std::vector< double > window2(Omega);

	//Temporary Chromosome for decoding
	ChromosomeT< double > dblchrom;

	//Minimization Task
	parents    .setMinimize();
	parents1   .setMinimize();
	parents2   .setMinimize();
	offsprings .setMinimize();
	offsprings1.setMinimize();
	offsprings2.setMinimize();

	//Initialization
	for (unsigned i = 0; i < PopSize; i++) {
		dynamic_cast< ChromosomeT< bool >& >(parents[i][0]).initialize();
	}

	// Evaluation
	for (unsigned i = 0; i < PopSize; i++) {
		if (MOOF1) {
			dblchrom.decodeBinary(parents[i][0], RangeOfValues, NumOfBits, UseGrayCode);
			if (i < SubPopSize) {
				parents[i].setFitness(SphereF1(dblchrom));
			}
			else {
				parents[i].setFitness(SphereF2(dblchrom));
			}
		}
		else if (MOOF2) {
			dblchrom.decodeBinary(parents[i][0], RangeOfValues, NumOfBits, UseGrayCode);
			if (i < SubPopSize) {
				parents[i].setFitness(DebConvexF1(dblchrom));
			}
			else {
				parents[i].setFitness(DebConvexF2(dblchrom));
			}
		}
		else if (MOOF3) {
			dblchrom.decodeBinary(parents[i][0], RangeOfValues, NumOfBits, UseGrayCode);
			if (i < SubPopSize) {
				parents[i].setFitness(DebConcaveF1(dblchrom));
			}
			else {
				parents[i].setFitness(DebConcaveF2(dblchrom));
			}
		}
		else if (MOOF4) {
			dblchrom.decodeBinary(parents[i][0], RangeOfValues, NumOfBits, UseGrayCode);
			if (i < SubPopSize) {
				parents[i].setFitness(DebDiscreteF1(dblchrom));
			}
			else {
				parents[i].setFitness(DebDiscreteF2(dblchrom));
			}
		}
		else if (MOOF5) {
			dblchrom.decodeBinary(parents[i][0], RangeOfValues, NumOfBits, UseGrayCode);
			if (i < SubPopSize) {
				parents[i].setFitness(FonsecaConcaveF1(dblchrom));
			}
			else {
				parents[i].setFitness(FonsecaConcaveF2(dblchrom));
			}
		}
	}

	// Iteration
	for (unsigned t = 1; t <= Iterations; t++) {

		std::cout << "Generation = " << t << std::endl;

		// Copy an offspring to a parent
		offsprings = parents;

		// Divide
		for (unsigned i = 0; i < SubPopSize; i++) {
			parents1   [ i ] = parents   [ i ];
			parents2   [ i ] = parents   [ i + SubPopSize ];
			offsprings1[ i ] = offsprings[ i ];
			offsprings2[ i ] = offsprings[ i + SubPopSize ];
		}

		// Selection
		offsprings1.linearDynamicScaling(window1, t);
		parents1.selectProportional(offsprings1, NElitists);
		offsprings2.linearDynamicScaling(window2, t);
		parents2.selectProportional(offsprings2, NElitists);

		std::cout << "   best value ( 1 ) = " << parents1.best().fitnessValue() << std::endl;
		std::cout << "   best value ( 2 ) = " << parents2.best().fitnessValue() << std::endl;

		// Gather
		for (unsigned i = 0; i < SubPopSize; i++) {
			parents[ i              ] = parents1[ i ];
			parents[ i + SubPopSize ] = parents2[ i ];
		}

		// Store Data
		offsprings = parents;

		// Shuffle
		if (VEGA) {
			parents.shuffle();
		}

		// Crossover
		for (unsigned i = 0; i < PopSize - 1; i += 2) {
			if (Rng::coinToss(CrossProb)) {
				parents[i][0].crossover(parents[i+1][0], CrossPoints);
			}
		}

		// Mutation
		for (unsigned i = 0; i < PopSize; i++) {
			dynamic_cast< ChromosomeT< bool >& >(parents[i][0]).flip(FlipProb);
		}

		// Evaluation
		for (unsigned i = 0; i < PopSize; i++) {
			if (MOOF1) {
				dblchrom.decodeBinary(parents[i][0], RangeOfValues, NumOfBits, UseGrayCode);
				if (i < SubPopSize) {
					parents[i].setFitness(SphereF1(dblchrom));
				}
				else {
					parents[i].setFitness(SphereF2(dblchrom));
				}
			}
			else if (MOOF2) {
				dblchrom.decodeBinary(parents[i][0], RangeOfValues, NumOfBits, UseGrayCode);
				if (i < SubPopSize) {
					parents[i].setFitness(DebConvexF1(dblchrom));
				}
				else {
					parents[i].setFitness(DebConvexF2(dblchrom));
				}
			}
			else if (MOOF3) {
				dblchrom.decodeBinary(parents[i][0], RangeOfValues, NumOfBits, UseGrayCode);
				if (i < SubPopSize) {
					parents[i].setFitness(DebConcaveF1(dblchrom));
				}
				else {
					parents[i].setFitness(DebConcaveF2(dblchrom));
				}
			}
			else if (MOOF4) {
				dblchrom.decodeBinary(parents[i][0], RangeOfValues, NumOfBits, UseGrayCode);
				if (i < SubPopSize) {
					parents[i].setFitness(DebDiscreteF1(dblchrom));
				}
				else {
					parents[i].setFitness(DebDiscreteF2(dblchrom));
				}
			}
			else if (MOOF5) {
				dblchrom.decodeBinary(parents[i][0], RangeOfValues, NumOfBits, UseGrayCode);
				if (i < SubPopSize) {
					parents[i].setFitness(FonsecaConcaveF1(dblchrom));
				}
				else {
					parents[i].setFitness(FonsecaConcaveF2(dblchrom));
				}
			}
		}

		// Data Store for Display
		for (unsigned i = 0; i < PopSize; i++) {
			if (MOOF1) {
				dblchrom.decodeBinary(parents[i][0], RangeOfValues, NumOfBits, UseGrayCode);
				X1[ i ] = SphereF1(dblchrom);
				Y1[ i ] = SphereF2(dblchrom);
				dblchrom.decodeBinary(offsprings[i][0], RangeOfValues, NumOfBits, UseGrayCode);
				X2[ i ] = SphereF1(dblchrom);
				Y2[ i ] = SphereF2(dblchrom);
			}
			else if (MOOF2) {
				dblchrom.decodeBinary(parents[i][0], RangeOfValues, NumOfBits, UseGrayCode);
				X1[ i ] = DebConvexF1(dblchrom);
				Y1[ i ] = DebConvexF2(dblchrom);
				dblchrom.decodeBinary(offsprings[i][0], RangeOfValues, NumOfBits, UseGrayCode);
				X2[ i ] = DebConvexF1(dblchrom);
				Y2[ i ] = DebConvexF2(dblchrom);
			}
			else if (MOOF3) {
				dblchrom.decodeBinary(parents[i][0], RangeOfValues, NumOfBits, UseGrayCode);
				X1[ i ] = DebConcaveF1(dblchrom);
				Y1[ i ] = DebConcaveF2(dblchrom);
				dblchrom.decodeBinary(offsprings[i][0], RangeOfValues, NumOfBits, UseGrayCode);
				X2[ i ] = DebConcaveF1(dblchrom);
				Y2[ i ] = DebConcaveF2(dblchrom);
			}
			else if (MOOF4) {
				dblchrom.decodeBinary(parents[i][0], RangeOfValues, NumOfBits, UseGrayCode);
				X1[ i ] = DebDiscreteF1(dblchrom);
				Y1[ i ] = DebDiscreteF2(dblchrom);
				dblchrom.decodeBinary(offsprings[i][0], RangeOfValues, NumOfBits, UseGrayCode);
				X2[ i ] = DebDiscreteF1(dblchrom);
				Y2[ i ] = DebDiscreteF2(dblchrom);
			}
			else if (MOOF5) {
				dblchrom.decodeBinary(parents[i][0], RangeOfValues, NumOfBits, UseGrayCode);
				X1[ i ] = FonsecaConcaveF1(dblchrom);
				Y1[ i ] = FonsecaConcaveF2(dblchrom);
				dblchrom.decodeBinary(offsprings[i][0], RangeOfValues, NumOfBits, UseGrayCode);
				X2[ i ] = FonsecaConcaveF1(dblchrom);
				Y2[ i ] = FonsecaConcaveF2(dblchrom);
			}
		}
		archive.cleanArchive();
		for (unsigned i = 0; i < PopSize; i++) {
			IndividualMOO TEMP(parents[i]);
			TEMP.setNoOfObj(2);
			TEMP.setMOOFitnessValues(X1[i], Y1[i]);
			archive.addArchive(TEMP);
		}
		archive.nonDominatedSolutions();
	}

	char filename[50];
	unsigned fileno = FileNumber;
	// unsigned popsizef = PopSize;
	sprintf(filename, "Archive%i.txt", fileno);
	archive.saveArchive(filename);
	std::cout << "\nThe result is stored in " << filename << "\n" << std::endl;
	std::cout << "Number of solutions" << std::endl;
	std::cout << "Number of objectives" << std::endl;
	std::cout << "Solution 1 (f1,f2,...)" << std::endl;
	std::cout << "Solution 2 (f1,f2,...)\n" << std::endl;

	std::cout << "Hit return key for continue" << std::endl;
	fgetc(stdin);

	return 0;
}


