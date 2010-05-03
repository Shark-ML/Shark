// ======================================================================
/*!
 *  \file VEGA.cpp
 *  \date 2004-11-15
 *
 *  \brief Sample Program for MOO-ES
 *
 *  \author Stefan Roth
 *
 *  \par Maintained by:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
 *
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
/* This file is derived from VEGA-SCAT.cpp which was written by
Tatsuya Okabe <tatsuya.okabe@honda-ri.de>.
**********************************************************/


#include <iostream>
#include <string>
#include <fstream>

#include <MOO-EALib/PopulationMOO.h>
#include <MOO-EALib/ArchiveMOO.h>
#include <MOO-EALib/TestFunction.h>
#include <Array/Array.h>
#include <FileUtil/Params.h>
#include <SharkDefs.h>

// My own derived class for managing my configuration files:
//
class MyParams : public Params
{
public:

	MyParams(int argc, char **argv) : Params(argc, argv)
	{ }

	~MyParams()
	{ }

	void readParams()
	{
		// Call the main program with parameters "-conf [filename]"
		if (scanFrom(confFile.c_str())) {
			std::cout << "Name of the configuration file: " << confFile << std::endl;
		}
		else {
		  throw SHARKEXCEPTION("No valid configuration file given! ");
		}
	}

	void io(std::istream& is, std::ostream& os, FileUtil::iotype type)
	{
		FileUtil::io(is, os, "Seed  "       , Seed		, 1234u	, type);
		FileUtil::io(is, os, "PopSize "     , PopSize	, 100u	, type);
		FileUtil::io(is, os, "Dimension "   , Dimension	, 2u	, type);
		FileUtil::io(is, os, "Iterations "  , Iterations	, 1000u	, type);
		FileUtil::io(is, os, "FileNumber "  , FileNumber	, 1u	, type);
		FileUtil::io(is, os, "NumOfBits "   , NumOfBits	, 10u	, type);
		FileUtil::io(is, os, "DspInterval " , DspInterval	, 100u	, type);
		FileUtil::io(is, os, "Omega "       , Omega	, 5u	, type);
		FileUtil::io(is, os, "CrossPoints " , CrossPoints	, 2u	, type);
		FileUtil::io(is, os, "MOOF "        , MOOF	        , 1u	, type);
		FileUtil::io(is, os, "Stopping "    , Stopping	, false	, type);
		FileUtil::io(is, os, "UseGrayCode " , UseGrayCode	, true	, type);
		FileUtil::io(is, os, "VEGA "	 , VEGA		, true	, type);
		FileUtil::io(is, os, "DOUBLEGA "    , DOUBLEGA	, false	, type);
		FileUtil::io(is, os, "CrossProb "   , CrossProb	, 0.6	, type);
		FileUtil::io(is, os, "FlipProb "    , FlipProb	, 0.005	, type);
		FileUtil::io(is, os, "Start "       , Start	, 0.	, type);
		FileUtil::io(is, os, "End "         , End		, 1.	, type);
	}

	// Method to show the current content of the class variable:
	void monitor()
	{
		std::cout << "Seed        " << Seed		<< std::endl;//1234);
		std::cout << "PopSize     " << PopSize	<< std::endl;//100);
		std::cout << "Dimension   " << Dimension	<< std::endl;//2);
		std::cout << "Iterations  " << Iterations	<< std::endl;//1000);
		std::cout << "FileNumber  " << FileNumber	<< std::endl;//1);
		std::cout << "NumOfBits   " << NumOfBits	<< std::endl;//10);
		std::cout << "DspInterval " << DspInterval	<< std::endl;//10);
		std::cout << "Omega       " << Omega	<< std::endl;//5);
		std::cout << "CrossPoints " << CrossPoints	<< std::endl;//2);
		std::cout << "MOOF       "  << MOOF	        << std::endl;//true);
		std::cout << "Stopping    " << Stopping	<< std::endl;//false);
		std::cout << "UseGrayCode " << UseGrayCode	<< std::endl;//true);
		std::cout << "VEGA        "	<< VEGA		<< std::endl;//false);
		std::cout << "DOUBLEGA    "	<< DOUBLEGA	<< std::endl;//true);
		std::cout << "CrossProb   " << CrossProb	<< std::endl;//0.6);
		std::cout << "FlipProb    " << FlipProb	<< std::endl;//0.002);
		std::cout << "Start       " << Start	<< std::endl;//-2);
		std::cout << "End         " << End		<< std::endl;//2);
	}

	unsigned Seed;
	unsigned PopSize;
	unsigned Dimension;
	unsigned Iterations;
	unsigned FileNumber;
	unsigned NumOfBits;
	unsigned DspInterval;
	unsigned Omega;
	unsigned CrossPoints;
	unsigned MOOF;
	bool Stopping;
	bool UseGrayCode;
	bool Goldberg;
	bool Fonseca;
	bool VEGA;
	bool DOUBLEGA;
	double CrossProb;
	double FlipProb;
	double Start;
	double End;
};

// main program
int main(int argc, char* argv[])
{
	unsigned i, t;

	MyParams param(argc, argv);

	param.setDefault();
	//param.readParams();
	param.monitor();

	const unsigned NElitists = 0;

	//Other Variable
	const Interval RangeOfValues(param.Start, param.End);
	const unsigned ChromLen    = param.Dimension * param.NumOfBits;
	const unsigned SubPopSize  = param.PopSize / 2;

	double *X1 = new double[(unsigned)param.PopSize ];
	double *X2 = new double[(unsigned)param.PopSize ];
	double *Y1 = new double[(unsigned)param.PopSize ];
	double *Y2 = new double[(unsigned)param.PopSize ];

	// initialize random number generator
	Rng::seed(param.Seed);

	//Define Populations
	Population parents(param.PopSize, ChromosomeT< bool >(ChromLen));
	Population parents1(SubPopSize, ChromosomeT< bool >(ChromLen));
	Population parents2(SubPopSize, ChromosomeT< bool >(ChromLen));
	Population offsprings(param.PopSize, ChromosomeT< bool >(ChromLen));
	Population offsprings1(SubPopSize, ChromosomeT< bool >(ChromLen));
	Population offsprings2(SubPopSize, ChromosomeT< bool >(ChromLen));

	// Archive for output
	ArchiveMOO archive(param.PopSize);
	archive.minimize();

	//Scaling Window
	std::vector< double > window1(param.Omega);
	std::vector< double > window2(param.Omega);

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
	for (i = 0; i < param.PopSize; i++) {
		dynamic_cast< ChromosomeT< bool >& >(parents[i][0]).initialize();
	}

	// Evaluation
	for (i = 0; i < param.PopSize; i++) {
		switch (param.MOOF) {
		case 1:
			dblchrom.decodeBinary(parents[i][0], RangeOfValues, param.NumOfBits, param.UseGrayCode);
			if (i < SubPopSize) {
				parents[i].setFitness(SphereF1(dblchrom));
			}
			else {
				parents[i].setFitness(SphereF2(dblchrom));
			}
			break;
		case 2:
			dblchrom.decodeBinary(parents[i][0], RangeOfValues, param.NumOfBits, param.UseGrayCode);
			if (i < SubPopSize) {
				parents[i].setFitness(DebConvexF1(dblchrom));
			}
			else {
				parents[i].setFitness(DebConvexF2(dblchrom));
			}
			break;
		case 3:
			dblchrom.decodeBinary(parents[i][0], RangeOfValues, param.NumOfBits, param.UseGrayCode);
			if (i < SubPopSize) {
				parents[i].setFitness(DebConcaveF1(dblchrom));
			}
			else {
				parents[i].setFitness(DebConcaveF2(dblchrom));
			}
			break;
		case 4:
			dblchrom.decodeBinary(parents[i][0], RangeOfValues, param.NumOfBits, param.UseGrayCode);
			if (i < SubPopSize) {
				parents[i].setFitness(DebDiscreteF1(dblchrom));
			}
			else {
				parents[i].setFitness(DebDiscreteF2(dblchrom));
			}
			break;
		case 5:
			dblchrom.decodeBinary(parents[i][0], RangeOfValues, param.NumOfBits, param.UseGrayCode);
			if (i < SubPopSize) {
				parents[i].setFitness(FonsecaConcaveF1(dblchrom));
			}
			else {
				parents[i].setFitness(FonsecaConcaveF2(dblchrom));
			}
			break;
		default:
			std::cerr << "Specify fitness function. Current setting of MOOF (1-5):" << param.MOOF << std::endl;
			break;
		}
	}

	// Iteration
	for (t = 1; t <= param.Iterations; t++) {

		std::cout << "Generation = " << t << std::endl;

		// Copy an offspring to a parent
		offsprings = parents;

		// Divide
		for (i = 0; i < SubPopSize; i++) {
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
		for (i = 0; i < SubPopSize; i++) {
			parents[ i              ] = parents1[ i ];
			parents[ i + SubPopSize ] = parents2[ i ];
		}

		// Store Data
		offsprings = parents;

		// Shuffle
		if (param.VEGA) {
			parents.shuffle();
		}

		// Crossover
		for (i = 0; i < param.PopSize - 1; i += 2) {
			if (Rng::coinToss(param.CrossProb)) {
				parents[i][0].crossover(parents[i+1][0], param.CrossPoints);
			}
		}

		// Mutation
		for (i = 0; i < param.PopSize; i++) {
			dynamic_cast< ChromosomeT< bool >& >(parents[i][0]).flip(param.FlipProb);
		}

		// Evaluation
		for (i = 0; i < param.PopSize; i++) {
			switch (param.MOOF) {
			case 1:
				dblchrom.decodeBinary(parents[i][0], RangeOfValues, param.NumOfBits, param.UseGrayCode);
				if (i < SubPopSize) {
					parents[i].setFitness(SphereF1(dblchrom));
				}
				else {
					parents[i].setFitness(SphereF2(dblchrom));
				}
				break;
			case 2:
				dblchrom.decodeBinary(parents[i][0], RangeOfValues, param.NumOfBits, param.UseGrayCode);
				if (i < SubPopSize) {
					parents[i].setFitness(DebConvexF1(dblchrom));
				}
				else {
					parents[i].setFitness(DebConvexF2(dblchrom));
				}
				break;
			case 3:
				dblchrom.decodeBinary(parents[i][0], RangeOfValues, param.NumOfBits, param.UseGrayCode);
				if (i < SubPopSize) {
					parents[i].setFitness(DebConcaveF1(dblchrom));
				}
				else {
					parents[i].setFitness(DebConcaveF2(dblchrom));
				}
				break;
			case 4:
				dblchrom.decodeBinary(parents[i][0], RangeOfValues, param.NumOfBits, param.UseGrayCode);
				if (i < SubPopSize) {
					parents[i].setFitness(DebDiscreteF1(dblchrom));
				}
				else {
					parents[i].setFitness(DebDiscreteF2(dblchrom));
				}
				break;
			case 5:
				dblchrom.decodeBinary(parents[i][0], RangeOfValues, param.NumOfBits, param.UseGrayCode);
				if (i < SubPopSize) {
					parents[i].setFitness(FonsecaConcaveF1(dblchrom));
				}
				else {
					parents[i].setFitness(FonsecaConcaveF2(dblchrom));
				}
				break;
			default:
				std::cerr << "Specify fitness function. Current setting of MOOF (1-5):" << param.MOOF << std::endl;
				break;
			}
		}

		// Data Store for Display
		for (i = 0; i < param.PopSize; i++) {
			switch (param.MOOF) {
			case 1:
				dblchrom.decodeBinary(parents[i][0], RangeOfValues, param.NumOfBits, param.UseGrayCode);
				X1[ i ] = SphereF1(dblchrom);
				Y1[ i ] = SphereF2(dblchrom);
				dblchrom.decodeBinary(offsprings[i][0], RangeOfValues, param.NumOfBits, param.UseGrayCode);
				X2[ i ] = SphereF1(dblchrom);
				Y2[ i ] = SphereF2(dblchrom);
				break;
			case 2:
				dblchrom.decodeBinary(parents[i][0], RangeOfValues, param.NumOfBits, param.UseGrayCode);
				X1[ i ] = DebConvexF1(dblchrom);
				Y1[ i ] = DebConvexF2(dblchrom);
				dblchrom.decodeBinary(offsprings[i][0], RangeOfValues, param.NumOfBits, param.UseGrayCode);
				X2[ i ] = DebConvexF1(dblchrom);
				Y2[ i ] = DebConvexF2(dblchrom);
				break;
			case 3:
				dblchrom.decodeBinary(parents[i][0], RangeOfValues, param.NumOfBits, param.UseGrayCode);
				X1[ i ] = DebConcaveF1(dblchrom);
				Y1[ i ] = DebConcaveF2(dblchrom);
				dblchrom.decodeBinary(offsprings[i][0], RangeOfValues, param.NumOfBits, param.UseGrayCode);
				X2[ i ] = DebConcaveF1(dblchrom);
				Y2[ i ] = DebConcaveF2(dblchrom);
				break;
			case 4:
				dblchrom.decodeBinary(parents[i][0], RangeOfValues, param.NumOfBits, param.UseGrayCode);
				X1[ i ] = DebDiscreteF1(dblchrom);
				Y1[ i ] = DebDiscreteF2(dblchrom);
				dblchrom.decodeBinary(offsprings[i][0], RangeOfValues, param.NumOfBits, param.UseGrayCode);
				X2[ i ] = DebDiscreteF1(dblchrom);
				Y2[ i ] = DebDiscreteF2(dblchrom);
				break;
			case 5:
				dblchrom.decodeBinary(parents[i][0], RangeOfValues, param.NumOfBits, param.UseGrayCode);
				X1[ i ] = FonsecaConcaveF1(dblchrom);
				Y1[ i ] = FonsecaConcaveF2(dblchrom);
				dblchrom.decodeBinary(offsprings[i][0], RangeOfValues, param.NumOfBits, param.UseGrayCode);
				X2[ i ] = FonsecaConcaveF1(dblchrom);
				Y2[ i ] = FonsecaConcaveF2(dblchrom);
				break;
			default:
				std::cerr << "Specify fitness function. Current setting of MOOF (1-5):" << param.MOOF << std::endl;
				break;
			}
			archive.cleanArchive();
			for (i = 0; i < param.PopSize; i++) {
				IndividualMOO TEMP(parents[i]);
				TEMP.setNoOfObj(2);
				TEMP.setMOOFitnessValues(X1[i], Y1[i]);
				archive.addArchive(TEMP);
			}
			archive.nonDominatedSolutions();
		}
	}

	char filename[50];
	unsigned fileno = param.FileNumber;
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

	delete [] X1;
	delete [] X2;
	delete [] Y1;
	delete [] Y2;

	return 0;
}


