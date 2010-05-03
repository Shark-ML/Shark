/*! ======================================================================
 *
 *  \file  MOGA.cpp
 *  \date  2004-11-15
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
/* This file is derived from MOGA-SCAT.cpp which was written by
Tatsuya Okabe <tatsuya.okabe@honda-ri.de>.
**********************************************************/

#include <iostream>
#include <iomanip>
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
		FileUtil::io(is, os, "numArchive "  , numArchive	, 200u	, type);
		FileUtil::io(is, os, "FileNumber "  , FileNumber	, 1u	, type);
		FileUtil::io(is, os, "NumOfBits "   , NumOfBits	, 10u	, type);
		FileUtil::io(is, os, "DspInterval " , DspInterval	, 10u	, type);
		FileUtil::io(is, os, "NElitists "   , NElitists	, 0u	, type);
		FileUtil::io(is, os, "Omega "       , Omega	, 5u	, type);
		FileUtil::io(is, os, "MOOF "        , MOOF	        , 1u	, type);
		FileUtil::io(is, os, "Stopping "    , Stopping	, false	, type);
		FileUtil::io(is, os, "UseGrayCode " , UseGrayCode	, true	, type);
		FileUtil::io(is, os, "Goldberg "    , Goldberg	, true	, type);
		FileUtil::io(is, os, "Fonseca "     , Fonseca	, false	, type);
		FileUtil::io(is, os, "CrossProb "   , CrossProb	, 0.6	, type);
		FileUtil::io(is, os, "FlipProb "    , FlipProb	, 0.002	, type);
		FileUtil::io(is, os, "Start "       , Start	, -2.	, type);
		FileUtil::io(is, os, "End "         , End		, 2.	, type);
		FileUtil::io(is, os, "Sharing "     , Sharing	, 0.1	, type);
	}

	// Method to show the current content of the class variable:
	void monitor()
	{
		std::cout << "Seed        " << Seed		<< std::endl;//1234);
		std::cout << "PopSize     " << PopSize	<< std::endl;//100);
		std::cout << "Dimension   " << Dimension	<< std::endl;//2);
		std::cout << "Iterations  " << Iterations	<< std::endl;//1000);
		std::cout << "numArchive  " << numArchive	<< std::endl;//200);
		std::cout << "FileNumber  " << FileNumber	<< std::endl;//1);
		std::cout << "NumOfBits   " << NumOfBits	<< std::endl;//10);
		std::cout << "DspInterval " << DspInterval	<< std::endl;//10);
		std::cout << "NElitists   " << NElitists	<< std::endl;//0);
		std::cout << "Omega       " << Omega	<< std::endl;//5);
		std::cout << "MOOF        " << MOOF  	<< std::endl;//true);
		std::cout << "Stopping    " << Stopping	<< std::endl;//false);
		std::cout << "UseGrayCode " << UseGrayCode	<< std::endl;//true);
		std::cout << "Goldberg    " << Goldberg	<< std::endl;//true);
		std::cout << "Fonseca     " << Fonseca	<< std::endl;//false);
		std::cout << "CrossProb   " << CrossProb	<< std::endl;//0.6);
		std::cout << "FlipProb    " << FlipProb	<< std::endl;//0.002);
		std::cout << "Start       " << Start	<< std::endl;//-2);
		std::cout << "End         " << End		<< std::endl;//2);
		std::cout << "Sharing     " << Sharing	<< std::endl;//0.1);
	}

	unsigned Seed;
	unsigned PopSize;
	unsigned Dimension;
	unsigned Iterations;
	unsigned numArchive;
	unsigned FileNumber;
	unsigned NumOfBits;
	unsigned DspInterval;
	unsigned NElitists;
	unsigned Omega;
	unsigned MOOF;
	bool Stopping;
	bool UseGrayCode;
	bool Goldberg;
	bool Fonseca;
	double CrossProb;
	double FlipProb;
	double Start;
	double End;
	double Sharing;
};

// main program
int main(int argc, char* argv[])
{
	unsigned i, t;

	MyParams param(argc, argv);

	param.setDefault();
	//param.readParams();
	param.monitor();

	//Other Variable
	const Interval RangeOfValues(param.Start, param.End);
	double f1 = 0, f2 = 0;

	double *PF1 = new double[(unsigned)param.PopSize ];
	double *PF2 = new double[(unsigned)param.PopSize ];
	double *OF1 = new double[(unsigned)param.PopSize ];
	double *OF2 = new double[(unsigned)param.PopSize ];

	// initialize random number generator
	Rng::seed(param.Seed);

	// define populations
	PopulationMOO parents(param.PopSize, ChromosomeT< bool >(param.Dimension * param.NumOfBits));
	PopulationMOO offsprings(param.PopSize, ChromosomeT< bool >(param.Dimension * param.NumOfBits));

	// Set Minimization Task
	parents   .setMinimize();
	offsprings.setMinimize();

	// Set No of Objective funstions
	parents   .setNoOfObj(2);
	offsprings.setNoOfObj(2);

	// Archive
	ArchiveMOO archive(param.numArchive);
	archive.minimize();

	// Temporary chromosomes for decoding
	ChromosomeT< double > dblchrom;
	// initialize all chromosomes of parent population
	for (i = 0; i < parents.size(); ++i) {
		dynamic_cast< ChromosomeT< bool >& >(parents[ i ][ 0 ]).initialize();
	}

	// evaluate parents (only needed for elitist strategy)
	if (param.NElitists > 0) {
		for (i = 0; i < parents.size(); ++i) {
			dblchrom.decodeBinary(parents[ i ][ 0 ], RangeOfValues, param.NumOfBits, param.UseGrayCode);
			switch (param.MOOF) {
			case 1:
				f1 = SphereF1(dblchrom);
				f2 = SphereF2(dblchrom);
				break;
			case 2:
				f1 = DebConvexF1(dblchrom);
				f2 = DebConvexF2(dblchrom);
				break;
			case 3:
				f1 = DebConcaveF1(dblchrom);
				f2 = DebConcaveF2(dblchrom);
				break;
			case 4:
				f1 = DebDiscreteF1(dblchrom);
				f2 = DebDiscreteF2(dblchrom);
				break;
			case 5:
				f1 = FonsecaConcaveF1(dblchrom);
				f2 = FonsecaConcaveF2(dblchrom);
				break;
			default:
				std::cerr << "Specify fitness function. Current setting of MOOF (1-5):" << param.MOOF << std::endl;
				break;
			}
			parents[ i ].setMOOFitnessValues(f1, f2);
			PF1[ i ] = f1;
			PF2[ i ] = f2;
		}
		parents.NicheCountPFN2(param.Sharing);
		if (param.Goldberg) {
			parents.MOGAGoldbergRank();
		}
		else if (param.Fonseca) {
			parents.MOGAFonsecaRank();
		}
		parents.MOORankToFitness();
		parents.SelectProbMichalewicz();
		parents.SharingSelProb();
	}
	// iterate
	for (t = 1; t < param.Iterations + 1; ++t) {
		std::cout << "Generations : " << t << std::endl;

		// copy parents to offsprings
		offsprings.reproduce(parents);

		// recombine by crossing over two parents
		for (i = 0; i < offsprings.size() - 1; i += 2) {
			if (Rng::coinToss(param.CrossProb)) {
				offsprings[ i ][ 0 ].crossoverUniform(offsprings[ i+1 ][ 0 ]);
			}
		}

		// mutate by flipping bits
		for (i = 0; i < offsprings.size(); ++i) {
			dynamic_cast< ChromosomeT< bool >& >(offsprings[ i ][ 0 ]).flip(param.FlipProb);
		}

		// evaluate objective function
		for (i = 0; i < offsprings.size(); ++i) {
			dblchrom.decodeBinary(offsprings[ i ][ 0 ], RangeOfValues, param.NumOfBits, param.UseGrayCode);
			switch (param.MOOF) {
			case 1:
				f1 = SphereF1(dblchrom);
				f2 = SphereF2(dblchrom);
				break;
			case 2:
				f1 = DebConvexF1(dblchrom);
				f2 = DebConvexF2(dblchrom);
				break;
			case 3:
				f1 = DebConcaveF1(dblchrom);
				f2 = DebConcaveF2(dblchrom);
				break;
			case 4:
				f1 = DebDiscreteF1(dblchrom);
				f2 = DebDiscreteF2(dblchrom);
				break;
			case 5:
				f1 = FonsecaConcaveF1(dblchrom);
				f2 = FonsecaConcaveF2(dblchrom);
				break;
			default:
				std::cerr << "Specify fitness function. Current setting of MOOF (1-5):" << param.MOOF << std::endl;
				break;
			}
			offsprings[ i ].setMOOFitnessValues(f1, f2);
			OF1[ i ] = f1;
			OF2[ i ] = f2;
		}
		offsprings.NicheCountPFN2(param.Sharing);
		if (param.Goldberg) {
			offsprings.MOGAGoldbergRank();
		}
		else if (param.Fonseca) {
			offsprings.MOGAFonsecaRank();
		}
		offsprings.MOORankToFitness();
		offsprings.SelectProbMichalewicz();
		offsprings.SharingSelProb();
		// Archive
		for (i = 0; i < offsprings.size(); i++) {
			int dominateIA = archive.Dominate(offsprings[ i ]);
			if (dominateIA >= 4) {
				archive.cleanArchive();
				archive.addArchive(offsprings[ i ]);
			}
			else if (dominateIA == 3) {
				archive.delDominateArchive(offsprings[ i ]);
				archive.addArchive(offsprings[ i ]);
			}
			else if (dominateIA == 2) {
				if (archive.getCapacity() > 0) {
					archive.addArchive(offsprings[ i ]);
				}
				else {
					double mindisIA = archive.distanceOnFitness(offsprings[ i ]);
					double mindisAA = archive.minDistanceOnFitness();
					if (mindisIA > mindisAA) {
						archive.delSharingWorst();
						archive.addArchive(offsprings[ i ]);
					}
				}
			}
		}

		// selection
		parents.selectElitists(offsprings, param.NElitists);
		parents.SelectByRoulette(offsprings, param.NElitists);
		// Evaluate again ( Selection Copy does not work well )
		for (i = 0; i < parents.size(); ++i) {
			dblchrom.decodeBinary(parents[ i ][ 0 ], RangeOfValues, param.NumOfBits, param.UseGrayCode);
			switch (param.MOOF) {
			case 1:
				f1 = SphereF1(dblchrom);
				f2 = SphereF2(dblchrom);
				break;
			case 2:
				f1 = DebConvexF1(dblchrom);
				f2 = DebConvexF2(dblchrom);
				break;
			case 3:
				f1 = DebConcaveF1(dblchrom);
				f2 = DebConcaveF2(dblchrom);
				break;
			case 4:
				f1 = DebDiscreteF1(dblchrom);
				f2 = DebDiscreteF2(dblchrom);
				break;
			case 5:
				f1 = FonsecaConcaveF1(dblchrom);
				f2 = FonsecaConcaveF2(dblchrom);
				break;
			default:
				std::cerr << "Specify fitness function. Current setting of MOOF (1-5):" << param.MOOF << std::endl;
				break;
			}
			parents[ i ].setMOOFitnessValues(f1, f2);
			PF1[ i ] = f1;
			PF2[ i ] = f2;
		}
		parents.NicheCountPFN2(param.Sharing);
		if (param.Goldberg) {
			parents.MOGAGoldbergRank();
		}
		else if (param.Fonseca) {
			parents.MOGAFonsecaRank();
		}
		parents.MOORankToFitness();
		parents.SelectProbMichalewicz();
		parents.SharingSelProb();

		if (t % param.DspInterval == 0) {
			std::cout << "Archive Size = " << archive.size() << std::endl;
			if (param.Stopping) {
				std::cout << "\nHit return key" << std::endl;
				fgetc(stdin);
			}
		}
	}

	char filename[ 80 ];
	unsigned filenumber = param.FileNumber;
	sprintf(filename, "Archive%i.txt", filenumber);
	archive.saveArchive(filename);
	std::cout << "\nThe result is stored in " << filename << "\n" << std::endl;
	std::cout << "Number of solutions" << std::endl;
	std::cout << "Number of objectives" << std::endl;
	std::cout << "Solution 1 (f1,f2,...)" << std::endl;
	std::cout << "Solution 2 (f1,f2,...)\n" << std::endl;

	std::cout << "\nHit return key" << std::endl;

	fgetc(stdin);

	delete [] PF1;
	delete [] PF2;
	delete [] OF1;
	delete [] OF2;

	return 0;

}

