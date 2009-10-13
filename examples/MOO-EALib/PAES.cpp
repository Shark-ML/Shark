// ======================================================================
/*!
 *  \file PAES.cpp
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
/* This file is derived from PAES-SCAT.cpp which was written by
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
		if (scanFrom(confFile.c_str())) {
			// Call the main program with parameters "-conf [filename]"
			std::cout << "Name of the configuration file: " << confFile << std::endl;
		}
		else {
		  throw SHARKEXCEPTION("No valid configuration file given! ");
		}
	}

	void io(std::istream& is, std::ostream& os, FileUtil::iotype type)
	{
		FileUtil::io(is, os, "baseSize"     , baseSize, 1350u, type);
		FileUtil::io(is, os, "Seed"         , Seed,  1234u, type);
		FileUtil::io(is, os, "Dimension"    , Dimension, 2u, type);
		FileUtil::io(is, os, "Iterations"   , Iterations, 2000u, type);
		FileUtil::io(is, os, "numArchive"   , numArchive, 200u, type);
		FileUtil::io(is, os, "Interval"     , Interval, 100u, type);
		FileUtil::io(is, os, "FileName"     , FileName, 1u, type);
		FileUtil::io(is, os, "Standard1"    , Standard1, true, type);
		FileUtil::io(is, os, "Standard2"    , Standard2, false, type);
		FileUtil::io(is, os, "Derandomize"  , Derandomize, false, type);
		FileUtil::io(is, os, "Rotate"       , Rotate, false, type);
		FileUtil::io(is, os, "MOOF"         , MOOF, 1u, type);
		FileUtil::io(is, os, "Recombine"    , Recombine, true, type);
		FileUtil::io(is, os, "CheckSigma"   , CheckSigma, true, type);
		FileUtil::io(is, os, "Stopping"     , Stopping, false, type);
		FileUtil::io(is, os, "SigmaLower"   , SigmaLower, 0.004, type);
		FileUtil::io(is, os, "Crowded"        , Crowded, 0.1, type);
		FileUtil::io(is, os, "MinInit"      , MinInit, -2., type);
		FileUtil::io(is, os, "MaxInit"      , MaxInit, 2., type);
		FileUtil::io(is, os, "SigmaMin"     , SigmaMin, .1, type);
		FileUtil::io(is, os, "SigmaMax"     , SigmaMax, .1, type);
	}

	// Method to show the current content of the class variable:
	void monitor()
	{
		std::cout << "baseSize "	<< baseSize	<< std::endl;
		std::cout << "Seed  "	<< Seed		<< std::endl;
		std::cout << "Dimension "	<< Dimension	<< std::endl;
		std::cout << "Iterations "	<< Iterations	<< std::endl;
		std::cout << "numArchive "	<< numArchive	<< std::endl;
		std::cout << "Interval "	<< Interval	<< std::endl;
		std::cout << "FileName "	<< FileName	<< std::endl;
		std::cout << "Standard1 "	<< Standard1	<< std::endl;
		std::cout << "Standard2 "	<< Standard2	<< std::endl;
		std::cout << "Derandomize "	<< Derandomize	<< std::endl;
		std::cout << "Rotate "	<< Rotate	<< std::endl;
		std::cout << "MOOF "	<< MOOF 	<< std::endl;
		std::cout << "Recombine "	<< Recombine	<< std::endl;
		std::cout << "CheckSigma "	<< CheckSigma	<< std::endl;
		std::cout << "Stopping "	<< Stopping	<< std::endl;
		std::cout << "SigmaLower "	<< SigmaLower	<< std::endl;
		std::cout << "Crowded "	<< Crowded	<< std::endl;
		std::cout << "MinInit "	<< MinInit	<< std::endl;
		std::cout << "MaxInit "	<< MaxInit	<< std::endl;
		std::cout << "SigmaMin "	<< SigmaMin	<< std::endl;
		std::cout << "SigmaMax "	<< SigmaMax	<< std::endl;
	}

	unsigned baseSize;
	unsigned Seed;
	unsigned Dimension;
	unsigned Iterations;
	unsigned numArchive;
	unsigned Interval;
	unsigned FileName;
	bool Standard1;
	bool Standard2;
	bool Derandomize;
	bool Rotate;
	unsigned MOOF;
	bool Recombine;
	bool CheckSigma;
	bool Stopping;
	double SigmaLower;
	double Crowded;
	double MinInit;
	double MaxInit;
	double SigmaMin;
	double SigmaMax;
};

// main program
int main(int argc, char* argv[])
{
	unsigned m, t, j;

	MyParams param(argc, argv);

	param.setDefault();
	//param.readParams();
	param.monitor();

	unsigned NSigma;

	//Other Variables
	if (param.Standard1) {
		NSigma = param.Dimension;
	}
	else if (param.Standard2) {
		NSigma = param.Dimension;
	}
	else if (param.Derandomize) {
	  throw SHARKEXCEPTION("Option Derandomized is not defined");
	}
	else if (param.Rotate) {
		NSigma = (unsigned)(0.5 * (1 + param.Dimension) * param.Dimension);
	}
	else {
	  throw SHARKEXCEPTION("Unknown option");
	}

	double           XP[ 1 ], YP[ 1 ], XO[ 1 ], YO[ 1 ];

	//double           XA[ (unsigned)numArchive ], YA[ (unsigned)numArchive ];

	double           f1 = 0, f2 = 0;
	int              dominateII;
	int              dominateIA;

	//Random Seed
	Rng::seed(param.Seed);

	//Define Populations
	PopulationMOO parents(1,
						  ChromosomeT< double >(param.Dimension), ChromosomeT< double >(NSigma));
	PopulationMOO offsprings(1,
							 ChromosomeT< double >(param.Dimension), ChromosomeT< double >(NSigma));

	//Set the Number of Objective Functions
	parents   .setNoOfObj(2);
	offsprings.setNoOfObj(2);

	//Task ( Minimize or Maximize )
	parents.   setMinimize();
	offsprings.setMinimize();

	//Archive
	ArchiveMOO archive(param.numArchive);
	archive.minimize();

	//Initialize Parent Population
	dynamic_cast< ChromosomeT< double >& >(parents[0][0]).initialize(param.MinInit, param.MaxInit);
	if (param.Standard1) {
		dynamic_cast< ChromosomeT< double >& >(parents[0][1]).initialize(param.SigmaMin, param.SigmaMax);
	}
	else if (param.Standard2) {
		dynamic_cast< ChromosomeT< double >& >(parents[0][1]).initialize(param.SigmaMin, param.SigmaMax);
	}
	else if (param.Derandomize) {
	  throw SHARKEXCEPTION("Option Derandomized is not defined");
	}
	else if (param.Rotate) {
		dynamic_cast< ChromosomeT< double >& >(parents[0][1]).initializeRotate(param.SigmaMin, param.SigmaMax);
	}
	else {
	  throw SHARKEXCEPTION("Unknown option");
	}

	//Evaluate Parents ( only needed for elitist strategy )
	switch (param.MOOF) {
	case 1:
		f1 = SphereF1(dynamic_cast< ChromosomeT< double >& >(parents[0][0]));
		f2 = SphereF2(dynamic_cast< ChromosomeT< double >& >(parents[0][0]));
		break;
	case 2:
		f1 = DebConvexF1(dynamic_cast< ChromosomeT< double >& >(parents[0][0]));
		f2 = DebConvexF2(dynamic_cast< ChromosomeT< double >& >(parents[0][0]));
		break;
	case 3:
		f1 = DebConcaveF1(dynamic_cast< ChromosomeT< double >& >(parents[0][0]));
		f2 = DebConcaveF2(dynamic_cast< ChromosomeT< double >& >(parents[0][0]));
		break;
	case 4:
		f1 = DebDiscreteF1(dynamic_cast< ChromosomeT< double >& >(parents[0][0]));
		f2 = DebDiscreteF2(dynamic_cast< ChromosomeT< double >& >(parents[0][0]));
		break;
	case 5:
		f1 = FonsecaConcaveF1(dynamic_cast< ChromosomeT< double >& >(parents[0][0]));
		f2 = FonsecaConcaveF2(dynamic_cast< ChromosomeT< double >& >(parents[0][0]));
		break;
	default:
		std::cerr << "Specify fitness function. Current setting of MOOF (1-5):" << param.MOOF << std::endl;
		break;
	}

	parents[0].setMOOFitnessValues(f1, f2);
	XP[ 0 ] = f1;
	YP[ 0 ] = f2;
	archive.addArchive(parents[0]);

	//Iterate
	for (t = 1; t < param.Iterations + 1; t++) {

		//Print Generation
		std::cout << "Generation  = " << t << " Archive = " << archive.size() << std::endl;

		//Generate New Offsprings
		Individual& pap = parents[0];

		//Define Temporary References for Convenience
		ChromosomeT< double >& objvar = dynamic_cast< ChromosomeT< double >& >(offsprings[0][0]);
		ChromosomeT< double >& sigma  = dynamic_cast< ChromosomeT< double >& >(offsprings[0][1]);

		for (j = 0; j < param.Dimension; j++) {
			objvar[j] = dynamic_cast< ChromosomeT< double >& >(pap[0])[j];
		}
		for (j = 0; j < NSigma; j++) {
			sigma[j]  = dynamic_cast< ChromosomeT< double >& >(pap[1])[j];
		}

		//Mutation
		if (param.Standard1) {
			double tau0 = 1.0 / sqrt(2.0 * (double)param.Dimension);
			double tau1 = 1.0 / sqrt(2.0 * sqrt((double)param.Dimension));
			sigma.mutateLogNormal(tau0, tau1);
			if (param.CheckSigma) {
				for (j = 0; j < sigma.size(); j++) {
					if (sigma[ j ] < param.SigmaLower * fabs(objvar[ j ])) {
						sigma[ j ] = param.SigmaLower * fabs(objvar[ j ]);
					}
				}
			}
			objvar.mutateNormal(sigma, true);
		}
		else if (param.Standard2) {
			double xi_prob = 0.5;
			sigma.mutateMSR(xi_prob);
			for (j = 0; j < sigma.size(); j++) {
				sigma[ j ] = sigma[ j ] / sqrt(double(sigma.size()));
			}
			if (param.CheckSigma) {
				for (j = 0; j < sigma.size(); j++) {
					if (sigma[ j ] < param.SigmaLower * fabs(objvar[ j ])) {
						sigma[ j ] = param.SigmaLower * fabs(objvar[ j ]);
					}
				}
			}
			objvar.mutateNormal(sigma, true);
			for (j = 0; j < sigma.size(); j++) {
				sigma[ j ] = sigma[ j ] * sqrt(double(sigma.size()));
			}
		}
		else if (param.Derandomize) {
		  throw SHARKEXCEPTION("Option Derandomized is not defined");
		}
		else if (param.Rotate) {
			objvar.mutateRotate(sigma);
		}
		else {
		  throw SHARKEXCEPTION("Unknown option");
		}

		//Evaluate Objective Function ( Parameters in Chromosome #0 )
		switch (param.MOOF) {
		case 1:
			f1 = SphereF1(dynamic_cast< ChromosomeT< double >& >(offsprings[0][0]));
			f2 = SphereF2(dynamic_cast< ChromosomeT< double >& >(offsprings[0][0]));
			break;
		case 2:
			f1 = DebConvexF1(dynamic_cast< ChromosomeT< double >& >(offsprings[0][0]));
			f2 = DebConvexF2(dynamic_cast< ChromosomeT< double >& >(offsprings[0][0]));
			break;
		case 3:
			f1 = DebConcaveF1(dynamic_cast< ChromosomeT< double >& >(offsprings[0][0]));
			f2 = DebConcaveF2(dynamic_cast< ChromosomeT< double >& >(offsprings[0][0]));
			break;
		case 4:
			f1 = DebDiscreteF1(dynamic_cast< ChromosomeT< double >& >(offsprings[0][0]));
			f2 = DebDiscreteF2(dynamic_cast< ChromosomeT< double >& >(offsprings[0][0]));
			break;
		case 5:
			f1 = FonsecaConcaveF1(dynamic_cast< ChromosomeT< double >& >(offsprings[0][0]));
			f2 = FonsecaConcaveF2(dynamic_cast< ChromosomeT< double >& >(offsprings[0][0]));
			break;
		default:
			std::cerr << "Specify fitness function. Current setting of MOOF (1-5):" << param.MOOF << std::endl;
			break;
		}

		offsprings[0].setMOOFitnessValues(f1, f2);
		XO[ 0 ] = f1;
		YO[ 0 ] = f2;

		//Select ( 1 + 1 )
		dominateII = archive.Dominate(parents[0], offsprings[0]);

		//A parent dominates an offspring
	if (dominateII > 0) {}

		//Program Error
		else if (dominateII == 0) {
		  throw SHARKEXCEPTION("Error: neither a parent nor an offsprings dominates on of the other");
		}

		//An offspring dominates a parent
		else if (dominateII < -1) {
			archive.delDominateArchive(offsprings[0]);
			archive.addArchive(offsprings[0]);
			parents = offsprings;
			XP[ 0 ] = parents[ 0 ].getMOOFitness(0);
			YP[ 0 ] = parents[ 0 ].getMOOFitness(1);
		}

		//Trade-OFF
		else if (dominateII == -1) {
			dominateIA = archive.Dominate(offsprings[0]);
			if (dominateIA > 2) {
				archive.delDominateArchive(offsprings[ 0 ]);
				archive.addArchive(offsprings[0]);
				parents = offsprings;
				XP[ 0 ] = parents[ 0 ].getMOOFitness(0);
				YP[ 0 ] = parents[ 0 ].getMOOFitness(1);
			}
			else if (dominateIA == 2) {
				if (archive.getCapacity() > 0) {
					archive.addArchive(offsprings[ 0 ]);
					unsigned numpar = archive.crowded(parents[ 0 ], param.Crowded) - 1;
					unsigned numoff = archive.crowded(offsprings[ 0 ], param.Crowded) - 1;
					if (numpar > numoff) {
						parents = offsprings;
						XP[ 0 ] = parents[ 0 ].getMOOFitness(0);
						YP[ 0 ] = parents[ 0 ].getMOOFitness(1);
					}
				}
				else {
					unsigned worst = archive.sharingWorst(param.Crowded);
					bool samecluster = true;
					int clusoff, cluswor;
					for (m = 0; m < archive.readArchive(worst).getNoOfObj(); m++) {
						clusoff = (int)floor(offsprings[0].getMOOFitness(m) / param.Crowded);
						cluswor = (int)floor(archive.readArchive(worst).getMOOFitness(m) / param.Crowded);
						if (clusoff != cluswor) {
							samecluster = false;
							break;
						}
					}
					if (samecluster) {}
					else {
						archive.delSharingWorst(param.Crowded);
						archive.addArchive(offsprings[0]);
						bool lostparent = true;
						for (unsigned lost = 0; lost < archive.getMaxArchive(); lost++) {
							if (parents[0] == archive.readArchive(lost)) {
								lostparent = false;
							}
						}
						if (lostparent == true) {
							parents = offsprings;
							XP[ 0 ] = parents[ 0 ].getMOOFitness(0);
							YP[ 0 ] = parents[ 0 ].getMOOFitness(1);
						}
						else {
							unsigned numpar = archive.crowded(parents[0], param.Crowded) - 1;
							unsigned numoff = archive.crowded(offsprings[0], param.Crowded);
							if (numpar > numoff) {
								parents = offsprings;
								XP[ 0 ] = parents[ 0 ].getMOOFitness(0);
								YP[ 0 ] = parents[ 0 ].getMOOFitness(1);
							}
						}
					}
				}
			}
			else if (dominateIA == 1) {}
			else if (dominateIA == 0) {
			  throw SHARKEXCEPTION("Error: neither a parent nor an offsprings dominates on of the other");
			}
			else if (dominateIA == -1) {}
			else {}
		}

		//Error
		else {
		  throw SHARKEXCEPTION("Error");
		}
		if (t % param.Interval == 0) {
			if (param.Stopping) {
				std::cout << "\n Hit Return Key for Continue" << std::endl;
				fgetc(stdin);
			}
		}
	}

	char dummy[ 80 ];
	unsigned filenumber = param.FileName;
	sprintf(dummy, "Archive%i.txt", filenumber);
	archive.saveArchive(dummy);
	std::cout << "\nThe result is stored in " << dummy << "\n" << std::endl;
	std::cout << "Number of Solutions" << std::endl;
	std::cout << "Number of Objectives" << std::endl;
	std::cout << "Solution 1 (f1,f2,...)" << std::endl;
	std::cout << "Solution 2 (f1,f2,...)\n" << std::endl;

	std::cout << "Hit Return Key for End" << std::endl;
	fgetc(stdin);

	return EXIT_SUCCESS;

}


