
/* ======================================================================
 *
 *  \filePAES-SCAT.cpp
 *
 *  \brief Sample Program for MOO-ES
 *  \author  Tatsuya Okabe <tatsuya.okabe@honda-ri.de>
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
 *      $RCSfile: PAES-SCAT.cpp,v $<BR>
 *      $Revision: 1.4 $<BR>
 *      $Date: 2007/09/19 13:22:11 $
 *
 *  \par Changes:
 *      $Log: PAES-SCAT.cpp,v $
 *      Revision 1.4  2007/09/19 13:22:11  leninga
 *      complete header
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
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <Array/Array.h>
#include <SpreadCAT/SpreadCAT.h>
#include "SharkDefs.h"

//=======================================================================
// main program
//=======================================================================
int main(void)
{
	//Constants & Variables
	VarT< bool     > Standard1("Standard1", TRUE);
	VarT< bool     > Standard2("Standard2", FALSE);
	VarT< bool     > Derandomize("Derandomize", FALSE);
	VarT< bool     > Rotate("Rotate", FALSE);
	VarT< bool     > GSA("GSA", FALSE);
	VarT< bool     > IDA("IDA", FALSE);
	VarT< bool     > CMA("CMA", FALSE);
	VarT< unsigned > baseSize("baseSize", 1350);
	VarT< unsigned > Seed("Seed", 1234);
	VarT< bool     > MOOF1("MOOF1", TRUE);
	VarT< bool     > MOOF2("MOOF2", FALSE);
	VarT< bool     > MOOF3("MOOF3", FALSE);
	VarT< bool     > MOOF4("MOOF4", FALSE);
	VarT< bool     > MOOF5("MOOF5", FALSE);
	VarT< bool     > Recombine("Recombine", true);
	VarT< unsigned > Dimension("Dimension", 2);
	VarT< unsigned > Iterations("Iterations", 2000);
	VarT< bool     > CheckSigma("CheckSigma", true);
	VarT< double   > SigmaLower("SigmaLower", 0.004);
	VarT< unsigned > numArchive("numArchive", 200);
	VarT< double   > Crowded("Crowded", 0.1);
	VarT< double   > MinInit("MinInit", -2);
	VarT< double   > MaxInit("MaxInit", 2);
	VarT< double   > SigmaMin("SigmaMin", 0.1);
	VarT< double   > SigmaMax("SigmaMax", 0.1);
	VarT< unsigned > Interval("Interval", 100);
	VarT< bool     > Stopping("Stopping", FALSE);
	VarT< unsigned > FileName("FileName", 1);

	unsigned NSigma;
	// char MatlabCommand[200];
	//double Var1, Var2, Var3, Var4;

	//SpreadCAT
	SpreadCAT mySpread("PAES.conf");
	//Other Variables
	if (Standard1) {
		NSigma = Dimension;
	}
	else if (Standard2) {
		NSigma = Dimension;
	}
	else if (Derandomize) {
	  throw SHARKEXCEPTION("Option Derandomized is not defined");
	}
	else if (Rotate) {
		NSigma = (unsigned)(0.5 * (1 + Dimension) * Dimension);
	}
	else if (GSA) {
		NSigma = 1 + baseSize * Dimension;
	}
	else if (IDA) {
		NSigma = 2 + 3 * Dimension;
	}
	else if (CMA) {
		NSigma = 1 + (2 + Dimension) * Dimension;
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
	Rng::seed(Seed);
	//Define Populations
	PopulationMOO parents(1,
						  ChromosomeT< double >(Dimension), ChromosomeT< double >(NSigma));
	PopulationMOO offsprings(1,
							 ChromosomeT< double >(Dimension), ChromosomeT< double >(NSigma));
	//Set the Number of Objective Functions
	parents   .setNoOfObj(2);
	offsprings.setNoOfObj(2);
	//Task ( Minimize or Maximize )
	parents.   setMinimize();
	offsprings.setMinimize();
	//Archive
	ArchiveMOO archive(numArchive);
	archive.minimize();
	//Initialize Parent Population
	dynamic_cast< ChromosomeT< double >& >(parents[0][0]).initialize(MinInit, MaxInit);
	if (Standard1) {
		dynamic_cast< ChromosomeT< double >& >(parents[0][1]).initialize(SigmaMin, SigmaMax);
	}
	else if (Standard2) {
		dynamic_cast< ChromosomeT< double >& >(parents[0][1]).initialize(SigmaMin, SigmaMax);
	}
	else if (Derandomize) {
	  throw SHARKEXCEPTION("Option Derandomized is not defined");
	}
	else if (Rotate) {
		dynamic_cast< ChromosomeT< double >& >(parents[0][1]).initializeRotate(SigmaMin, SigmaMax);
	}
	else if (GSA) {
		dynamic_cast< ChromosomeT< double >& >(parents[0][1]).initializeGSA(SigmaMin, SigmaMax, baseSize);
	}
	else if (IDA) {
		dynamic_cast< ChromosomeT< double >& >(parents[0][1]).initializeIDA(SigmaMin, SigmaMax);
	}
	else if (CMA) {
		dynamic_cast< ChromosomeT< double >& >(parents[0][1]).initializeCMA(SigmaMin, SigmaMax);
	}
	else {
	  throw SHARKEXCEPTION("Unknown option");
	}
	//Evaluate Parents ( only needed for elitist strategy )
	if (MOOF1) {
		f1 = SphereF1(dynamic_cast< ChromosomeT< double >& >(parents[0][0]));
		f2 = SphereF2(dynamic_cast< ChromosomeT< double >& >(parents[0][0]));
	}
	if (MOOF2) {
		f1 = DebConvexF1(dynamic_cast< ChromosomeT< double >& >(parents[0][0]));
		f2 = DebConvexF2(dynamic_cast< ChromosomeT< double >& >(parents[0][0]));
	}
	if (MOOF3) {
		f1 = DebConcaveF1(dynamic_cast< ChromosomeT< double >& >(parents[0][0]));
		f2 = DebConcaveF2(dynamic_cast< ChromosomeT< double >& >(parents[0][0]));
	}
	if (MOOF4) {
		f1 = DebDiscreteF1(dynamic_cast< ChromosomeT< double >& >(parents[0][0]));
		f2 = DebDiscreteF2(dynamic_cast< ChromosomeT< double >& >(parents[0][0]));
	}
	if (MOOF5) {
		f1 = FonsecaConcaveF1(dynamic_cast< ChromosomeT< double >& >(parents[0][0]));
		f2 = FonsecaConcaveF2(dynamic_cast< ChromosomeT< double >& >(parents[0][0]));
	}
	parents[0].setMOOFitnessValues(f1, f2);
	XP[ 0 ] = f1;
	YP[ 0 ] = f2;
	archive.addArchive(parents[0]);

	//Iterate
	for (unsigned t = 1; t < Iterations + 1; t++) {
		//Print Generation
		std::cout << "Generation  = " << t << " Archive = " << archive.size() << std::endl;
		//Generate New Offsprings
		Individual& pap = parents[0];
		//Define Temporary References for Convenience
		ChromosomeT< double >& objvar = dynamic_cast< ChromosomeT< double >& >(offsprings[0][0]);
		ChromosomeT< double >& sigma  = dynamic_cast< ChromosomeT< double >& >(offsprings[0][1]);
		for (unsigned j = 0; j < Dimension; j++) {
			objvar[j] = dynamic_cast< ChromosomeT< double >& >(pap[0])[j];
		}
		for (unsigned j = 0; j < NSigma; j++) {
			sigma[j]  = dynamic_cast< ChromosomeT< double >& >(pap[1])[j];
		}
		//Mutation
		if (Standard1) {
			double tau0 = 1.0 / sqrt(2.0 * (double)Dimension);
			double tau1 = 1.0 / sqrt(2.0 * sqrt((double)Dimension));
			sigma.mutateLogNormal(tau0, tau1);
			if (CheckSigma) {
				for (unsigned j = 0; j < sigma.size(); j++) {
					if (sigma[ j ] < SigmaLower * fabs(objvar[ j ])) {
						sigma[ j ] = SigmaLower * fabs(objvar[ j ]);
					}
				}
			}
			objvar.mutateNormal(sigma, true);
		}
		else if (Standard2) {
			double xi_prob = 0.5;
			sigma.mutateMSR(xi_prob);
			for (unsigned j = 0; j < sigma.size(); j++) {
				sigma[ j ] = sigma[ j ] / sqrt(sigma.size());
			}
			if (CheckSigma) {
				for (unsigned j = 0; j < sigma.size(); j++) {
					if (sigma[ j ] < SigmaLower * fabs(objvar[ j ])) {
						sigma[ j ] = SigmaLower * fabs(objvar[ j ]);
					}
				}
			}
			objvar.mutateNormal(sigma, true);
			for (unsigned j = 0; j < sigma.size(); j++) {
				sigma[ j ] = sigma[ j ] * sqrt(sigma.size());
			}
		}
		else if (Derandomize) {
		  throw SHARKEXCEPTION("Option Derandomized is not defined");
		}
		else if (Rotate) {
			objvar.mutateRotate(sigma);
		}
		else if (GSA) {
			objvar.mutateGSA(sigma);
		}
		else if (IDA) {
			objvar.mutateIDA(sigma);
		}
		else if (CMA) {
			objvar.mutateCMA(sigma);
		}
		else {
			  throw SHARKEXCEPTION("Unknown option");
		}
		//Evaluate Objective Function ( Parameters in Chromosome #0 )
		if (MOOF1) {
			f1 = SphereF1(dynamic_cast< ChromosomeT< double >& >(offsprings[0][0]));
			f2 = SphereF2(dynamic_cast< ChromosomeT< double >& >(offsprings[0][0]));
		}
		if (MOOF2) {
			f1 = DebConvexF1(dynamic_cast< ChromosomeT< double >& >(offsprings[0][0]));
			f2 = DebConvexF2(dynamic_cast< ChromosomeT< double >& >(offsprings[0][0]));
		}
		if (MOOF3) {
			f1 = DebConcaveF1(dynamic_cast< ChromosomeT< double >& >(offsprings[0][0]));
			f2 = DebConcaveF2(dynamic_cast< ChromosomeT< double >& >(offsprings[0][0]));
		}
		if (MOOF4) {
			f1 = DebDiscreteF1(dynamic_cast< ChromosomeT< double >& >(offsprings[0][0]));
			f2 = DebDiscreteF2(dynamic_cast< ChromosomeT< double >& >(offsprings[0][0]));
		}
		if (MOOF5) {
			f1 = FonsecaConcaveF1(dynamic_cast< ChromosomeT< double >& >(offsprings[0][0]));
			f2 = FonsecaConcaveF2(dynamic_cast< ChromosomeT< double >& >(offsprings[0][0]));
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
					unsigned numpar = archive.crowded(parents[ 0 ], Crowded) - 1;
					unsigned numoff = archive.crowded(offsprings[ 0 ], Crowded) - 1;
					if (numpar > numoff) {
						parents = offsprings;
						XP[ 0 ] = parents[ 0 ].getMOOFitness(0);
						YP[ 0 ] = parents[ 0 ].getMOOFitness(1);
					}
				}
				else {
					unsigned worst = archive.sharingWorst(Crowded);
					bool samecluster = true;
					int clusoff, cluswor;
					for (unsigned m = 0; m < archive.readArchive(worst).getNoOfObj(); m++) {
						clusoff = (int)floor(offsprings[0].getMOOFitness(m) / Crowded);
						cluswor = (int)floor(archive.readArchive(worst).getMOOFitness(m) / Crowded);
						if (clusoff != cluswor) {
							samecluster = false;
							break;
						}
					}
					if (samecluster) {}
					else {
						archive.delSharingWorst(Crowded);
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
							unsigned numpar = archive.crowded(parents[0], Crowded) - 1;
							unsigned numoff = archive.crowded(offsprings[0], Crowded);
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
		if (t % Interval == 0) {
			if (Stopping) {
				std::cout << "\n Hit Return Key for Continue" << std::endl;
				fgetc(stdin);
			}
		}
	}

	char dummy[ 80 ];
	unsigned filenumber = FileName;
	sprintf(dummy, "Archive%i.txt", filenumber);
	archive.saveArchive(dummy);
	std::cout << "\nThe result is stored in " << dummy << "\n" << std::endl;
	std::cout << "Number of Solutions" << std::endl;
	std::cout << "Number of Objectives" << std::endl;
	std::cout << "Solution 1 (f1,f2,...)" << std::endl;
	std::cout << "Solution 2 (f1,f2,...)\n" << std::endl;

	std::cout << "Hit Return Key for End" << std::endl;
	fgetc(stdin);

	return 0;

}


