// ======================================================================
/*!
 *  \file realcodedNSGA2.cpp
 *  \date 2004-12-29
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

#include <stdlib.h>
#include <stdio.h>

#include <Array/Array.h>
#include <FileUtil/Params.h>
#include <MOO-EALib/TestFunction.h>
#include <MOO-EALib/PopulationMOO.h>
#include <MOO-EALib/ArchiveMOO.h>
#include <SharkDefs.h>

using namespace std;

// My own derived class for managing my configuration files:
//
class MyParams : public Params
{
public:

	MyParams(int argc, char **argv) : Params(argc, argv)
	{}

	~MyParams()
	{}

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
		FileUtil::io(is, os, "Task "        , Task 	   , 1u		, type);
		FileUtil::io(is, os, "ArchiveSize " , ArchiveSize , 100u	, type);
		FileUtil::io(is, os, "Dimension "   , Dimension   , 10u	, type);
		FileUtil::io(is, os, "Iterations "  , Iterations  , 500u	, type);
		FileUtil::io(is, os, "DisplInt "    , DisplInt    , 100u	, type);
		FileUtil::io(is, os, "MinInit "     , MinInit	   , -25.	, type);
		FileUtil::io(is, os, "MaxInit "     , MaxInit	   , 25.        , type);
		FileUtil::io(is, os, "Seed  "       , Seed	   , 1u		, type);
		FileUtil::io(is, os, "rotBasis "    , rotBasis	   , true	, type);
		FileUtil::io(is, os, "rpcon1 "      , rpcon1	   , 1000.	, type);
		FileUtil::io(is, os, "rpcon2 "      , rpcon2	   , 2.		, type);
		FileUtil::io(is, os, "nc "	     , nc	   , 20u	, type);
		FileUtil::io(is, os, "nm "	     , nm	   , 20.	, type);
		FileUtil::io(is, os, "pbc "	     , pbc	   , 0.5	, type);
		FileUtil::io(is, os, "pc "	     , pc	   , 0.9	, type);
		FileUtil::io(is, os, "pm "	     , pm	   , 1. / Dimension, type);
	}

	// Method to show the current content of the class variable:
	void monitor()
	{
		std::cout << "Task        " << Task 		<< std::endl;
		std::cout << "ArchiveSize " << ArchiveSize	<< std::endl;
		std::cout << "Dimension   " << Dimension	<< std::endl;
		std::cout << "Iterations  " << Iterations	<< std::endl;
		std::cout << "DisplInt    " << DisplInt		<< std::endl;
		std::cout << "MinInit     " << MinInit		<< std::endl;
		std::cout << "MaxInit     " << MaxInit		<< std::endl;
		std::cout << "rotBasis    " << rotBasis		<< std::endl;
		std::cout << "rpcon1      " << rpcon1		<< std::endl;
		std::cout << "rpcon2      " << rpcon2		<< std::endl;
		std::cout << "Seed        " << Seed		<< std::endl;
		std::cout << "nc          " << nc		<< std::endl;
		std::cout << "nm          " << nm		<< std::endl;
		std::cout << "pc          " << pc		<< std::endl;
		std::cout << "pbc         " << pbc		<< std::endl;
		std::cout << "pm          " << pm		<< std::endl;
	}

	unsigned Seed;
	unsigned ArchiveSize;
	unsigned Dimension;
	unsigned Iterations;
	unsigned Task;
	unsigned DisplInt;
	double	MinInit;
	double	MaxInit;
	bool	rotBasis;
	double	rpcon1;
	double	rpcon2;
	unsigned nc;
	double	pc;
	double	pbc;
	double   nm;
	double   pm;
};

inline void calcFitness(IndividualMOO &i,
						unsigned Task,
						vector<double>& Min,
						vector<double>& Max,
						Array<double> & Basis,
						double rotparco1 = 1000.,
						double rotparco2 = 2.,
						bool initBasis = false
					   )
{
	double f1 = 0, f2 = 0;unsigned j;
	ChromosomeT< double >& dblchrom = dynamic_cast< ChromosomeT< double >& >(i[ 0 ]);
	switch (Task) {
	case 1:
		Basis = 0; for (j = 0;j < Basis.dim(0);j++) Basis(j, j) = 1;
		f1 = RotParF1(dblchrom, Basis, Min, Max, 1.);
		f2 = RotParF2(dblchrom, Basis, Min, Max, 1., rotparco2);
		break;
	case 2:
		f1 = FonsecaConcaveF1(dblchrom);
		f2 = FonsecaConcaveF2(dblchrom);
		break;
	case 3:
		f1 = MessacConcaveF1(dblchrom);
		f2 = MessacConcaveF2(dblchrom);
		break;
	case 4:
		f1 = ZDT1F1(dblchrom);
		f2 = ZDT1F2(dblchrom);
		break;
	case 5:
		f1 = ZDT2F1(dblchrom);
		f2 = ZDT2F2(dblchrom);
		break;
	case 6:
		f1 = ZDT3F1(dblchrom);
		f2 = ZDT3F2(dblchrom);
		break;
	case 7:
		f1 = ZDT4F1(dblchrom);
		f2 = ZDT4F2(dblchrom);
		break;
	case 8:
		f1 = ZDT6F1(dblchrom);
		f2 = ZDT6F2(dblchrom);
		break;
	case 9:
		if (initBasis) generateBasis(dblchrom.size(), Basis);
		f1 = RotParF1(dblchrom, Basis, Min, Max, rotparco1);
		f2 = RotParF2(dblchrom, Basis, Min, Max, rotparco1, rotparco2);
		break;
	case 10:
		if (initBasis) generateBasis(dblchrom.size(), Basis);
		f1 = DebRotatedF1(dblchrom, Basis);
		f2 = DebRotatedF2(dblchrom, Basis);
		break;
	case 11:
		if (initBasis) generateBasis(dblchrom.size(), Basis);
		f1 = RotCigarF1(dblchrom, Basis, Min, Max, rotparco1);
		f2 = RotCigarF2(dblchrom, Basis, Min, Max, rotparco1, rotparco2);
		break;
	case 12:
		if (initBasis) generateBasis(dblchrom.size(), Basis);
		f1 = RotTabletF1(dblchrom, Basis, Min, Max, rotparco1);
		f2 = RotTabletF2(dblchrom, Basis, Min, Max, rotparco1, rotparco2);
		break;
	default:
		std::cerr << "unknown task:" << Task << std::endl;
		break;
	}

	i.setMOOFitnessValues(f1, f2);
}

// main program
int main(int argc, char* argv[])
{
	unsigned i, ii, t;

	MyParams param(argc, argv);

	param.setDefault();
	//param.readParams();

	if (argc > 1) {
		for (int i = 0; i < argc; ++i)
			if (strcmp(argv[ i ], "-Seed") == 0 && i + 1 < argc)
				param.Seed = atoi(argv[ i+1 ]);
	}

	param.monitor();

	Array<double> B(param.Dimension, param.Dimension);
	B = 0; for (i = 0;i < B.dim(0);i++) B(i, i) = 1;

	// initialize random number generator
	Rng::seed(param.Seed);

	Array<double>	pf, sf;

	cout << endl;
	switch (param.Task) {
	case 1:
		cout << "Sphere" << endl;
		cout << "shift " << param.rpcon2 << endl;
		RotParSampleFront(pf, param.Dimension, 1., param.rpcon2);
		if (param.Dimension == 2) RotParSample(sf, param.Dimension, 1., param.rpcon2);
		break;
	case 2:
		cout << "FonsecaConcave" << endl;
		if (param.MinInit < -4. || param.MaxInit > 4.) {
			cerr << "Warning: parameter initialization is out of bounds, resetting ..." << endl;
			param.MinInit = -4.; param.MaxInit = 4.;
		}
		FonsecaConcaveSampleFront(pf, param.Dimension);
		if (param.Dimension == 2) FonsecaConcaveSample(sf, param.Dimension);
		break;
	case 3:
		cout << "MessacConcave" << endl;
		if (param.Dimension == 2)
			MessacConcaveSample(sf, param.Dimension, param.MinInit, param.MaxInit);
		if (param.Dimension > 1) {
		  throw SHARKEXCEPTION("Warning: pareto sampling is not implemented for dim > 1 ...");
		}
		if (param.Dimension == 1)
			MessacConcaveSampleFront(sf, param.Dimension, param.MinInit, param.MaxInit);
		break;
	case 4:
		cout << "ZDT1" << endl;
		ZDT1SampleFront(pf);
		break;
	case 5:
		cout << "ZDT2" << endl;
		ZDT2SampleFront(pf);
		break;
	case 6:
		cout << "ZDT3" << endl;
		ZDT3SampleFront(pf);
		break;
	case 7:
		cout << "ZDT4" << endl;
		ZDT4SampleFront(pf);
		break;
	case 8:
		cout << "ZDT6" << endl;
		ZDT6SampleFront(pf);
		break;
	case 9:
		cout << "Rotated Paraboloid" << endl;
		cout << "rotated Basis " << param.rotBasis << ", scaling " << param.rpcon1 << ", shift " << param.rpcon2 << endl;
		RotParSampleFront(pf, param.Dimension, param.rpcon1, param.rpcon2);
		if (param.Dimension == 2)
			RotParSample(sf, param.Dimension, param.rpcon1, param.rpcon2);
		break;
	case 10:
		cout << "Deb's Rotated test function" << endl;
		cout << "rotated Basis " << param.rotBasis << endl;
		DebRotatedSampleFront(pf, param.Dimension);
		break;
	case 11:
		cout << "Rotated Cigar" << endl;
		cout << "rotated Basis " << param.rotBasis << ", scaling " << param.rpcon1 << ", shift " << param.rpcon2 << endl;
		RotCigarSampleFront(pf, param.Dimension, param.rpcon1, param.rpcon2);
		break;
	case 12:
		cout << "Rotated Tablet" << endl;
		cout << "rotated Basis " << param.rotBasis << ", scaling " << param.rpcon1 << ", shift " << param.rpcon2 << endl;
		RotTabletSampleFront(pf, param.Dimension, param.rpcon1, param.rpcon2);
		break;
	default:
	  throw SHARKEXCEPTION("Unknown task ");
	}

	char filename[500];

	if (param.Task != 3 || param.Dimension == 1) {
		sprintf(filename, "pareto_%d_%d.arv", param.Task, param.Dimension);
		ofstream PF(filename);
		for (ii = 0;ii < pf.dim(0);ii++) {
			PF << pf(ii, 0) << " " << pf(ii, 1) << endl;
			//cerr<< ii << " " << pf(ii,0) << " " << pf(ii,1) << endl;
		}
		PF.close();
	}

	if ((param.Task < 4 || param.Task == 9) && param.Dimension == 2) {
		sprintf(filename, "sample_%d_%d.arv", param.Task, param.Dimension);
		ofstream SF(filename);
		for (ii = 0;ii < sf.dim(0);ii++) {
			SF << sf(ii, 0) << " " << sf(ii, 1) << endl;
			//cerr<< ii << " " << pf(ii,0) << " " << pf(ii,1) << endl;
		}
		SF.close();
	}

	// define populations
	PopulationMOO parents(param.ArchiveSize, ChromosomeT< double >(param.Dimension));
	PopulationMOO offsprings(param.ArchiveSize, ChromosomeT< double >(param.Dimension));

	// Set Minimization Task
	parents   .setMinimize();
	offsprings.setMinimize();

	// Set No of Objective funstions
	parents   .setNoOfObj(2);
	offsprings.setNoOfObj(2);

	ArchiveMOO archive(param.ArchiveSize);
	archive.minimize();
	archive.setMaxArchive(param.ArchiveSize);

	vector<double> lower(param.Dimension), upper(param.Dimension);

	for (i = 0; i < lower.size(); ++i) {
		lower[i] = param.MinInit;
		upper[i] = param.MaxInit;
	}

	// initialize all chromosomes of parent population
	for (i = 0; i < parents.size(); ++i)
		dynamic_cast< ChromosomeT< double >& >(parents[ i ][ 0 ]).initialize(param.MinInit, param.MaxInit);

	// evaluate parents
	for (i = 0; i < parents.size(); ++i)
		calcFitness(parents[ i ], param.Task, lower, upper, B, param.rpcon1, param.rpcon2, param.rotBasis);

	parents.crowdedDistance();

	// iterate
	for (t = 1; t < param.Iterations + 1; ++t) {

		if (t % param.DisplInt == 1)
			cout << "Generation: " << t << endl;

		// copy parents to offsprings
		offsprings.selectBinaryTournamentMOO(parents);

		// recombine by crossing over two parents
		for (i = 0; i < offsprings.size(); i += 2)
			if (Rng::coinToss(param.pc))
				dynamic_cast< ChromosomeT< double >& >(offsprings[i][0]).SBX(dynamic_cast< ChromosomeT< double >& >(offsprings[i+1][0]), lower, upper, param.nc, param.pbc);

		// mutate by flipping bits and evaluate objective function
		for (i = 0; i < offsprings.size(); ++i) {
			// modify
			dynamic_cast< ChromosomeT< double >& >(offsprings[i][0]).mutatePolynomial(param.MinInit, param.MaxInit, param.nm, param.pm);

			// evaluate objective function
			calcFitness(offsprings[ i ], param.Task, lower, upper, B, param.rpcon1, param.rpcon2);
		}

		// Selection
		parents.selectCrowdedMuPlusLambda(offsprings);

		if (!(t % 10)) {
			archive.cleanArchive();
			for (i = 0; i < parents.size(); i++)
				archive.addArchive(parents[i]);
			archive.nonDominatedSolutions();
		}
	} // Iteration
	cout << "Generation: " << t << endl;

	// Data Output
	archive.cleanArchive();
	for (ii = 0; ii < param.ArchiveSize; ii++) {
		archive.addArchive(parents[ ii ]);
	}

	archive.nonDominatedSolutions();

	cout << endl;
	sprintf(filename, "nsga2_%d_%d_%d_%d_%d.arv", param.Task, param.ArchiveSize, param.Dimension, param.Iterations, param.Seed);
	cout	<< "Size of the Archive: "  << archive.size()
	<< ", filename of archive: " << filename << endl << endl;

	archive.saveArchiveGPT(filename);

	/*
	for(i=0;i< archive.size();i++)
	{	
		for(ii=0;ii<(uint)archive.readArchive(i)[0].size();ii++)
			std::cout << (static_cast<ChromosomeT<double>&>(archive.readArchive( i )[0]))[ii] << " " << std::flush;
		std::cout << std::endl;
	}
	*/

	return 0;
}

