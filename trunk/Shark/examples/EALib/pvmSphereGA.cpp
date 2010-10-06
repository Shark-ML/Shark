/*! pvmSphereGA.cpp
* ======================================================================
*
*  \file pvmSphereGA.cpp
*  \date 2004
*
*  \author Stefan Wiegand
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
*  \par Project:
*      EALib
*
*
*  <BR><HR>
*  This file is part of the EALib. This library is free software;
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

#include <stdio.h>
#include <iostream>

#include <SharkDefs.h>
#include <EALib/PVMinterface.h>
#include <Array/Array.h>
#include <EALib/Population.h>


/////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// HOWTO START THIS PARALLEL OPTIMIZATION EXAMPLE:
//
// Setup your PVM-environent and recompile this source.
// Start the pvm console, i.e., call the command
//  '... /PVM-binpath/pvm'.
// For example, add 3 hosts from your local network to your parallel virtual machine, i.e.,
//  'add host1 host2 host3'.
// Start the pvmSphereGA, i.e,
//  'spawn -> -4 ${SHARKHOME}/EALib/examples/pvmSphereGA_${ARCH} [random-seed]'.
//
// Assuming equal random seed initializations, however, there might be different results compared to
// 'sphereGA' or even to another restart of 'pvmSphereGA'. This is due to different orderings of
// Individuals within the Population which might occur in the parallel case using the roulette wheel
// selection within the proportional selection scheme.


//=======================================================================
//=======================================================================
//
// Some tools for readers convenience
//
//=======================================================================
//=======================================================================

// fitness function: sphere model
//
double sphere(const std::vector< double >& x);

// send-and-receive routine of the master process
//
void pvmMasterSendReceive(Population& pop          ,
						  char*       PVM_groupname,
						  const int   PVM_groupsize,
						  const int   msgTagSend   ,
						  const int   msgTagReceive
						 );

// receive-evaluate-and-send routine of the slave processes
//
void pvmSlaveReceiveSend(char*          PVM_groupname,
						 const int      PVM_groupsize,
						 const int      PVM_MasterID ,
						 const int      msgTagExit   ,
						 const int      msgTagSend   ,
						 const int      msgTagReceive,
						 const Interval RangeOfValues,
						 const unsigned NumOfBits    ,
						 const bool     UseGrayCode
						);

//=======================================================================
//=======================================================================
//
// Main programm
//
//=======================================================================
//=======================================================================


int main(int argn, char *argv[])
{

	std::cout << "# PVM process attemps to start ... "
	<< std::endl;

	////////////////////////////////
	//
	// Parameter initialization
	//

	// PVM constants and variables
	//
	char*       PVM_groupname      = strdup("PVM-Test");
	const int   PVM_groupsize      = 4;
	const int   PVM_sendExitTag    = 3;
	const int   PVM_sendFitnessTag = 2;
	const int   PVM_sendEvalTag    = 1;
	const int   PVM_MasterID       = 0;
	int inum, tid;

	// EALib constants and variables
	//
	const unsigned PopSize     = 50;
	const unsigned Dimension   = 20;
	const unsigned NumOfBits   = 10;
	const unsigned Iterations  = 2000;
	const unsigned DspInterval = 10;
	const unsigned NElitists   = 1;
	const unsigned Omega       = 5;
	const unsigned CrossPoints = 2;
	const double   CrossProb   = 0.6;
	const double   FlipProb    = 1. / (Dimension * NumOfBits);
	const bool     UseGrayCode = true;
	const Interval RangeOfValues(-3, + 5);

	unsigned i, t;

	int myseed = (argn > 1 ? atoi(argv[ 1 ]) : 1234);

	////////////////////////////////
	//
	// Parallel processing
	//

	// require a unique task-ID and apply for the PVM-group
	//
	tid  = pvm_mytid();

	if (tid < 0) {
		throw SHARKEXCEPTION("There is something going wrong ... \nmaybe you forgot to spawn processes from the pvm-console.\nPlease try the following. \nStart the pvm-console, i.e., call the command '... /$(PVM-binpath)/pvm'.\nAfter the pvm-console has started, for example, add 3 hosts from your local network\nto your parallel virtual machine (PVM), i.e., call 'add host1 host2 host3'.\nStart this program from the pvm console, i.e, call\n'spawn -> -4 $(SHARKHOME)/EALib/examples/pvmSphereGa<extension for your operating system> [optional random-seed]'.");
		//		exit(-1);
	}
	else
		std::cout << "# ... process with TID " << tid << " started"
		<< " successfully. Please hold on for results."
		<< std::endl;

	inum = pvm_joingroup(PVM_groupname);

	// synchronize all processes
	//
	pvm_barrier(PVM_groupname, PVM_groupsize);

	// distinction between master and slave processes
	//
	if (inum == PVM_MasterID) {
		//
		//  Master process
		//
		Rng::seed(myseed);

		// define populations
		//
		static Population parents(PopSize, ChromosomeT< bool >(Dimension * NumOfBits));
		static Population offspring(PopSize, ChromosomeT< bool >(Dimension * NumOfBits));

		// scaling window
		//
		std::vector< double > window(Omega);

		// minimization task
		//
		parents  .setMinimize();
		offspring.setMinimize();

		// initialize all chromosomes of parent population
		//
		for (i = 0; i < parents.size(); ++i)
			dynamic_cast< ChromosomeT< bool >& >(parents[ i ][ 0 ]).initialize();

		// send and receive individuals
		//
		if (NElitists > 0)
			pvmMasterSendReceive(parents, PVM_groupname, PVM_groupsize, PVM_sendEvalTag, PVM_sendFitnessTag);

		for (t = 0; t < Iterations; ++t) {
			// recombine by crossing over two parents
			//
			offspring = parents;
			for (i = 0; i < offspring.size() - 1; i += 2)
				if (Rng::coinToss(CrossProb))
					offspring[ i ][ 0 ].crossover(offspring[ i+1 ][ 0 ],
												  CrossPoints);

			// mutate by flipping bits
			//
			for (i = 0; i < offspring.size(); ++i)
				dynamic_cast< ChromosomeT< bool >& >(offspring[ i ][ 0 ]).flip(FlipProb);

			// send and receive individuals
			//
			pvmMasterSendReceive(offspring,
								 PVM_groupname,
								 PVM_groupsize,
								 PVM_sendEvalTag,
								 PVM_sendFitnessTag);

			// scale fitness values and use proportional selection
			//
			offspring.linearDynamicScaling(window, t);
			parents.selectProportional(offspring, NElitists);

			// print out best value found so far
			//
			if (t % DspInterval == 0)
				std::cout << t << "\tbest value = "
				<< parents.best().fitnessValue() << "\n";
		}

		// exit slave processess
		//
		for (int i = 1; i < PVM_groupsize; i++) {
			pvm_initsend(PvmDataDefault);
			pvm_send(pvm_gettid(PVM_groupname, i), PVM_sendExitTag);
		}
	}
	else {
		//
		//  Slave processess
		//

		// Note, if random sampling is required it should be ensured
		// that all slaves apply different random seed initializations
		//
		// Rng::seed(myseed + inum);

		// receive, evaluate and send parents (only needed for elitist strategy)
		//
		pvmSlaveReceiveSend(PVM_groupname,
							PVM_groupsize,
							PVM_MasterID,
							PVM_sendExitTag,
							PVM_sendFitnessTag,
							PVM_sendEvalTag,
							RangeOfValues,
							NumOfBits,
							UseGrayCode
						   );
	}

	// synchronize all processes
	//
	pvm_barrier(PVM_groupname, PVM_groupsize);

	////////////////////////////////
	//
	// Finish all processes
	//
	std::cout << "# PVM process with TID " << tid << " will halt now... " << std::endl
	<< std::endl;

	pvm_exit();
}



//=======================================================================
//
// fitness function: sphere model
//
double sphere(const std::vector< double >& x)
{
	unsigned i;
	double   sum;
	for (sum = 0., i = 0; i < x.size(); i++)
		sum += Shark::sqr(x[ i ]);
	return sum;
}

//=======================================================================
//
// send and receive routine of the master
//
void pvmMasterSendReceive(Population& pop          ,
						  char*       PVM_groupname,
						  const int   PVM_groupsize,
						  const int   msgTagSend   ,
						  const int   msgTagReceive
						 )
{
	unsigned i;

	for (i = 0; i < pop.size(); ++i) {
		// send all individuals of pop to slave processes for evaluation
		//
		pvm_initsend(PvmDataDefault);
		pop[i].pvm_pkind();
		pvm_send(pvm_gettid(PVM_groupname, i % (PVM_groupsize - 1) + 1), msgTagSend);
	}

	int bufid = 0; int counter = pop.size();

	while (counter > 0) {
		// check whether any individual has returned from parallel evaluation
		//
		do {
			bufid = pvm_probe(-1, msgTagReceive);
		}
		while (bufid == 0);

		// in case, integrate the evaluated individual in the Population pop
		//
		if (bufid > 0) {
			// extract obligatory information from the message
			//
			int bytes; int msgtag; int tid;
			pvm_bufinfo(bufid, &bytes, &msgtag, &tid);

			// receive individual and overwrite not evaluated individuals with evaluated ones
			//
			pvm_recv(tid, msgtag);
			pop[pop.size()-counter].pvm_upkind();
			counter--;
		}
	}
}

//=======================================================================
//
// receive, evaluate and send routine of the slave
//
void pvmSlaveReceiveSend(char*          PVM_groupname,
						 const int      PVM_groupsize,
						 const int      PVM_MasterID ,
						 const int      msgTagExit   ,
						 const int      msgTagSend   ,
						 const int      msgTagReceive,
						 const Interval RangeOfValues,
						 const unsigned NumOfBits    ,
						 const bool     UseGrayCode
						)
{
	int bufid, bufidExit = 0;

	while (bufidExit == 0) {
		// check whether any individual has arrived from the master process
		//
		do {
			bufid     = pvm_probe(pvm_gettid(PVM_groupname, PVM_MasterID), msgTagReceive);
			bufidExit = pvm_probe(pvm_gettid(PVM_groupname, PVM_MasterID), msgTagExit)   ;
			if (bufidExit != 0)
				pvm_recv(pvm_gettid(PVM_groupname, PVM_MasterID), msgTagExit)              ;
		}
		while ((bufid == 0) && (bufidExit == 0));

		// in case, evaluate individual and return it to the sender
		//
		if (bufid != 0) {
			// extract obligatory information from the message
			//
			int bytes; int msgtag; int tid;
			pvm_bufinfo(bufid, &bytes, &msgtag, &tid);

			// receive individual
			//
			Individual ind(ChromosomeT<bool>(1));
			pvm_recv(tid, msgtag);
			ind.pvm_upkind();

			// evaluate objective function
			//
			ChromosomeT< double > dblchrom;
			dblchrom.decodeBinary(ind[ 0 ], RangeOfValues, NumOfBits, UseGrayCode);
			ind.setFitness(sphere(dblchrom));

			// return evaluated individua to the sender
			//
			pvm_initsend(PvmDataDefault);
			ind.pvm_pkind();
			pvm_send(tid, msgTagSend);
		}
	}
}



