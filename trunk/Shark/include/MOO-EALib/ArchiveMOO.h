/*! ======================================================================
 *
 *  \file ArchiveMOO.h
 *
 *  \brief External Archive of candidates for pareto optimality
 * 
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
 *
 *  \par Project:
 *      MOO-EALib
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


#ifndef _ARCHIVEMOO_H_
#define _ARCHIVEMOO_H_

#include <SharkDefs.h>
#include <Array/Array.h>
#include <MOO-EALib/PopulationMOO.h>
#include <MOO-EALib/IndividualMOO.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>


//!
//! \brief Archive of candidates for pareto optimality
//!
class ArchiveMOO : private std::vector< IndividualMOO * >
{
public:
	//*******************************************
	//** Constructor
	//*******************************************
	//***** TO-AM-001
	ArchiveMOO();
	//***** TO-AM-002
	ArchiveMOO(bool strategy);
	//***** TO-AM-003
	ArchiveMOO(unsigned max);
	//***** TO-AM-004
	ArchiveMOO(unsigned max, bool strategy);

	//*******************************************
	//** Destructor
	//*******************************************
	//***** TO-AM-005
	~ArchiveMOO();

	//*******************************************
	//** Internal variables I/O
	//*******************************************
	//***** TO-AM-010
	unsigned getMaxArchive();
	//***** TO-AM-011
	void     setMaxArchive(unsigned max);
	//***** TO-AM-012
	unsigned getCapacity();
	//***** TO-AM-013
	unsigned size();
	//***** TO-AM-014
	bool     getStrategy();
	//***** TO-AM-015
	void     setStrategy(bool strategy);
	//***** TO-AM-016
	void     minimize();
	//***** TO-AM-017
	void     maximize();

	//*******************************************
	//** Archive I/O
	//*******************************************
	//***** TO-AM-050
	void addArchive(IndividualMOO& indmoo);
	//***** TO-AM-051
	IndividualMOO& readArchive(unsigned i);
	//***** TO-AM-052
	void delArchive(unsigned i);
	//***** TO-AM-053
	void delArchive(std::vector< unsigned >& v);
	//***** TO-AM-054
	void delArchive(Array< unsigned >& a);
	//***** TO-AM-055
	void cleanArchive();
	//***** TO-AM-056
	void delSharingWorst();
	//***** TO-AM-057
	void delSharingWorst(double div);
	//***** TO-AM-058
	IndividualMOO& readBestArchive();
	//***** TO-AM-059
	void nonDominatedSolutions();

	//*******************************************
	//** Dominate
	//*******************************************
	//***** TO-AM-100
	int Dominate(IndividualMOO& im1, IndividualMOO& im2);
	//***** TO-AM-101
	int Dominate(IndividualMOO& im1);
	//***** TO-AM-102
	void delDominateArchive(IndividualMOO& im1);


	//*******************************************
	//** Distance on Fitness Space
	//*******************************************
	//***** TO-AM-150
	double distanceOnFitness(unsigned i1, unsigned i2);
	//***** TO-AM-151
	double distanceOnFitness(IndividualMOO& im1);
	//***** TO-AM-152
	Array< double > distanceDataOnFitness();
	//***** TO-AM-153
	unsigned sharingWorst();
	//***** TO-AM-154
	unsigned sharingWorst(double div);
	//***** TO-AM-155
	unsigned sharingBest();
	//***** TO-AM-156
	double minDistanceOnFitness();
	//***** TO-AM-157
	double minDistanceOnFitness(unsigned i1);


	//*******************************************
	//** PAES
	//*******************************************
	//***** TO-AM-300
	unsigned crowded(IndividualMOO& im, double div);



	//*******************************************
	//** For Metrix
	//*******************************************
	//***** TO-AM-1000
	void saveArchive(char *filename);

	//***** SW-AM-1001
	void saveArchiveGPT(char *filename);



protected:
	unsigned MaxArchive;
	bool     Strategy;



};


#endif

