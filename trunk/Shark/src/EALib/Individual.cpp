/*!
*  \file Individual.cpp
*
*  \author Martin Kreutz
*
*  \brief Base class for individuals in a population 
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
*    \par Project:
*        EALib
*
*
*  <BR>
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

#include "EALib/Individual.h"
#include "SharkDefs.h"

using namespace std;

//===========================================================================
//
// constructors
//
Individual::Individual(unsigned n) : vector<Chromosome*>(n),
fitness( 0 ),
scaledFitness( 0 ),
evalFlg( true ),
feasible( 0 ),
selProb( 0 ),
numCopies( 0 ),
elitist( false ),
age( 0 ),
learnTime( 0 ) {
	for (unsigned i = 0; i < size(); i++ )
		*(begin() + i) = new ChromosomeT< char >;
}

Individual::Individual(unsigned n, const Chromosome& chrom )  : vector<Chromosome*>(n),
fitness( 0 ),
scaledFitness( 0 ),
evalFlg( true ),
feasible( 0 ),
selProb( 0 ),
numCopies( 0 ),
elitist( false ),
age( 0 ),
learnTime( 0 ) {
	for (unsigned i = 0; i < size(); i++ )
		*(begin() + i) = chrom.clone();
}

Individual::Individual(const Chromosome& chrom0)  : vector<Chromosome*>( 1 ),
fitness( 0 ),
scaledFitness( 0 ),
evalFlg( true ),
feasible( 0 ),
selProb( 0 ),
numCopies( 0 ),
elitist( false ),
age( 0 ),
learnTime( 0 ) {
	*(begin()) = chrom0.clone();
}

Individual::Individual(const Chromosome& chrom0,
					   const Chromosome& chrom1)  : vector<Chromosome*>(2),
fitness( 0 ),
scaledFitness( 0 ),
evalFlg( true ),
feasible( 0 ),
selProb( 0 ),
numCopies( 0 ),
elitist( false ),
age( 0 ),
learnTime( 0 ) {
	*(begin()) = chrom0.clone();
	*(begin() + 1) = chrom1.clone();
}

Individual::Individual(const Chromosome& chrom0,
					   const Chromosome& chrom1,
					   const Chromosome& chrom2)  : vector<Chromosome*>( 3 ),
fitness( 0 ),
scaledFitness( 0 ),
evalFlg( true ),
feasible( 0 ),
selProb( 0 ),
numCopies( 0 ),
elitist( false ),
age( 0 ),
learnTime( 0 ) {
	*(begin()) = chrom0.clone();
	*(begin() + 1) = chrom1.clone();
	*(begin() + 2) = chrom2.clone();
}

Individual::Individual(const Chromosome& chrom0,
					   const Chromosome& chrom1,
					   const Chromosome& chrom2,
					   const Chromosome& chrom3)  : vector<Chromosome*>(4),
fitness( 0 ),
scaledFitness( 0 ),
evalFlg( true ),
feasible( 0 ),
selProb( 0 ),
numCopies( 0 ),
elitist( false ),
age( 0 ),
learnTime( 0 ) {
	*(begin()) = chrom0.clone();
	*(begin() + 1) = chrom1.clone();
	*(begin() + 2) = chrom2.clone();
	*(begin() + 3) = chrom3.clone();
}

Individual::Individual(const Chromosome& chrom0,
					   const Chromosome& chrom1,
					   const Chromosome& chrom2,
					   const Chromosome& chrom3,
					   const Chromosome& chrom4)  : vector<Chromosome*>(5),
fitness( 0 ),
scaledFitness( 0 ),
evalFlg( true ),
feasible( 0 ),
selProb( 0 ),
numCopies( 0 ),
elitist( false ),
age( 0 ),
learnTime( 0 ) {
	*(begin()) = chrom0.clone();
	*(begin() + 1) = chrom1.clone();
	*(begin() + 2) = chrom2.clone();
	*(begin() + 3) = chrom3.clone();
	*(begin() + 4) = chrom4.clone();
}

Individual::Individual(const Chromosome& chrom0,
					   const Chromosome& chrom1,
					   const Chromosome& chrom2,
					   const Chromosome& chrom3,
					   const Chromosome& chrom4,
					   const Chromosome& chrom5)  : vector<Chromosome*>(6),
fitness( 0 ),
scaledFitness( 0 ),
evalFlg( true ),
feasible( 0 ),
selProb( 0 ),
numCopies( 0 ),
elitist( false ),
age( 0 ),
learnTime( 0 ) {
	*(begin()) = chrom0.clone();
	*(begin() + 1) = chrom1.clone();
	*(begin() + 2) = chrom2.clone();
	*(begin() + 3) = chrom3.clone();
	*(begin() + 4) = chrom4.clone();
	*(begin() + 5) = chrom5.clone();
}

Individual::Individual(const Chromosome& chrom0,
					   const Chromosome& chrom1,
					   const Chromosome& chrom2,
					   const Chromosome& chrom3,
					   const Chromosome& chrom4,
					   const Chromosome& chrom5,
					   const Chromosome& chrom6)  : vector<Chromosome*>(7),
fitness( 0 ),
scaledFitness( 0 ),
evalFlg( true ),
feasible( 0 ),
selProb( 0 ),
numCopies( 0 ),
elitist( false ),
age( 0 ),
learnTime( 0 ) {
	*(begin()) = chrom0.clone();
	*(begin() + 1) = chrom1.clone();
	*(begin() + 2) = chrom2.clone();
	*(begin() + 3) = chrom3.clone();
	*(begin() + 4) = chrom4.clone();
	*(begin() + 5) = chrom5.clone();
	*(begin() + 6) = chrom6.clone();
}

Individual::Individual(const Chromosome& chrom0,
					   const Chromosome& chrom1,
					   const Chromosome& chrom2,
					   const Chromosome& chrom3,
					   const Chromosome& chrom4,
					   const Chromosome& chrom5,
					   const Chromosome& chrom6,
					   const Chromosome& chrom7)  : vector<Chromosome*>(8),
fitness( 0 ),
scaledFitness( 0 ),
evalFlg( true ),
feasible( 0 ),
selProb( 0 ),
numCopies( 0 ),
elitist( false ),
age( 0 ),
learnTime( 0 ) {
	*(begin()) = chrom0.clone();
	*(begin() + 1) = chrom1.clone();
	*(begin() + 2) = chrom2.clone();
	*(begin() + 3) = chrom3.clone();
	*(begin() + 4) = chrom4.clone();
	*(begin() + 5) = chrom5.clone();
	*(begin() + 6) = chrom6.clone();
	*(begin() + 7) = chrom7.clone();
}

Individual::Individual(const vector< Chromosome * >& chrom)  : vector<Chromosome*>( chrom ),
fitness( 0 ),
scaledFitness( 0 ),
evalFlg( true ),
feasible( 0 ),
selProb( 0 ),
numCopies( 0 ),
elitist( false ),
age( 0 ),
learnTime( 0 ) {
}

Individual::Individual(const Individual& indiv)  : vector<Chromosome*>( indiv.size() ),
fitness( indiv.fitness ),
scaledFitness( indiv.scaledFitness ),
evalFlg( indiv.evalFlg ),
feasible( indiv.feasible ),
selProb( indiv.selProb ),
numCopies( indiv.numCopies ),
elitist( indiv.elitist ),
age( indiv.age ),
learnTime( indiv.learnTime ) {
	for( unsigned int i = 0; i < size(); i++ )
		*(begin() + i) = indiv[ i ].clone();
}

//===========================================================================
//
// destructor
//
Individual::~Individual() {
	for( unsigned int i = 0; i < size(); i++ )
		delete *(begin() + i);
}

//===========================================================================

unsigned Individual::totalSize() const
{
	unsigned s = 0;
	for (unsigned i = size(); i--; s += (*(begin() + i))->size());
	return s;
}

//===========================================================================
// changed by Marc Toussaint and Stefan Wiegand (INI) 20.11.2002
Individual& Individual::operator = (const Individual& indiv)
{
	unsigned i;

	for (i = size(); i--;)
		delete *(begin() + i);
	vector< Chromosome * >::operator = (indiv);
	for (i = size(); i--;)
		*(begin() + i) = indiv[ i ].clone();

	fitness       = indiv.fitness;
	scaledFitness = indiv.scaledFitness;
	evalFlg       = indiv.evalFlg;
	feasible      = indiv.feasible;
	selProb       = indiv.selProb;
	numCopies     = indiv.numCopies;
	elitist       = indiv.elitist;
	age           = indiv.age;
	learnTime     = indiv.learnTime;

	return *this;
}

//===========================================================================

void Individual::replace(unsigned i, const Chromosome& chrom)
{
	RANGE_CHECK(i < size())
	delete *(begin() + i);
	*(begin() + i) = chrom.clone();
}

void Individual::insert(unsigned i, const Chromosome& chrom)
{
	RANGE_CHECK(i <= size())
	vector< Chromosome * >::insert(begin() + i, chrom.clone());
}

void Individual::append(const Chromosome& chrom)
{
	vector< Chromosome * >::push_back(chrom.clone());
}

void Individual::remove(unsigned i)
{
	RANGE_CHECK(i < size())
	delete *(begin() + i);
	vector< Chromosome * >::erase(begin() + i);
}

void Individual::remove(unsigned i, unsigned k)
{
	if (i <= k) {
		RANGE_CHECK(k < size())
		for (unsigned j = i; j < k; j++)
			delete *(begin() + j);
		vector< Chromosome * >::erase(begin() + i, begin() + k);
	}
}

//===========================================================================

bool Individual::operator == (const Individual& ind) const
{
	if (size() == ind.size()) {
		for (unsigned i = 0; i < size(); ++i)
			if (!((*this)[ i ] == ind[ i ])) return false;

		return fitness       == ind.fitness       &&
			   scaledFitness == ind.scaledFitness &&
			   evalFlg       == ind.evalFlg       &&
			   feasible      == ind.feasible      &&
			   selProb       == ind.selProb       &&
			   numCopies     == ind.numCopies     &&
			   elitist       == ind.elitist       &&
			   age           == ind.age           &&
			   learnTime     == ind.learnTime    ;
	}
	return false;
}

bool Individual::operator < (const Individual& ind) const
{
	if (size() == ind.size()) {
		bool less = false;

		for (unsigned i = 0; i < size(); ++i)
			if (ind[ i ] < (*this)[ i ])
				return false;
			else if (! less && (*this)[ i ] < ind[ i ])
				less = true;

		return less/*                             &&
			   	       fitness       <= ind.fitness       &&
			   	       scaledFitness <= ind.scaledFitness &&
			   	       evalFlg       <= ind.evalFlg       &&
			   	       feasible      <= ind.feasible      &&
			   	       selProb       <= ind.selProb       &&
			   	       numCopies     <= ind.numCopies     &&
			   	       elitist       <= ind.elitist       &&
			   	       age           <= ind.age*/;
	}

	return size() < ind.size();
}

//===========================================================================
// added by Marc Toussaint and Stefan Wiegand (INI) 20.11.2002

int Individual::pvm_pkind()
{
	//cout << "\t Individual_pk" << endl;

	unsigned i;

	unsigned *s = new unsigned;
	*s = this->size();
	pvm_pkuint(s, 1, 1);
	delete s;

	for (i = 0; i < this->size(); i++)
		((*this)[i]).pvm_pkchrom();

	unsigned int u[6];
	u[0] = feasible;
	u[1] = elitist;
	u[2] = evalFlg;
	u[3] = numCopies;
	u[4] = age;
	u[5] = learnTime;
	pvm_pkuint(u, 6, 1);

	/*double *f = new double[displVal.size()+3];
	unsigned jj = 0;
	for (;jj < displVal.size();jj++)
		f[jj]   = getDisplVal(jj);
	f[jj]   = fitness;
	f[++jj] = scaledFitness;
	f[++jj] = selProb;
	pvm_pkdouble(f, displVal.size() + 3, 1);
	delete[] f;*/
	
	double f[3];
	f[0] = fitness;
	f[1] = scaledFitness;
	f[2] = selProb;
	pvm_pkdouble(f, 3, 1);

	return 1;
}

int Individual::pvm_upkind()
{
	//cout << "\t Individual_upk" << endl;

	unsigned i;

	unsigned *s = new unsigned;
	pvm_upkuint(s, 1, 1);
	if (this->size() != *s) {

	  throw SHARKEXCEPTION("EALib/Individual.cpp: the individual which has called pvm_upkind() is of unexpected size!\nPlease initialize this individual prototypically with chromosomes of appropriate type.");
	}
	delete s;

	for (i = 0; i < this->size(); i++)
		((*this)[i]).pvm_upkchrom();

	unsigned u[6];
	pvm_upkuint(u, 7, 1);
	feasible  = (u[0] != 0);
	elitist   = (u[1] != 0);
	evalFlg   = (u[2] != 0);
	numCopies = u[3];
	age       = u[4];
	learnTime = u[5];

	/*double *f = new double[noDisplVal+3];
	pvm_upkdouble(f, noDisplVal + 3, 1);
	unsigned jj = 0;
	for (;jj < noDisplVal ;jj++)
		setDisplVal(jj, f[jj]);
	fitness       = f[jj];
	scaledFitness = f[++jj];
	selProb       = f[++jj];
	delete[] f;*/
	double f[3];
	pvm_upkdouble( f, 3, 1);
	fitness       = f[0];
	scaledFitness = f[1];
	selProb       = f[2];

	return 1;
}

//===========================================================================

