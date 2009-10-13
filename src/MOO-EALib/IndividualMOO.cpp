// ======================================================================
/*!
 *  \file IndividualMOO.cpp
 *  \date 2002-01-23
 *
 *  \brief Multi-objective version of an individual consisting of chromosomes
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


#include <MOO-EALib/IndividualMOO.h>
#include <EALib/Individual.h>
#include <typeinfo>


using namespace std;


// ----------------------------------------------------------------------------
// TO-IM-001
IndividualMOO::IndividualMOO()
		: Individual()
{
	MOOFitness.resize( 1, 0 );
	UnpenalizedMOOFitness.resize( 1, 0 );
	setMOORank(0);
	setMOOShare(0.0);
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-IM-002
IndividualMOO::IndividualMOO(unsigned n)
		: Individual(n)
{
	MOOFitness.resize( 1, 0 );
	UnpenalizedMOOFitness.resize( 1, 0 );
	setMOORank(0);
	setMOOShare(0.0);
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-IM-003
IndividualMOO::IndividualMOO(unsigned n, const Chromosome& chrom1)
		: Individual(n, chrom1)
{
	MOOFitness.resize( 1, 0 );
	UnpenalizedMOOFitness.resize( 1, 0 );
	setMOORank(0);
	setMOOShare(0.0);
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-IM-004
IndividualMOO::IndividualMOO(const Chromosome& chrom1)
		: Individual(chrom1)
{
	MOOFitness.resize( 1, 0 );
	UnpenalizedMOOFitness.resize( 1, 0 );
	setMOORank(0);
	setMOOShare(0.0);
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-IM-005
IndividualMOO::IndividualMOO(const Chromosome& chrom1,
							 const Chromosome& chrom2)
		: Individual(chrom1, chrom2)
{
	MOOFitness.resize( 1, 0 );
	UnpenalizedMOOFitness.resize( 1, 0 );
	setMOORank(0);
	setMOOShare(0.0);
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-IM-006
IndividualMOO::IndividualMOO(const Chromosome& chrom1,
							 const Chromosome& chrom2,
							 const Chromosome& chrom3)
		: Individual(chrom1, chrom2, chrom3)
{
	MOOFitness.resize( 1, 0 );
	UnpenalizedMOOFitness.resize( 1, 0 );
	setMOORank(0);
	setMOOShare(0.0);
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-IM-007
IndividualMOO::IndividualMOO(const Chromosome& chrom1,
							 const Chromosome& chrom2,
							 const Chromosome& chrom3,
							 const Chromosome& chrom4)
		: Individual(chrom1, chrom2, chrom3, chrom4)
{
	MOOFitness.resize( 1, 0 );
	UnpenalizedMOOFitness.resize( 1, 0 );
	setMOORank(0);
	setMOOShare(0.0);
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-IM-008
IndividualMOO::IndividualMOO(const Chromosome& chrom1,
							 const Chromosome& chrom2,
							 const Chromosome& chrom3,
							 const Chromosome& chrom4,
							 const Chromosome& chrom5)
		: Individual(chrom1, chrom2, chrom3, chrom4, chrom5)
{
	MOOFitness.resize( 1, 0 );
	UnpenalizedMOOFitness.resize( 1, 0 );
	setMOORank(0);
	setMOOShare(0.0);
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-IM-009
IndividualMOO::IndividualMOO(const Chromosome& chrom1,
							 const Chromosome& chrom2,
							 const Chromosome& chrom3,
							 const Chromosome& chrom4,
							 const Chromosome& chrom5,
							 const Chromosome& chrom6)
		: Individual(chrom1, chrom2, chrom3, chrom4, chrom5, chrom6)
{
	MOOFitness.resize( 1, 0 );
	UnpenalizedMOOFitness.resize( 1, 0 );
	setMOORank(0);
	setMOOShare(0.0);
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-IM-010
IndividualMOO::IndividualMOO(const Chromosome& chrom1,
							 const Chromosome& chrom2,
							 const Chromosome& chrom3,
							 const Chromosome& chrom4,
							 const Chromosome& chrom5,
							 const Chromosome& chrom6,
							 const Chromosome& chrom7)
		: Individual(chrom1, chrom2, chrom3, chrom4, chrom5, chrom6, chrom7)
{
	MOOFitness.resize( 1, 0 );
	UnpenalizedMOOFitness.resize( 1, 0 );
	setMOORank(0);
	setMOOShare(0.0);
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-IM-011
IndividualMOO::IndividualMOO(const Chromosome& chrom1,
							 const Chromosome& chrom2,
							 const Chromosome& chrom3,
							 const Chromosome& chrom4,
							 const Chromosome& chrom5,
							 const Chromosome& chrom6,
							 const Chromosome& chrom7,
							 const Chromosome& chrom8)
		: Individual(chrom1, chrom2, chrom3, chrom4, chrom5, chrom6, chrom7, chrom8)
{
	MOOFitness.resize( 1, 0 );
	UnpenalizedMOOFitness.resize( 1, 0 );
	setMOORank(0);
	setMOOShare(0.0);
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-IM-012
IndividualMOO::IndividualMOO(const vector< Chromosome* >& chrom1)
		: Individual(chrom1)
{
	MOOFitness.resize( 1, 0 );
	UnpenalizedMOOFitness.resize( 1, 0 );
	setMOORank(0);
	setMOOShare(0.0);
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-IM-013
IndividualMOO::IndividualMOO(const Individual& indiv1)
		: Individual(indiv1)
{
	MOOFitness.resize( 1, 0 );
	UnpenalizedMOOFitness.resize( 1, 0 );
	setMOORank(0);
	setMOOShare(0.0);
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-IM-014
IndividualMOO::IndividualMOO(const IndividualMOO& indmoo)
		: Individual(dynamic_cast< const Individual& >(indmoo))
{
	unsigned i;
	// copy data
	MOOFitness.resize(indmoo.MOOFitness.size());
	for (i = indmoo.MOOFitness.size(); i--;)
	{
		setMOOFitness(i, indmoo.MOOFitness[i]);
	}
	UnpenalizedMOOFitness.resize(indmoo.UnpenalizedMOOFitness.size());
	for (i = indmoo.UnpenalizedMOOFitness.size(); i--;)
	{
		setUnpenalizedMOOFitness(i, indmoo.UnpenalizedMOOFitness[i]);
	}
	setMOORank(indmoo.MOORank);
	setMOOShare(indmoo.MOOShare);
	fitness       = indmoo.fitness;
	scaledFitness = indmoo.scaledFitness;
	evalFlg       = indmoo.evalFlg;
	feasible      = indmoo.feasible;
	selProb       = indmoo.selProb;
	numCopies     = indmoo.numCopies;
	elitist       = indmoo.elitist;
	age           = indmoo.age;
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-IM-015
IndividualMOO::~IndividualMOO()
{ }

// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-IM-021
unsigned IndividualMOO::totalSize() const
{
	return Individual::totalSize();
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-IM-047
void IndividualMOO::setEvalFlg(bool flg)
{
	if (flg)
	{
		Individual::setEvaluationFlag();
	}
	else
	{
		Individual::clearEvaluationFlag();
	}
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// SW-IM-050
void IndividualMOO::setLearnTime(unsigned lt)
{
	Individual::setLearnTime(lt);
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// SW-IM-051
unsigned IndividualMOO::getLearnTime() const
{
	return Individual::getLearnTime();
}
// ----------------------------

// ----------------------------------------------------------------------------
// TO-IM-060
void IndividualMOO::setNoOfObj(unsigned n)
{
	MOOFitness.resize(n);
	UnpenalizedMOOFitness.resize(n);
	initializeMOOFitness(0.0);
	initializeUnpenalizedMOOFitness(0.0);
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-IM-061
unsigned IndividualMOO::getNoOfObj() const
{
	return MOOFitness.size();
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-IM-062
void IndividualMOO::setMOORank(unsigned n)
{
	MOORank = n;
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-IM-063
unsigned IndividualMOO::getMOORank() const
{
	return MOORank;
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-IM-064
void IndividualMOO::setMOOShare(double n)
{
	MOOShare = n;
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-IM-065
double IndividualMOO::getMOOShare() const
{
	return MOOShare;
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Tommy
const std::vector<double> & IndividualMOO::getMOOFitnessValues( bool unpenalized ) const {
	return (unpenalized ? getUnpenalizedMOOFitnessValues() : getMOOFitnessValues());
}

std::vector<double> & IndividualMOO::getMOOFitnessValues(bool unpenalized) {
	return (unpenalized ? getUnpenalizedMOOFitnessValues() : getMOOFitnessValues());
}

// ----------------------------------------------------------------------------
// TO-IM-066
void IndividualMOO::setMOOFitness(unsigned nof, double fit)
{
	RANGE_CHECK(nof < MOOFitness.size())
	MOOFitness[nof] = fit;
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-IM-067
double IndividualMOO::getMOOFitness(unsigned nof)
{
	RANGE_CHECK(nof < MOOFitness.size())
	return MOOFitness[nof];
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-IM-068
void IndividualMOO::setMOOFitnessValues(double f0)
{
	RANGE_CHECK(0 < MOOFitness.size())
	MOOFitness[0] = f0;
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-IM-069
void IndividualMOO::setMOOFitnessValues(double f0, double f1)
{
	RANGE_CHECK(1 < MOOFitness.size())
	MOOFitness[0] = f0;
	MOOFitness[1] = f1;
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-IM-070
void IndividualMOO::setMOOFitnessValues(double f0, double f1,
										double f2)
{
	RANGE_CHECK(2 < MOOFitness.size())
	MOOFitness[0] = f0;
	MOOFitness[1] = f1;
	MOOFitness[2] = f2;
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-IM-071
void IndividualMOO::setMOOFitnessValues(double f0, double f1,
										double f2, double f3)
{
	RANGE_CHECK(3 < MOOFitness.size())
	MOOFitness[0] = f0;
	MOOFitness[1] = f1;
	MOOFitness[2] = f2;
	MOOFitness[3] = f3;
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-IM-072
void IndividualMOO::setMOOFitnessValues(double f0, double f1,
										double f2, double f3,
										double f4)
{
	RANGE_CHECK(4 < MOOFitness.size())
	MOOFitness[0] = f0;
	MOOFitness[1] = f1;
	MOOFitness[2] = f2;
	MOOFitness[3] = f3;
	MOOFitness[4] = f4;
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-IM-073
void IndividualMOO::setMOOFitnessValues(double f0, double f1,
										double f2, double f3,
										double f4, double f5)
{
	RANGE_CHECK(5 < MOOFitness.size())
	MOOFitness[0] = f0;
	MOOFitness[1] = f1;
	MOOFitness[2] = f2;
	MOOFitness[3] = f3;
	MOOFitness[4] = f4;
	MOOFitness[5] = f5;
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-IM-074
void IndividualMOO::setMOOFitnessValues(double f0, double f1,
										double f2, double f3,
										double f4, double f5,
										double f6)
{
	RANGE_CHECK(6 < MOOFitness.size())
	MOOFitness[0] = f0;
	MOOFitness[1] = f1;
	MOOFitness[2] = f2;
	MOOFitness[3] = f3;
	MOOFitness[4] = f4;
	MOOFitness[5] = f5;
	MOOFitness[6] = f6;
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-IM-075
void IndividualMOO::setMOOFitnessValues(double f0, double f1,
										double f2, double f3,
										double f4, double f5,
										double f6, double f7)
{
	RANGE_CHECK(7 < MOOFitness.size())
	MOOFitness[0] = f0;
	MOOFitness[1] = f1;
	MOOFitness[2] = f2;
	MOOFitness[3] = f3;
	MOOFitness[4] = f4;
	MOOFitness[5] = f5;
	MOOFitness[6] = f6;
	MOOFitness[7] = f7;
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-IM-076
void IndividualMOO::setMOOFitnessValues(vector< double >& fit)
{
	RANGE_CHECK(fit.size() == MOOFitness.size())
	for (unsigned i = fit.size(); i--;)
	{
		MOOFitness[i] = fit[i];
	}
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-IM-077
const vector<double> & IndividualMOO::getMOOFitnessValues() const {
	return MOOFitness;
}

vector< double >& IndividualMOO::getMOOFitnessValues() {
	return MOOFitness;
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-IM-078
void IndividualMOO::initializeMOOFitness(double x)
{
	for (unsigned i = MOOFitness.size(); i--;)
	{
		MOOFitness[i] = x;
	}
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// SR-IM-079
void IndividualMOO::setUnpenalizedMOOFitness(unsigned nof, double fit)
{
	RANGE_CHECK(nof < UnpenalizedMOOFitness.size())
	UnpenalizedMOOFitness[nof] = fit;
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// SR-IM-080
double IndividualMOO::getUnpenalizedMOOFitness(unsigned nof)
{
	RANGE_CHECK(nof < UnpenalizedMOOFitness.size())
	return UnpenalizedMOOFitness[nof];
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// SR-IM-081
void IndividualMOO::setUnpenalizedMOOFitnessValues(double f0)
{
	RANGE_CHECK(0 < UnpenalizedMOOFitness.size())
	UnpenalizedMOOFitness[0] = f0;
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// SR-IM-082
void IndividualMOO::setUnpenalizedMOOFitnessValues(double f0, double f1)
{
	RANGE_CHECK(1 < UnpenalizedMOOFitness.size())
	UnpenalizedMOOFitness[0] = f0;
	UnpenalizedMOOFitness[1] = f1;
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// SR-IM-083
void IndividualMOO::setUnpenalizedMOOFitnessValues(double f0, double f1,
		double f2)
{
	RANGE_CHECK(2 < UnpenalizedMOOFitness.size())
	UnpenalizedMOOFitness[0] = f0;
	UnpenalizedMOOFitness[1] = f1;
	UnpenalizedMOOFitness[2] = f2;
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// SR-IM-084
void IndividualMOO::setUnpenalizedMOOFitnessValues(double f0, double f1,
		double f2, double f3)
{
	RANGE_CHECK(3 < UnpenalizedMOOFitness.size())
	UnpenalizedMOOFitness[0] = f0;
	UnpenalizedMOOFitness[1] = f1;
	UnpenalizedMOOFitness[2] = f2;
	UnpenalizedMOOFitness[3] = f3;
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// SR-IM-085
void IndividualMOO::setUnpenalizedMOOFitnessValues(double f0, double f1,
		double f2, double f3,
		double f4)
{
	RANGE_CHECK(4 < UnpenalizedMOOFitness.size())
	UnpenalizedMOOFitness[0] = f0;
	UnpenalizedMOOFitness[1] = f1;
	UnpenalizedMOOFitness[2] = f2;
	UnpenalizedMOOFitness[3] = f3;
	UnpenalizedMOOFitness[4] = f4;
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// SR-IM-086
void IndividualMOO::setUnpenalizedMOOFitnessValues(double f0, double f1,
		double f2, double f3,
		double f4, double f5)
{
	RANGE_CHECK(5 < UnpenalizedMOOFitness.size())
	UnpenalizedMOOFitness[0] = f0;
	UnpenalizedMOOFitness[1] = f1;
	UnpenalizedMOOFitness[2] = f2;
	UnpenalizedMOOFitness[3] = f3;
	UnpenalizedMOOFitness[4] = f4;
	UnpenalizedMOOFitness[5] = f5;
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// SR-IM-087
void IndividualMOO::setUnpenalizedMOOFitnessValues(double f0, double f1,
		double f2, double f3,
		double f4, double f5,
		double f6)
{
	RANGE_CHECK(6 < UnpenalizedMOOFitness.size())
	UnpenalizedMOOFitness[0] = f0;
	UnpenalizedMOOFitness[1] = f1;
	UnpenalizedMOOFitness[2] = f2;
	UnpenalizedMOOFitness[3] = f3;
	UnpenalizedMOOFitness[4] = f4;
	UnpenalizedMOOFitness[5] = f5;
	UnpenalizedMOOFitness[6] = f6;
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// SR-IM-088
void IndividualMOO::setUnpenalizedMOOFitnessValues(double f0, double f1,
		double f2, double f3,
		double f4, double f5,
		double f6, double f7)
{
	RANGE_CHECK(7 < UnpenalizedMOOFitness.size())
	UnpenalizedMOOFitness[0] = f0;
	UnpenalizedMOOFitness[1] = f1;
	UnpenalizedMOOFitness[2] = f2;
	UnpenalizedMOOFitness[3] = f3;
	UnpenalizedMOOFitness[4] = f4;
	UnpenalizedMOOFitness[5] = f5;
	UnpenalizedMOOFitness[6] = f6;
	UnpenalizedMOOFitness[7] = f7;
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// SR-IM-089
void IndividualMOO::setUnpenalizedMOOFitnessValues(vector< double >& fit)
{
	RANGE_CHECK(fit.size() < UnpenalizedMOOFitness.size())
	for (unsigned i = fit.size(); i--;)
	{
		UnpenalizedMOOFitness[i] = fit[i];
	}
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// SR-IM-090
const vector<double> & IndividualMOO::getUnpenalizedMOOFitnessValues() const {
	return UnpenalizedMOOFitness;
}

vector<double> & IndividualMOO::getUnpenalizedMOOFitnessValues() {
	return UnpenalizedMOOFitness;
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// SR-IM-091
void IndividualMOO::initializeUnpenalizedMOOFitness(double x)
{
	for (unsigned i = UnpenalizedMOOFitness.size(); i--;)
	{
		UnpenalizedMOOFitness[i] = x;
	}
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-IM-100
Chromosome& IndividualMOO::operator [ ](unsigned i)
{
	return Individual::operator [ ](i);
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-IM-101
const Chromosome& IndividualMOO::operator [ ](unsigned i) const
{
	return Individual::operator [ ](i);
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-IM-102
IndividualMOO& IndividualMOO::operator = (const IndividualMOO& indmoo)
{
	//
	// replaced by Stefan Wiegand (INI) 14.1.2004
	//
	unsigned i;

	for (i = size(); i--;)
		delete *(begin() + i);

	vector< Chromosome * >::operator = (indmoo);

	for (i = size(); i--;)
		*(begin() + i) = indmoo[ i ].clone();

#ifdef EALIB_REGISTER_INDIVIDUAL	
	for (i = size(); i--;)
		(*(begin() + i))->registerIndividual(*this, i);
#endif	

	if (MOOFitness.size() != indmoo.MOOFitness.size())
		setNoOfObj(indmoo.MOOFitness.size());

	fitness       = indmoo.fitness;
	scaledFitness = indmoo.scaledFitness;
	evalFlg       = indmoo.evalFlg;
	feasible      = indmoo.feasible;
	selProb       = indmoo.selProb;
	numCopies     = indmoo.numCopies;
	elitist       = indmoo.elitist;
	age           = indmoo.age;
	learnTime     = indmoo.getLearnTime();
	MOORank       = indmoo.MOORank;
	MOOShare      = indmoo.MOOShare;

	for (i = getNoOfObj(); i--;)
	{
		MOOFitness[i] = indmoo.MOOFitness[i];
	}

	for (i = getNoOfObj(); i--;)
	{
		UnpenalizedMOOFitness[i] = indmoo.UnpenalizedMOOFitness[i];
	}
	
	return *this;
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-IM-103
bool IndividualMOO::operator == (const IndividualMOO& indmoo) const
{
	unsigned i;
	if (size() == indmoo.size())
	{
		// ***** check Chromosome
		for (i = size(); i--;)
		{
			if (!((*this)[i] == indmoo[i]))
			{
				return false;
			}
		}
		// ***** check MOO fitness
		if (MOOFitness.size() != indmoo.MOOFitness.size() || UnpenalizedMOOFitness.size() != indmoo.UnpenalizedMOOFitness.size())
		{
			return false;
		}

		for (i = MOOFitness.size(); i--;)
		{
			if (!(MOOFitness[i] == indmoo.MOOFitness[i]))
			{
				return false;
			}
		}

		for (i = UnpenalizedMOOFitness.size(); i--;)
		{
			if (!(UnpenalizedMOOFitness[i] == indmoo.UnpenalizedMOOFitness[i]))
			{
				return false;
			}
		}

		// ***** check internal variables
		if (fitness       == indmoo.fitness       &&
				scaledFitness == indmoo.scaledFitness &&
				evalFlg       == indmoo.evalFlg       &&
				feasible      == indmoo.feasible      &&
				selProb       == indmoo.selProb       &&
				numCopies     == indmoo.numCopies     &&
				elitist       == indmoo.elitist       &&
				age           == indmoo.age           &&
				MOORank       == indmoo.MOORank       &&
				MOOShare      == indmoo.MOOShare)
		{
			return true;
		}
	}
	return false;
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-IM-104
IndividualMOO& IndividualMOO::operator = (const Individual& ind)
{
	//
	// replaced by Stefan Wiegand (INI) 14.1.2004
	//
	unsigned i;

	for (i = size(); i--;)
		delete *(begin() + i);

	vector< Chromosome * >::operator = (ind);

	for (i = size(); i--;)
		*(begin() + i) = ind[ i ].clone();

#ifdef EALIB_REGISTER_INDIVIDUAL	
	for (i = size(); i--;)
		(*(begin() + i))->registerIndividual(*this, i);
#endif

	setNoOfObj(1);
	initializeMOOFitness(0);
	initializeUnpenalizedMOOFitness(0);


	fitness       = ind.fitnessValue();
	scaledFitness = ind.getScaledFitness();
	evalFlg       = ind.needEvaluation();
	feasible      = ind.isFeasible();
	selProb       = ind.selectionProbability();
	numCopies     = ind.numberOfCopies();
	elitist       = ind.isElitist();
	age           = ind.getAge();
	learnTime     = ind.getLearnTime();
	MOORank       = 0;
	MOOShare      = 0;

	return *this;
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-IM-110
void IndividualMOO::replace(unsigned i, const Chromosome& chrom)
{
	Individual::replace(i, chrom);
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-IM-111
void IndividualMOO::insert(unsigned i, const Chromosome& chrom)
{
	Individual::insert(i, chrom);
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-IM-112
void IndividualMOO::append(const Chromosome& chrom)
{
	//Individual::append( chrom );
	vector< Chromosome * >::push_back(chrom.clone());
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-IM-113
void IndividualMOO::remove (unsigned i)
{
	Individual::remove (i);
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-IM-114
void IndividualMOO::remove (unsigned i, unsigned j)
{
	Individual::remove (i, j);
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-IM-200
double IndividualMOO::aggregation(const vector< double >& weight)
{
	if (MOOFitness.size() > weight.size())
	{
		cout << "\n ***** The size is not match in TO-IM-200 *****\n" << endl;
		return 0.0;
	}
	double sum = 0.0;
	for (unsigned i = MOOFitness.size(); i--;)
	{
		sum += getMOOFitness(i) * weight[ i ];
	}
	setFitness(sum);
	return sum;
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-IM-201
double IndividualMOO::simplesum()
{
	double sum = 0.0;
	for (unsigned i = MOOFitness.size(); i--;)
	{
		sum += getMOOFitness(i);
	}
	setFitness(sum);
	return sum;
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TO-IM-500
void IndividualMOO::printIM()
{
	unsigned i;
	cout << "\n\n********** IndividualMOO **********\n";
	cout << "No. of Chromosomes         : " << size() << "\n";
	cout << "Variable ( fitness )       : " << fitness << "\n";
	cout << "Variable ( scaledFitness ) : " << scaledFitness << "\n";
	cout << "Variable ( evalFlg )       : " << evalFlg << "\n";
	cout << "Variable ( feasible )      : " << feasible << "\n";
	cout << "Variable ( selProb )       : " << selProb << "\n";
	cout << "Variable ( numCopies )     : " << numCopies << "\n";
	cout << "Variable ( elitist )       : " << elitist << "\n";
	cout << "Variable ( age )           : " << age << "\n";
	cout << "Variable ( MOOShare )      : " << MOOShare << "\n";
	cout << "Variable ( MOORank )       : " << MOORank << "\n";
	cout << "No. of fitness functions   : " << MOOFitness.size()
	<< "\n";
	for (i = 0; i < MOOFitness.size(); i++)
	{
		cout << "Value of fitness function  : " << MOOFitness[i]
		<< " ( fun. = " << i << " )\n";
	}

	for (i = 0; i < UnpenalizedMOOFitness.size(); i++)
	{
		cout << "Value of penalized fitness function  : " << UnpenalizedMOOFitness[i]
		<< " ( fun. = " << i << " )\n";
	}

	if (size() != 0)
	{
		for (unsigned i = 0; i < size(); i++)
		{
			cout << "No. Of Alleles             : " << (*this)[i].size()
			<< " ( Chr. = " << i << " )\n";
		}
	}
	cout << endl;
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// SW-IM-600
int IndividualMOO::pvm_pkind()
{
	//cout << "\t IndividualMOO_pk" << endl;

	unsigned i;

	unsigned *s = new unsigned;
	*s = this->size();
	pvm_pkuint(s, 1, 1);
	delete s;

	for (i = 0; i < this->size(); i++)
		((*this)[i]).pvm_pkchrom();

	uint u[8];
	u[0] = feasible;
	u[1] = elitist;
	u[2] = evalFlg;
	u[3] = numCopies;
	u[4] = age;
	u[5] = learnTime;
	u[6] = MOORank;
	u[7] = getNoOfObj();
	pvm_pkuint(u, 8, 1);

	double *g = new double[2*getNoOfObj()];
	unsigned jj = 0;
	for (;jj < getNoOfObj();jj++)
		g[jj] = MOOFitness[jj];
	for (;jj < 2*getNoOfObj();jj++)
		g[jj] = UnpenalizedMOOFitness[jj];
	pvm_pkdouble(g, 2*getNoOfObj(), 1);
	delete[] g;

	double f[4];
	f[0] = fitness;
	f[1] = scaledFitness;
	f[2] = selProb;
	f[3] = MOOShare;
	pvm_pkdouble(f, 4, 1);

	return 1;
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// SW-IM-601
int IndividualMOO::pvm_upkind()
{
	//cout << "\t Individual_upk" << endl;

	unsigned i;

	unsigned *s = new unsigned;
	pvm_upkuint(s, 1, 1);
	if (this->size() != *s) throw SHARKEXCEPTION("[IndividualMOO::pvm_upkind] unexpected size");
	delete s;

	for (i = 0; i < this->size(); i++)
		((*this)[i]).pvm_upkchrom();

	uint *o = new uint;
	uint u[8];
	pvm_upkuint(u, 8, 1);
	feasible  = u[0] ? 1 : 0;
	elitist   = u[1] ? 1 : 0;
	evalFlg   = u[2] ? 1 : 0;
	numCopies = u[3];
	age       = u[4];
	learnTime = u[5];
	MOORank   = u[6];
	*o         = u[7];

	setNoOfObj(*o);

	double *g = new double[2 * getNoOfObj()];
	pvm_upkdouble(g, 2 * getNoOfObj(), 1);
	unsigned jj = 0;
	for (;jj < getNoOfObj();jj++)
		MOOFitness[jj] = g[jj];
	for (;jj < 2 * getNoOfObj();jj++)
		UnpenalizedMOOFitness[jj] = g[jj];
	delete[] g;
	delete[] o;

	double f[4];
	pvm_upkdouble(f, 4, 1);
	fitness       = f[0];
	scaledFitness = f[1];
	selProb       = f[2];
	MOOShare      = f[3];

	return 1;
}
// ----------------------------------------------------------------------------






