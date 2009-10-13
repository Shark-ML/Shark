/*! ======================================================================
 *
 *  \file IndividualMOO.h
 *
 *  \brief Multi-objective version of an individual consisting of chromosomes
 * 
 *  \author Tatsuya Okabe &lt;tatsuya.okabe@honda-ri.de&gt;
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
 *
 *  <BR><HR>
 *  This file is part of Shark. This library is free software;
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



#ifndef _INDIVIDUALMOO_H_
#define _INDIVIDUALMOO_H_


#include <EALib/Individual.h>
#include <EALib/ChromosomeFactory.h>


//!
//! \brief Multi objective version of an individual consisting of chromosomes
//!
class IndividualMOO : public Individual
{

public:

	// ---------------------------------------------------
	// constructor
	// ---------------------------------------------------
	// TO-IM-001
	IndividualMOO();
	// TO-IM-002
	IndividualMOO(unsigned);
	// TO-IM-003
	IndividualMOO(unsigned, const Chromosome&);
	// TO-IM-004
	IndividualMOO(const Chromosome&);
	// TO-IM-005
	IndividualMOO(const Chromosome&,
				  const Chromosome&);
	// TO-IM-006
	IndividualMOO(const Chromosome&,
				  const Chromosome&,
				  const Chromosome&);
	// TO-IM-007
	IndividualMOO(const Chromosome&,
				  const Chromosome&,
				  const Chromosome&,
				  const Chromosome&);
	// TO-IM-008
	IndividualMOO(const Chromosome&,
				  const Chromosome&,
				  const Chromosome&,
				  const Chromosome&,
				  const Chromosome&);
	// TO-IM-009
	IndividualMOO(const Chromosome&,
				  const Chromosome&,
				  const Chromosome&,
				  const Chromosome&,
				  const Chromosome&,
				  const Chromosome&);
	// TO-IM-010
	IndividualMOO(const Chromosome&,
				  const Chromosome&,
				  const Chromosome&,
				  const Chromosome&,
				  const Chromosome&,
				  const Chromosome&,
				  const Chromosome&);
	// TO-IM-011
	IndividualMOO(const Chromosome&,
				  const Chromosome&,
				  const Chromosome&,
				  const Chromosome&,
				  const Chromosome&,
				  const Chromosome&,
				  const Chromosome&,
				  const Chromosome&);
	// TO-IM-012
	IndividualMOO(const std::vector< Chromosome* >&);
	// TO-IM-013
	IndividualMOO(const Individual&);
	// TO-IM-014
	IndividualMOO(const IndividualMOO&);

	// ---------------------------------------------------
	// destructor
	// ---------------------------------------------------
	// TO-IM-015
	~IndividualMOO();

	// ---------------------------------------------------
	// structure
	// ---------------------------------------------------
	// TO-IM-021
	unsigned totalSize() const;

	// ---------------------------------------------------
	// internal variables
	// ---------------------------------------------------
	// TO-IM-047
	void setEvalFlg(bool);

	// parameter for embedded learning
	// SW-IM-050
	void setLearnTime(unsigned lt = 0);
	// SW-IM-051
	unsigned getLearnTime() const;

	// ---------------------------------------------------
	// internal variables ( IndividualMOO )
	// ---------------------------------------------------

	// Number of objectives
	// TO-IM-060
	void setNoOfObj(unsigned);
	// TO-IM-061
	unsigned getNoOfObj() const;

	// MOORank
	// TO-IM-062
	void setMOORank(unsigned);
	// TO-IM-063
	unsigned getMOORank() const;

	// MOOShare
	// TO-IM-064
	void setMOOShare(double);
	// TO-IM-065
	double getMOOShare() const;

	// MOOFitness
	const std::vector<double> & getMOOFitnessValues(bool unpenalized) const;
	std::vector<double> & getMOOFitnessValues(bool unpenalized);

	// TO-IM-066
	void setMOOFitness(unsigned, double);
	// TO-IM-067
	double getMOOFitness(unsigned);
	// TO-IM-068
	void setMOOFitnessValues(double);
	// TO-IM-069
	void setMOOFitnessValues(double, double);
	// TO-IM-070
	void setMOOFitnessValues(double, double, double);
	// TO-IM-071
	void setMOOFitnessValues(double, double, double,
							 double);
	// TO-IM-072
	void setMOOFitnessValues(double, double, double,
							 double, double);
	// TO-IM-073
	void setMOOFitnessValues(double, double, double,
							 double, double, double);
	// TO-IM-074
	void setMOOFitnessValues(double, double, double,
							 double, double, double,
							 double);
	// TO-IM-075
	void setMOOFitnessValues(double, double, double,
							 double, double, double,
							 double, double);
	// TO-IM-076
	void setMOOFitnessValues(std::vector< double >&);
	// TO-IM-077
	const std::vector< double >& getMOOFitnessValues() const;
	std::vector<double> & getMOOFitnessValues();
	// TO-IM-078
	void initializeMOOFitness(double = 0.0);


	// Unpenalized MOOFitness
	// SR-IM-079
	void setUnpenalizedMOOFitness(unsigned, double);
	// SR-IM-080
	double getUnpenalizedMOOFitness(unsigned);
	// SR-IM-081
	void setUnpenalizedMOOFitnessValues(double);
	// SR-IM-082
	void setUnpenalizedMOOFitnessValues(double, double);
	// SR-IM-083
	void setUnpenalizedMOOFitnessValues(double, double, double);
	// SR-IM-084
	void setUnpenalizedMOOFitnessValues(double, double, double,
										double);
	// SR-IM-085
	void setUnpenalizedMOOFitnessValues(double, double, double,
										double, double);
	// SR-IM-086
	void setUnpenalizedMOOFitnessValues(double, double, double,
										double, double, double);
	// SR-IM-087
	void setUnpenalizedMOOFitnessValues(double, double, double,
										double, double, double,
										double);
	// SR-IM-088
	void setUnpenalizedMOOFitnessValues(double, double, double,
										double, double, double,
										double, double);
	// SR-IM-089
	void setUnpenalizedMOOFitnessValues(std::vector< double >&);
	// SR-IM-090
	const std::vector< double >& getUnpenalizedMOOFitnessValues() const;
	std::vector< double >& getUnpenalizedMOOFitnessValues();
	// SR-IM-091
	void initializeUnpenalizedMOOFitness(double = 0.0);

	// ---------------------------------------------------
	// operator
	// ---------------------------------------------------
	// TO-IM-100
	Chromosome& operator [ ](unsigned);
	// TO-IM-101
	const Chromosome& operator [ ](unsigned) const;
	// TO-IM-102
	IndividualMOO& operator = (const IndividualMOO&);
	// TO-IM-103
	bool operator == (const IndividualMOO&) const;
	// TO-IM-104
	IndividualMOO& operator = (const Individual&);

	// ---------------------------------------------------
	// Change Chromosomes
	// ---------------------------------------------------
	// TO-IM-110
	void replace(unsigned, const Chromosome&);
	// TO-IM-111
	void insert(unsigned, const Chromosome&);
	// TO-IM-112
	void append(const Chromosome&);
	// TO-IM-113
	void remove (unsigned);
	// TO-IM-114
	void remove (unsigned, unsigned);

	// SW-IM-110
	// possibly conflicting with 'to-im-112': see vc-compiler message
	// template<class ChromosomeTemplate> void append(const ChromosomeTemplate& chrom);

	// ---------------------------------------------------
	// calculation for fitness
	// ---------------------------------------------------
	// TO-IM-200
	double aggregation(const std::vector< double >&);
	// TO-IM-201
	double simplesum();

	// ---------------------------------------------------
	// Print data
	// ---------------------------------------------------
	// TO-IM-500
	void printIM();

	// ---------------------------------------------------
	// PVM Interface
	// ---------------------------------------------------

	// SW-IM-600
	/*! Part of PVM-send routine for MOO-individuals */
	int pvm_pkind();

	// SW-IM-601
	/*! Part of PVM-receive routine for MOO-individuals */
	int pvm_upkind();


#ifndef __NO_GENERIC_IOSTREAM
	friend std::ostream& operator << (std::ostream& os, const IndividualMOO& ind)
	{
//		printf("IndividualMOO::operator << \n"); fflush(stdout);

		os << "IndividualMOO(" << ind.size() << ")\n"
		<< ind.fitness       << '\n'
		<< ind.scaledFitness << '\n'
		<< ind.evalFlg       << '\n'
		<< ind.feasible      << '\n'
		<< ind.selProb       << '\n'
		<< ind.numCopies     << '\n'
		<< ind.elitist       << '\n'
		<< ind.age           << '\n'
		<< ind.learnTime     << '\n'
		<< ind.MOORank				<< '\n'
		<< ind.MOOShare;

		int r, rc;

		rc = ind.MOOFitness.size();
		os << '\n' << rc;
		for (r = 0; r < rc; r++)
			os << '\n' << ind.MOOFitness[r];

		rc = ind.UnpenalizedMOOFitness.size();
		os << '\n' << rc;
		for (r = 0; r < rc; r++)
			os << '\n' << ind.UnpenalizedMOOFitness[r];

		for (unsigned i = 0; i < ind.size(); ++i)
			os << '\n' << ind[ i ];
		os << std::endl;

		return os;
	}

	friend std::istream& operator >> (std::istream& is, IndividualMOO& ind)
	{
//		printf("IndividualMOO::operator >> \n"); fflush(stdout);

		unsigned i, indSize(0);
		std::string s, t;

		is >> s;
		is.get();    // skip end of line

		if (is.good() &&
				s.substr(0, 14) == "IndividualMOO(" &&
				s.find(')') != std::string::npos)
		{
			// Extract the size indication from the string:
			t = s.substr(s.find('(') + 1, s.find(')') - s.find('(') - 1);
			indSize = atoi(t.c_str());

			// Adapt size of Individual:
			ind.resize(indSize);

			is >> ind.fitness
			>> ind.scaledFitness
			>> ind.evalFlg
			>> ind.feasible
			>> ind.selProb
			>> ind.numCopies
			>> ind.elitist
			>> ind.age
			>> ind.learnTime
			>> ind.MOORank
			>> ind.MOOShare;

			int r, rc;

			is >> rc;
			ind.MOOFitness.resize(rc);
			for (r = 0; r < rc; r++)
				is >> ind.MOOFitness[r];

			is >> rc;
			ind.UnpenalizedMOOFitness.resize(rc);
			for (r = 0; r < rc; r++)
				is >> ind.UnpenalizedMOOFitness[r];

			for (i = 0; i < ind.size(); ++i)
			{
				int pos = is.tellg();
				is >> s;
				is.seekg(pos, std::ios_base::beg);
				t = s.substr(s.find('<') + 1, s.find('>') - s.find('<') - 1);
				const char* type = t.c_str();

				Chromosome* pC = CreateChromosome(type);
				ind.replace(i, *pC);
				is >> ind[ i ];
			}
		}
		return is;
	}
#endif // !__NO_GENERIC_IOSTREAM


protected:
	std::vector< double > MOOFitness;
	std::vector< double > UnpenalizedMOOFitness;
	unsigned         MOORank;
	double           MOOShare;
};




#endif /* !__INDIVIDUALMOO_H */

