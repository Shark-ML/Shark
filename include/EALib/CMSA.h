//===========================================================================
/*!
 *  \file CMSA.h
 *
 *  \brief Implements the non-elitist CMSA-ES
 *
 *  The algorithm is described in:
 *
 *  Covariance Matrix Adaptation Revisited - the CMSA Evolution
 *  Strategy - by Hans-Georg Beyer and Bernhard Senhoff, PPSN X, LNCS,
 *  Springer-Verlag, 2008
 *
 *  \par Copyright (c) 1998-2008: Institut
 *  f&uuml;r Neuroinformatik<BR> Ruhr-Universit&auml;t Bochum<BR>
 *  D-44780 Bochum, Germany<BR> Phone: +49-234-32-25558<BR> Fax:
 *  +49-234-32-14209<BR> eMail:
 *  shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR> www:
 *  http://www.neuroinformatik.ruhr-uni-bochum.de<BR> <BR>
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
//===========================================================================


#ifndef _CMSA_H_
#define _CMSA_H_


#include <SharkDefs.h>
#include <EALib/Population.h>
#include <EALib/PopulationT.h>
#include <EALib/SearchAlgorithm.h>
#include <Array/ArrayOp.h>
#include <Array/ArrayIo.h>
#include <LinAlg/LinAlg.h>


//!
//! \brief Implements the most recent version of the non-elitist CMSA-ES
//!
class CMSA
{
public:
	CMSA()
	{}

	virtual ~CMSA()
	{}

	typedef enum { equal, linear, superlinear } RecombType;

	static unsigned suggestLambda(unsigned dimension);
	static unsigned suggestMu(unsigned lambda, RecombType recomb = equal);
	void init(unsigned dimension,
						std::vector<double > var, double _sigma,
						Population &p,
						RecombType recomb  = equal);
	void init(unsigned dimension, double _sigma,  Population &p, 
						RecombType recomb  = equal);
	//
	// calculate weighted mean
	//
	void cog(ChromosomeT<double >& a, Population &p, unsigned c = 0) const;
	void cog(double& a, Population &p, unsigned c = 0) const;

	//
	// mutation after global intermediate recombination
	//
	void create(Individual &o);

	//
	// do the CMSA
	//
	void updateStrategyParameters(Population &p, double lowerBound = .0);

	double getSigma() const;
	void   setSigma(double x);
	double getCondition() const;
	const  Array<double> &getC() const;
	const  Array<double> &getLambda() const;

protected:
	unsigned n;
	double tauopt;
	double tauc;
	double sigma;

	ChromosomeT<double> x; // weighted center of mass of the population

	Array<double> z;      // standard normally distributed random vector 
	Array<double> C;      // covariance matrix
	Array<double> Z;      // rank-mu update matrix
	Array<double> lambda; // eigenvalues of C
	Array<double> B;      // eigenvectors of C
	Array<double> w;      // weights for weighted recombination
};

/*! \brief Non-elitist CMSA-ES implementing the interface EvolutionaryAlgorithm. */
class CMSASearch : public EvolutionaryAlgorithm<double*>
{
public:
	CMSASearch();
	~CMSASearch();

	inline CMSA& getCMSA() { return m_cma; }
	inline const CMSA& getCMSA() const { return m_cma; }
	inline const PopulationT<double>* parents() const { return m_parents; }
	inline const PopulationT<double>* offspring() const { return m_offspring; }

	void init(ObjectiveFunctionVS<double>& fitness, unsigned lambda = 0, CMSA::RecombType recomb = CMSA::equal);
	void init(ObjectiveFunctionVS<double>& fitness, const Array<double>& start, double stepsize, unsigned lambda = 0, CMSA::RecombType recomb = CMSA::equal);
	void run();
	void bestSolutions(std::vector<double*>& points);
	void bestSolutionsFitness(Array<double>& fitness);

protected:
	CMSA m_cma;
	ObjectiveFunctionVS<double>* m_fitness;
	PopulationT<double>* m_parents;
	PopulationT<double>* m_offspring;
};


#endif
