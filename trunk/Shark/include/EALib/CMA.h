//===========================================================================
/*!
 *  \file CMA.h
 *
 *  \brief Implements the most recent version of the non-elitist CMA-ES
 *
 *  The algorithm is described in
 *
 *  Hansen, N., S. Kern (2004). Evaluating the CMA Evolution Strategy
 *  on Multimodal Test Functions. In Proceedings of the Eighth
 *  International Conference on Parallel Problem Solving from Nature
 *  (PPSN VIII), pp. 282-291, LNCS, Springer-Verlag
 *
 *  \par Copyright (c) 1998-2008:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
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
//===========================================================================


#ifndef _CMA_H_
#define _CMA_H_


#include <SharkDefs.h>
#include <EALib/Population.h>
#include <EALib/PopulationT.h>
#include <EALib/SearchAlgorithm.h>
#include <Array/ArrayOp.h>
#include <Array/ArrayIo.h>
#include <LinAlg/LinAlg.h>

#include<iostream>


//!
//! \brief Implements the most recent version of the non-elitist CMA-ES
//!
class CMA
{
public:
	CMA()
	{}

	virtual ~CMA()
	{}

	typedef enum { equal, linear, superlinear } RecombType;
	typedef enum { rankone, rankmu }            UpdateType;

	static unsigned suggestLambda(unsigned dimension);
	static unsigned suggestMu(unsigned lambda, RecombType recomb = superlinear);
	void init(unsigned dimension,
						std::vector<double > var, double _sigma,
						Population &p,
						RecombType recomb  = superlinear,
						UpdateType cupdate = rankmu);
	void init(unsigned dimension, double _sigma,  Population &p,
						RecombType recomb  = superlinear, UpdateType cupdate = rankmu);
	//
	// calculate weighted mean
	//
	void cog(ChromosomeT<double >& a, Population &p, unsigned c = 0) const;

	//
	// mutation after global intermediate recombination
	//
	void create(Individual &o);

	//
	// do the CMA
	//
	void updateStrategyParameters(Population &p, double lowerBound = .0);

	double getSigma() const;
	void   setSigma(double x);
	void   setChi_n(double x);
	double getCondition() const;
	const  Array<double> &getC() const;
	const  Array<double> &getLambda() const;


	friend std::ostream & operator<<( std::ostream & stream, const CMA& cma);
	friend std::istream & operator>>( std::istream & stream, CMA& cma);

protected:
	unsigned n;
	double sigma;
	double chi_n;
	double cc;
	double cs;
	double csu;
	double ccu;
	double ccov;
	double d;
	double mueff;
	double mucov;

	ChromosomeT<double> x;
	ChromosomeT<double> xPrime;
	ChromosomeT<double> meanz;

	Array<double> z;
	Array<double> pc;
	Array<double> ps;
	Array<double> C;
	Array<double> Z;
	Array<double> lambda;
	Array<double> B;
	Array<double> w;
	Array<double> theVector;
};

std::ostream & operator<<( std::ostream & stream, const CMA& cma);
std::istream & operator>>( std::istream & stream, CMA& cma);

//!
//! \brief Non-elitist CMA-ES implementing the interface EvolutionaryAlgorithm
//!
class CMASearch : public EvolutionaryAlgorithm<double*>
{
public:
	CMASearch();
	~CMASearch();

	inline CMA& getCMA() { return m_cma; }
	inline const CMA& getCMA() const { return m_cma; }
	inline const PopulationT<double>* parents() const { return m_parents; }
	inline const PopulationT<double>* offspring() const { return m_offspring; }

	void init(ObjectiveFunctionVS<double>& fitness, CMA::RecombType recomb = CMA::superlinear, CMA::UpdateType cupdate = CMA::rankmu);
	void init(ObjectiveFunctionVS<double>& fitness, const Array<double>& start, double stepsize, CMA::RecombType recomb = CMA::superlinear, CMA::UpdateType cupdate = CMA::rankmu);
	void init(ObjectiveFunctionVS<double>& fitness, unsigned int mu, unsigned int lambda, const Array<double>& start, double stepsize, CMA::RecombType recomb = CMA::superlinear, CMA::UpdateType cupdate = CMA::rankmu);
	void init(ObjectiveFunctionVS<double>& fitness, unsigned int mu, unsigned int lambda, const Array<double>& start, const Array<double>& stepsize, CMA::RecombType recomb = CMA::superlinear, CMA::UpdateType cupdate = CMA::rankmu);
	void run();
	void bestSolutions(std::vector<double*>& points);
	void bestSolutionsFitness(Array<double>& fitness);

	friend std::ostream & operator<<( std::ostream & stream, const CMASearch& search);
	friend std::istream & operator>>( std::istream & stream, CMASearch& search);

protected:
	CMA m_cma;
	ObjectiveFunctionVS<double>* m_fitness;
	PopulationT<double>* m_parents;
	PopulationT<double>* m_offspring;
};
std::ostream & operator<<( std::ostream & stream, const CMASearch& search);
std::istream & operator>>( std::istream & stream, CMASearch& search);
#endif
