//===========================================================================
/*!
*  \file CMAOptimizer.h
*
*  \brief The CMA-ES as a ReClaM Optimizer
*
*  \author  T. Glasmachers
*  \date    2006
*
*  \par Copyright (c) 1999-2006:
*      Institut f&uuml;r Neuroinformatik<BR>
*      Ruhr-Universit&auml;t Bochum<BR>
*      D-44780 Bochum, Germany<BR>
*      Phone: +49-234-32-25558<BR>
*      Fax:   +49-234-32-14209<BR>
*      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
*      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
*
*  \par Project:
*      ReClaM
*
*
*  <BR>
*
*
*  <BR><HR>
*  This file is part of ReClaM. This library is free software;
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

#ifndef CMA_OPTIMIZER_H
#define CMA_OPTIMIZER_H


#include <ReClaM/Optimizer.h>
#include <EALib/CMA.h>
#include <EALib/ElitistCMA.h>
#include <EALib/UncertaintyQuantification.h>


//!
//! \brief The CMA-ES as a ReClaM Optimizer
//!
//! For a detailed description please refer to the
//! EALib module.
//!
class CMAOptimizer : public Optimizer
{
public:
	enum eMode
	{
		modeRankMuUpdate = 1,
		modeRankOneUpdate = 2,
		modeOnePlusOne = 4,
	};


	//! Constructor
	CMAOptimizer(int verbosity = 0);

	//! Destructor
	~CMAOptimizer();


	//! basic initialization with default parameters
	//! \param  model  model to optimize
	void init(Model& model);

	//! initialization with additional parameters
	//! \param  model   model to optimize
	//! \param  sigma   initial step size
	//! \param  mode    CMA mode, see EALib for details
	//! \param  best    if true, return the best individual ever evaluated
	//! \param  lambda  number of offspring per generation, a value of 0 indicates the default depending on the problem dimension - not used in 1+1-mode
	//! \param  mu      number of parents, a value of 0 indicates the default depending on lambda - not used in 1+1-mode
	void init(Model& model, double sigma, eMode mode = modeRankMuUpdate, bool best = true, int lambda = 0, int mu = 0);

	//! initialization with additional parameters
	//! \param  model  model to optimize
	//! \param  sigma  initial step size for each coordinate
	//! \param  mode   CMA mode, see EALib for details
	//! \param  best   if true, return the best individual ever evaluated
	//! \param  lambda  number of offspring per generation, a value of 0 indicates the default depending on the problem dimension - not used in 1+1-mode
	//! \param  mu      number of parents, a value of 0 indicates the default depending on lambda - not used in 1+1-mode
	void init(Model& model, const Array<double>& sigma, eMode mode = modeRankMuUpdate, bool best = true, int lambda = 0, int mu = 0);

	//! initialization with additional parameter
	//! for CMA with uncertainty handling
	//! \param  model     model to optimize
	//! \param  sigma     initial step size
	//! \param  maxEvals  maximum number of function evaluation for one fitness computation
	//! \param  alpha     speed of adaptation of number of function evaluations
	//! \param  theta     uncertainty threshold
	//! \param  mode      CMA mode, see EALib for details
	//! \param  lambda    number of offspring per generation, a value of 0 indicates the default depending on the problem dimension - not used in 1+1-mode
	//! \param  mu        number of parents, a value of 0 indicates the default depending on lambda - not used in 1+1-mode
	void initUncertainty(Model& model, double sigma = 0.01, unsigned int maxEvals = 1000, double alpha = 1.5, double theta = 0.1, eMode mode = modeRankMuUpdate, int lambda = 0, int mu = 0);

	//! initialization with additional parameter
	//! \param  model   model to optimize
	//! \param  sigma   initial step size
	//! \param  maxEvals  maximum number of function evaluation for one fitness computation
	//! \param  alpha     speed of adaptation of number of function evaluations
	//! \param  theta     uncertainty threshold
	//! \param  mode    CMA mode, see EALib for details
	//! \param  lambda  number of offspring per generation, a value of 0 indicates the default depending on the problem dimension - not used in 1+1-mode
	//! \param  mu      number of parents, a value of 0 indicates the default depending on lambda - not used in 1+1-mode
	void initUncertainty(Model& model, const Array<double>& sigma, unsigned int maxEvals = 1000, double alpha = 1.5, double theta = 0.1, eMode mode = modeRankMuUpdate, int lambda = 0, int mu = 0);

	//! create and select one CMA-ES generation
	double optimize(Model& model, ErrorFunction& errorfunction, const Array<double>& input, const Array<double>& target);

	//! return the offspring size,
	//! that is, the number of fitness
	//! evaluations per generation.
	int getLambda();

	inline double getSigma() const
	{
		if ((cmaMode == modeRankMuUpdate) || (cmaMode == modeRankOneUpdate)) return cma.getSigma();
		else if (cmaMode == modeOnePlusOne) return (*(PopulationCT<ChromosomeCMA>*)parents)[0][0].getSigma();
		else { throw SHARKEXCEPTION("[CMAOptimizer::getSigma] invalid mode"); return 0.0; }
	}

protected:
	//! Inner class that handles NoisyFitnessFunctions as a ReClaM-model.
	class ModelFitness : public NoisyFitnessFunction
	{
	public:
		void Set(Model& model, ErrorFunction& errorfunction, const Array<double>& input, const Array<double>& target);
		double fitness(const std::vector<double>& v);

	protected:
		Model* m;
		ErrorFunction* e;
		const Array<double>* i;
		const Array<double>* t;
	};
	ModelFitness objective;

	void Ind2Model(Individual& ind, Model& model);

	//! is this the first iteration?
	bool bFirstIteration;

	//! CMA object from EALib
	CMA cma;

	//! ElitistCMA object from EALib
	ElitistCMA ecma;

	//! parent population
	Population* parents;

	//! offspring population
	Population* offspring;

	//! CMA mode
	eMode cmaMode;

	//! if true, always return the best known individual
	bool returnBestIndividual;

	//! best fitness
	double bestFitness;

	//! parameters leading to the best fitness
	Array<double> bestParameters;

	//! verbosity level
	int verbosity;

	//! enable uncertainty handling?
	bool uncertaintyHandling;

	//! maximum number of function evaluations
	unsigned int maxEvals;

	//! speed of strategy adaptation for uncertainty handling
	double alpha;

	//! uncertainty threshold parameter
	double theta;
};


#endif
