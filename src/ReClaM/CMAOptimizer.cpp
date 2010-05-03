//===========================================================================
/*!
*  \file CMAOptimizer.cpp
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
*
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
*
*
*/
//===========================================================================


#include <ReClaM/CMAOptimizer.h>


CMAOptimizer::CMAOptimizer(int verbosity)
{
	parents = NULL;
	offspring = NULL;

	this->verbosity = verbosity;
}

CMAOptimizer::~CMAOptimizer()
{
	if (parents != NULL) delete parents;
	if (offspring != NULL) delete offspring;
}


void CMAOptimizer::init(Model& model)
{
	init(model, 0.01);
}

void CMAOptimizer::init(Model& model, double sigma, eMode mode, bool best, int lambda, int mu)
{
	Array<double> s(model.getParameterDimension());
	s = sigma;
	init(model, s, mode, best, lambda, mu);
}

void CMAOptimizer::init(Model& model, const Array<double>& sigma, eMode mode, bool best, int lambda, int mu)
{
	if (parents != NULL) delete parents;
	if (offspring != NULL) delete offspring;

	cmaMode = mode;
	returnBestIndividual = best;

	int i;
	int dim = model.getParameterDimension();

	if ((cmaMode == modeRankMuUpdate) || (cmaMode == modeRankOneUpdate))
	{
		if (lambda <= 0) lambda = cma.suggestLambda(dim);
		if (mu <= 0) mu = cma.suggestMu(lambda);
		if (verbosity > 0) printf("[CMA] lambda=%d mu=%d\n", lambda, mu);

		ChromosomeT<double> chrom_w(dim);
		ChromosomeT<double> chrom_0(dim);
		std::vector<double> stdv(dim);
		for (i = 0; i < dim; i++)
		{
			chrom_w[i] = model.getParameter(i);
			chrom_0[i] = 0.0;
			stdv[i] = sigma(i);
		}

		parents = new Population(mu, chrom_w, chrom_0);
		parents->setMinimize();
		offspring = new Population(lambda, chrom_w, chrom_0);
		offspring->setMinimize();

		if (cmaMode == modeRankMuUpdate)
			cma.init(dim, stdv, 1.0, *parents, CMA::superlinear, CMA::rankmu);
		else if (cmaMode == modeRankOneUpdate)
			cma.init(dim, stdv, 1.0, *parents, CMA::superlinear, CMA::rankone);
	}
	else if (cmaMode == modeOnePlusOne)
	{
		ChromosomeCMA chrom(dim);
		for (i = 0; i < dim; i++)
		{
			chrom[i] = model.getParameter(i);
		}

		parents = new PopulationCT<ChromosomeCMA>(1, chrom);
		parents->setMinimize();
		offspring = new PopulationCT<ChromosomeCMA>(1, chrom);
		offspring->setMinimize();

		ecma.init((*(PopulationCT<ChromosomeCMA>*)parents)[0], sigma, 1);
	}

	bestFitness = 1e100;
	bestParameters.resize(dim, false);

	uncertaintyHandling = false;
	bFirstIteration = true;
}

void CMAOptimizer::initUncertainty(Model& model, double sigma, unsigned int maxEvals, double alpha, double theta, eMode mode, int lambda, int mu)
{
	init(model, sigma, mode, false, lambda, mu);

	uncertaintyHandling = true;

	this->maxEvals = maxEvals;
	this->alpha = alpha;
	this->theta = theta;

	objective.resetCount();
}

void CMAOptimizer::initUncertainty(Model& model, const Array<double>& sigma, unsigned int maxEvals, double alpha, double theta, eMode mode, int lambda, int mu)
{
	init(model, sigma, mode, false, lambda, mu);

	uncertaintyHandling = true;

	this->maxEvals = maxEvals;
	this->alpha = alpha;
	this->theta = theta;

	objective.resetCount();
}

double CMAOptimizer::optimize(Model& model, ErrorFunction& errorfunction, const Array<double>& input, const Array<double>& target)
{
	objective.Set(model, errorfunction, input, target);

	int dim = model.getParameterDimension();
	int i, o;
	Individual* pI;
	ChromosomeCMA* pC;

	if (bFirstIteration)
	{
		// ensure that the parents have a valid fitness value
		int i, ic = parents->size();
		for (i = 0; i < ic; i++)
		{
// 			Ind2Model((*parents)[i], model);
// 			(*parents)[i].setFitness(errorfunction.error(model, input, target));
			pI = &((*offspring)[0]);
			pC = (ChromosomeCMA*)(&((*pI)[0]));
			(*parents)[i].setFitness(objective.fitness(*pC));
		}
		bFirstIteration = false;
		objective.resetCount();
	}

	if ((cmaMode == modeRankMuUpdate) || (cmaMode == modeRankOneUpdate))
	{
		int lambda = offspring->size();

		// create lambda feasible offspring
		for (o = 0; o < lambda; o++)
		{
			pI =& (offspring->operator [](o));
			do
			{
				cma.create(*pI);
				Ind2Model(*pI, model);
			}
			while (! model.isFeasible());
			pC = (ChromosomeCMA*)(&((*pI)[0]));
			double f = objective.fitness(*pC);
// 			printf("[%g]", f);
			pI->setFitness(f);
		}

		if (uncertaintyHandling)
		{
			double uncertainty = UncertaintyQuantification(*offspring, objective, theta);
			unsigned int n = objective.getN();
			if (verbosity > 1) { printf("[n=%d]", n); fflush(stdout); }
			unsigned int new_n = n;
			if (uncertainty > 0.0)
			{
				new_n = (unsigned int)(alpha * n);
				if (new_n == n) new_n++;
				if (new_n > maxEvals) new_n = maxEvals;
			}
			if (uncertainty < 0.0)
			{
				new_n = (unsigned int)(n / alpha);
				if (new_n == n) new_n--;
				if (new_n == 0) new_n++;
			}
			objective.setN(new_n);
		}

		// selection
		parents->selectMuLambda(*offspring, 0);

		// strategy adaptation
		cma.updateStrategyParameters(*parents);
	}
	else if (cmaMode == modeOnePlusOne)
	{
		// create one feasible offspring
		do
		{
			ecma.Mutate((*(PopulationCT<ChromosomeCMA>*)parents)[0], *(PopulationCT<ChromosomeCMA>*)offspring);
			pI = &((*(PopulationCT<ChromosomeCMA>*)offspring)[0]);
			Ind2Model(*pI, model);
		}
		while (! model.isFeasible());
		pC = (ChromosomeCMA*)(&((*pI)[0]));
		pI->setFitness(objective.fitness(*pC));

		// selection and strategy adaptation
		ecma.SelectAndUpdateStrategyParameters((*(PopulationCT<ChromosomeCMA>*)parents)[0], *(PopulationCT<ChromosomeCMA>*)offspring);
	}
	else
	{
		throw SHARKEXCEPTION("[CMA::optimize] invalid mode");
	}

	// If this is the best solution ever seen
	// remember the parameters and the fitness
	pI = &((*parents)[0]);
	pC = (ChromosomeCMA*)(&((*pI)[0]));
	double f = pI->fitnessValue();

	if (returnBestIndividual)
	{
		if (verbosity > 0) printf("[CMA] current fitness: %g best fitness: %g\n", f, bestFitness);
		if (f < bestFitness)
		{
			bestFitness = f;
			for (i = 0; i < dim; i++) bestParameters(i) = (*pC)[i];
		}

		// Copy the globally best solution to the model
		// and return the corresponding error value.
		for (i = 0; i < dim; i++) model.setParameter(i, bestParameters(i));
		return bestFitness;
	}
	else
	{
		if (verbosity > 0) printf("[CMA] current fitness: %g\n", f);

		// Copy the best current solution to the model
		// and return the corresponding error value.
		for (i = 0; i < dim; i++) model.setParameter(i, (*pC)[i]);
		return f;
	}
}

int CMAOptimizer::getLambda()
{
	return offspring->size();
}

void CMAOptimizer::Ind2Model(Individual& ind, Model& model)
{
	int i, ic = model.getParameterDimension();
	ChromosomeCMA* pC = (ChromosomeCMA*)(&ind[0]);

	SIZE_CHECK(ic == (int)pC->size());

	for (i = 0; i < ic; i++) model.setParameter(i, (*pC)[i]);
}


////////////////////////////////////////////////////////////


void CMAOptimizer::ModelFitness::Set(Model& model, ErrorFunction& errorfunction, const Array<double>& input, const Array<double>& target)
{
	m = &model;
	e = &errorfunction;
	i = &input;
	t = &target;
}

double CMAOptimizer::ModelFitness::fitness(const std::vector<double>& v)
{
	unsigned int p, pc = v.size();
	for (p=0; p<pc; p++) m->setParameter(p, v[p]);
	double ret = 0.0;
	for (p=0; p<evals; p++) ret += e->error(*m, *i, *t);
	return ret / (double)evals;
}
