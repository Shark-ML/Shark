//===========================================================================
/*!
 *  \file ElitistCMA.cpp
 *
 *  \brief Implements the the elitist version of the CMA-ES
 *
 *  \par Copyright (c) 2006:
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



#include <EALib/ElitistCMA.h>


// static
void ElitistCMA::init(IndividualCT<ChromosomeCMA>& parent, double sigma, int lambda)
{
	Array<double> s(parent[0].size());
	s = sigma;
	init(parent, s, lambda);
}

// static
void ElitistCMA::init(IndividualCT<ChromosomeCMA>& parent, const Array<double>& sigma, int lambda)
{
	unsigned int i, ic = parent[0].size();
	std::vector<double> stdv(ic);
	ChromosomeT<double> parentchrom(dynamic_cast<const std::vector<double>&>(parent[0]));
	for (i=0; i<ic; i++) stdv[i] = sigma(i);
	parent[0].init(ic, stdv, 1.0, parentchrom, parentchrom, lambda);
}

// static
void ElitistCMA::Mutate(IndividualCT<ChromosomeCMA>& parent, PopulationCT<ChromosomeCMA>& offspring)
{
	unsigned int i; 
	for(i=0; i<offspring.size(); i++)
	{
		offspring[i] = parent;
		offspring[i][0].mutate();
	}
}

// static
void ElitistCMA::SelectAndUpdateStrategyParameters(IndividualCT<ChromosomeCMA>& parent, PopulationCT<ChromosomeCMA>& offspring)
{
	Individual oldParent = parent;
	unsigned int i, lambdaSucc = 0;
	for(i=0; i<offspring.size(); i++)
	{
		if (offspring[i].getFitness() <= oldParent.getFitness())
		{
			lambdaSucc++;
			if (offspring[i].getFitness() < parent.getFitness())
			{
				parent = offspring[i];
			}
		}
	}

	// update covariance matrix
	if (lambdaSucc > 0)
	{
		parent[0].updateCovariance(oldParent);
	}

	// update step size
	parent[0].updateGlobalStepsize(lambdaSucc);
}


////////////////////////////////////////////////////////////


CMAElitistSearch::CMAElitistSearch()
{
	m_name = "Elitist-CMA-ES";

	m_parents = NULL;
	m_offspring = NULL;
}

CMAElitistSearch::~CMAElitistSearch()
{
	if (m_parents != NULL) delete m_parents;
	if (m_offspring != NULL) delete m_offspring;
}


void CMAElitistSearch::init(ObjectiveFunctionVS<double>& fitness, unsigned int lambda)
{
	m_fitness = &fitness;
	m_mu = 1;
	m_lambda = lambda;

	unsigned int i, dim = fitness.dimension();

	m_parents = new PopulationCT<ChromosomeCMA>(1, 1);
	m_offspring = new PopulationCT<ChromosomeCMA>(lambda, 1);

	m_parents->setMinimize();
	m_offspring->setMinimize();

	// Sample three initial points and determine the
	// initial step size as the median of their distances.
	Vector start1(dim);
	Vector start2(dim);
	Vector start3(dim);
	double* p;
	p = &start1(0);
	if (! fitness.ProposeStartingPoint(p)) throw SHARKEXCEPTION("[CMASearch::init] The fitness function must propose a starting point");
	p = &start2(0);
	if (! fitness.ProposeStartingPoint(p)) throw SHARKEXCEPTION("[CMASearch::init] The fitness function must propose a starting point");
	p = &start3(0);
	if (! fitness.ProposeStartingPoint(p)) throw SHARKEXCEPTION("[CMASearch::init] The fitness function must propose a starting point");
	double d[3];
	d[0] = (start2 - start1).norm();
	d[1] = (start3 - start1).norm();
	d[2] = (start3 - start2).norm();
	std::sort(d, d + 3);
	double stepsize = d[1]; if (stepsize == 0.0) stepsize = 1.0;

	(*m_parents)[0][0].init(dim, stepsize, 0.0, 0.0, lambda);
	for (i=0; i<dim; i++) (*m_parents)[0][0][i] = start1(i);
	(*m_parents)[0].setFitness(fitness((*m_parents)[0][0]));
	m_bIsParentFitnessValid = true;
}

void CMAElitistSearch::init(ObjectiveFunctionVS<double>& fitness, const Array<double>& start, double stepsize, unsigned int lambda)
{
	unsigned int i, dim = fitness.dimension();
	SIZE_CHECK(start.ndim() == 1);
	SIZE_CHECK(start.dim(0) == dim);

	m_fitness = &fitness;
	m_mu = 1;
	m_lambda = lambda;

	m_parents = new PopulationCT<ChromosomeCMA>(1, 1);
	m_offspring = new PopulationCT<ChromosomeCMA>(lambda, 1);

// 	ChromosomeCMA point(dim);
// 	point.init(dim, stepsize, 0.0, 0.0, lambda);
// 	for (i=0; i<dim; i++) point[i] = start(i);

// 	m_parents = new PopulationCT<ChromosomeCMA>(m_mu, point);
// 	m_offspring = new PopulationCT<ChromosomeCMA>(m_lambda, point);

// 	m_bIsParentFitnessValid = false;

	m_parents->setMinimize();
	m_offspring->setMinimize();

	(*m_parents)[0][0].init(dim, stepsize, 0.0, 0.0, lambda);
	for (i=0; i<dim; i++) (*m_parents)[0][0][i] = start(i);
	(*m_parents)[0].setFitness(fitness((*m_parents)[0][0]));
	m_bIsParentFitnessValid = true;
}

void CMAElitistSearch::run()
{
	int lambdaSucc = 0;
	IndividualCT<ChromosomeCMA> parentIndiv = (*m_parents)[0];

	if (! m_bIsParentFitnessValid)
	{
		// first iteration, parent has not been evaluated yet
		(*m_parents)[0].setFitness((*m_fitness)(&(*m_parents)[0][0][0]));
		m_bIsParentFitnessValid = true;
	}

	// generate lambda offspring
	unsigned int i;
	for (i=0; i<m_lambda; i++) {
		while (true)
		{
			(*m_offspring)[i] = parentIndiv;
			(*m_offspring)[i][0].mutate();
			if (m_fitness->isFeasible((*m_offspring)[i][0])) break;
			if (m_fitness->ProposeStartingPoint((*m_offspring)[i][0])) break;
		}
		(*m_offspring)[i].setFitness((*m_fitness)(&(*m_offspring)[i][0][0]));
		if ((*m_offspring)[i].getFitness() <= parentIndiv.getFitness()) {
			lambdaSucc++;
			if ((*m_offspring)[i].getFitness() <= (*m_parents)[0].getFitness()) (*m_parents)[0] = (*m_offspring)[i];
		}
	}

	// update strategy parameters
	(*m_parents)[0][0].updateGlobalStepsize(lambdaSucc);
	if (lambdaSucc) (*m_parents)[0][0].updateCovariance(parentIndiv);

	SearchAlgorithm<double*>::run();
}

void CMAElitistSearch::bestSolutions(std::vector<double*>& points)
{
	points.resize(1);
	points[0] = &(*m_parents)[0][0][0];
}

void CMAElitistSearch::bestSolutionsFitness(Array<double>& fitness)
{
	fitness.resize(1, 1, false);
	fitness(0, 0) = (*m_parents)[0].getFitness();
}
