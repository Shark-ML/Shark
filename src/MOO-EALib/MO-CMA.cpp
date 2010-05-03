//===========================================================================
/*!
 *  \file MO-CMA.cpp
 *
 *  \brief Implementation of the CMA-ES for multi-objective optimization
 *
 *  \author  Tobias Glasmachers
 *  \date    2008
 *
 *  \par Copyright (c) 2008:
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


#include <MOO-EALib/MO-CMA.h>

//! \brief Interprets integers as indices in a population and returns a comparison of
//! the corresponding individuals based on their rank and share.
struct IndexComparator {
	IndexComparator( PopulationMOO & pop ) : m_pop( pop ) {}
	
	bool operator()( unsigned int i, unsigned int j ) {
		return( PopulationMOO::compareRankShare( &m_pop[i], &m_pop[j] ) );
	}
	
	PopulationMOO & m_pop;
};

MOCMASearch::MOCMASearch()
{
	m_pop = NULL;
	m_penaltyFactor = 0.00001;
}

MOCMASearch::~MOCMASearch()
{
	if (m_pop != NULL) delete m_pop;
}


void MOCMASearch::init(ObjectiveFunctionVS<double>& fitness, unsigned int mu, unsigned int lambda)
{
	if (m_pop != NULL) delete m_pop;

	unsigned int i, j;
	unsigned int dim = fitness.dimension();
	m_objectives = fitness.objectives();

	m_fitness = &fitness;
	m_mu = mu;
	m_lambda = lambda;

	// create a joint population
	ChromosomeCMA chrom(dim);
	IndividualMOO prot(chrom);
	m_pop = new PopulationMOO(mu + lambda, prot);
	m_pop->setMinimize();
	m_pop->setNoOfObj(m_objectives);

	if (m_fitness->getConstraintHandler() != NULL && dynamic_cast<const BoxConstraintHandler*>(m_fitness->getConstraintHandler()) != NULL)
	{
		// initialize the search distribution according to the constraints
		const BoxConstraintHandler* bch = static_cast<const BoxConstraintHandler*>(m_fitness->getConstraintHandler());

		ChromosomeT<double> min(dim);
		ChromosomeT<double> max(dim);
		std::vector<double> stddev(dim);
		Vector pt(dim);
		double* p = &pt(0);
		for (i=0; i<dim; ++i )
		{
			min[i] = bch->lowerBound(i);
			max[i] = bch->upperBound(i);
			stddev[i] = (max[i] - min[i]) / 3.0;
		}
		for (i=0; i<m_mu; ++i )
		{
			(static_cast<ChromosomeCMA&>((*m_pop)[i][0])).init(dim, stddev, 1.0, min, max);
			if (! fitness.ProposeStartingPoint(p)) throw SHARKEXCEPTION("[MOCMASearch::init] The fitness function must propose a starting point");
			for (j=0; j<dim; j++) (static_cast<ChromosomeCMA&>((*m_pop)[i][0]))[j] = pt(j);
			eval((*m_pop)[i]);
		}
	}
	else
	{
		// Sample three initial points and determine the
		// initial step size as the median of their distances.
		Vector start1(dim);
		Vector start2(dim);
		Vector start3(dim);
		double* p;
		p = &start1(0);
		if (! fitness.ProposeStartingPoint(p)) throw SHARKEXCEPTION("[MOCMASearch::init] The fitness function must propose a starting point");
		p = &start2(0);
		if (! fitness.ProposeStartingPoint(p)) throw SHARKEXCEPTION("[MOCMASearch::init] The fitness function must propose a starting point");
		p = &start3(0);
		if (! fitness.ProposeStartingPoint(p)) throw SHARKEXCEPTION("[MOCMASearch::init] The fitness function must propose a starting point");
		double d[3];
		d[0] = (start2 - start1).norm();
		d[1] = (start3 - start1).norm();
		d[2] = (start3 - start2).norm();
		std::sort(d, d + 3);
		double stepsize = d[1]; if (stepsize == 0.0) stepsize = 1.0;

		// initialize parents
		Vector pt(dim);
		p = &pt(0);
		for (i=0; i<m_mu; ++i )
		{
			(static_cast<ChromosomeCMA&>((*m_pop)[i][0])).init(dim, stepsize, 0.0, 0.0);
			if (! fitness.ProposeStartingPoint(p)) throw SHARKEXCEPTION("[MOCMASearch::init] The fitness function must propose a starting point");
			for (j=0; j<dim; j++) (static_cast<ChromosomeCMA&>((*m_pop)[i][0]))[j] = pt(j);
			eval((*m_pop)[i]);
		}
	}
}

void MOCMASearch::init(ObjectiveFunctionVS<double>& fitness, double stepsize, unsigned int mu, unsigned int lambda)
{
	if (m_pop != NULL) delete m_pop;

	unsigned int i, j;
	unsigned int dim = fitness.dimension();
	m_objectives = fitness.objectives();

	m_fitness = &fitness;
	m_mu = mu;
	m_lambda = lambda;

	// create a joint population
	ChromosomeCMA chrom(dim);
	IndividualMOO prot(chrom);
	m_pop = new PopulationMOO(mu + lambda, prot);
	m_pop->setMinimize();
	m_pop->setNoOfObj(m_objectives);

	// initialize parents
	Vector pt(dim);
	double* p = &pt(0);
	for (i=0; i<m_mu; ++i )
	{
		(dynamic_cast<ChromosomeCMA&>((*m_pop)[i][0])).init(dim, stepsize, 0.0, 0.0);
		if (! fitness.ProposeStartingPoint(p)) throw SHARKEXCEPTION("[MOCMASearch::init] The fitness function must propose a starting point");
		for (j=0; j<dim; j++) (dynamic_cast<ChromosomeCMA&>((*m_pop)[i][0]))[j] = pt(j);
		eval((*m_pop)[i]);
	}
}

void MOCMASearch::run()
{
	unsigned int i;

	// reproduce, modify, and evaluate
	for (i=0; i<m_lambda; i++)
	{
		IndividualMOO& ind = (*m_pop)[m_mu + i];
		while (true)
		{
			ind = (*m_pop)[i % m_mu];
			ChromosomeCMA& chrom = static_cast<ChromosomeCMA&>(ind[0]);
			chrom.mutate();
			if (m_fitness->isFeasible(chrom)) break;

			// resample only if repair is not supported
			ChromosomeCMA tmp = chrom;
			if (m_fitness->closestFeasible(tmp)) break;
		}
		eval(ind);
	}

	// compute rank and second level sorting criterion
	m_pop->SMeasure();		// use hypervolume for sorting

	// check success of offspring	
	std::vector<unsigned int> indices;
	for( unsigned int i = 0; i < m_pop->size(); i++ ) {
		indices[i] = i;
	}
	
	std::sort( indices.begin(), indices.end(), IndexComparator( *m_pop ) );
	
	std::vector<unsigned int>::iterator it;
	for( unsigned int i = 0; i < m_pop->size(); i++ ) {
		if( indices[i] < m_mu ) {
			
			it = std::find( indices.begin(), indices.begin() + m_mu, indices[i] + m_mu );
			if( it == indices.end() )
				continue;
			
			(static_cast<ChromosomeCMA&>((*m_pop)[i][0])).updateLambdaSucc( true );
		} else {
			(static_cast<ChromosomeCMA&>((*m_pop)[i][0])).updateLambdaSucc( true );
			
		}
		
		
	}
	
	// environmental selection
	std::sort(m_pop->begin(), m_pop->end(), PopulationMOO::compareRankShare);

	// update strategy parameters
	for (i=0; i<m_mu; i++)
	{
		ChromosomeCMA& c = static_cast<ChromosomeCMA&>((*m_pop)[i][0]);
		if (c.covarianceUpdateNeeded()) c.updateCovariance();
		c.updateGlobalStepsize();
	}

	// call superclass
	SearchAlgorithm<double*>::run();
}

// return the non-dominated solutions
void MOCMASearch::bestSolutions(std::vector<double*>& points)
{
	PopulationMOO nondom;
	m_pop->getNonDominated(nondom, true);
	unsigned int i, ic = nondom.size();
	points.resize(ic);
	for (i=0; i<ic; i++) points[i] = &(static_cast<ChromosomeCMA&>(nondom[i][0]))[0];
}

// return the fitness vectors of the whole population
void MOCMASearch::bestSolutionsFitness(Array<double>& fitness)
{
	PopulationMOO nondom;
	m_pop->getNonDominated(nondom, true);
	unsigned int i, ic = nondom.size();
	unsigned int d;
	fitness.resize(ic, m_objectives, false);
	for (i=0; i<ic; i++)
	{
		std::vector<double>& fit = nondom[i].getMOOFitnessValues(true);
		for (d=0; d<m_objectives; d++) fitness(i, d) = fit[d];
	}
}

void MOCMASearch::parents(PopulationMOO& parents) const
{
	parents.resize(m_mu);
	unsigned int i;
	for (i=0; i<m_mu; i++) parents[i] = (*m_pop)[i];
}

// Evaluate an individual.
// Due to the penalty concept this must include constraint handling.
void MOCMASearch::eval(IndividualMOO& ind)
{
	ChromosomeCMA chromosome1 = static_cast<ChromosomeCMA&>(ind[0]);
	ChromosomeCMA chromosome2 = chromosome1;
	if (! m_fitness->isFeasible(chromosome1))
	{
		if (! m_fitness->closestFeasible(chromosome1))
			throw SHARKEXCEPTION("[MOCMASearch::eval] fitness function must implement closestFeasible(...)");
	}

	std::vector<double> fit(m_objectives);
	m_fitness->result(chromosome1, fit);
	ind.setUnpenalizedMOOFitnessValues(fit);
	unsigned int j;
	double penalty = 0.0;
	for (j = 0; j < chromosome1.size(); j++)
	{
		penalty += Shark::sqr(chromosome1[j] - chromosome2[j]);
	}
	for (j = 0; j < fit.size(); j++)
	{
		fit[j] += m_penaltyFactor * penalty;
	}
	ind.setMOOFitnessValues(fit);
}
