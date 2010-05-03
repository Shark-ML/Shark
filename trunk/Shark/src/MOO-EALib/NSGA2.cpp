/*!
 *  \file NSGA2.cpp
 *
 *  \author T. Glasmachers
 *
 *  \brief NSGA-2 algorithm for multi-objective optimization
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
 *  <BR>
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


#include <MOO-EALib/NSGA2.h>


NSGA2Search::NSGA2Search()
{
	m_parents = NULL;
	m_offspring = NULL;

	m_penaltyFactor = 0.00001;
}

NSGA2Search::~NSGA2Search()
{
	if (m_parents != NULL) delete m_parents;
	if (m_offspring != NULL) delete m_offspring;
}


void NSGA2Search::init(ObjectiveFunctionVS<double>& fitness, unsigned int mu, double nm, double nc, double pc)
{
	unsigned int dim = fitness.dimension();
	unsigned int i;

	m_fitness = &fitness;
	if (m_parents != NULL) delete m_parents;
	if (m_offspring != NULL) delete m_offspring;

	const ConstraintHandler<double*>* ch = m_fitness->getConstraintHandler();
	if (ch == NULL) throw SHARKEXCEPTION("[NSGA2Search::init] The fitness function must provide a BoxConstraintHandler.");
	const BoxConstraintHandler* bch = dynamic_cast<const BoxConstraintHandler*>(ch);
	if (bch == NULL) throw SHARKEXCEPTION("[NSGA2Search::init] The fitness function must provide a BoxConstraintHandler.");

	m_mu = mu;
	m_lambda = mu;
	m_objectives = fitness.objectives();
	m_nm = nm;
	m_nc = nc;
	m_pc = pc;
	m_pm = 1.0 / fitness.dimension();

	m_lower.resize(dim);
	m_upper.resize(dim);
	for (i=0; i<dim; i++)
	{
		m_lower[i] = bch->lowerBound(i);
		m_upper[i] = bch->upperBound(i);
	}

	m_parents = new PopulationMOO(m_mu, ChromosomeT<double>(dim));
	m_parents->setMinimize();
	m_parents->setNoOfObj(m_objectives);
	m_offspring = new PopulationMOO(m_mu, ChromosomeT<double>(dim));
	m_offspring->setMinimize();
	m_offspring->setNoOfObj(m_objectives);

	// initialize all chromosomes of parent population
	Vector pt(dim);
	double* p = &pt(0);
	unsigned int j;
	for (i=0; i<m_mu; ++i )
	{
		if (! fitness.ProposeStartingPoint(p)) throw SHARKEXCEPTION("[NSGA2Search::init] The fitness function must propose a starting point");
		for (j=0; j<dim; j++) (static_cast<ChromosomeT<double>&>((*m_parents)[i][0]))[j] = pt(j);
		eval((*m_parents)[i]);
	}

	m_parents->crowdedDistance();
}

void NSGA2Search::run()
{
	unsigned int i;

	// copy parents to (*m_offspring)
	m_offspring->selectBinaryTournamentMOO(*m_parents);

	// recombine by crossing over two parents
	for (i=0; i < m_offspring->size(); i+=2)
		if (Rng::coinToss(m_pc))
			(dynamic_cast<ChromosomeT<double>&>((*m_offspring)[i][0])).SBX(dynamic_cast<ChromosomeT<double>&>((*m_offspring)[i+1][0]), m_lower, m_upper, m_nc, 0.5);

	for (i=0; i < (*m_offspring).size(); i++) {
		// mutate by flipping bits
		(dynamic_cast<ChromosomeT<double>&>((*m_offspring)[i][0])).mutatePolynomial(m_lower, m_upper, m_nm, m_pm);

		// evaluate objective function
		eval((*m_offspring)[i]);
	}

	// selection
	m_parents->selectCrowdedMuPlusLambda(*m_offspring);

	// call superclass
	SearchAlgorithm<double*>::run();
}

// return the non-dominated solutions
void NSGA2Search::bestSolutions(std::vector<double*>& points)
{
	PopulationMOO nondom;
	m_parents->getNonDominated(nondom, false);
	unsigned int i, ic = nondom.size();
	points.resize(ic);
	for (i=0; i<ic; i++) points[i] = &(static_cast<ChromosomeCMA&>(nondom[i][0]))[0];
}

// return the fitness vectors of the whole population
void NSGA2Search::bestSolutionsFitness(Array<double>& fitness)
{
	PopulationMOO nondom;
	m_parents->getNonDominated(nondom, false);
	unsigned int i, ic = nondom.size();
	unsigned int d;
	fitness.resize(ic, m_objectives, false);
	for (i=0; i<ic; i++)
	{
		std::vector<double>& fit = nondom[i].getMOOFitnessValues(false);
		for (d=0; d<m_objectives; d++) fitness(i, d) = fit[d];
	}
}

// evaluate an individual
void NSGA2Search::eval(IndividualMOO& ind)
{
	std::vector<double> fit(m_objectives);
	m_fitness->result(static_cast<ChromosomeT<double>&>(ind[0]), fit);
	ind.setMOOFitnessValues(fit);
}
