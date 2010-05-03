//===========================================================================
/*!
 *  \file OnePlusOneES.cpp
 *
 *  \brief simple 1+1-ES
 *
 *  \author Tobias Glasmachers
 *  \date 2008
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


#include <EALib/OnePlusOneES.h>


OnePlusOneES::OnePlusOneES()
: m_parent(ChromosomeT<double>(0), ChromosomeT<double>(1))
{
	m_name = "1+1-ES";
	m_mode = SelfAdaptation;
}

OnePlusOneES::~OnePlusOneES()
{
}


void OnePlusOneES::init(eStepSizeControl mode, ObjectiveFunctionVS<double>& fitness)
{
	unsigned int dim = fitness.dimension();

	// Sample three initial points and determine the
	// initial step size as the median of their distances.
	Vector start1(dim);
	Vector start2(dim);
	Vector start3(dim);
	double* p;
	p = &start1(0);
	if (! fitness.ProposeStartingPoint(p)) throw SHARKEXCEPTION("[OnePlusOneES::init] The fitness function must propose a starting point");
	p = &start2(0);
	if (! fitness.ProposeStartingPoint(p)) throw SHARKEXCEPTION("[OnePlusOneES::init] The fitness function must propose a starting point");
	p = &start3(0);
	if (! fitness.ProposeStartingPoint(p)) throw SHARKEXCEPTION("[OnePlusOneES::init] The fitness function must propose a starting point");
	double d[3];
	d[0] = (start2 - start1).norm();
	d[1] = (start3 - start1).norm();
	d[2] = (start3 - start2).norm();
	std::sort(d, d + 3);
	double stepsize = d[1]; if (stepsize == 0.0) stepsize = 1.0;

	init(mode, fitness, start1, stepsize);
}

void OnePlusOneES::init(eStepSizeControl mode, ObjectiveFunctionVS<double>& fitness, const Array<double>& start, double stepsize)
{
	m_mode = mode;
	m_fitness = &fitness;
	unsigned int i, dim = fitness.dimension();

	m_parent[0].resize(dim);
	for (i=0; i<dim; i++) m_parent[0][i] = start(i);
	m_parent.setFitness(fitness(m_parent[0]));
	m_parent[1][0] = stepsize;

	m_logStepSizeStdDev = 0.5;
	m_logStepSizeShift = 0.0;
}

void OnePlusOneES::SampleUnitVector(Vector& v)
{
	double len2 = 0.0;
	unsigned int i, dim = v.dim(0);
	for (i=0; i<dim; i++)
	{
		double g = Rng::gauss();
		len2 += g * g;
		v(i) = g;
	}
	double len = sqrt(len2);
	for (i=0; i<dim; i++) v(i) /= len;
}

void OnePlusOneES::run()
{
	if (m_mode == SelfAdaptation)
	{
		DoSelfAdaptation();
	}
	else if (m_mode == OneFifth)
	{
		DoOneFifth();
	}
	else if (m_mode == SymmetricOneFifth)
	{
		DoSymmetricOneFifth();
	}
}

void OnePlusOneES::DoSelfAdaptation()
{
	unsigned int i, dim = m_fitness->dimension();

	// sample one offspring
	IndividualT<double> offspring(m_parent);
	double tau = sqrt(0.5 / dim);
	double gauss = Rng::gauss();
	offspring[1][0] *= exp(tau * gauss);
	for (i=0; i<dim; i++) offspring[0][i] += offspring[1][0] * Rng::gauss();

	// evaluate the offspring
	offspring.setFitness((*m_fitness)(offspring[0]));

	// select
	if (offspring.getFitness() < m_parent.getFitness())
	{
		m_parent = offspring;
	}
	else if (offspring.getFitness() == m_parent.getFitness())
	{
		m_parent = offspring;
	}
}

void OnePlusOneES::DoOneFifth()
{
	unsigned int i, dim = m_fitness->dimension();

	// evaluate the offspring
	IndividualT<double> offspring(m_parent);
	for (i=0; i<dim; i++) offspring[0][i] += offspring[1][0] * Rng::gauss();
	offspring.setFitness((*m_fitness)(offspring[0]));

	// select
	double success = 0.0;
	if (offspring.getFitness() < m_parent.getFitness())
	{
		m_parent = offspring;
		success = 1.0;
	}
	else if (offspring.getFitness() == m_parent.getFitness())
	{
		m_parent = offspring;
		success = 0.2;
	}

	// explicit strategy update
	m_parent[1][0] *= exp(success - 0.2);
}

void OnePlusOneES::DoSymmetricOneFifth()
{
	unsigned int i, dim = m_fitness->dimension();

	// sample one offspring
	IndividualT<double> offspring(m_parent);
	Vector u(dim);
	SampleUnitVector(u);

	double ls = log(offspring[1][0]) + Rng::gauss(0.0, m_logStepSizeStdDev);
	if (Rng::coinToss()) ls += m_logStepSizeShift; else ls -= m_logStepSizeShift;
	double s = exp(ls);
	for (i=0; i<dim; i++) offspring[0][i] += s * u(i);

	// evaluate the offspring
	offspring.setFitness((*m_fitness)(offspring[0]));

	// selection and explicit strategy update
	if (offspring.getFitness() <= m_parent.getFitness())
	{
		m_parent = offspring;
		m_parent[1][0] = s;
		m_logStepSizeShift = 0.0;
		m_logStepSizeStdDev = 0.5;
	}
	else
	{
		m_logStepSizeShift += 0.01;
		m_logStepSizeStdDev += 0.01;
	}
}

void OnePlusOneES::bestSolutions(std::vector<double*>& points)
{
	points.resize(1);
	points[0] = &m_parent[0][0];
}

void OnePlusOneES::bestSolutionsFitness(Array<double>& fitness)
{
	fitness.resize(1, 1, false);
	fitness(0, 0) = m_parent.getFitness();
}
