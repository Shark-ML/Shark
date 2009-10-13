//===========================================================================
/*!
 *  \file CMSA.cpp
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



#include <EALib/CMSA.h>


// static
unsigned CMSA::suggestLambda(unsigned dimension) {
	unsigned lambda = unsigned(4. + floor(3. * log((double) dimension)));
	// heuristics for small search spaces
	if (lambda < 8) lambda = 8; // H.G. Beyer's lower bound
	return lambda;
}

// static
unsigned CMSA::suggestMu(unsigned lambda, RecombType recomb) {
	if (recomb == equal) return  unsigned(floor(lambda / 4.));
	return  unsigned(floor(lambda / 2.));
}

void CMSA::init(unsigned dimension,
							 std::vector<double > var, double _sigma,
							 Population &p,
							 RecombType recomb) {
	unsigned int i, j;

	unsigned mu = p.size();
	
	n     = dimension;
	sigma = _sigma;

	tauopt = 1./sqrt((double)(2*n));
	tauc = 1. + (n*(n+1))/(2.*mu);
	
	w.resize(mu);     // weights for weighted recombination
	x.resize(n);      // weighted center of mass of the population

	z.resize(n);      // standard normally distributed random vector 
	Z.resize(n, n);   // rank-mu update matrix
	C.resize(n, n);   // covariance matrix
	B.resize(n, n);   // eigenvectors of C
	lambda.resize(n); // eigenvalues of C

	
	switch (recomb) {
	case equal:
		for (i = 0; i < mu; i++) w(i) = 1;
		break;
	case linear:
		for (i = 0; i < mu; i++) w(i) = mu - i;
		break;
	case superlinear:
		for (i = 0; i < mu; i++) w(i) = log(mu + 1.) - log(1. + i);
		break;
	}
	
	double wSum    = 0;
	double wSumSqr = 0;
	for (i = 0; i < mu; i++) {
		wSum += w(i);
		wSumSqr += Shark::sqr(w(i));
	}
	w /= wSum; // normalizing weights

	// init COG
	cog(x, p);

	// init C, eigenvalues lambda and eigenvector matrix B
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			if (i != j) C(i, j) = 0;
			else C(i, j) = var[i];
		}
	}
	eigensymm(C, B, lambda);
}

void CMSA::init(unsigned dimension, double _sigma,  Population &p, 
							 RecombType recomb) {
	std::vector<double> var(dimension);
	unsigned int i;
	for (i = 0; i < dimension; i++) var[i] = 1;
	init(dimension, var, _sigma, p, recomb);
}

//
//! calculate weighted mean for intermediate recombination
//
void CMSA::cog(ChromosomeT<double >& a, Population &p, unsigned c) const {
	SIZE_CHECK(n == dynamic_cast<ChromosomeT<double >& >(p[0][c]).size());
	unsigned int i, j;
	for (j = 0; j < n; j++) {
		a[j] = dynamic_cast< ChromosomeT< double >& >(p[0][c])[j] * w(0);
		for (i = 1; i < p.size(); i++) {
			a[j] += dynamic_cast< ChromosomeT< double >& >(p[i][c])[j] * w(i);
		}
	}
}

//
//! calculate scalar weighted mean for intermediate recombination
//
void CMSA::cog(double &a, Population &p, unsigned c) const {
	unsigned int i;
	a = dynamic_cast< ChromosomeT< double >& >(p[0][c])[0] * w(0);
	for (i = 1; i < p.size(); i++) {
		a += dynamic_cast< ChromosomeT< double >& >(p[i][c])[0] * w(i);
	}
}


//! mutation after global intermediate recombination
//
void CMSA::create(Individual &o)
{
	unsigned int i, j;

	// draw random vector
	for (i = 0; i < n; i++)	z(i) = Rng::gauss(0, 1);

	// compute individual sigma (R1)
	dynamic_cast< ChromosomeT< double >& >(o[2])[0] = sigma * exp(tauopt * Rng::gauss(0, 1));

	// mutate objective variables
	for (i = 0; i < n; i++) {
		dynamic_cast< ChromosomeT< double >& >(o[1])[i] = 0.;
		// apply sqrt(C) (R2)
		for (j = 0; j < n; j++) {
			dynamic_cast< ChromosomeT< double >& >(o[1])[i] += B(i, j) * sqrt(fabs(lambda(j))) * z(j);
		}
	}
	// add mutation (R3, R4)
	for (i = 0; i < n; i++) {
		dynamic_cast< ChromosomeT< double >& >(o[0])[i] = x[i] +
			dynamic_cast< ChromosomeT< double >& >(o[2])[0] * dynamic_cast< ChromosomeT< double >& >(o[1])[i];
	}

}

//
//! do the update of the covariance matrix and the global step size
//
void CMSA::updateStrategyParameters(Population &p, double lowerBound) {
	unsigned int i, j, k;

	// COG of new parents
	cog(x, p);
	cog(sigma, p, 2);

	// rank mu update matrix
	for (i = 0; i < n; i++)
		for (j = 0; j < n; j++)
			for (Z(i, j) = 0., k = 0; k < p.size(); k++)
				Z(i, j) += w(k) * (dynamic_cast< ChromosomeT< double >& >(p[k][1])[i]) *
					(dynamic_cast< ChromosomeT< double >& >(p[k][1])[j]);
	
	// (R7)
	for (i = 0; i < n; i++)
		for (j = 0; j < n; j++)
			C(i, j) = (1. - 1./tauc ) * C(i, j) +  1./tauc *  Z(i, j);

	eigensymm(C, B, lambda);
}

//
//! get global setp size \f$\sigma\f$
//
double CMSA::getSigma() const {
	return sigma;
}

//
//! set global setp size \f$\sigma\f$
//
void CMSA::setSigma(double x) {
	sigma = x;
}

//
//! get condition number of covariance matrix
//
double CMSA::getCondition() const {
	return lambda(0) / lambda(n - 1);
}

//
//! get covariance matrix
//
const Array<double> &CMSA::getC() const {
	return C;
}

//
//! get eigenvalues of covariance matrix
//
const  Array<double> &CMSA::getLambda() const {
	return lambda;
}


////////////////////////////////////////////////////////////


CMSASearch::CMSASearch()
{
	m_name = "CMSA-ES";

	m_parents = NULL;
	m_offspring = NULL;
}

CMSASearch::~CMSASearch()
{
	if (m_parents != NULL) delete m_parents;
	if (m_offspring != NULL) delete m_offspring;
}


void CMSASearch::init(ObjectiveFunctionVS<double>& fitness, unsigned lambda, CMSA::RecombType recomb)
{
	unsigned int i, dim = fitness.dimension();

	m_fitness = &fitness;
	if(lambda)
		m_lambda = lambda;
	else
		m_lambda = m_cma.suggestLambda(dim);
	m_mu = m_cma.suggestMu(m_lambda, recomb);

	// Sample three initial points and determine the
	// initial step size as the median of their distances.
	Vector start1(dim);
	Vector start2(dim);
	Vector start3(dim);
	double* p;
	p = &start1(0);
	if (! fitness.ProposeStartingPoint(p)) throw SHARKEXCEPTION("[CMSASearch::init] The fitness function must propose a starting point");
	p = &start2(0);
	if (! fitness.ProposeStartingPoint(p)) throw SHARKEXCEPTION("[CMSASearch::init] The fitness function must propose a starting point");
	p = &start3(0);
	if (! fitness.ProposeStartingPoint(p)) throw SHARKEXCEPTION("[CMSASearch::init] The fitness function must propose a starting point");
	double d[3];
	d[0] = (start2 - start1).norm();
	d[1] = (start3 - start1).norm();
	d[2] = (start3 - start2).norm();
	std::sort(d, d + 3);
	double stepsize = d[1]; if (stepsize == 0.0) stepsize = 1.0;

	ChromosomeT<double> point(dim);
	for (i=0; i<dim; i++) point[i] = start1(i);
	ChromosomeT<double> individualSigma(1);
	individualSigma[0] = stepsize;

	m_parents = new PopulationT<double>(m_mu, point, ChromosomeT<double>(dim), individualSigma);
	m_offspring = new PopulationT<double>(m_lambda, point, ChromosomeT<double>(dim), individualSigma);

	m_parents->setMinimize();
	m_offspring->setMinimize();

	m_cma.init(dim, stepsize, *m_parents, recomb);
}

void CMSASearch::init(ObjectiveFunctionVS<double>& fitness, const Array<double>& start, double stepsize, unsigned lambda, CMSA::RecombType recomb)
{
	unsigned int i, dim = fitness.dimension();
	SIZE_CHECK(start.ndim() == 1);
	SIZE_CHECK(start.dim(0) == dim);

	m_fitness = &fitness;
	if(lambda)
		m_lambda = lambda;
	else
		m_lambda = m_cma.suggestLambda(dim);
	m_mu = m_cma.suggestMu(m_lambda, recomb);

	std::cout <<m_mu << " " <<  m_lambda << std::endl;

	ChromosomeT<double> point(dim);
	for (i=0; i<dim; i++) point[i] = start(i);
	ChromosomeT<double> individualSigma(1);
	individualSigma[0] = stepsize;

	m_parents = new PopulationT<double>(m_mu, point, ChromosomeT<double>(dim), individualSigma);
	m_offspring = new PopulationT<double>(m_lambda, point, ChromosomeT<double>(dim), individualSigma);

	m_parents->setMinimize();
	m_offspring->setMinimize();

	m_cma.init(dim, stepsize, *m_parents, recomb);
}

void CMSASearch::run()
{
	unsigned int o;
	for (o=0; o<m_offspring->size(); o++) {
		while (true)
		{
			m_cma.create((*m_offspring)[o]);
			if (m_fitness->isFeasible((*m_offspring)[o][0])) break;
			if (m_fitness->closestFeasible((*m_offspring)[o][0])) break;
		}
		double f = (*m_fitness)((*m_offspring)[o][0]);
		(*m_offspring)[o].setFitness(f);
	}

	m_parents->selectMuLambda(*m_offspring, 0u);

	m_cma.updateStrategyParameters(*m_parents);

	SearchAlgorithm<double*>::run();
}

void CMSASearch::bestSolutions(std::vector<double*>& points)
{
	points.resize(1);
	points[0] = &((*m_parents)[0][0][0]);
}

void CMSASearch::bestSolutionsFitness(Array<double>& fitness)
{
	fitness.resize(1, 1, false);
	fitness(0, 0) = (*m_parents)[0].fitnessValue();
}
