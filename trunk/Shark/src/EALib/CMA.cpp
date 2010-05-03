//===========================================================================
/*!
 *  \file CMA.cpp
 *
 *  \brief Implements the most recent version of the non-elitist CMA-ES
 *
 *  The algorithm is described in:
 *
 *  Hansen, N., S. Kern (2004). Evaluating the CMA Evolution Strategy
 *  on Multimodal Test Functions. In Proceedings of the Eighth
 *  International Conference on Parallel Problem Solving from Nature
 *  (PPSN VIII), pp. 282-291, LNCS, Springer-Verlag
 *
 *  The parameters were updated according to:
 *
 *  Hansen, N., A. S. P. Niederberger, L. Guzzella, and
 *  P. Koumoutsakos. A Method for Handling Uncertainty in Evolutionary
 *  Optimization with an Application to Feedback Control of
 *  Combustion, IEEE Transactions on Evolutionary Computation
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



#include <EALib/CMA.h>


// static
unsigned CMA::suggestLambda(unsigned dimension) {
	unsigned lambda = unsigned(4. + floor(3. * log((double) dimension)));
	// heuristics for small search spaces
	if (lambda > dimension) lambda = dimension; // CI's golden rule :-)
	if (lambda < 5) lambda = 5; // Hansen & Ostermeier's lower bound
	return lambda;
}

// static
unsigned CMA::suggestMu(unsigned lambda, RecombType recomb) {
	if (recomb == equal) return  unsigned(floor(lambda / 4.));
	return  unsigned(floor(lambda / 2.));
}

void CMA::init(unsigned dimension,
							 std::vector<double > var, double _sigma,
							 Population &p,
							 RecombType recomb,
							 UpdateType cupdate) {
	unsigned int i, j;

	unsigned mu = p.size();

	n     = dimension;
	sigma = _sigma;

	w.resize(mu);
	x.resize(n);
	xPrime.resize(n);
	z.resize(n);
	pc.resize(n);
	ps.resize(n);
	Z.resize(n, n);
	C.resize(n, n);
	B.resize(n, n);
	lambda.resize(n);
	theVector.resize(n);
	meanz.resize(n);

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
	wSumSqr /= Shark::sqr(wSum);
	mueff   = 1 / wSumSqr;

	// step size control
	cs      = (mueff + 2.)/(n + mueff + 3.);
	d       = 1. + 2. * Shark::max(0., sqrt( (mueff-1.)/(n+1) ) - 1.) + cs;

	// covariance matrix adaptation
	mucov   = mueff;
	if (cupdate == rankone) mucov = 1.;
	cc      = 4. / (4. + n);
	ccov    = 1. / mucov * 2. / Shark::sqr(n + sqrt(2.))
		+ (1 - 1. / mucov) * Shark::min(1., (2 * mueff - 1) / (Shark::sqr(n + 2) + mueff));

	ccu     = sqrt((2. - cc) * cc);
	csu     = sqrt((2. - cs) * cs);
	chi_n   = sqrt(double(n)) * (1 - 1. / (4. * n) +  1. / (21. * Shark::sqr(double(n))));

	// init COG
	cog(x, p);

	// init paths
	for (i = 0; i < n; i++) {
		pc(i) = ps(i) = 0.;
		for (j = 0; j < n; j++) {
			if (i != j) C(i, j) = 0;
			else C(i, j) = var[i];
		}
	}

	// eigenvalues lambda and eigenvector matrix B
	eigensymm(C, B, lambda);
}

void CMA::init(unsigned dimension, double _sigma,  Population &p,
							 RecombType recomb, UpdateType cupdate) {
	std::vector<double> var(dimension);
	unsigned int i;
	for (i = 0; i < dimension; i++) var[i] = 1;
	init(dimension, var, _sigma, p, recomb, cupdate);
}

//
//! calculate weighted mean for intermediate recombination
//
void CMA::cog(ChromosomeT<double >& a, Population &p, unsigned c) const {
	SIZE_CHECK(n == dynamic_cast<ChromosomeT<double >& >(p[0][c]).size());
	unsigned int i, j;
	for (j = 0; j < n; j++) {
		a[j] = dynamic_cast< ChromosomeT< double >& >(p[0][c])[j] * w(0);
		for (i = 1; i < p.size(); i++) {
			a[j] += dynamic_cast< ChromosomeT< double >& >(p[i][c])[j] * w(i);
		}
	}
}

//! mutation after global intermediate recombination
//
void CMA::create(Individual &o)
{
	unsigned int i, j;

	for (i = 0; i < n; i++) {
		// draw random vector
		z(i) = Rng::gauss(0, 1);
		dynamic_cast< ChromosomeT< double >& >(o[1])[i] = z(i);
		// global intermediate recombination, Eq. (1)
		dynamic_cast< ChromosomeT< double >& >(o[0])[i] = x[i];
	}

	// mutate objective variables, Eq. (1)
	for (i = 0; i < n; i++)
		for (j = 0; j < n; j++)
			dynamic_cast< ChromosomeT< double >& >(o[0])[i] += sigma * B(i, j) * sqrt(fabs(lambda(j))) * z(j);
}

//
//! do the update of the covariance matrix and the global step size
//
void CMA::updateStrategyParameters(Population &p, double lowerBound) {
	unsigned int i, j, k;
	double normPS = 0.;

	// COG of new parents
	cog(xPrime, p);
	cog(meanz , p, 1);

	theVector = 0;
	for (i = 0; i < n; i++)
		for (j = 0; j < n; j++)
			theVector(i) += B(i, j) * meanz[j];

	// Eq. (2) & Eq. (4)
	for (i = 0; i < n; i++) {
		pc(i) = (1 - cc) * pc(i) + ccu * sqrt(mueff) / sigma * (xPrime[i] - x[i]);
		ps(i) = (1 - cs) * ps(i) + csu * sqrt(mueff) * theVector(i);
		normPS += Shark::sqr(ps(i));
	}
	normPS  = sqrt(normPS);

	// Eq. (3)
	for (i = 0; i < n; i++)
		for (j = 0; j < n; j++)
			for (Z(i, j) = 0., k = 0; k < p.size(); k++)
				Z(i, j) += w(k) * (dynamic_cast< ChromosomeT< double >& >(p[k][0])[i] - x[i]) *
					(dynamic_cast< ChromosomeT< double >& >(p[k][0])[j] - x[j]);

	for (i = 0; i < n; i++)
		for (j = 0; j < n; j++)
			C(i, j) = (1 - ccov) * C(i, j) + ccov *
				(1. / mucov * pc(i) * pc(j) + (1. - 1. / mucov) * 1. / Shark::sqr(sigma) * Z(i, j));

	// Eq. (5)
	sigma *= exp((cs / d) * (normPS / chi_n - 1));

	// eigenvalues lambda and eigenvector matrix B
	eigensymm(C, B, lambda);

	// lower bound
	if ((sigma * sqrt(fabs(lambda(n - 1)))) < lowerBound)
		sigma = lowerBound / sqrt(fabs(lambda(n - 1)));

	// new COG becomes old COG
	x = xPrime;
}

//
//! get global setp size \f$\sigma\f$
//
double CMA::getSigma() const {
	return sigma;
}

//
//! set global setp size \f$\sigma\f$
//
void CMA::setSigma(double x) {
	sigma = x;
}

//
//! set different approximation for expectation of \f$\chi_n\f$ distribution
//
void CMA::setChi_n(double x) {
	chi_n = x;
}

//
//! get condition number of covariance matrix
//
double CMA::getCondition() const {
	return lambda(0) / lambda(n - 1);
}

//
//! get covariance matrix
//
const Array<double> &CMA::getC() const {
	return C;
}

//
//! get eigenvalues of covariance matrix
//
const  Array<double> &CMA::getLambda() const {
	return lambda;
}

std::ostream & operator<<( std::ostream & stream, const CMA& cma)
{
    //change precision and save the old precision
    std::streamsize precision=stream.precision(21);

    //writing states
    stream<<cma.n<<" ";
	stream<<cma.sigma<<" ";
	stream<<cma.chi_n<<" ";
	stream<<cma.cc<<" ";
	stream<<cma.cs<<" ";
	stream<<cma.csu<<" ";
	stream<<cma.ccu<<" ";
	stream<<cma.ccov<<" ";
	stream<<cma.d<<" ";
	stream<<cma.mueff<<" ";
	stream<<cma.mucov<<" ";

    //reading chromosomes
	stream<<cma.x<<" ";
	stream<<cma.xPrime<<" ";
	stream<<cma.meanz<<" ";

    //reading arrays
    stream<<cma.z<<" ";
	stream<<cma.pc<<" ";
	stream<<cma.ps<<" ";
	stream<<cma.C<<" ";
	stream<<cma.Z<<" ";
	stream<<cma.lambda<<" ";
	stream<<cma.B<<" ";
	stream<<cma.w<<" ";
	stream<<cma.theVector<<" ";

	//reset precision
	stream.precision(precision);
    return stream;
}
std::istream & operator>>( std::istream & stream, CMA& cma)
{
    //change precision and save the old precision
    double precision=stream.precision(21);
    //reading states
    stream>>cma.n;
	stream>>cma.sigma;
	stream>>cma.chi_n;
	stream>>cma.cc;
	stream>>cma.cs;
	stream>>cma.csu;
	stream>>cma.ccu;
	stream>>cma.ccov;
	stream>>cma.d;
	stream>>cma.mueff;
	stream>>cma.mucov;

    //reading chromosomes
	stream>>cma.x;
	stream>>cma.xPrime;
	stream>>cma.meanz;

    //reading arrays
    stream>>cma.z;
	stream>>cma.pc;
	stream>>cma.ps;
	stream>>cma.C;
	stream>>cma.Z;
	stream>>cma.lambda;
	stream>>cma.B;
	stream>>cma.w;
	stream>>cma.theVector;

	//reset precision state
	stream.precision(precision);
    return stream;
}


////////////////////////////////////////////////////////////


CMASearch::CMASearch()
{
	m_name = "CMA-ES";

	m_parents = NULL;
	m_offspring = NULL;
}

CMASearch::~CMASearch()
{
	if (m_parents != NULL) delete m_parents;
	if (m_offspring != NULL) delete m_offspring;
}


void CMASearch::init(ObjectiveFunctionVS<double>& fitness, CMA::RecombType recomb, CMA::UpdateType cupdate)
{
	unsigned int i, dim = fitness.dimension();

	m_fitness = &fitness;
	m_lambda = m_cma.suggestLambda(dim);
	m_mu = m_cma.suggestMu(m_lambda, recomb);

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

	ChromosomeT<double> point(dim);
	for (i=0; i<dim; i++) point[i] = start1(i);

	m_parents = new PopulationT<double>(m_mu, point, ChromosomeT<double>(dim));
	m_offspring = new PopulationT<double>(m_lambda, point, ChromosomeT<double>(dim));

	m_parents->setMinimize();
	m_offspring->setMinimize();

	m_cma.init(dim, stepsize, *m_parents, recomb, cupdate);
}

void CMASearch::init(ObjectiveFunctionVS<double>& fitness, const Array<double>& start, double stepsize, CMA::RecombType recomb, CMA::UpdateType cupdate)
{
	unsigned int i, dim = fitness.dimension();
	SIZE_CHECK(start.ndim() == 1);
	SIZE_CHECK(start.dim(0) == dim);

	m_fitness = &fitness;
	m_lambda = m_cma.suggestLambda(dim);
	m_mu = m_cma.suggestMu(m_lambda, recomb);

	ChromosomeT<double> point(dim);
	for (i=0; i<dim; i++) point[i] = start(i);

	m_parents = new PopulationT<double>(m_mu, point, ChromosomeT<double>(dim));
	m_offspring = new PopulationT<double>(m_lambda, point, ChromosomeT<double>(dim));

	m_parents->setMinimize();
	m_offspring->setMinimize();

	m_cma.init(dim, stepsize, *m_parents, recomb, cupdate);
}

void CMASearch::init(ObjectiveFunctionVS<double>& fitness, unsigned int mu, unsigned int lambda, const Array<double>& start, double stepsize, CMA::RecombType recomb, CMA::UpdateType cupdate)
{
	unsigned int i, dim = fitness.dimension();
	SIZE_CHECK(start.ndim() == 1);
	SIZE_CHECK(start.dim(0) == dim);

	m_fitness = &fitness;
	m_lambda = lambda;
	m_mu = mu;

	ChromosomeT<double> point(dim);
	for (i=0; i<dim; i++) point[i] = start(i);

	m_parents = new PopulationT<double>(m_mu, point, ChromosomeT<double>(dim));
	m_offspring = new PopulationT<double>(m_lambda, point, ChromosomeT<double>(dim));

	m_parents->setMinimize();
	m_offspring->setMinimize();

	m_cma.init(dim, stepsize, *m_parents, recomb, cupdate);
}

void CMASearch::init(ObjectiveFunctionVS<double>& fitness, unsigned int mu, unsigned int lambda, const Array<double>& start, const Array<double>& stepsize, CMA::RecombType recomb, CMA::UpdateType cupdate)
{
	unsigned int i, dim = fitness.dimension();
	SIZE_CHECK(start.ndim() == 1);
	SIZE_CHECK(stepsize.ndim() == 1);
	SIZE_CHECK(start.dim(0) == dim);
	SIZE_CHECK(stepsize.dim(0) == dim);

	m_fitness = &fitness;
	m_lambda = lambda;
	m_mu = mu;

	ChromosomeT<double> point(dim);
	for (i=0; i<dim; i++) point[i] = start(i);

	m_parents = new PopulationT<double>(m_mu, point, ChromosomeT<double>(dim));
	m_offspring = new PopulationT<double>(m_lambda, point, ChromosomeT<double>(dim));

	m_parents->setMinimize();
	m_offspring->setMinimize();

	std::vector<double> var(dim);
	for (i=0; i<dim; i++) var[i] = stepsize(i) * stepsize(i);
	m_cma.init(dim, var, 1.0, *m_parents, recomb, cupdate);
}

void CMASearch::run()
{
	unsigned int o;
	for (o=0; o<m_offspring->size(); o++) {
		while (true) {
			m_cma.create((*m_offspring)[o]);
			if (m_fitness->isFeasible((*m_offspring)[o][0])) break;
			if (m_fitness->closestFeasible((*m_offspring)[o][0])) break;
		}
		double f = (*m_fitness)(&(*m_offspring)[o][0][0]);
		(*m_offspring)[o].setFitness(f);
	}

	m_parents->selectMuLambda(*m_offspring, 0u);

	m_cma.updateStrategyParameters(*m_parents);

	SearchAlgorithm<double*>::run();
}

void CMASearch::bestSolutions(std::vector<double*>& points)
{
	points.resize(1);
	points[0] = &((*m_parents)[0][0][0]);
}

void CMASearch::bestSolutionsFitness(Array<double>& fitness)
{
	fitness.resize(1, 1, false);
	fitness(0, 0) = (*m_parents)[0].fitnessValue();
}

std::ostream & operator<<( std::ostream & stream, const CMASearch& search)
{
    double precision=stream.precision(21);

    stream<<search.m_cma<<" "<<*search.m_parents<<" "<<*search.m_offspring<<std::flush;

    stream.precision(precision);
    return stream;
}
std::istream & operator>>( std::istream & stream, CMASearch& search)
{
    double precision=stream.precision(21);

    if(!search.m_parents)
        search.m_parents=new PopulationT<double>();
    if(!search.m_offspring)
        search.m_offspring=new PopulationT<double>();

    stream>>search.m_cma
          >>*search.m_parents
          >>*search.m_offspring;

    stream.precision(precision);
    return stream;
}
