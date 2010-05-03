//===========================================================================
/*!
 *  \file ChromosomeCMA.h
 *
 *  \brief Elitist CMA evolution strategy
 *
 *  The algorithm is described in
 *
 *  Christian Igel, Thorsten Suttorp, and Nikolaus Hansen. A
 *  Computational Efficient Covariance Matrix Update and a (1+1)-CMA
 *  for Evolution Strategies. Proceedings of the Genetic and
 *  Evolutionary Computation Conference (GECCO 2006), ACM Press, 2006
 *
 *  and
 *
 *  Christian Igel, Nikolaus Hansen, and Stefan Roth. Covariance
 *  Matrix Adaptation for Multi-objective Optimization. Evolutionary
 *  Computation 15(1), pp. 1-28, 2007
 *
 *
 *  \author  Stefan Roth, Christian Igel
 *  \date    2004-11-24
 *
 *  \par Copyright (c) 2005:
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

#ifndef __CHROMOSOME_CMA_H
#define __CHROMOSOME_CMA_H

#include <SharkDefs.h>

#include <EALib/ChromosomeT.h>
#include <EALib/Population.h>
#include <Array/ArrayOp.h>
#include <Array/ArrayIo.h>
#include <LinAlg/LinAlg.h>


//!
//! \brief Elitist CMA evolution strategy
//!
class ChromosomeCMA : public ChromosomeT< double >
{
public:
	ChromosomeCMA()
	{};
	explicit ChromosomeCMA(unsigned l) : ChromosomeT< double >(l)
	{};
	ChromosomeCMA(unsigned l, const double& v): ChromosomeT< double >(l, v)
	{};
	ChromosomeCMA(const std::vector< double >& v) : ChromosomeT< double >(v)
	{};
	~ChromosomeCMA()
	{};

	void init(unsigned             dimension,
			  std::vector<double > stdv,
			  double               initialSigma,
			  ChromosomeT<double> &MinInit,
			  ChromosomeT<double> &MaxInit,
			  int                  noOffspring = 1,
			  double               lower       = 0.)
	{
		unsigned i;

		n     = dimension;
		sigma = initialSigma;

		(*this).resize(n);

		pc.resize(n);
		C.resize(n, n, false);
		B.resize(n, n, false);
		eigenvalues.resize(n);
		z.resize(n);
		lastStep.resize(n);

		initialize(MinInit, MaxInit);

		lambda = noOffspring;
		cc      = 2. / (2. + n);
		ccov    = 2. / (n * n + 6.);

		lambdaSucc = 0.;
		noOffspringSinceUpdate = 0;

		lowerBound = lower;

		// succprob, p_succ
		psPrime = 1. / (5. + sqrt((double)lambda) / 2.);
		//psPrime = 1. / 2. * 1./(5. + (double)lambda/4.);
		ps      = psPrime;

		cp = (psPrime * lambda) / (2 +  psPrime * lambda);

		d = 1 + n / (lambda * 2);

		// init paths
		for (i = 0; i < n; i++) {
			pc(i) = 0.;
			for (j = 0; j < n; j++) {
				if (i != j) C(i, j) = 0;
				else C(i, j) = Shark::sqr(stdv[i]);
			}
		}
		// eigenvalues and eigenvector matrix B
		eigensymm(C, B, eigenvalues);
		needCovarianceUpdate = false;
	}

	void init(unsigned dimension,
			  double   initialSigma,
			  double   MinInit,
			  double   MaxInit,
			  int      lambda = 1)
	{
		std::vector<double> stdv(dimension);
		ChromosomeT<double> chromeMinInit(dimension), chromeMaxInit(dimension);

		for (i = 0; i < dimension; i++) {
			stdv         [i] = 1;
			chromeMinInit[i] = MinInit;
			chromeMaxInit[i] = MaxInit;
		}
		init(dimension, stdv, initialSigma, chromeMinInit, chromeMaxInit, lambda);
	}

	void mutate()
	{
		// draw random vector
		for (i = 0; i < n; i++) z(i) = Rng::gauss(0, 1);
		lastStep = 0;

		// mutate objective variables, Eq. (1)
		for (i = 0; i < n; i++)
			for (j = 0; j < n; j++)
				lastStep(i) += B(i, j) * sqrt(fabs(eigenvalues(j))) * z(j);

		for (i = 0; i < n; i++)
			(*this)[i] += sigma * lastStep(i);

		needCovarianceUpdate = true;
	};

	void updateGlobalStepsize()
	{
		updateGlobalStepsize(lambdaSucc / noOffspringSinceUpdate);
		lambdaSucc = 0.;
		noOffspringSinceUpdate = 0;
	}

	void updateLambdaSucc(bool better)
	{
		noOffspringSinceUpdate++;
		if (better) lambdaSucc++;
	}

	void updateGlobalStepsize(double nsucc)
	{
		// Step size update by accumulated success probability
		ps	  = (1 - cp) * ps + cp * (nsucc / lambda);
		sigma *= exp((ps - (psPrime / (1 - psPrime)) * (1 - ps)) / d);

		if ((sigma * sqrt(fabs(eigenvalues(n - 1)))) < lowerBound)
			sigma = lowerBound / sqrt(fabs(eigenvalues(n - 1)));
	};

	void updateCovariance(Individual& parent)
	{
		// Eq. (2) & Eq. (4)
		for (i = 0; i < n; i++) {
			if (ps < 0.44) pc(i) = (1 - cc) * pc(i)
									   + sqrt(cc * (2 - cc))
									   * ((*this)[i] - (dynamic_cast<std::vector<double > & >(parent[0]))[i]) / sigma;
			else pc(i) = (1 - cc) * pc(i);
		}

		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {
				if (ps >= 0.44) C(i, j) = (1 - ccov) * C(i, j)
											  + ccov * (pc(i) * pc(j) + cc * (2 - cc) * C(i, j));
				else C(i, j) = (1 - ccov) * C(i, j)
								   + ccov * pc(i) * pc(j);
			}
		}

		// eigenvalues eigenvalues and eigenvector matrix B
		eigensymm(C, B, eigenvalues);

		needCovarianceUpdate = false;
	};

	void updateCovariance()
	{
		// Eq. (2) & Eq. (4)
		for (i = 0; i < n; i++) {
			if (ps < 0.44) pc(i) = (1 - cc) * pc(i) + sqrt(cc * (2 - cc)) * lastStep(i);
			else pc(i) = (1 - cc) * pc(i);
		}

		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {
				if (ps >= 0.44) C(i, j) = (1 - ccov) * C(i, j)
											  + ccov * (pc(i) * pc(j) + cc * (2 - cc) * C(i, j));
				else C(i, j) = (1 - ccov) * C(i, j)
								   + ccov * pc(i) * pc(j);
			}
		}

		// eigenvalues eigenvalues and eigenvector matrix B
		eigensymm(C, B, eigenvalues);

		needCovarianceUpdate = false;
	};

	const Array<double> &getC() const
	{
		return C;
	}

	double getSigma() const
	{
		return sigma;
	};

	double getCondition() const
	{
		return eigenvalues(0) / eigenvalues(n - 1);
	}

	void setLower(double lower)
	{
		lowerBound = lower;
	}

	bool covarianceUpdateNeeded()
	{
		return needCovarianceUpdate;
	}

protected:
	Chromosome* clone() const
	{
		return new ChromosomeCMA(*this);
	}
	Chromosome* empty() const
	{
		return new ChromosomeCMA;
	}

	int pvm_pkchrom()
	{
		unsigned  i;
		double*   u;
		unsigned* s;

		this->ChromosomeT<double>::pvm_pkchrom();

		s = new unsigned[2];
		s[0] = n;
		s[1] = lambda;
		pvm_pkuint(s, 2, 1);
		delete[] s;

		u = new double[Shark::min(7u,n*n)];
		u[0] = sigma;
		u[1] = psPrime;
		u[2] = cc;
		u[3] = ccov;
		u[4] = d;
		u[5] = ps;
		u[6] = cp;
		pvm_pkdouble(u, 7, 1);

		for (i = 0; i < n; i++) u[i] = pc(i);
		pvm_pkdouble(u, n, 1);

		for (i = 0; i < n; i++) u[i] = eigenvalues(i);
		pvm_pkdouble(u, n, 1);

		for (i = 0; i < n*n; i++) u[i] = C.elem(i);
		pvm_pkdouble(u, n*n, 1);

		for (i = 0; i < n*n; i++) u[i] = B.elem(i);
		pvm_pkdouble(u, n*n, 1);

		delete[] u;
		return 1;
	};

	int pvm_upkchrom()
	{
		unsigned  i;
		double*   u;
		unsigned* s;

		this->ChromosomeT<double>::pvm_upkchrom();

		s = new unsigned[2];
		pvm_pkuint(s, 2, 1);
		n	    = s[0];
		lambda  = s[1];
		delete[] s;

		if (n != this->size()) throw SHARKEXCEPTION("[ChromosomeCMA::pvm_upkchrom] size missmatch");

		eigenvalues.resize(n);
		z.     resize(n);
		pc.resize(n);
		C.resize(n, n);
		B.resize(n, n);

		u = new double[Shark::min(7u,n*n)];
		pvm_upkdouble(u, 7, 1);
		sigma   = u[0];
		psPrime = u[1];
		cc      = u[2];
		ccov    = u[3];
		d       = u[4];
		ps      = u[5];
		cp      = u[6];

		pvm_upkdouble(u, n, 1);
		for (i = 0; i < n; i++) pc(i) = u[i];

		pvm_upkdouble(u, n, 1);
		for (i = 0; i < n; i++) eigenvalues(i) = u[i];

		pvm_upkdouble(u, n*n, 1);
		for (i = 0; i < n*n; i++) C.elem(i) = u[i];

		pvm_upkdouble(u, n*n, 1);
		for (i = 0; i < n*n; i++) B.elem(i) = u[i];

		delete[] u;
		return 1;
	};

private:
	unsigned n;		// Dimension // problem dimension / chromosome length
	unsigned lambda;      //	     // size of offspring population
	unsigned i, j, k;	//	     // iterators
	double sigma;		// delta     // the current global stepsize
	double psPrime;	//	     // target success probability; strategy constant for global stepsize update
	double cc;		// c	     // strategy constant for cma culmulation update
	double ccov;		//	     //	strategy constant for cma update
	double d;
	double cp;
	double ps;		// avsuccess // global stepsize culmulation path
	Array<double> pc;	// s	     // evo cumulation path
	Array2D<double> C;	//	     // the current covariance matrix
	Array2D<double> B;	//	     // eigenvector matrix C for offspring sampling
	Array<double> z;      // standard normally distributed mutation vector
	Array<double> eigenvalues;	     // eigenvalue vector of C for offspring sampling
	Array<double> lastStep;      //

	unsigned      noOffspringSinceUpdate;
	double        lambdaSucc;
	double        lowerBound;
	bool needCovarianceUpdate;
};
#endif


