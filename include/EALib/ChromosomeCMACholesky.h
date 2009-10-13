//===========================================================================
/*!
 *  \file ChromosomeCMACholesky.h
 *
 *  \brief Elitist CMA evolution strategy with "Cholesky Update" 
 * 
 *  The algorithm is described in
 *
 *  Christian Igel, Thorsten Suttorp, and Nikolaus Hansen. A
 *  Computational Efficient Covariance Matrix Update and a (1+1)-CMA
 *  for Evolution Strategies. Proceedings of the Genetic and
 *  Evolutionary Computation Conference (GECCO 2006), ACM Press, 2006
 *
 *  \author  Christian Igel
 *  \date    2005
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
 *  \par Project:
 *      EALib
 *  <BR>
 *
 *
 *  <BR><HR>
 *  This file is part of EALib. This library is free software;
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

#ifndef __CHROMOSOMECMACHOLESKY_H
#define __CHROMOSOMECMACHOLESKY_H

#include <SharkDefs.h>
#include <EALib/ChromosomeT.h>
#include <EALib/Population.h>
#include <Array/ArrayOp.h>
#include <Array/ArrayIo.h>
#include <LinAlg/LinAlg.h>


//!
//! \brief Elitist CMA evolution strategy with "Cholesky Update"
//!
class ChromosomeCMACholesky : public ChromosomeT< double >
{
public:
	ChromosomeCMACholesky()
	{}
	explicit ChromosomeCMACholesky(unsigned l) : ChromosomeT< double >(l)
	{}
	ChromosomeCMACholesky(unsigned l, const double& v): ChromosomeT< double >(l, v)
	{}
	ChromosomeCMACholesky(const std::vector< double >& v) : ChromosomeT< double >(v)
	{}
	~ChromosomeCMACholesky()
	{};

	void init(unsigned             dimension,
			  std::vector<double > stdv,
			  double               initialSigma,
			  ChromosomeT<double> &MinInit,
			  ChromosomeT<double> &MaxInit,
			  int                  noOffspring = 1)
	{
		unsigned i;

		n     = dimension;
		sigma = initialSigma;

		(*this).resize(n);
		pc.resize(n);
		B.resize(n, n);
		z.resize(n);

		initialize(MinInit, MaxInit);

		lambda = noOffspring;
		ccov    = 2. / (n * n + 6.);


		// succprob, p_succ
		psPrime = 1. / (5. + sqrt((double)lambda) / 2.);
		//psPrime = 1. / 2. * 1./(5. + (double)lambda/4.);
		ps      = psPrime;

		cp = (psPrime * lambda) / (2 +  psPrime * lambda);

		d = 1 + n / (lambda * 2);

		lambdaSucc = 0.;
		noOffspringSinceUpdate = 0;

		// eigenvalues and eigenvector matrix B

		B = 0.;

		for (i = 0; i < n; i++) B(i, i) = stdv[i];
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

	void init(unsigned dimension,
			  std::vector<double > &stdv,
			  double   initialSigma,
			  double   MinInit,
			  double   MaxInit,
			  int      lambda = 1)
	{
		ChromosomeT<double> chromeMinInit(dimension), chromeMaxInit(dimension);

		for (i = 0; i < dimension; i++) {
			chromeMinInit[i] = MinInit;
			chromeMaxInit[i] = MaxInit;
		}
		init(dimension, stdv, initialSigma, chromeMinInit, chromeMaxInit, lambda);
	}

	void mutate()
	{
		// draw random vector
		for (i = 0; i < n; i++) z(i) = Rng::gauss(0, 1);

		// mutate objective variables, Eq. (1)
		for (i = 0; i < n; i++)
			for (j = 0; j < n; j++)
				(*this)[i] += sigma * B(i, j)  * z(j);

		needCovarianceUpdate = true;
	};

	void updateGlobalStepsize(double nsucc)
	{
		// Step size update by accumulated success probability
		ps	  = (1 - cp) * ps + cp * (nsucc / lambda);
		sigma *= exp((ps - (psPrime / (1 - psPrime)) * (1 - ps)) / d);
	};

	void updateCovariance()
	{
		// Eq. (2) & Eq. (4)
		if (ps < 0.44) 
			rankOneUpdate(B, z, 1 - ccov, ccov);

		needCovarianceUpdate = false;
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

	bool covarianceUpdateNeeded()
	{
		return needCovarianceUpdate;
	}

	const Array<double> &getB()
	{
		return B;
	}

	double getSigma()
	{
		return sigma;
	};

	double getCondition()
	{
		return 1;
	}

	double getPath()
	{
		return ps;
	};


protected:
	void rankOneUpdate(Array<double> &A, const Array<double> &v, double alpha, double beta)
	{
		Array<double> H;
		double s, wNormSqr;
		wNormSqr = scalarProduct(v, v) * beta / alpha;
		s = -1 / wNormSqr * (1 - sqrt(1 + wNormSqr));
		H = innerProduct(A, v);
		H = outerProduct(H, v);
		A = sqrt(alpha) * A + s * H * (beta / sqrt(alpha));
	}

	Chromosome* clone() const
	{
		return new ChromosomeCMACholesky(*this);
	}
	Chromosome* empty() const
	{
		return new ChromosomeCMACholesky;
	}

private:
	unsigned n;		// Dimension // problem dimension / chromosome length
	unsigned lambda;      //	     // size of offspring population
	unsigned i, j, k;	//	     // iterators
	double sigma;		// delta     // the current global stepsize
	double psPrime;	//	     // target success probability; strategy constant for global stepsize update
	double ccov;		//	     //	strategy constant for cma update
	double d;
	double cp;
	double ps;		// avsuccess // global stepsize culmulation path
	Array<double> pc;	// s	     // evo cumulation path
	Array2D<double> B;	//	     // decomposition of covariance matrix
	Array<double> z;      // standard normally distributed mutation vector

	unsigned      noOffspringSinceUpdate;
	double        lambdaSucc;
	bool needCovarianceUpdate;
};
#endif


