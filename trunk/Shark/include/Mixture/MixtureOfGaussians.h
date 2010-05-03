// !!! maxIter und precision tauschen
//===========================================================================
/*!
 *  \file MixtureOfGaussians.h
 *
 *  \brief Core class implementing a sum of Gaussians
 *
 *  \author  Martin Kreutz
 *  \date    1995-01-01
 *
 *  \par Copyright (c) 1995,2002:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
 *
 *  \par Project:
 *      Mixture
 *
 *
 *  <BR>
 *
 *
 *  <BR><HR>
 *  This file is part of Mixture. This library is free software;
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

#ifndef __MIXTUREOFGAUSSIANS_H
#define __MIXTUREOFGAUSSIANS_H

#include "Rng/Uniform.h"
#include "Rng/Normal.h"
#include "Mixture/CodeBook.h"
#include "Mixture/MixtureModel.h"


//! \brief Core class implementing a sum of Gaussians
class MixtureOfGaussians : /*virtual*/ public MixtureModel< double >,
			/*virtual*/ /*protected*/ public CodeBook
{
public:
	bool isfinite() const;

public:
	static void join
	(
		const Array< double >& x,
		const Array< double >& y,
		Array< double >& z
	);

protected:
	void firstLayer
	(
		const Array< double >& x,
		Array< double >& y
	) const;

	/*
	virtual void gradientMSE( ); // overloaded in RBFN and RBFN_PTO
	void gradientLL ( );
	virtual void gradientCurve( ); // overloaded in RBFN
	*/

	double deficientP(const Array< double >&,
					  const Array< double >&,
					  unsigned,
					  double = 1e+30) const;

	double deficientP(const Array< double >&,
					  const Array< double >&,
					  double = 1e+30) const;

	friend class RBFN;
	friend class RBFN_PTO;

	//private:
public:
	//
	// needed for curvature
	//
	double A(unsigned r) const;
	double B(unsigned r, unsigned t) const;
	double E(unsigned i, unsigned j) const;
	double U(unsigned i, unsigned j, unsigned r) const;
	double V(unsigned i, unsigned j, unsigned r, unsigned t) const;

	double Delta1(unsigned s, unsigned j, unsigned q) const;
	double Delta2(unsigned s, unsigned j, unsigned q) const;
	double Delta3(unsigned s, unsigned j, unsigned q, unsigned k) const;
	double Delta4(unsigned s, unsigned j, unsigned q, unsigned k) const;

	double dAda(unsigned r, unsigned s) const;
	double dAdb(unsigned r, unsigned s) const;
	double dAdm(unsigned r, unsigned s, unsigned q) const;
	double dAds(unsigned r, unsigned s, unsigned q) const;

	double dBda(unsigned r, unsigned t, unsigned s) const;
	double dBdb(unsigned r, unsigned t, unsigned s) const;
	double dBdm(unsigned r, unsigned t, unsigned s, unsigned q) const;
	double dBds(unsigned r, unsigned t, unsigned s, unsigned q) const;

	void  dCurve(Array< double >& db,
				 Array< double >& dm,
				 Array< double >& ds) const;

	double q(const Array< double >& x,
			 const MixtureOfGaussians& old) const;

	void  dq(const Array< double >& x,
			 const MixtureOfGaussians& old,
			 Array< double >& da,
			 Array< double >& dm,
			 Array< double >& ds) const;

	//protected:
public:
	Array< double > v;
	Normal          gauss;

	void  dCurveams(Array< double >& da,
					Array< double >& dm,
					Array< double >& ds) const;

	double   overlap(unsigned i, unsigned j) const;
	unsigned maxOverlap(unsigned i) const;

	double   volume(unsigned i) const;
	unsigned minVolume() const;
	unsigned maxVolume() const;

	void     splitDataset(const Array< double >& x,
						  std::vector< Array< double > > & z);

	void     merge(unsigned i, unsigned j,
				   std::vector< Array< double > > & z);
	void     split(unsigned i,
				   std::vector< Array< double > > & z);

	void     splitBroadest(const Array< double >&);
	void     splitRandom(const Array< double >&);

	void     mergeSmallest(const Array< double >&);
	void     mergeRandom(const Array< double >&);

	void     splitBroadest(const Array< double >&, const Array< double >&);
	void     splitRandom(const Array< double >&, const Array< double >&);

	void     mergeSmallest(const Array< double >&, const Array< double >&);
	void     mergeRandom(const Array< double >&, const Array< double >&);

	double   p(const Array< double >&, unsigned) const;

	double p(const Array< double >& x) const
	{
		return MixtureModel< double >::p(x);
	}

public:
	//! Returns the dimension of the model
	CodeBook::dim;

	//! Contructs a mixture model with given number of kernels
	MixtureOfGaussians
	(
		unsigned numA  = 0,
		unsigned dimA  = 0
	);

	virtual ~MixtureOfGaussians()
	{ }

	//! Contructs a mixture model and initializes it with given parameter vectors
	MixtureOfGaussians
	(
		const Array< double >& alphaA,
		const Array< double >& muA,
		const Array< double >& sigmaSqrA
	);

	double& mean(unsigned i, unsigned j)
	{
		return m(i, j);
	}
	double& var(unsigned i, unsigned j)
	{
		return v(i, j);
	}
	double  mean(unsigned i, unsigned j) const
	{
		return m(i, j);
	}
	double  var(unsigned i, unsigned j) const
	{
		return v(i, j);
	}

	//! Returns the array of mean vectors (read-only)
	const Array< double >& mean() const
	{
		return m;
	}

	//! Returns the array of variance vectors (read-only)
	const Array< double >& var() const
	{
		return v;
	}

	//! Returns the array of mean vectors
	Array< double >& mean()
	{
		return m;
	}

	//! Returns the array of variance vectors
	Array< double >& var()
	{
		return v;
	}

	//! Changes the number of Gaussian kernels
	void resize
	(
		unsigned	numA,
		bool		copyA = false
	);

	//! Changes the number and dimensionality of Gaussian kernels
	void resize
	(
		unsigned	numA,
		unsigned	dimA,
		bool		copyA = false
	);

	//! Appends all kernels of a given mixture model to the current model
	void append(const MixtureOfGaussians&, bool = true);

	//! Inserts a new kernel initialized with given parameter vectors
	void insertKernel
	(
		double,
		const Array< double >&,
		const Array< double >&,
		bool norm = true
	);

	void            insertKernel(const Array< double >&,
								 const Array< double >&, bool norm = true);
	void            insertKernel(bool = true);
	void            deleteKernel(unsigned, bool = true);
	void            deleteKernel(unsigned, unsigned, bool = true);

	void            deleteInput(unsigned i);

	unsigned        removeDuplicates(double eps  = 1e-6,  bool norm = true);
	unsigned        removeMinVar(double minv = 1e-30, bool norm = true);

	//! Returns the number of Gaussian kernels
	unsigned size() const
	{
		return MixtureModel< double >::size();
	}

	//
	// initialization
	//
	void initialize
	(
		const Array< double >& x
	);

	virtual void initialize
	(
		const Array< double >& x,
		const Array< double >& y
	);

	void estimateRadii();

	void kmc
	(
		const Array< double >& x,
		double prec      = 1e-6,
		unsigned maxiter = 0
	);

	void kmc
	(
		const Array< double >& x,
		const Array< double >& y,
		double prec      = 1e-6,
		unsigned maxiter = 0
	);

	//
	// learning rules
	//
	bool em
	(
		const Array< double >& x,
		double   prec    = 1e-6,
		unsigned maxiter = 1000,
		double   minvar  = 1e-30,
		double   epsilon = 1,
		double   beta    = 1
	);

	void noisy_em
	(
		const Array< double >& x,
		const Array< double >& vx,
		unsigned numSamples = 1000,
		double   prec    = 1e-6,
		unsigned maxiter = 1000,
		double   minvar  = 1e-30,
		double   epsilon = 1,
		double   beta    = 1
	);

	void em_mask
	(
		const Array< double >& x,
		const Array< bool >& amask,
		const Array< bool >& mmask,
		const Array< bool >& vmask,
		double   prec    = 1e-6,
		unsigned maxiter = 1000,
		double   minvar  = 1e-30
	);

	void jack_knife_em
	(
		const Array< double >& x,
		double   prec    = 1e-6,
		unsigned maxiter = 1000,
		double   minvar  = 1e-30,
		double   beta    = 1
	);

	void stochastic_em
	(
		const Array< double >& x,
		double   prec    = 1e-6,
		unsigned maxiter = 1000,
		double   minvar  = 1e-30,
		double   epsilon = 1
	);

	void annealing_em
	(
		const Array< double >& x,
		double   prec    = 1e-6,
		unsigned maxiter = 1000,
		double   minvar  = 1e-30,
		double   epsilon = 1,
		double   minBeta = 0.5,
		double   incBeta = 1.2
	);

	void em_deficient
	(
		const Array< double >& x,
		const Array< double >& xvar,
		double   prec    = 1e-6,
		unsigned maxiter = 1000,
		double   minvar  = 1e-30,
		double   maxvar  = 1e+30
	);

	void bsom_update
	(
		const Array< double >& x,
		double alpha,
		double beta
	);

	void bsom
	(
		const Array< double >& x,
		double   alphai = 0.5,
		double   alphaf = 0.005,
		double   betai  = 1,
		double   betaf  = 1,
		unsigned tmax   = 10000
	);

	//
	// routines for structure optimization
	//
	void            crossover(MixtureOfGaussians& mate, bool = true);

	void            deleteSmallest(bool = true);
	void            deleteRandom(bool = true);
	void            insertRandom(bool = true);

	double          entropy2();
	double          Renyi() const;
	double          kernelEntropy(const Array< double >&) const;
	double          kernelEntropy(const Array< double >&,
								  const Array< double >&) const;
	virtual double  curvature() const;

	double          sqrDistance(const MixtureOfGaussians& mix) const;

	Array< double > operator()();

	//
	// prediction
	//
	Array< double > max() const;
	MixtureOfGaussians condDensity(const Array< double >& x) const;
	MixtureOfGaussians marginalDensity(const Array< unsigned >& idx) const;
	MixtureOfGaussians marginalDensity(unsigned) const;
	MixtureOfGaussians marginalDensity(unsigned, unsigned) const;
	MixtureOfGaussians marginalDensity(unsigned, unsigned, unsigned) const;

	//! Computes the mean squared error of the regression based on the mixture model
	double mse
	(
		const Array< double >& x,
		const Array< double >& y
	) const;

	//! Computes the regression based on the mixture model
	virtual void recall
	(
		const Array< double >&,
		Array< double >&
	) const;

	//
	// conditional expectation and variance
	//

	//! Computes the conditional expectation
	Array< double > condExpectation
	(
		const Array< double >& x
	) const;

	//! Computes the conditional expectation under a linear transformation
	Array< double > condExpectation
	(
		const Array< double >& x,
		const Array< double >& A,
		const Array< double >& b
	) const;

	//! Computes the conditional variance
	Array< double > condVariance
	(
		const Array< double >& x
	) const;

	//! Computes the overall expectation of the model
	Array< double > overallExpectation() const;

	//! Computes the overall variance of the model
	Array< double > overallVariance() const;

	double RenyiLikelihood(const Array< double >&) const;

	double RenyiLikelihood(const Array< double >&,
						   const Array< double >&) const;

	double jointLogLikelihood(const Array< double >&,
							  const Array< double >&) const;

	double condLogLikelihood(const Array< double >&,
							 const Array< double >&) const;

	double deficientLogLikelihood(const Array< double >&,
								  const Array< double >&,
								  double = 1e+30) const;

	bool operator == (const MixtureOfGaussians&) const;

	// automatically generated
	//bool operator != ( const MixtureOfGaussians& ) const;

	//
	// dummy operator (for template ChromosomeT)
	//
	bool operator < (const MixtureOfGaussians&) const
	{
		return false;
	}

	//! Writes the parameters of the mixture model to an output stream
	void writeMathematica(std::ostream& os) const;
};

#endif /* !__MIXTUREOFGAUSSIANS_H */

