//===========================================================================
/*!
 *  \file KernelDensity.h
 *
 *  \brief Class for kernel density estimators
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

#ifndef __KERNELDENSITY_H
#define __KERNELDENSITY_H

#include "Mixture/MixtureOfGaussians.h"

//===========================================================================
/*!
 *  \brief A container class for a kernel density estimator.
 *
 *  This class implements a kernel density estimator with Gaussian kernels.
 *  The class is derived from MixtureOfGaussians since kernel density
 *  estimators can be viewed as special mixture models with identical models
 *  and one model centered on each data point.
 *
 *  \author  M. Kreutz
 *  \date    1995-01-01
 *
 *  \par Changes:
 *      none
 *
 *  \par Status:
 *      stable
 */
class KernelDensity : public MixtureOfGaussians
{
public:
	//! Constructs an empty kernel density estimator with a fixed variance.
	/*
	 *  If the given value for the variance is zero the global variance is
	 *  marked as uninitialized and is estimated when data is assigned to
	 *  the model.
	 */
	KernelDensity(double va = 0)
			: var(va)
	{}

	//! Constructs a kernel density estimators with given data points.
	/*
	 *  The centers of the kernels are set to the given data points
	 *  (as in member function set). If a non-zero value for the variance
	 *  is provided the global variance is set to this value, otherwise
	 *  it is estimated on the basis of the given data points (as in
	 *  member function setVar and estimateVariance).
	 */
	KernelDensity
	(
		const Array< double >& x,
		double                 va   = 0
	)
			: MixtureOfGaussians(x.dim(0), x.dim(1)), var(va)
	{
		a = 1. / size();
		m = x;

		if (var == 0) {
			estimateVariance();
		}

		v = var;
	}

	//! Sets the global variance of all kernels.
	/*
	 *  If a non-zero value for the variance is provided the global
	 *  variance is set to this value, otherwise it is estimated on the
	 *  basis of the given data points (stored in the centers of the kernels).
	 *
	 *  \sa estimateVariance
	 */
	void setVar(double va)
	{
		if (va == 0 && size() > 0) {
			estimateVariance();
		}
		else {
			var = va;
		}

		v = var;
	}

	//! Returns the global variance.
	double variance() const
	{
		return var;
	}

	//! Sets the centers of the kernels.
	/*
	 *  In the first step the number and dimension of the centers of the
	 *  kernels is adjusted according to the given data. Then, a mixture
	 *  model (implemented in the base class) is initialized with
	 *  equi-probable priors and Gaussian models centered at each data
	 *  point. Finally, the global variance of all models is estimated if
	 *  was not set before to a non-zero value.
	 */
	void set(const Array< double >& x)
	{
		resize(x.dim(0), x.dim(1));

		a = 1. / size();
		m = x;

		if (var == 0) {
			estimateVariance();
		}

		v = var;
	}

	//! Creates a pseudo random sample from the estimated distribution.
	Array< double > sample();

	//! Computes the value of the estimated pdf for a given data point.
	double p(const Array< double >& pat) const;

protected:
	//! Estimates the global variance on the basis of the given data.
	void estimateVariance();

private:
	//! Contains the value of the global variance
	double var;
};

#endif /* !__KERNELDENSITY_H */

