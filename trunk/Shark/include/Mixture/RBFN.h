//===========================================================================
/*!
 *  \file RBFN.h
 *
 *  \brief Radial Basis Function Network
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

#ifndef __RBFN_H
#define __RBFN_H

#include "Mixture/MixtureOfGaussians.h"


//! \brief Radial Basis Function Network
class RBFN : public MixtureOfGaussians
{
public:
	RBFN(unsigned nInputs  = 0,
		 unsigned nOutputs = 0,
		 unsigned nCenters = 0);

	double& weight(unsigned i, unsigned j)
	{
		return A(i, j);
	}
	double& bias(unsigned i)
	{
		return b(i);
	}
	double  weight(unsigned i, unsigned j) const
	{
		return A(i, j);
	}
	double  bias(unsigned i) const
	{
		return b(i);
	}

	const Array< double >& weight() const
	{
		return A;
	}
	const Array< double >& bias() const
	{
		return b;
	}

	void setParams(const Array< double >& w);
	void getParams(Array< double >& w) const;

	unsigned odim() const
	{
		return b.nelem();
	}

	void initialize(const Array< double >& x)
	{
		MixtureOfGaussians::initialize(x);
	}

	void initialize(const Array< double >& x,
					const Array< double >& y);

	void initialize_linear(double min = -1, double max = 1);

	double curvature();

	void   insertRBFData(const Array< double >& input,
						 const Array< double >& output,
						 const Array< double >& minInput,
						 const Array< double >& maxInput);

	void   insertRBF(const Array< double >&,
					 const Array< double >&,
					 const Array< double >&);
	void   deleteRBF(unsigned);

	void   resize(unsigned, bool = false);
	void   resize(unsigned, unsigned, unsigned, bool = false);

	void   train_linear(const Array< double >&,
						const Array< double >&);

	/*
	void   train_ran     ( const Array< double >&,
	  const Array< double >&,
	  double epsilon, double delta );
	*/

	void   recall(const Array< double >&, Array< double >&) const;

	void   estimateFisherInformation(const Array< double >& input,
									 const Array< double >& output,
									 Array< double >& A);

	void   estimateInvFisher(const Array< double >& input,
							 const Array< double >& output,
							 Array< double >& invA,
							 Array< double >& transInvA,
							 double &S2);

	double estimateVariance(const Array< double >& input,
							const Array< double >& invA);

	double estimateVarianceChange(const Array< double >& input,
								  const Array< double >& invA,
								  const Array< double >& transInvA,
								  double S2);

	double overallVariance(const Array< double >& input,
						   const Array< double >& output);

protected:
	Array< double > A;
	Array< double > b;

	double  gradientMSE(const Array< double >& in,
						const Array< double >& out,
						Array< double >& de);
	void    gradientOut(const Array< double >& in,
						Array< double >& dw);

	virtual void firstLayer(const Array< double >&,
							Array< double >&) const;

	void   gradientCurve(Array< double >& de);
};

#endif /* !__RBFN_H */

