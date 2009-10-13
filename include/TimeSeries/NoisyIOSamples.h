//===========================================================================
/*!
 *  \file NoisyIOSamples.h
 *
 *
 *  \author  Martin Kreutz
 *  \date    21.09.1998
 *
 *  \par Copyright (c) 1998-2003:
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
 *      TestData
 *
 *
 *  <BR>
 *
 *
 *  <BR><HR>
 *  This file is part of TestData. This library is free software;
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


#ifndef __cplusplus
#error Must use C++.
#endif

#ifndef __NOISYIOSAMPLES_H
#define __NOISYIOSAMPLES_H

#ifdef __GNUC__
#pragma interface
#endif

#include <Rng/Bernoulli.h>
#include <Rng/Normal.h>
#include <TimeSeries/IOSamples.h>

//===========================================================================

class NoisyIOSamples : public IOSamples< double >
{
public:
	NoisyIOSamples(const Generator< double >& gen,
				   unsigned idim,
				   unsigned ilag,
				   unsigned odim,
				   unsigned olag,
				   unsigned bufsize = 100);

	NoisyIOSamples(const Generator< double >& gen,
				   const Array< unsigned >& ilag,
				   const Array< unsigned >& olag,
				   unsigned bufsize = 100);

	void reset();
	void operator()(Array< double >& in, Array< double >& out);

	void operator()(Array< double >& in,    Array< double >& out,
					Array< double >& invar, Array< double >& outvar);

	void setMissingProb(double p)
	{
		coinToss.prob(p);
	}
	void setSNR(double s)
	{
		snr     = s;
	}
	void setWindowSize(double w)
	{
		winSize = w;
	}
	void setVariance(double v)
	{
		gauss.variance(v);
	}

protected:
	double snr;
	double winSize;
	double estVar;
	Array< double > inMean, outMean;
	Array< double > estInVar, estOutVar;

	Bernoulli coinToss;
	Normal    gauss;

	IOGenerator< Array< double > > * clone() const;
};

//===========================================================================

#endif /* !__NOISYIOSAMPLES_H */

