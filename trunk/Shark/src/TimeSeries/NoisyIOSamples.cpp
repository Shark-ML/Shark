//===========================================================================
/*!
 *  \file NoisyIOSamples.cpp
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
 *
 *
 */
//===========================================================================





#ifdef __GNUC__
#pragma implementation
#endif


#include <cfloat>
#include <SharkDefs.h>
#include <Array/ArrayOp.h>
#include <TimeSeries/NoisyIOSamples.h>


NoisyIOSamples::NoisyIOSamples(const Generator< double >& gen,
							   unsigned idim,
							   unsigned ilag,
							   unsigned odim,
							   unsigned olag,
							   unsigned bufsize)
		: IOSamples< double >(gen, idim, ilag, odim, olag, bufsize),
		snr(DBL_MAX),
		winSize(1000),
		coinToss(0),
		gauss(0, 0)
{}

NoisyIOSamples::NoisyIOSamples(const Generator< double >& gen,
							   const Array< unsigned >& ilag,
							   const Array< unsigned >& olag,
							   unsigned bufsize)
		: IOSamples< double >(gen, ilag, olag, bufsize)
{}

void NoisyIOSamples::reset()
{
	IOSamples< double >::reset();
	estVar = 0;
}

void NoisyIOSamples::operator()(Array< double >& in,
								Array< double >& out)
{
	Array< double > invar, outvar;
	operator()(in, out, invar, outvar);
}

void NoisyIOSamples::operator()(Array< double >& in,
								Array< double >& out,
								Array< double >& invar,
								Array< double >& outvar)
{
	unsigned i;

	IOSamples< double >::operator()(in, out);

	if (! in .samedim(inMean) ||
			! in .samedim(estInVar) ||
			! out.samedim(outMean) ||
			! out.samedim(estOutVar)) {
		inMean    = in;
		outMean   = out;
		estInVar  = 0;
		estOutVar = 0;
	}
	else {
		inMean    += (in  - inMean) / double(winSize);
		outMean   += (out - outMean) / double(winSize);
		estInVar  += (Shark::sqr(in  - inMean) - estInVar) / double(winSize);
		estOutVar += (Shark::sqr(out - outMean) - estOutVar) / double(winSize);
	}

	invar .resize(in, false);
	outvar.resize(out, false);

	if (snr < DBL_MAX) {}
	else if (gauss.variance() > 0) {
		invar  = gauss.variance();
		outvar = gauss.variance();

		for (i = 0; i < in.nelem(); ++i)
			in.elem(i) += gauss();

		for (i = 0; i < out.nelem(); ++i)
			out.elem(i) += gauss();
	}

	if (coinToss.prob() > 0) {
		for (i = 0; i < in.nelem(); ++i)
			if (coinToss()) {
				in   .elem(i) = inMean.elem(i);
				invar.elem(i) = DBL_MAX;
			}

		for (i = 0; i < out.nelem(); ++i)
			if (coinToss()) {
				out   .elem(i) = outMean.elem(i);
				outvar.elem(i) = DBL_MAX;
			}
	}
}

IOGenerator< Array< double > > * NoisyIOSamples::clone() const
{
	return new NoisyIOSamples(*this);
}

