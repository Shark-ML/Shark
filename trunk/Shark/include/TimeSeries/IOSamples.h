//===========================================================================
/*!
 *  \file IOSamples.h
 *
 *
 *  \author  Martin Kreutz
 *  \date    16.09.1998
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
 *   <BR>
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

#ifndef __IOSAMPLES_H
#define __IOSAMPLES_H

#include <Array/ArrayOp.h>
#include <TimeSeries/Generator.h>
#include <TimeSeries/IOGenerator.h>

//===========================================================================

template < class T >
class IOSamples : public IOGenerator< Array< T > >
{
public:
	IOSamples(const Generator< T >& gen,
			  unsigned idim,
			  unsigned ilag,
			  unsigned odim,
			  unsigned olag,
			  unsigned bufsize = 200)
			: generator(gen.clone()),
			buf(bufsize),
			step(0)
	{
		setEmbedding(idim, ilag, odim, olag);
	}

	IOSamples(const Generator< T >& gen,
			  const Array< unsigned >& ilag,
			  const Array< unsigned >& olag,
			  unsigned bufsize = 100)
			: generator(gen.clone()),
			buf(bufsize),
			step(0)
	{
		setEmbedding(ilag, olag);
	}

	IOSamples(const IOSamples< T >& io)
			: generator(io.generator->clone()),
			inLag(io.inLag),
			outLag(io.outLag),
			buf(io.buf),
			step(io.step)
	{}

	~IOSamples()
	{
		delete generator;
	}

	unsigned idim() const
	{
		return inLag.nelem() + 1;
	}

	unsigned odim() const
	{
		return outLag.nelem();
	}

	void setEmbedding(unsigned id,
					  unsigned ilag,
					  unsigned od,
					  unsigned olag)
	{
		unsigned i, l;

		inLag .resize(id > 0 ? id - 1 : 0);
		outLag.resize(od);

		for (l = 0, i = inLag.nelem(); i--; inLag(i) = (l += ilag));
		for (l = i = 0; i < outLag.nelem(); outLag(i++) = (l += olag));

		readahead();
	}

	void setEmbedding(const Array< unsigned >& ilag,
					  const Array< unsigned >& olag)
	{
		inLag  = ilag;
		outLag = olag;
		readahead();
	}

	void readahead()
	{
		unsigned d = 0;

		omax = outLag.nelem() > 0 ? maxElement(outLag) : 0;

		if (inLag.nelem() > 0)
			d += maxElement(inLag);
		d += omax;

		RANGE_CHECK(d <= buf.nelem())
		while (step < d)
			buf(step++ % buf.nelem()) = (*generator)();
	}

	void reset()
	{
		generator->reset();
		step = 0;
	}

	void operator()(Array< T >& in, Array< T >& out)
	{
		unsigned j;

		in .resize(idim());
		out.resize(odim());

		buf(step % buf.nelem()) = (*generator)();

		for (j = 0; j < inLag.nelem(); ++j)
			in(j) = buf((step - omax - inLag(j)) % buf.nelem());
		in(j) = buf((step - omax) % buf.nelem());

		for (j = 0; j < outLag.nelem(); ++j)
			out(j) = buf((step - omax + outLag(j)) % buf.nelem());

		step++;
	}

protected:
	Generator< T >* generator;
	Array< unsigned > inLag;
	Array< unsigned > outLag;
	Array< T > buf;
	unsigned step;
	unsigned omax;

	IOGenerator< Array< T > > * clone() const
	{
		return new IOSamples< T >(*this);
	}
};

//===========================================================================

#endif /* !__IOSAMPLES_H */

