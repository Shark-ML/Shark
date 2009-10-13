//===========================================================================
/*!
 *  \file Embedding.h
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

#ifndef __EMBEDDING_H
#define __EMBEDDING_H

#include <Array/ArrayOp.h>
#include <TimeSeries/Generator.h>


//===========================================================================

template < class T >
class Embedding : public Generator< Array< T > >
{
public:
	Embedding(const Generator< T >& gen,
			  unsigned d,
			  unsigned l,
			  unsigned bufsize = 100)
			: generator(gen.clone()),
			buf(bufsize),
			step(0)
	{
		setEmbedding(d, l);
	}

	Embedding(const Generator< T >& gen,
			  const Array< unsigned >& lagvec,
			  unsigned bufsize = 100)
			: generator(gen.clone()),
			buf(bufsize),
			step(0)
	{
		setEmbedding(lagvec);
	}

	Embedding(const Embedding< T >& embed)
			: generator(embed.generator->clone()),
			lag(embed.lag),
			buf(embed.buf),
			step(embed.step)
	{}

	~Embedding()
	{
		delete generator;
	}

	unsigned dim() const
	{
		return lag.nelem() + 1;
	}

	void setDim(unsigned d)
	{
		setEmbedding(d, 1);
	}

	void setLag(unsigned l)
	{
		setEmbedding(dim(), l);
	}

	void setEmbedding(unsigned d, unsigned l)
	{
		lag.resize(d > 0 ? d - 1 : 0);
		for (unsigned j = 0, i = lag.nelem(); i--; lag(i) = (j += l));
		readahead();
	}

	void setEmbedding(const Array< unsigned >& lagvec)
	{
		lag = lagvec;
		readahead();
	}

	void readahead()
	{
		if (lag.nelem() > 0)
			while (step < maxElement(lag))
				buf(step++ % buf.nelem()) = (*generator)();
	}

	void reset()
	{
		generator->reset();
		step = 0;
	}

	Array< T > operator()()
	{
		unsigned j;
		Array< T > v(dim());

		buf(step % buf.nelem()) = (*generator)();

		for (j = 0; j < lag.nelem(); ++j)
			v(j) = buf((step - lag(j)) % buf.nelem());
		v(j) = buf(step % buf.nelem());

		step++;

		return v;
	}

protected:
	Generator< T >* generator;
	Array< unsigned > lag;
	Array< T > buf;
	unsigned step;

	Generator< Array< T > > * clone() const
	{
		return new Embedding< T >(*this);
	}
};

//===========================================================================

#endif /* __EMBEDDING_H */

