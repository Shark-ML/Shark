//===========================================================================
/*!
 *  \file CodeBook.h
 *
 *  \brief Class for storing and generation code book vectors.
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

#ifndef __CODEBOOK_H
#define __CODEBOOK_H

#include "Array/Array.h"
#include "Rng/Uniform.h"

//===========================================================================
/*!
 *  \brief A container class for code book vectors.
 *
 *  This class implements a container for code book vectors which may be
 *  used e.g. for vector quantization. This class serves also as a base
 *  class for mixture models and kernel density estimators which use the
 *  code book vectors as centers for the models and kernels, respectively.
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
class CodeBook
{
public:
	//=======================================================================
	//
	// constructor and destructor
	//

	//! Contructs an object with an empty set of code book vectors.
	/*
	 *  In order to assign vectors to this object, the member function
	 *  resize has to be called.
	 *  Note that the dimension of the code book vectors (not the number of
	 *  vectors) is automatically adjusted if the function initialized
	 *  is called.
	 */
	CodeBook()
	{}

	//! Contructs an object with a given number and dimension of vectors.
	/*
	 *  All elements of the vectors are initialized with uniformly
	 *  distributed pseudo random numbers in order to assign valid values
	 *  to them. If this is not wanted (e.g. for efficiency reasons) an
	 *  empty object may be constructed and resized after creation.
	 *  The function resize never initializes new elements.
	 */
	CodeBook(unsigned num, unsigned dim);

	//! Destructs the current object
	virtual ~CodeBook()
	{}

	//=======================================================================
	//
	// find the best match in the code book
	//

	//! Finds the index of the best matching vector in the code book.
	unsigned nearest
	(
		const Array< double >& x
	) const;

	//! Finds the best matching vector in the code book.
	void nearest
	(
		const Array< double >& x,
		Array< double >& y
	) const;

	//=======================================================================
	//
	// return size and dimension
	//

	//! Returns the number of code book vectors.
	unsigned size() const
	{
		return m.ndim() == 2 ? m.dim(0) : 0;
	}

	//! Returns the dimension of the code book vectors.
	unsigned dim() const
	{
		return m.ndim() == 2 ? m.dim(1) : 0;
	}

	//=======================================================================
	//
	// change size (number of code book vectors), should be virtual since
	// derived classes may need a special version of 'resize' (e.g. if they
	// have to resize other items than only the code book vectors)
	//

	//! Changes the number of code book vectors without affection the dimension.
	virtual void resize
	(
		unsigned n,
		bool copy = false
	);

	//! Changes the number and dimension of the code book vectors.
	virtual void resize
	(
		unsigned n,
		unsigned d,
		bool copy = false
	);

	//=======================================================================
	//
	// initialization
	//

	//! Initializes the code book vectors with uniformly distributed numbers.
	void initialize
	(
		const Array< double >& min,
		const Array< double >& max
	);

	//! Initializes the code book vectors with uniformly distributed numbers.
	void initialize
	(
		const Array< double >& x
	);

	//! Performs a k-means-clustering.
	void kmc
	(
		const Array< double >&	x,
		double					prec	= 1e-6,
		unsigned				maxiter	= 0
	);

protected:
	//! An n x d matrix that contains n code book vectors of dimension d.
	Array< double > m;
};

#endif /* !__CODEBOOK_H */

