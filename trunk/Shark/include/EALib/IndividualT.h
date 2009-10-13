//===========================================================================
/*!
 *  \file IndividualT.h
 *
 *  \brief Templates for typesafe uniform individuals
 *
 *  \author Tobias Glasmachers
 *  \date 2008
 *
 *  \par Copyright (c) 2008:
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


#ifndef _IndividualT_H_
#define _IndividualT_H_


#include <EALib/Individual.h>
#include <EALib/ChromosomeT.h>


//!
//! \brief Individual with uniform chromosome type CT
//!
template<typename CT>
class IndividualCT : public Individual
{
public:
	explicit IndividualCT(unsigned int noChromosomes = 0)
	: Individual( noChromosomes ) {
		unsigned int i;
		for (i=0; i<noChromosomes; i++) *(this->begin() + i) = new CT();
	}

	IndividualCT(unsigned int noChromosomes, const CT & c)
	: Individual( noChromosomes, c )
	{ }

	IndividualCT(const CT & c)
	: Individual( c )
	{ }

	IndividualCT(const CT & c0, const CT & c1)
	: Individual( c0, c1 )
	{ }

	IndividualCT(const CT & c0, const CT & c1, const CT & c2)
	: Individual( c0, c1, c2 )
	{ }

	IndividualCT( const CT & c0, const CT & c1, const CT & c2, const CT & c3 )
	: Individual( c0, c1, c2, c3 )
	{ }

	IndividualCT(const CT & c0, const CT & c1, const CT & c2, const CT & c3, const CT & c4)
	: Individual( c0, c1, c2, c3, c4 )
	{ }

	IndividualCT(const CT & c0,
			   const CT & c1,
			   const CT & c2,
			   const CT & c3,
			   const CT & c4,
				const CT & c5 )
	: Individual( c0, c1, c2, c3, c4, c5 )
	{ }

	IndividualCT(const CT & c0,
			   const CT & c1,
			   const CT & c2,
			   const CT & c3,
			   const CT & c4,
			   const CT & c5,
				const CT & c6 )
	: Individual( c0, c1, c2, c3, c4, c5, c6 )
	{ }

	IndividualCT(const CT & c0,
			   const CT & c1,
			   const CT & c2,
			   const CT & c3,
			   const CT & c4,
			   const CT & c5,
			   const CT & c6,
				const CT & c7 )
	: Individual( c0, c1, c2, c3, c4, c5, c6, c7 )
	{ }

	IndividualCT(const std::vector< CT > & v)
	: Individual( v )
	{ }

	~IndividualCT() { }


	CT & operator[]( unsigned i ) {
		return dynamic_cast<CT&> (Individual::operator [] (i));
	}

	const CT & operator[]( unsigned i ) const {
		return dynamic_cast<const CT&> (Individual::operator [] (i));
	}

	//=======================================================================

	IndividualCT<CT>& operator = ( const IndividualCT<CT>& other) {
		Individual::operator = (other);
		return *this;
	}
};


// workaround because
//  template <typename T> typedef IndividualCT< ChromosomeT<T> > IndividualT<T>;
// is not legal in the current C++ standard.

//!
//! \brief Individual with uniform chromosome type ChromosomeT &lt; T &gt;
//!
template<typename T>
class IndividualT : public IndividualCT< ChromosomeT<T> >
{
	typedef IndividualCT< ChromosomeT<T> > super;

public:
	explicit IndividualT(unsigned int noChromosomes = 0)
	: super( noChromosomes )
	{ }

	IndividualT(const ChromosomeT<T> & c)
	: super( c )
	{ }

	IndividualT(const ChromosomeT<T> & c0, const ChromosomeT<T> & c1)
	: super( c0, c1 )
	{ }

	IndividualT(const ChromosomeT<T> & c0, const ChromosomeT<T> & c1, const ChromosomeT<T> & c2)
	: super( c0, c1, c2 )
	{ }

	IndividualT(const ChromosomeT<T> & c0, const ChromosomeT<T> & c1, const ChromosomeT<T> & c2, const ChromosomeT<T> & c3)
	: super( c0, c1, c2, c3 )
	{ }

	IndividualT(const ChromosomeT<T> & c0, const ChromosomeT<T> & c1, const ChromosomeT<T> & c2, const ChromosomeT<T> & c3, const ChromosomeT<T> & c4)
	: super( c0, c1, c2, c3, c4 )
	{ }

	IndividualT(const ChromosomeT<T> & c0, const ChromosomeT<T> & c1, const ChromosomeT<T> & c2, const ChromosomeT<T> & c3, const ChromosomeT<T> & c4, const ChromosomeT<T> & c5)
	: super( c0, c1, c2, c3, c4, c5 )
	{ }

	IndividualT(const ChromosomeT<T> & c0, const ChromosomeT<T> & c1, const ChromosomeT<T> & c2, const ChromosomeT<T> & c3, const ChromosomeT<T> & c4, const ChromosomeT<T> & c5, const ChromosomeT<T> & c6)
	: super( c0, c1, c2, c3, c4, c5, c6 )
	{ }

	IndividualT(const ChromosomeT<T> & c0, const ChromosomeT<T> & c1, const ChromosomeT<T> & c2, const ChromosomeT<T> & c3, const ChromosomeT<T> & c4, const ChromosomeT<T> & c5, const ChromosomeT<T> & c6, const ChromosomeT<T> & c7)
	: super( c0, c1, c2, c3, c4, c5, c6, c7 )
	{ }

	~IndividualT()
	{ }


	ChromosomeT<T> & operator[]( unsigned i ) {
		return dynamic_cast<ChromosomeT<T>&> (super::operator [] (i));
	}

	const ChromosomeT<T> & operator[]( unsigned i ) const {
		return dynamic_cast<const ChromosomeT<T>&> (super::operator [] (i));
	}

	//=======================================================================

	IndividualT<T>& operator = ( const IndividualT<T>& other) {
		super::operator = (other);
		return *this;
	}
};


#endif
