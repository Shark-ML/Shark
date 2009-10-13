/*!
*  \file ChromosomeT_bool.cpp
*
*  \brief Functions for Boolean chromosomes (i.e., bit strings).
*
*  \author  Martin Kreutz
*  \date    2005
*
*  \par
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
*
*
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

#include <EALib/ChromosomeT.h>

//===========================================================================

void ChromosomeT< bool >::initialize()
{
	for (unsigned i = size(); i--;)
		(*this)[ i ] = bool(Rng::coinToss(0.5));
}

void ChromosomeT< bool >::initialize(const unsigned pos)
{
	for (unsigned i = pos; i < size(); ++i)
		(*this)[ i ] = bool(Rng::coinToss(0.5));
}


//===========================================================================

inline unsigned long pow2(unsigned n)
{
	return 1UL << n;
}

void ChromosomeT< bool >::encode(double v,
								 const Interval& range,
								 unsigned nbits,
								 bool useGray)
{
	unsigned long l;

	l = (unsigned long)((pow2(nbits) - 1)
						* (v - range.lowerBound())
						/ range.width());

	if (useGray) l ^= l >> 1;

	resize(nbits);
	for (unsigned i = 0; i < nbits; i++, l >>= 1)
		(*this)[ i ] = l & 1;
}

//===========================================================================

double ChromosomeT< bool >::decode(const Interval& range, bool useGray) const
{
	unsigned long l, m;
	unsigned      i;

	if (useGray)
		for (l = m = 0, i = size(); i--;)
			l = (l << 1) | (m ^= ((*this)[ i ] ? 1 : 0));
	else
		for (l = 0, i = size(); i--;)
			l = (l << 1) | ((*this)[ i ] ? 1 : 0);

	return l * range.width() / (pow2(size()) - 1)
		   + range.lowerBound();
}

//===========================================================================

void ChromosomeT< bool >::encodeBinary(const std::vector< double >& src,
									   const Interval&         range,
									   unsigned                nbits,
									   bool                    useGray)
{
	double stepSize = (pow2(nbits) - 1) / range.width();
	unsigned i, j, k;
	unsigned long l;

	resize(src.size() * nbits);

	for (j = k = 0; j < src.size(); j++) {
		l = (unsigned long)((src[ j ] - range.lowerBound()) * stepSize);

		if (useGray) l ^= l >> 1;

		for (i = nbits; i--; l >>= 1)
			(*this)[ k++ ] = l & 1;
	}
}

//===========================================================================

void ChromosomeT< bool >::encodeBinary
(
	const Chromosome& chrom,
	const Interval&   range,
	unsigned          nbits,
	bool              useGray
)
{
	encodeBinary(dynamic_cast< const std::vector< double >& >(chrom),
				 range, nbits, useGray);
}

//===========================================================================
//
//
//
void ChromosomeT< bool >::flip(double p)
{
	for (unsigned i = size(); i--;)
		if (Rng::coinToss(p))
			//( *this )[ i ] ^= true;
			//
			// vector< bool > is now represented by a packed bit array
			//
			(*this)[ i ] = (*this)[ i ] ^ true;
}


//===========================================================================
//
//
//
void ChromosomeT< bool >::flip(const std::vector< double >& p, bool cycle)
{
	RANGE_CHECK(p.size() <= size())

	for (unsigned i = cycle ? size() : p.size(); i--;)
		if (Rng::coinToss(p[ i % p.size()]))
			//( *this )[ i ] ^= true;
			//
			// vector< bool > is now represented by a packed bit array
			//
			(*this)[ i ] = (*this)[ i ] ^ true;
}


//===========================================================================
//
//
//
void ChromosomeT< bool >::flip(const Chromosome& p, bool cycle)
{
	flip(dynamic_cast< const std::vector< double >& >(p), cycle);
}

//===========================================================================

bool ChromosomeT< bool >::operator == (const Chromosome& c) const
{
	//
	// this annoying static cast is necessary to satisfy
	// the pedantic VC++ 5.0
	//
	return static_cast< const std::vector< bool >& >(*this) == dynamic_cast< const std::vector< bool >& >(c);
}

bool ChromosomeT< bool >::operator < (const Chromosome& c) const
{
#ifdef __BOOL_COMPARE__
	return static_cast< const std::vector< bool >& >(*this) <  dynamic_cast< const std::vector< bool >& >(c);
#else
	UNDEFINED
	return false;
#endif
}

//===========================================================================

/*
#ifndef __NO_GENERIC_IOSTREAM

void ChromosomeT< bool >::writeTo( ostream& os ) const
{
    os << "ChromosomeT<" << typeid( bool ).name( ) << ">(" << size() << ")" << endl;
    for( unsigned i = 0; i < size( ); i++ )
        os.put( ( *this )[ i ] ? '1' : '0' );
    os << endl;
}

void ChromosomeT< bool >::readFrom( istream& is )
{
    string s;
  //is.getline( s );
    is >> s;
    is.get( );   // skip end of line

    if( is.good( ) &&
	s.substr( 0, 12 ) == "ChromosomeT<" &&
	s.find( '>' ) != string::npos &&
	s.substr( 12, s.find( '>' ) - 12 ) == typeid( bool ).name( ) ) {

        resize( atoi( s.substr( s.find( '>' ) + 2 ).c_str( ) ) );
	char c;
	for( unsigned i = 0; i < size( ); i++ ) {
	    is.get( c );
	    ( *this )[ i ] = c != '0';
	}
    } else
      is.setf( ios::failbit );
}

#endif // !__NO_GENERIC_IOSTREAM
*/

//===========================================================================

