/*!
*  \file ChromosomeT_int.cpp
* 
*  \author Martin Kreutz
*
*  \brief Functions for  chromosomes where the alleles are integers.
** 
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
*    \par Project:
*        EALib
*
*
*  <BR>
*
*
*  <BR><HR>
*  This file is part of the EALib. This library is free software;
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


#include <SharkDefs.h>
#include <EALib/ChromosomeT.h>


//===========================================================================

void ChromosomeT< int >::mutateDiffGeom(double s)
{
	double p = 1 - (s / size()) / (sqrt(1 + Shark::sqr(s / size())) + 1);

	for (unsigned i = size(); i--;)
		(*this)[ i ] += Rng::diffGeom(p);
}

//===========================================================================

void ChromosomeT< int >::mutateDiffGeom(const std::vector< double >& s,
										bool cycle)
{
	RANGE_CHECK(s.size() <= size())

	for (unsigned i = cycle ? size() : s.size(); i--;) {
		double t = s[ i % s.size()] / size();
		(*this)[ i ] += Rng::diffGeom(1 - t / (sqrt(1 + t * t) + 1));
	}
}

//===========================================================================

void ChromosomeT< int >::mutateDiffGeom(const ChromosomeT< double >& s,
										bool cycle)
{
	mutateDiffGeom(static_cast< const std::vector< double >& >(s), cycle);
}

//===========================================================================

void ChromosomeT< int >::mutateDiffGeom(const Chromosome& s, bool cycle)
{
	mutateDiffGeom(dynamic_cast< const std::vector< double >& >(s), cycle);
}

//===========================================================================

/*
#ifndef __NO_GENERIC_IOSTREAM

void ChromosomeT< int >::writeTo( std::ostream& os ) const
{
    os << "ChromosomeT<" << typeid( int ).name( ) << ">(" << size() << ")" << std::endl;
    for( unsigned i = 0; i < size( ); i++ ) {
        if( i ) os << '\t';
	os << ( *this )[ i ];
    }
    os << std::endl;
}

void ChromosomeT< int >::readFrom( std::istream& is )
{
    string s;
  //is.getline( s );
    is >> s;
    is.get( );   // skip end of line

    if( is.good( ) &&
	s.substr( 0, 12 ) == "ChromosomeT<" &&
	s.find( '>' ) != string::npos &&
	s.substr( 12, s.find( '>' ) - 12 ) == typeid( int ).name( ) ) {

        resize( atoi( s.substr( s.find( '>' ) + 2 ).c_str( ) ) );
	for( unsigned i = 0; i < size( ); i++ )
	    is >> ( *this )[ i ];
    } else
        is.setf( ios::failbit );
}

#endif // !__NO_GENERIC_IOSTREAM
*/

//===========================================================================

