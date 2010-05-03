//===========================================================================
/*!
 *  \file ArrayIo.h
 *
 *  \brief This file offers two methods for reading in the content of
 *         arrays from an input stream and for writing this content
 *         to an output stream.
 *
 *  \author  M. Kreutz
 *  \date    1995
 *
 *  \par Copyright (c) 1995, 1998:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
 *
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


#ifndef __ARRAYIO_H
#define __ARRAYIO_H

#include <Array/Array.h>



//! \ingroup InOut
//! Reads the content for array "arr" from input stream "is".
int readArray
(
	Array< double >&	arr,
	std::istream&		is,
	bool				seek			= false,
	const std::string	beginRecord		= "",
	const std::string	endRecord		= "\n",
	const std::string	beginComment	= ";",
	const std::string	endComment		= "\n",
	const std::string	separator		= " ,",
	const std::vector< unsigned > dimensions = std::vector< unsigned >(0)
);



//===================================================================
/*!
 *  \ingroup InOut
 *  \brief Writes the content of array "arr" to the
 *         output stream "os".
 *
 *  When writing arrays to the output stream, each array will be arranged
 *  by records. Each position of the first dimension with all its
 *  subdimensions is a single record and the records are written
 *  successively. Because all subdimensions of a position of the
 *  first dimension are processed by means of recursion, you will
 *  get a nested structure of the \em beginRecord and \em endRecord
 *  strings. <br>
 *  Guess, you have the following 3-dimensional array and
 *  call of the writeArray-function: <br>
 *  \code
 *  Array< double > test( 2, 2, 4 );
 *
 *  test( 0, 0, 0 ) = 1.;
 *  test( 0, 0, 1 ) = 2.;
 *  test( 0, 0, 2 ) = 3.;
 *  test( 0, 0, 3 ) = 4.;
 *  test( 0, 1, 0 ) = 5.;
 *  test( 0, 1, 1 ) = 6.;
 *  test( 0, 1, 2 ) = 7.;
 *  test( 0, 1, 3 ) = 8.;
 *  test( 1, 0, 0 ) = 9.;
 *  test( 1, 0, 1 ) = 10.;
 *  test( 1, 0, 2 ) = 11.;
 *  test( 1, 0, 3 ) = 12.;
 *  test( 1, 1, 0 ) = 13.;
 *  test( 1, 1, 1 ) = 14.;
 *  test( 1, 1, 2 ) = 15.;
 *  test( 1, 1, 3 ) = 16.;
 *
 *  writeArray( test, cout, "<RECORD BEGIN>", "<RECORD END>", '\t', '#' );
 *  \endcode
 *
 *  Then you will get the following output: <br>
 *
 *  \<RECORD BEGIN\>\<RECORD BEGIN\>\<RECORD BEGIN\>1    2    3    4\<RECORD END\>\<RECORD BEGIN\>5    6    7    8\<RECORD END\>\<RECORD END\>#\<RECORD BEGIN\>\<RECORD BEGIN\>9    10    11    12\<RECORD END\>\<RECORD BEGIN\>13    14    15    16\<RECORD END\>\<RECORD END\>\<RECORD END\>
 *
 *  \param arr             the array whose content will be written to \em os
 *  \param os              the output stream to which the arrays content is
 *                         written
 *  \param beginRecord     the string that will be written at the beginning
 *                         of the output of each subdimension of a single
 *                         record, the default is an empty string
 *  \param endRecord       the string that will be written at the end
 *                         of the output of each subdimension of a single
 *                         record, the default is the newline character
 *  \param separator       the character that will separate single
 *                         array elements, the default is the tabulator
 *                         character
 *  \param recordSeparator the character that separates single records, the
 *                         default is the null character for strings
 *
 *  \return
 *
 *  \author  M. Kreutz
 *  \date    1995
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
template < class T >
int writeArray
(
	const Array< T >&	arr,
	std::ostream&		os,
	const std::string	beginRecord     = "",
	const std::string	endRecord       = "\n",
	char				separator       = '\t',
	char				recordSeparator = '\0'
)
{
	if (arr.ndim() != 0 && os.good()) {
		os << beginRecord;
	}

	if (arr.ndim() <= 1) {
		for (unsigned i = 0; i < arr.nelem() && os.good(); ++i) {
			if (i) {
				os << separator;
			}
			os << arr.elem(i);
		}
	}
	else {
		for (unsigned i = 0; i < arr.dim(0) && os.good(); ++i) {
			if (i && recordSeparator) {
				os << recordSeparator;
			}
			writeArray(arr[ i ], os, beginRecord, endRecord, separator);
		}
	}

	if (arr.ndim() != 0 && os.good()) {
		os << endRecord;
	}

	return os.good() ? 0 : -1;
}



#endif /* !__ARRAYIO_H */

