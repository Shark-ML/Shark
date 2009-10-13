//===========================================================================
/*!
 *  \file Array.h
 *
 *  \brief Provides a data structure for arbitrary arrays.
 *
 *  The class array provides basic data structure for handling arrays
 *  of arbitrary types. It is implemented via a C++ template class
 *  Array< type >.  The array template offers more flexible than
 *  standard c++ arrays, for example it is possible to dynamically
 *  change the number of elements and dimensions.  Furthermore,
 *  parameter passing by value and assignment works properly (i.e.,
 *  the value is passed or assigned and not a pointer to the value)
 *  and the subscript operator [ ] may perform a range check at
 *  run-time. <br> If the preprocessor directive "-DNDEBUG" is used,
 *  the range checks are disabled! <br> For convenience subscript
 *  operators for arrays of known dimensions are defined, for example:
 *  \code ...  #include "Array/Array.h" // declarations ...  Array<
 *  int > x( 10, 5 ); // define an array of integer values Array< int
 *  > y; // define an array with unspecified size
 *
 *      y = x[ 2 ];                // y contains a 5-dimensional sub-array
 *      x( 4, 2 ) = -3;            // set element at position ( 4, 2 ) to -3
 *          ...
 *  \endcode
 *
 *  \author  M. Kreutz
 *  \date    1995-01-01
 *
 *  \par Copyright (c) 1995, 2002:
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

//! Setting this flag will disable I/O-methods writeTo, readFrom and
//! the input/output stream operators.
#define __ARRAY_NO_GENERIC_IOSTREAM

#ifndef __ARRAY_H
#define __ARRAY_H


////////////////////////////////////////////////////////////
// backwards compatibility
#define Array2D Array
#define Array2DReference ArrayReference


#include <SharkDefs.h>
#include <algorithm>
#include <vector>
#include <typeinfo>


//==========================================================================
//
// doxygen declarations begin
//
//==========================================================================

/*!
 *  \defgroup AllOps Operations for Arrays (Please notice the detailed description!)
 *
 *  The lists of methods for each operation type
 *  are not complete. In many classes of the library "Array" you can
 *  find further methods, that are so special, that they couldn't be
 *  associated to one of the operation types listed here. This list here
 *  will give you a survey only.
 *
 */

/*!
 *  \defgroup Create Creating Arrays
 *  \ingroup AllOps
 */

/*!
 *  \defgroup Information Information Retrieval methods
 *  \ingroup AllOps
 */

/*!
 *  \defgroup Resize Changing the size of an Array
 *  \ingroup AllOps
 */

/*!
 *  \defgroup Extract Extracting Array elements
 *  \ingroup AllOps
 */

/*!
 *  \defgroup Copy Copying Arrays or Array elements
 *  \ingroup AllOps
 */

/*!
 *  \defgroup Assign Assigning values to an Array
 *  \ingroup AllOps
 */

/*!
 *  \defgroup Add Adding content to an Array
 *  \ingroup AllOps
 */

/*!
 *  \defgroup Del Removing content from an Array
 *  \ingroup AllOps
 */

/*!
 *  \defgroup InOut Input/Output of Arrays
 *  \ingroup AllOps
 */

/*!
 *  \defgroup Compare Comparing Arrays
 *  \ingroup AllOps
 */

/*!
 *  \defgroup Math Mathematic Functions (Please notice the detailed description!)
 *
 *  \ingroup AllOps
 *
 *  The methods listed here are not complete. Most of the defined
 *  operators in file ArrayOp.h are removed from the reference,
 *  because the special way of defining these operators causes doxygen
 *  to create a corrupted reference. A list of these operators can be
 *  found in the detailed description of ArrayOp.h.
 *
 */



//==========================================================================
//
// doxygen declarations end
//
//==========================================================================


// forward declaration
template< class T >
class Array;

// forward declaration
class ArrayIndex;

// forward declaration
template< class T >
class ArrayReference;

//===========================================================================
/*!
 *  \brief Class ArrayBase serves as a base class for the template class
 *         Array< T >.
 *
 *  All members dealing with index operation are defined here.
 *  Type-specific members like memory allocation, stream-I/O and the
 *  representation itself are defined in Array< T >. These members are
 *  accessible from ArrayBase via virtual functions, which are
 *  declared here. <br> Here you will find methods for retrieving info
 *  about the structure of an Array object.
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
class ArrayBase
{
public:  // interface of public members

	//========================================================================
	/*!
	 *  \ingroup Information
	 *  \brief Returns the number of dimensions.
	 *
	 *  The number of dimensions, i.e. the length of the dimension vector 
	 *  #d, is returned.
	 *
	 *  \return the number of dimensions
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	unsigned ndim() const
	{
		return nd;
	}


	//========================================================================
	/*!
	 *  \ingroup Information
	 *  \brief Returns the total number of elements.
	 *
	 *  The total number of elements, i.e. the product over all dimensions,
	 *  is returned.
	 *
	 *  \return the total number of elements
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	unsigned nelem() const
	{
		return ne;
	}


	//========================================================================
	/*!
	 *  \ingroup Information
	 *  \brief Returns true if this array and array "v" have the same 
	 *         dimensions.
	 *
	 *  Checks whether two arrays have the same dimensions independent
	 *  of their types.
	 *
	 *  \param  v the second array, to which dimensions are compared 
	 *  \return "true", if the dimensions of both arrays are the same,
	 *          "false" otherwise
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	bool samedim(const ArrayBase& v) const
	{
		if (nd == v.nd && ne == v.ne)
		{
			unsigned i;
			for (i = 0; i < nd && d[ i ] == v.d[ i ]; i++);
			return i == nd;
		}
		return false;
	}

	//========================================================================
	/*!
	 *  \ingroup Resize
	 *  \brief Resizes an array to an one-dimensional array of size "i".
	 *
	 *  When the \em copy flag is set, then existing elements are
	 *  copied (if possible), otherwise all existing elements are
	 *  discarded.
	 *
	 *  \param  i    the new size of the one-dimensional array.
	 *  \param  copy if set to "true", then existing elements are
	 *               copied, otherwise they are discarded.
	 *  \return none
	 *  \throw  SharkException the type of the exception will be
	 *          "size mismatch" and indicates, that you've tried
	 *          to resize a static array reference and the new
	 *          size is not equal to the old size
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 *  \sa #resize_i(unsigned*, unsigned, bool)
	 *
	 */
	void resize(unsigned i, bool copy = false)
	{
		unsigned _d[ 1 ];
		_d[ 0 ] = i;
		resize_i(_d, 1, copy);
	}

	//========================================================================
	/*!
	 *  \ingroup Resize
	 *  \brief Resizes an array to a two-dimensional array of size i x j.
	 *
	 *  When the \em copy flag is set, then existing elements are
	 *  copied (if possible), otherwise all existing elements are
	 *  discarded.
	 *
	 *  \param  i    the size of the array in dimension one.
	 *  \param  j    the size of the array in dimension two.
	 *  \param  copy if set to "true", then existing elements are
	 *               copied, otherwise they are discarded.
	 *  \return none
	 *  \throw  SharkException the type of the exception will be
	 *          "size mismatch" and indicates, that you've tried
	 *          to resize a static array reference and the new
	 *          size is not equal to the old size
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 *  \sa #resize_i(unsigned*, unsigned, bool)
	 *
	 */
	void resize(unsigned i, unsigned j, bool copy = false)
	{
		unsigned _d[ 2 ];
		_d[ 0 ] = i;
		_d[ 1 ] = j;
		resize_i(_d, 2, copy);
	}

	//========================================================================
	/*!
	 *  \ingroup Resize
	 *  \brief Resizes an array to a three-dimensional array of size 
	 *         i x j x k.
	 *
	 *  When the \em copy flag is set, then existing elements are
	 *  copied (if possible), otherwise all existing elements are
	 *  discarded.
	 *
	 *  \param  i    the size of the array in dimension one.
	 *  \param  j    the size of the array in dimension two.
	 *  \param  k    the size of the array in dimension three.
	 *  \param  copy if set to "true", then existing elements are
	 *               copied, otherwise they are discarded.
	 *  \return none
	 *  \throw  SharkException the type of the exception will be
	 *          "size mismatch" and indicates, that you've tried
	 *          to resize a static array reference and the new
	 *          size is not equal to the old size
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 *  \sa #resize_i(unsigned*, unsigned, bool)
	 *
	 */
	void resize(unsigned i, unsigned j, unsigned k, bool copy = false)
	{
		unsigned _d[ 3 ];
		_d[ 0 ] = i;
		_d[ 1 ] = j;
		_d[ 2 ] = k;
		resize_i(_d, 3, copy);
	}

	//========================================================================
	/*!
	 *  \ingroup Resize
	 *  \brief Resizes an array to an array with dimensions defined in 
	 *         vector "i".
	 *
	 *  When the \em copy flag is set, then existing elements are
	 *  copied (if possible), otherwise all existing elements are
	 *  discarded.
	 *
	 *  \param  i    the vector with the sizes of the different
	 *               dimensions of the array.
	 *  \param  copy if set to "true", then existing elements are
	 *               copied, otherwise they are discarded.
	 *  \return none
	 *  \throw  SharkException the type of the exception will be
	 *          "size mismatch" and indicates, that you've tried
	 *          to resize a static array reference and the new
	 *          size is not equal to the old size
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 *  \sa #resize_i(unsigned*, unsigned, bool)
	 *
	 */
	void resize(const std::vector< unsigned >& i, bool copy = false)
	{
		unsigned* _d = new unsigned[ i.size()];

		for (unsigned j = 0; j < i.size(); ++j)
			_d[ j ] = i[ j ];

		resize_i(_d, i.size(), copy);

		delete[ ] _d;
	}

	//========================================================================
	/*!
	 *  \ingroup Resize
	 *  \brief Resizes an array to the dimensions of array "v".
	 *
	 *  The new dimensions of the current array are adopted from the
	 *  dimensions of array \em v. <br> When the \em copy flag is set,
	 *  existing elements are copied (if possible), otherwise all
	 *  existing elements are discarded.
	 *
	 *  \param  v    the array which dimensions are taken as new
	 *               dimensions of the current array.
	 *  \param  copy if set to "true", then existing elements of
	 *               the current array are copied, otherwise they 
	 *               are discarded.
	 *  \return none
	 *  \throw  SharkException the type of the exception will be
	 *          "size mismatch" and indicates, that you've tried
	 *          to resize a static array reference and the new
	 *          size is not equal to the old size
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 *  \sa #resize_i(unsigned*, unsigned, bool)
	 *
	 */
	void resize(const ArrayBase& v, bool copy = false)
	{
		resize_i(v.d, v.nd, copy);
	}

	//========================================================================
	/*!
	 *  \ingroup Information
	 *  \brief Returns the size of the i-th dimension of the array.
	 *
	 *  \param  i the dimension of which the size will be returned.
	 *  \return the size of dimension \em i.
	 *  \throw  SharkException the type of the exception will be
	 *          "range check error" and denotes, that the value
	 *          of \em i exceeds the number of dimensions of the array.
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	unsigned dim(unsigned i) const
	{
		RANGE_CHECK(i < nd)
		return d[ i ];
	}

	//! \ingroup Information
	//! \brief Returns the number of rows.
	inline unsigned int rows() const
	{
		if (ndim() == 0) return 0;
		else return dim(0);
	}

	//! \ingroup Information
	//! \brief Returns the number of columns.
	inline unsigned int cols() const
	{
		if (ndim() == 0) return 0;
		else return dim(ndim() - 1);
	}

	//========================================================================
	/*!
	 *  \brief Returns the pointer to the dimension vector ArrayBase::d.
	 *
	 *  This method allows you to read and manipulate the dimension
	 *  vector of the current array directly. <br> Please use this
	 *  method with care and prefer the methods #resize, #ndim and #dim
	 *  instead if possible.
	 *
	 *  \return the dimension vector #d.
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	unsigned* dimvec()
	{
		return d;
	}

	//========================================================================
	/*!
	 *  \brief Returns the pointer to the dimension vector ArrayBase::d for
	 *         constant objects.
	 *
	 *  Please use this method with care and prefer the methods
	 *  #resize, #ndim and #dim instead if possible.
	 *
	 *  \return the dimension vector #d.
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 *  \sa #dimvec( )
	 *
	 */
	const unsigned* dimvec() const
	{
		return d;
	}

	//========================================================================
	/*!
	 *  \brief Returns an identical copy of this array object.
	 * 
	 *  This method is pure virtual and is defined in template
	 *  class Array< T >.
	 *
	 *  \return a copy of the current array
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 *  \sa Array::clone
	 *
	 */
	virtual ArrayBase* clone() const = 0;

	//========================================================================
	/*!
	 *  \brief Returns an empty array with the same type as this array object.
	 * 
	 *  This method is pure virtual and is defined in template
	 *  class Array< T >.
	 *
	 *  \return an empty copy of the current array
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 *  \sa Array::empty
	 *
	 */
	virtual ArrayBase* empty() const = 0;

	//========================================================================
	/*!
	 *  \brief Destructs this ArrayBase object.
	 * 
	 *  Destruction is only performed if there is anything to destruct and
	 *  it is not a static reference (signalled by the flag #stat).
	 *
	 *  \return none
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	virtual ~ArrayBase()
	{
		if (! stat && nd)
		{
			delete[ ] d;
		}
	}

protected:  // interface of protected members (overloaded by array< T >)

	//========================================================================
	/*!
	 *  \brief Constructs an ArrayBase object.
	 * 
	 *  The default constructor, which should be never invoked directly
	 *  because ArrayBase contains pure virtual members.
	 *
	 *  \return none
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	ArrayBase()
	{
		d    = 0;
		nd   = 0;
		ne   = 0;
		stat = false;
	}

	//=======================================================================
	/*!
	 *  \brief Constructs a copy of an ArrayBase object.
	 *
	 *  The copy constructor should be never invoked directly since
	 *  ArrayBase contains pure virtual members.  Only the dimension
	 *  vector ArrayBase::d is copied, the data vector must be copied in
	 *  the copy contructor of template class Array< T >.  This copy
	 *  constructor is independent of the type of a particular template
	 *  object of class Array< T >.
	 *
	 *  \param v object which is copied by the constructor
	 *  \return none
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	ArrayBase(const ArrayBase& v)
	{
		nd    = v.nd;
		ne    = v.ne;
		stat  = false;

		if (nd)
		{
			d = new unsigned[ nd ];
			for (unsigned i = nd; i--; d[ i ] = v.d[ i ]);
		}
	}

	//=======================================================================
	/*!
	 *  \brief Handles memory allocation for the other resize methods.
	 *
	 *  This pure virtual function handles memory allocation of the
	 *  respective template type in case of resizing.
	 *
	 *  \param d one-dimensional vector of array-dimensions
	 *  \param nd number of array-dimensions
	 *  \param copy flag which indicates whether existing elements of
	 *             an array should be copied in case of resizing (as long
	 *             as possible)
	 *  \return none
	 *  \throw  SharkException the type of the exception will be 
	 *          "size mismatch" and indicates, that you've tried
	 *          to resize a static array reference with another size
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 *  \sa #resize(const ArrayBase&, bool),
	 *      #resize(const std::vector< unsigned >&, bool),
	 *      #resize(unsigned, unsigned, unsigned, bool),
	 *      #resize(unsigned, unsigned, bool),
	 *      #resize(unsigned, bool)
	 *      Array::resize_i(unsigned*, unsigned, bool)
	 *
	 */
	virtual void resize_i(unsigned* d, unsigned nd, bool copy) = 0;

#ifndef __ARRAY_NO_GENERIC_IOSTREAM

	//=======================================================================
	/*!
	 *  \brief Replaces the structure and content of the current array by
	 *         the information read from the input stream "is".
	 *
	 *  This pure virtual function is defined in the classes of the
	 *  respective template types. <br>
	 *  This method is only available, when the flag
	 *  #__ARRAY_NO_GENERIC_IOSTREAM is undefined at the beginning
	 *  of Array.h.
	 *
	 *  \param is input stream from which the array content is read.
	 *  \return none
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 *  \sa Array::readFrom
	 *
	 */
	virtual void readFrom(std::istream& is) = 0;

	//=======================================================================
	/*!
	 *  \brief Writes the structure and content of the current array to 
	 *         output stream "os".
	 *
	 *  This pure virtual function is defined in the classes of the
	 *  respective template types. <br> This method is only available,
	 *  when the flag #__ARRAY_NO_GENERIC_IOSTREAM is undefined at the
	 *  beginning of Array.h.
	 *
	 *  \param os output stream to which the array content is written.
	 *  \return none
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 *  \sa Array::writeTo
	 */
	virtual void writeTo(std::ostream& os) const = 0;
#endif // !__ARRAY_NO_GENERIC_IOSTREAM

	//=======================================================================
	/*!
	 *  \brief The dimension vector.
	 *
	 *  Internally, the major information about an array is stored in
	 *  two vectors, the element vector Array::e and the dimension vector. <br>
	 *  The dimension vector stores the sizes of all dimensions of
	 *  an array.
	 *
	 *  \par Example
	 *  \code
	 *  #include "Array/Array.h"
	 * 
	 *  using namespace std;
	 * 
	 *  void main()
	 *  {
	 *      Array< double >  test( 3, 2, 12 );  // test array
	 *      unsigned        *dvec,              // dimension vector
	 *                       no_dims,           // total number of dimensions
	 *                       curr_dim;          // number of current dimension
	 * 
	 *
	 *      // Get total number of dimensions: 
	 *      no_dims = test.ndim( );
	 *
	 *      // Get dimension vector, output of all dimension sizes: 
	 *      for ( curr_dim = 0, dvec = test.dimvec( ); 
	 *            curr_dim < no_dims; curr_dim++, dvec++ )
	 *      {
	 *          cout << "Size of dimension no. " << curr_dim + 1
	 *  	            << ": " << *dvec << endl;
	 *      } 
	 *  }
	 *  \endcode
	 * 
	 *  This program will produce the output:
	 *
	 *  \f$
	 *  \mbox{\ }\\ \noindent
	 *  \mbox{Size of dimension no. 1: 3}\\
	 *  \mbox{Size of dimension no. 2: 2}\\
	 *  \mbox{Size of dimension no. 3: 12}
	 *  \f$
	 *
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 *  \sa #readFrom
	 *
	 */
	unsigned* d;

	//! number of dimensions
	unsigned  nd;

	//! total number of elements
	unsigned  ne;

	//! flag which signals whether object is a static reference.
	bool      stat;

#ifndef __ARRAY_NO_GENERIC_IOSTREAM

	//=======================================================================
	/*!
	 *  \ingroup InOut
	 *  \brief Replaces the structure and content of the current array by
	 *         the information read from the input stream "is".
	 *
	 *  Interface for the standard C++ library input stream operator.
	 *  The virtual method #readFrom is called internally. <br>
	 *  This method is only available, when the flag
	 *  #__ARRAY_NO_GENERIC_IOSTREAM is undefined at the beginning
	 *  of Array.h.
	 *
	 *  \param is input stream from which the array content is read.
	 *  \param a  the array which content is read from \em is.
	 *  \return reference to the input stream
	 *  \throw SharkException if the type of the exception is "type mismatch",
	 *         then the data read from \em is has not the
	 *         right format (see #Array::readFrom for details about 
	 *         the format). If the type of the exception is "I/O error",
	 *         then there is a problem with the input stream \em is.
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 *  \sa #readFrom
	 *
	 */
	friend inline std::istream& operator >> (std::istream& is, ArrayBase& a)
	{
		a.readFrom(is);
		IO_CHECK(is)
		return is;
	}

	//=======================================================================
	/*!
	 *  \ingroup InOut
	 *  \brief Writes the structure and content of the current array to 
	 *         output stream "os".
	 *
	 *  Interface for the standard C++ library output stream operator.
	 *  The virtual method #writeTo is called internally. <br>
	 *  This method is only available, when the flag
	 *  #__ARRAY_NO_GENERIC_IOSTREAM is undefined at the beginning
	 *  of Array.h.
	 *
	 *  \param os output stream to which the array content is written.
	 *  \param a  the array which content is written to \em is.
	 *  \return reference to the output stream
	 *  \throw SharkException the type of the exception will be "I/O error" and
	 *         indicates, that there is a problem with the output stream \em os.
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 *  \sa #writeTo
	 *
	 */
	friend inline std::ostream& operator << (std::ostream& os, const ArrayBase& a)
	{
		a.writeTo(os);
		IO_CHECK(os)
		return os;
	}
#endif // __ARRAY_NO_GENERIC_IOSTREAM
};

//===========================================================================
/*!
 *  \brief This template class defines a data structure for general arrays
 *         of arbitrary types.
 *
 *  The class is derived from class ArrayBase. <br>
 *  Here you will find element-specific methods for access of elements,
 *  assigning elements, adding and removing elements, changing
 *  the order of elements and resizing arrays.
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
template< class T >
class Array : public ArrayBase
{
	friend class ArrayReference< T >;
	friend class ArrayIndex;

public:

	//! Pointer to a single array element. Used for compatibility with
	//! the other stdlib structures.
	typedef T* iterator;

	//! Constant pointer to a single array element. Used for compatibility
	//! with the other stdlib structures.
	typedef T* const_iterator;


	//=======================================================================
	/*!
	 *  \ingroup Constructor
	 *  \brief Creates a new empty Array object.
	 *
	 *  \return none
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	Array()
	{
		ne = 0;
		e = eEnd = 0;
	}


	//=======================================================================
	/*!
	 *  \ingroup Constructor
	 *  \brief Creates a new empty one-dimensional array of size "i".
	 *
	 *  Memory for an one-dimensional array of length \em i is allocated.
	 *
	 *  \param i size of the new array 
	 *  \return none
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	explicit Array(unsigned i)
	{
		d      = new unsigned[ 1 ];
		d[ 0 ] = i;
		nd     = 1;
		ne     = i;
		e      = 0;
		if (ne)
		{
			e = new T[ ne ];
		}
		stat = false;
		eEnd = e + ne;
	}

	//=======================================================================
	/*!
	 *  \ingroup Constructor
	 *  \brief Creates a new empty two-dimensional array.
	 *
	 *  Memory for an \f$i \times j\f$ array is allocated.
	 *
	 *  \param i size of the first dimension of the new array 
	 *  \param j size of the second dimension of the new array 
	 *  \return none
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	Array(unsigned i, unsigned j)
	{
		d      = new unsigned[ 2 ];
		d[ 0 ] = i;
		d[ 1 ] = j;
		nd     = 2;
		ne     = i * j;
		e      = 0;
		if (ne) e = new T[ ne ];
		stat = false;
		eEnd = e + ne;
	}

	//=======================================================================
	/*!
	 *  \ingroup Constructor
	 *  \brief Creates a new empty three-dimensional array.
	 *
	 *  Memory for an \f$i \times j \times k\f$ array is allocated.
	 *
	 *  \param i size of the first dimension of the new array 
	 *  \param j size of the second dimension of the new array 
	 *  \param k size of the third dimension of the new array 
	 *  \return none
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	Array(unsigned i, unsigned j, unsigned k)
	{
		d      = new unsigned[ 3 ];
		d[ 0 ] = i;
		d[ 1 ] = j;
		d[ 2 ] = k;
		nd     = 3;
		ne     = i * j * k;
		e      = 0;
		if (ne)
		{
			e = new T[ ne ];
		}
		stat = false;
		eEnd = e + ne;
	}

	//=======================================================================
	/*!
	 *  \ingroup Constructor
	 *  \brief Creates a new one-dimensional array with the content of "v".
	 *
	 *  \param v vector, whose content will be content of the new array
	 *  \return none
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	Array(std::vector< T >& v)
	{
		d      = new unsigned[ 1 ];
		d[ 0 ] = v.size();
		nd     = 1;
		ne     = v.size();
		e      = 0;
		if (ne)
		{
			e = new T[ ne ];
			for (unsigned i = ne; i--; e[ i ] = v[ i ]);
		}
		stat = false;
		eEnd = e + ne;
	}

	//=======================================================================
	/*!
	 *  \ingroup Constructor
	 *  \brief Creates a new array with the structure and content of array "v".
	 *
	 *  \param v array, whose content will be content of the new array,
	 *           the content of \em v will be deleted and \em v will
	 *           act as static reference to the new created array
	 *  \param ... dummy value
	 *  \return none
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	Array(Array< T >& v, bool)
	{
		d      = v.d;
		e      = v.e;
		nd     = v.nd;
		ne     = v.ne;
		stat   = v.stat;
		v.nd   = v.ne = 0;
		v.stat = true;
		eEnd   = e + ne;
	}

	//=======================================================================
	/*!
	 *  \ingroup Constructor
	 *  \brief Creates a new array with the structure and content of the
	 *         constant array "v".
	 *
	 *  \param v array, whose content will be copied to the new array
	 *  \return none
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	Array(const Array< T >& v) : ArrayBase(v)
	{
		if (ne)
		{
			e = new T[ ne ];
			for (unsigned i = ne; i--; e[ i ] = v.e[ i ]);
			eEnd = e + ne;
		}
		else
		{
			e = eEnd = 0;
		}
	}

	//=======================================================================
	/*!
	 *  \brief Destructs the current Array object.
	 *
	 *  Deletes the dimension and element vector Array::e 
	 *  and frees the corresponding memory.
	 * 
	 *  \return none
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 *  \sa ArrayBase::~ArrayBase()
	 *
	 */
	~Array()
	{
		if (! stat && ne)
		{
			if (ne)
			{
				delete[ ] e;
			}
		}
	}

	//! \ingroup Information
	//! \brief Returns the number of rows.
	inline unsigned int rows() const
	{
		return ArrayBase::rows();
	}

	//! \ingroup Information
	//! \brief Returns the number of columns.
	inline unsigned int cols() const
	{
		return ArrayBase::cols();
	}

	//=======================================================================
	/*!
	 *  \ingroup Extract
	 *  \brief Returns the first and only element of the current array.
	 *
	 *  The dimension of the array must be zero and the number of elements
	 *  must be "1".
	 * 
	 *  \return The first element of the array.
	 *  \throw SharkException if the type of the exception is "size mismatch",
	 *         then the array has one or more dimensions, if the type is 
	 *         "range check error", then the array contains more or less than
	 *         one element
	 * 
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	T& operator()()
	{
		SIZE_CHECK(nd == 0)
		RANGE_CHECK(ne == 1)
		return e[ 0 ];
	}

	//=======================================================================
	/*!
	 *  \ingroup Extract
	 *  \brief Returns the first and only element of the current array as
	 *         constant.
	 *
	 *  The dimension of the array must be zero and the number of elements
	 *  must be "1".
	 * 
	 *  \return The first element of the array.
	 *  \throw SharkException if the type of the exception is "size mismatch",
	 *         then the array has one or more dimensions, if the type is 
	 *         "range check error", then the array contains more or less than
	 *         one element
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	const T& operator()() const
	{
		SIZE_CHECK(nd == 0)
		RANGE_CHECK(ne == 1)
		return e[ 0 ];
	}

	//=======================================================================
	/*!
	 *  \ingroup Extract
	 *  \brief Returns the i-th element of the one-dimensional array.
	 *
	 *  \param i position of the element that will be returned.
	 *  \return The i-th array element
	 *  \throw SharkException if the type of the exception is "size mismatch",
	 *         then the array is not one-dimensional. If the type is
	 *         "range check error", then the value of \em i exceeds
	 *         the array size
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	T& operator()(unsigned i)
	{
		SIZE_CHECK(nd == 1)
		RANGE_CHECK(i < d[ 0 ])
		return e[ i ];
	}

	//=======================================================================
	/*!
	 *  \ingroup Extract
	 *  \brief Returns the i-th element of the one-dimensional array as
	 *         constant.
	 *
	 *  \param i position of the element that will be returned.
	 *  \return The i-th array element
	 *  \throw SharkException if the type of the exception is "size mismatch",
	 *         then the array is not one-dimensional. If the type is
	 *         "range check error", then the value of \em i exceeds
	 *         the array size
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	const T& operator()(unsigned i) const
	{
		SIZE_CHECK(nd == 1)
		RANGE_CHECK(i < d[ 0 ])
		return e[ i ];
	}

	//=======================================================================
	/*!
	 *  \ingroup Extract
	 *  \brief Returns the element at position "i" of dimension one
	 *         and position "j" at dimension two of the two-dimensional
	 *         array.
	 *
	 *  \param i row-position of the element that will be returned.
	 *  \param j column-position of the element that will be returned.
	 *  \return The i-th, j-th array element
	 *  \throw SharkException if the type of the exception is "size mismatch",
	 *         then the array is not 2-dimensional. If the type is
	 *         "range check error", then the value of \em i exceeds
	 *         the size of the first dimension and/or the value of
	 *         \em j exceeds the size of the second dimension
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	T& operator()(unsigned i, unsigned j)
	{
		SIZE_CHECK(nd == 2)
		RANGE_CHECK(i < d[ 0 ])
		RANGE_CHECK(j < d[ 1 ])
		return e[ d[ 1 ] * i + j ];
	}

	//=======================================================================
	/*!
	 *  \ingroup Extract
	 *  \brief Returns the element at position "i" of dimension one
	 *         and position "j" at dimension two of the two-dimensional
	 *         array as constant.
	 *
	 *  \param i row-position of the element that will be returned.
	 *  \param j column-position of the element that will be returned.
	 *  \return The i-th, j-th array element
	 *  \throw SharkException if the type of the exception is "size mismatch",
	 *         then the array is not 2-dimensional. If the type is
	 *         "range check error", then the value of \em i exceeds
	 *         the size of the first dimension and/or the value of
	 *         \em j exceeds the size of the second dimension
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	const T& operator()(unsigned i, unsigned j) const
	{
		SIZE_CHECK(nd == 2)
		RANGE_CHECK(i < d[ 0 ])
		RANGE_CHECK(j < d[ 1 ])
		return e[ d[ 1 ] * i + j ];
	}

	//=======================================================================
	/*!
	 *  \ingroup Extract
	 *  \brief Returns the element at position "i" of dimension one,
	 *         position "j" at dimension two and position "k" of dimension
	 *         three of the three-dimensional array.
	 *
	 *  \param i position in dimension 1 of the element that will be returned.
	 *  \param j position in dimension 2 of the element that will be returned.
	 *  \param k position in dimension 3 of the element that will be returned.
	 *  \return The i-th, j-th, k-th array element
	 *  \throw SharkException if the type of the exception is "size mismatch",
	 *         then the array is not 3-dimensional. If the type is
	 *         "range check error", then the value of \em i exceeds
	 *         the size of the first dimension and/or the value of
	 *         \em j exceeds the size of the second dimension and/or
	 *         the value of \em k exceeds the size of the third dimension
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	T& operator()(unsigned i, unsigned j, unsigned k)
	{
		SIZE_CHECK(nd == 3)
		RANGE_CHECK(i < d[ 0 ])
		RANGE_CHECK(j < d[ 1 ])
		RANGE_CHECK(k < d[ 2 ])
		return e[ d[ 2 ] * d[ 1 ] * i + d[ 2 ] * j + k ];
	}

	//=======================================================================
	/*!
	 *  \ingroup Extract
	 *  \brief Returns the element at position "i" of dimension one,
	 *         position "j" at dimension two and position "k" of dimension
	 *         three of the three-dimensional array as constant.
	 *
	 *  \param i position in dimension 1 of the element that will be returned.
	 *  \param j position in dimension 2 of the element that will be returned.
	 *  \param k position in dimension 3 of the element that will be returned.
	 *  \return The i-th, j-th, k-th array element
	 *  \throw SharkException if the type of the exception is "size mismatch",
	 *         then the array is not 3-dimensional. If the type is
	 *         "range check error", then the value of \em i exceeds
	 *         the size of the first dimension and/or the value of
	 *         \em j exceeds the size of the second dimension and/or
	 *         the value of \em k exceeds the size of the third dimension
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	const T& operator()(unsigned i, unsigned j, unsigned k) const
	{
		SIZE_CHECK(nd == 3)
		RANGE_CHECK(i < d[ 0 ])
		RANGE_CHECK(j < d[ 1 ])
		RANGE_CHECK(k < d[ 2 ])
		return e[ d[ 2 ] * d[ 1 ] * i + d[ 2 ] * j + k ];
	}


	//=======================================================================
	/*!
	 *  \ingroup Extract
	 *  \brief Returns the element identified by the dimension-positions in 
	 *         vector "i" from the current array.
	 *
	 *  \param  i the positions of the wished array element related to
	 *            the different dimensions
	 *  \return The array element with the indices stored in \em i
	 *  \throw SharkException if the type of the exception is "size mismatch",
	 *         then \em i contains more index values, than the current
	 *         array has dimensions. If the type is
	 *         "range check error", then at least one of the index values
	 *         in \em i exceeds the size of the corresponding array dimension
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	T& operator()(const std::vector< unsigned >& i)
	{
		unsigned l, m, n;
		SIZE_CHECK(i.size() == nd)
#ifndef NDEBUG
		for (l = nd; l--;)
		{
			RANGE_CHECK(i[ l ] < d[ l ])
		}
#endif
		for (l = nd, m = 0, n = 1; l--;)
		{
			m += i[ l ] * n;
			n *= d[ l ];
		}

		return e[ m ];
	}


	//=======================================================================
	/*!
	 *  \ingroup Extract
	 *  \brief Returns the element identified by the dimension-positions in 
	 *         vector "i" from the current array as constant.
	 *
	 *  \param  i the positions of the wished array element related to
	 *            the different dimensions
	 *  \return The array element with the indices stored in \em i
	 *  \throw SharkException if the type of the exception is "size mismatch",
	 *         then \em i contains more index values, than the current
	 *         array has dimensions. If the type is
	 *         "range check error", then at least one of the index values
	 *         in \em i exceeds the size of the corresponding array dimension
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	const T& operator()(const std::vector< unsigned >& i) const
	{
		unsigned l, m, n;
		SIZE_CHECK(i.size() == nd)
#ifndef NDEBUG
		for (l = nd; l--;)
		{
			RANGE_CHECK(i[ l ] < d[ l ])
		}
#endif
		for (l = nd, m = 0, n = 1; l--;)
		{
			m += i[ l ] * n;
			n *= d[ l ];
		}

		return e[ m ];
	}


	// Forward reference (see end of class ArrayReference for definition)
	ArrayReference< T > operator [ ](unsigned i);

	// Forward reference (see end of class ArrayReference for definition)
	const ArrayReference< T > operator [ ](unsigned i) const;

	//========================================================================
	/*!
	 *  \ingroup Assign
	 *  \brief Assigns the value "v" to all positions of the current array.
	 *
	 *  \param v the new value for all array elements
	 *  \return the array with the new values
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	Array< T >& operator = (const T& v)
	{
		for (unsigned i = ne; i--; e[ i ] = v);
		return *this;
	}




	//========================================================================
	/*!
	 *  \ingroup Assign
	 *  \brief Assigns the values of vector "v" to the current array.
	 *
	 *  The size of the array is adopted to the size of the vector \em v.
	 *
	 *  \param v vector with the new values for the array
	 *  \return the array with the new values
	 *  \throw SharkException if the type of the exception will be 
	 *         "size mismatch",
	 *         then the current array is a static array reference and has
	 *         not the same size then the array \em v
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	Array< T >& operator = (const std::vector< T >& v)
	{
		resize(v.size());
		for (unsigned i = ne; i--; e[ i ] = v[ i ]);
		return *this;
	}



	//========================================================================
	/*!
	 *  \ingroup Assign
	 *  \brief Assigns the values of array "v" to the current array.
	 *
	 *  The size of the current array is adopted to the size of array \em v.
	 *
	 *  \param v array with the new values for the current array
	 *  \return the current array with the new values
	 *  \throw SharkException the type of the exception will be 
	 *         "size mismatch" and indicates that the current
	 *         array is a static reference and its size is different
	 *         to array \em v
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	Array< T >& operator = (const Array< T >& v)
	{
		//
		// handle the special case of assigning an empty array
		// in order to avoid uninitialized memory read (purify)
		//
		if (v.nd)
		{
			resize_i(v.d, v.nd, false);
			for (unsigned i = ne; i--; e[ i ] = v.e[ i ]);
		}
		else
		{
			if (nd)
			{
				delete[ ] d;
			}
			if (ne)
			{
				delete[ ] e;
			}
			nd = ne = 0;
		}

		return *this;
	}



	//========================================================================
	/*!
	 *  \ingroup Extract
	 *  \brief Returns the i-th element of the arrays element vector Array::e.
	 *
	 *  \param i position of the element in the element vector
	 *  \return the i-th element of the element vector
	 *  \throw SharkException the type of the exception will be 
	 *         "range check error"
	 *         and indicates, that the value of \em i exceeds the total
	 *         number of array elements
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	T& elem(unsigned i)
	{
		RANGE_CHECK(i < ne)
		return e[ i ];
	}


	//========================================================================
	/*!
	 *  \ingroup Extract
	 *  \brief Returns the i-th element of the arrays element vector 
	 *         Array::e as constant.
	 *
	 *  \param i position of the element in the element vector
	 *  \return the i-th element of the element vector
	 *  \throw SharkException the type of the exception will be 
	 *         "range check error"
	 *         and indicates, that the value of \em i exceeds the total
	 *         number of array elements
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	const T& elem(unsigned i) const
	{
		RANGE_CHECK(i < ne)
		return e[ i ];
	}


	//========================================================================
	/*!
	 *  \brief Returns the element vector Array::e of the array.
	 *
	 *  \return the element vector of the current array 
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	T* elemvec()
	{
		return e;
	}


	//========================================================================
	/*!
	 *  \brief Returns the element vector Array::e of the array as constant.
	 *
	 *  \return the element vector of the current array 
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	const T* elemvec() const
	{
		return e;
	}


	//========================================================================
	/*!
	 *  \ingroup Add
	 *  \brief Appends value "w" to the current one-dimensional or empty array.
	 *
	 *  \em w will be appended to the end of the element vector Array::e.
	 *
	 *  \param w the value that will be appended.
	 *  \return the current array with the appended value
	 *  \throw SharkException the type of the exception will be "size mismatch"
	 *         and indicates, that the array has two or more dimensions
	 *         or that the current array is a static reference, that can not be
	 *         resized
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	Array< T >& append_elem(const T& w)
	{
		SIZE_CHECK(nd == 0 || nd == 1)

		resize(ne + 1, true);
		e[ ne - 1 ] = w;
		return *this;
	}


	//========================================================================
	/*!
	 *  \ingroup Add
	 *  \brief Appends the values in "w" to the current one-dimensional or 
	 *         empty array.
	 *
	 *  The values in \em w will be appended to the end of the element vector
	 *  Array::e.
	 *
	 *  \param w one-dimensional array with values that will be appended
	 *           to the current array
	 *  \return the current array with the appended values
	 *  \throw SharkException the type of the exception will be "size mismatch"
	 *         and indicates, that the current array has 2 or more
	 *         dimensions or that \em w is not one-dimensional or that
	 *         the current array is a static array reference, where no
	 *         elements can be appended
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	Array< T >& append_elems(const Array< T >& w)
	{
		SIZE_CHECK((nd == 0 || nd == 1) && w.nd == 1)

		unsigned i = 0, j = ne;

		resize(ne + w.ne, true);

		while (i < w.ne)
		{
			e[ j++ ] = w.e[ i++ ];
		}

		return *this;
	}



	//========================================================================
	/*!
	 *  \ingroup Add
	 *  \brief Appends the array "y" after the last row of the current array.
	 *
	 *  The current array must be empty or must have the same dimension
	 *  than array \em y or must have one dimension more than array
	 *  \em y. If the length or "row length" of array \em y is
	 *  less than the row length of the current array, the missing
	 *  "row positions" will be filled by zero values. <br>
	 *  "Row" here always refer to the first dimension of the array.
	 *
	 *  \param y the array that will be appended
	 *  \return the current array with the appended array \em y
	 *  \throw SharkException the type of the exception will be
	 *         "size mismatch" and indicates that the number of
	 *         dimensions of the current array is not compliant
	 *         to this operation
	 *
	 *  \par Example
	 *  Guess, the current array has the content <br>
	 *
	 *  \f$
	 *  \left(\begin{array}{llll}
	 *      10 & 11 & 12 & 13\\
	 *      14 & 15 & 16 & 17\\
	 *  \end{array}\right)
	 *  \f$
	 *
	 *  and the array \em y has the content \f$( 18 \mbox{\ \ } 19 )\f$. <br>
	 *  Then the resulting array will look like this: <br>
	 *
	 *  \f$
	 *  \left(\begin{array}{llll}
	 *      10 & 11 & 12 & 13\\
	 *      14 & 15 & 16 & 17\\
	 *      18 & 19 &  0 &  0
	 *  \end{array}\right)
	 *  \f$
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	Array< T >& append_rows(const Array< T >& y)
	{
		SIZE_CHECK(ndim() == 0         ||
				   ndim() == y.ndim() ||
				   ndim() == y.ndim() + 1)

		unsigned i, pos = nelem();
		Array< unsigned > da;//is used as to store the new dimensions

		// Append rows to empty array:
		if (ndim() == 0)
		{
			da.resize(y.ndim() + 1);
			da(0) = 1;
			for (i = 0; i < y.ndim(); ++i)
			{
				da(i + 1) = y.dim(i);
			}
		}
		// Append rows if dimensions are equal
		else if (ndim() == y.ndim())
		{
		  //dimarr() returns a copy of the current arrays 
		  //dimension vector as an array
			da = dimarr();
			da(0) = dim(0) + y.dim(0);
		}
		// Append rows if y's dimension is one less the 
		// the arrays dimension
		else if (ndim() == y.ndim() + 1)
		{
			da = dimarr();
			da(0) ++;
		}
		// Otherwise do nothing and return the original array
		else
		{
			return *this;
		}
		// resize the array, keep its elements and fill 
		// the new rows with zeros
		resize_i(da.elemvec(), da.nelem(), true);
		// add the new elements
		for (i = 0; i < y.nelem();)
		{
			elem(pos++) = y.elem(i++);
		}

		return *this;
	}




	//========================================================================
	/*!
	 *  \ingroup Del
	 *  \brief Removes row "i" from the current non-empty array.
	 *
	 *  "Row" refers here always to the first dimension of
	 *  the array.
	 *
	 *  \param i the number of the row that will be removed, must be less
	 *           than the size of the first dimension
	 *  \return the array without row \em i
	 *  \throw SharkException the type of the exception will be
	 *         "range check error" and indicates, that the
	 *         current array has no dimensions or that \em i
	 *         is greater than the size of the first dimension
	 *         of the current array
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	Array< T >& remove_row(unsigned i)
	{
		RANGE_CHECK(ndim() > 0 && i < dim(0))

		unsigned k;
		unsigned size = nelem() / dim(0);// number of rows

		Array< T        > y;// will be the new array
		Array< unsigned > da(dimarr());// stores the new dimensions

		da(0)--;

		y.resize_i(da.elemvec(), da.nelem(), false);
		// copy all rows with indices smaller than the deleted row
		for (k = 0; k < i * size; ++k)
		{
			y.elem(k) = elem(k);
		}
		// copy all rows with indices greater than the deleted row
		for (; k < y.nelem(); ++k)
		{
			y.elem(k) = elem(k + size);
		}
		// replace the current array by y
		return *this = y;
	}




	//========================================================================
	/*!
	 *  \ingroup Del
	 *  \brief Removes column "k" from the current non-empty array.
	 *
	 *  "Column" refers here always to the last dimension of
	 *  the array.
	 *
	 *  \param k the number of the column that will be removed, must
	 *           be less than the size of the last dimension
	 *  \return the array without column \em i
	 *  \throw  SharkException the type of the exception will be
	 *          "range check error" and indicates that the current
	 *          array has no dimensions or that \em k exceeds the
	 *          lats dimension of the current array
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	Array< T > remove_col(unsigned k) const
	{
		RANGE_CHECK(ndim() > 0 && k < dim(ndim() - 1))

		unsigned i, j, xi, zi;
		unsigned xlast = dim(ndim() - 1);
		//number of columns
		unsigned num   = nelem() / xlast;
		//the new array
		Array< T > z;
		//stores the dimensions of the new array
		Array< unsigned > da(dimarr());
		//new array has one column less
		da(ndim() - 1) = xlast - 1;
		//resize z according to da
		z.resize_i(da.elemvec(), da.nelem(), false);
		// copy all kept columns in z 
		for (xi = zi = i = 0; i < num; ++i)
		{
		  // copy all columns with a lower index than k
			for (j = 0; j < k; ++j)
			{
				z.elem(zi++) = elem(xi++);
			}
		  // copy all columns with a greater index than k
			for (++xi, ++j; j < xlast; ++j)
			{
				z.elem(zi++) = elem(xi++);
			}
		}
		//return new array
		return Array< T > (z, true);
	}




	//========================================================================
	/*!
	 *  \ingroup Del
	 *  \brief Removes the columns with the indices stored in array
	 *         "idx" from the current non-empty array.
	 *
	 *  "Column" refers here always to the last dimension of
	 *  the array.
	 *
	 *  \param idx one-dimensional array with the indices of the
	 *             columns that will be removed
	 *  \return the array without the columns defined in \em idx
	 *  \throw SharkException the type of the exception will be
	 *         "range check error" and indicates that the current
	 *         array has no dimensions or that \em idx is not
	 *         one-dimensional or that \em idx contains values,
	 *         that are greater than the size of the last dimension of the
	 *         current array
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 */
	Array<T>& remove_cols(const Array<unsigned> idx)
	{
		RANGE_CHECK(ndim() > 0 && idx.ndim() == 1);

		if (idx.nelem() == 0) return *this;
		std::sort(idx.begin(), idx.end());

		T* source = e;
		T* dest = e;
		int i;
		int lastDim = ndim() - 1;
		int r, rc = idx.nelem();
		int f, fc = dim(lastDim);
		RANGE_CHECK(idx(rc - 1) < fc);
		for (i=0, r=0, f=0; i<ne; i++)
		{
			if (r < rc && f == idx(r))
			{
				source++;
				r++;
			}
			else
			{
				*dest = *source;
				source++;
				dest++;
			}
			f++;
			if (f == fc)
			{
				f = 0;
				r = 0;
			}
		}

		std::vector<unsigned> newdim(ndim());
		for (i=0; i<lastDim; i++) newdim[i] = d[i];
		newdim[lastDim] = fc - rc;
		resize_i(&newdim[0], ndim(), true);

		return *this;
	}




	//========================================================================
	/*!
	 *  \ingroup Del
	 *  \brief Removes the two columns with the indices "i" and "j" 
	 *         from the current non-empty array.
	 *
	 *  "Column" refers here always to the last dimension of
	 *  the array.
	 *
	 *  \param i index of the first column that will be removed
	 *  \param j index of the second column that will be removed
	 *  \return the array without the columns \em i and \em j
	 *  \throw SharkException the type of the exception will be
	 *         "range check error" and indicates that the current
	 *         array has no dimensions or that \em i or \em j
	 *         are greater than the size of the last dimension of the
	 *         current array
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 *  \sa #remove_cols(const Array< unsigned >)
	 *
	 */
	Array< T > & remove_cols(unsigned i, unsigned j)
	{
		Array< unsigned > idx(2);

		idx(0) = i;
		idx(1) = j;

		return remove_cols(idx);
	}



	//========================================================================
	/*!
	 *  \ingroup Del
	 *  \brief Removes the three columns with the indices "i", "j" and "k"
	 *         from the current non-empty array.
	 *
	 *  "Column" refers here always to the last dimension of
	 *  the array.
	 *
	 *  \param i index of the first column that will be removed
	 *  \param j index of the second column that will be removed
	 *  \param k index of the third column that will be removed
	 *  \return the array without the columns \em i, \em j and \em k
	 *  \throw SharkException the type of the exception will be
	 *         "range check error" and indicates that the current
	 *         array has no dimensions or that \em i or \em j or \em k
	 *         are greater than the size of the last dimension of the
	 *         current array
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 *  \sa #remove_cols(const Array< unsigned >)
	 *
	 */
	Array< T > & remove_cols(unsigned i, unsigned j, unsigned k)
	{
		Array< unsigned > idx(3);

		idx(0) = i;
		idx(1) = j;
		idx(2) = k;

		return remove_cols(idx);
	}



	//========================================================================
	/*!
	 *  \ingroup Del
	 *  \brief Removes the four columns with the indices "i", "j", "k"
	 *         and "l" from the current non-empty array.
	 *
	 *  "Column" refers here always to the last dimension of
	 *  the array.
	 *
	 *  \param i index of the first column that will be removed
	 *  \param j index of the second column that will be removed
	 *  \param k index of the third column that will be removed
	 *  \param l index of the 4-th column that will be removed
	 *  \return the array without the columns \em i, \em j, \em k and \em l
	 *  \throw SharkException the type of the exception will be
	 *         "range check error" and indicates that the current
	 *         array has no dimensions or that \em i, \em j, \em k
	 *         or \em l are greater than the size of the last dimension of the
	 *         current array
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 *  \sa #remove_cols(const Array< unsigned >)
	 *
	 */
	Array< T > & remove_cols(unsigned i, unsigned j, unsigned k,
						   unsigned l)
	{
		Array< unsigned > idx(4);

		idx(0) = i;
		idx(1) = j;
		idx(2) = k;
		idx(3) = l;

		return remove_cols(idx);
	}



	//========================================================================
	/*!
	 *  \ingroup Del
	 *  \brief Removes the five columns with the indices "i", "j", "k",
	 *         "l" and "m" from the current non-empty array.
	 *
	 *  "Column" refers here always to the last dimension of
	 *  the array.
	 *
	 *  \param i index of the first column that will be removed
	 *  \param j index of the second column that will be removed
	 *  \param k index of the third column that will be removed
	 *  \param l index of the 4-th column that will be removed
	 *  \param m index of the 5-th column that will be removed
	 *  \return the array without the columns \em i, \em j, \em k, \em l
	 *          and \em m
	 *  \throw SharkException the type of the exception will be
	 *         "range check error" and indicates that the current
	 *         array has no dimensions or that \em i, \em j, \em k, \em l
	 *         or \em m are greater than the size of the last dimension of the
	 *         current array
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 *  \sa #remove_cols(const Array< unsigned >)
	 *
	 */
	Array< T > & remove_cols(unsigned i, unsigned j, unsigned k,
						   unsigned l, unsigned m)
	{
		Array< unsigned > idx(5);

		idx(0) = i;
		idx(1) = j;
		idx(2) = k;
		idx(3) = l;
		idx(4) = m;

		return remove_cols(idx);
	}



	//========================================================================
	/*!
	 *  \ingroup Del
	 *  \brief Removes the six columns with the indices "i", "j", "k",
	 *         "l", "m" and "n" from the current non-empty array.
	 *
	 *  "Column" refers here always to the last dimension of
	 *  the array.
	 *
	 *  \param i index of the first column that will be removed
	 *  \param j index of the second column that will be removed
	 *  \param k index of the third column that will be removed
	 *  \param l index of the 4-th column that will be removed
	 *  \param m index of the 5-th column that will be removed
	 *  \param n index of the 6-th column that will be removed
	 *  \return the array without the columns \em i, \em j, \em k, \em l,
	 *          \em m and \em n
	 *  \throw SharkException the type of the exception will be
	 *         "range check error" and indicates that the current
	 *         array has no dimensions or that \em i, \em j, \em k, \em l,
	 *         \em m or \em n are greater than the size of the last 
	 *         dimension of the current array
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 *  \sa #remove_cols(const Array< unsigned >)
	 *
	 */
	Array< T > & remove_cols(unsigned i, unsigned j, unsigned k,
						   unsigned l, unsigned m, unsigned n)
	{
		Array< unsigned > idx(6);

		idx(0) = i;
		idx(1) = j;
		idx(2) = k;
		idx(3) = l;
		idx(4) = m;
		idx(5) = n;

		return remove_cols(idx);
	}



	//========================================================================
	/*!
	 *  \ingroup Del
	 *  \brief Removes the seven columns with the indices "i", "j", "k",
	 *         "l", "m", "n" and "o" from the current non-empty array.
	 *
	 *  "Column" refers here always to the last dimension of
	 *  the array.
	 *
	 *  \param i index of the first column that will be removed
	 *  \param j index of the second column that will be removed
	 *  \param k index of the third column that will be removed
	 *  \param l index of the 4-th column that will be removed
	 *  \param m index of the 5-th column that will be removed
	 *  \param n index of the 6-th column that will be removed
	 *  \param o index of the 7-th column that will be removed
	 *  \return the array without the columns \em i, \em j, \em k, \em l,
	 *          \em m, \em n and \em o
	 *  \throw SharkException the type of the exception will be
	 *         "range check error" and indicates that the current
	 *         array has no dimensions or that \em i, \em j, \em k, \em l,
	 *         \em m, \em n or \em o are greater than the size of the last 
	 *         dimension of the current array
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 *  \sa #remove_cols(const Array< unsigned >)
	 *
	 */
	Array< T > & remove_cols(unsigned i, unsigned j, unsigned k,
						   unsigned l, unsigned m, unsigned n,
						   unsigned o)
	{
		Array< unsigned > idx(7);

		idx(0) = i;
		idx(1) = j;
		idx(2) = k;
		idx(3) = l;
		idx(4) = m;
		idx(5) = n;
		idx(6) = o;

		return remove_cols(idx);
	}



	//========================================================================
	/*!
	 *  \ingroup Del
	 *  \brief Removes the eight columns with the indices "i", "j", "k",
	 *         "l", "m", "n", "o" and "p" from the current non-empty array.
	 *
	 *  "Column" refers here always to the last dimension of
	 *  the array.
	 *
	 *  \param i index of the first column that will be removed
	 *  \param j index of the second column that will be removed
	 *  \param k index of the third column that will be removed
	 *  \param l index of the 4-th column that will be removed
	 *  \param m index of the 5-th column that will be removed
	 *  \param n index of the 6-th column that will be removed
	 *  \param o index of the 7-th column that will be removed
	 *  \param p index of the 8-th column that will be removed
	 *  \return the array without the columns \em i, \em j, \em k, \em l,
	 *          \em m, \em n, \em o and \em p
	 *  \throw SharkException the type of the exception will be
	 *         "range check error" and indicates that the current
	 *         array has no dimensions or that \em i, \em j, \em k, \em l,
	 *         \em m, \em n, \em o or \em p are greater than the size of 
	 *         the last dimension of the current array
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 *  \sa #remove_cols(const Array< unsigned >)
	 *
	 */
	Array< T > & remove_cols(unsigned i, unsigned j, unsigned k,
						   unsigned l, unsigned m, unsigned n,
						   unsigned o, unsigned p)
	{
		Array< unsigned > idx(8);

		idx(0) = i;
		idx(1) = j;
		idx(2) = k;
		idx(3) = l;
		idx(4) = m;
		idx(5) = n;
		idx(6) = o;
		idx(7) = p;

		return remove_cols(idx);
	}


	//========================================================================
	/*!
	 *  \ingroup Add
	 *  \brief Appends the array "y" after the last column of the current array.
	 *
	 *  The current array must be empty or must have the same dimension
	 *  than array \em y. <br>
	 *  "Column" here always refer to the last dimension of the array.
	 *
	 *  \param y the array that will be appended
	 *  \return the current array with the appended array \em y
	 *  \throw SharkException the type of the exception will be
	 *         "size mismatch" and indicates that the current array
	 *         has one or more dimensions with the number of dimensions
	 *         different to that of \em y
	 *
	 *  \par Example
	 *  Guess, the current array has the content <br>
	 *
	 *  \f$
	 *  \left(\begin{array}{llll}
	 *      10 & 11 & 12 & 13\\
	 *      14 & 15 & 16 & 17\\
	 *  \end{array}\right)
	 *  \f$
	 *
	 *  and the array \em y has the content 
	 *
	 *  \f$
	 *  \left(\begin{array}{ll}
	 *      18 & 19\\
	 *      20 & 21
	 *  \end{array}\right)
	 *  \f$
	 *
	 *  Then the resulting array will look like this: <br>
	 *
	 *  \f$
	 *  \left(\begin{array}{llllll}
	 *      10 & 11 & 12 & 13 & 18 & 19\\
	 *      14 & 15 & 16 & 17 & 20 & 21
	 *  \end{array}\right)
	 *  \f$
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	Array< T > append_cols(const Array< T >& y) const
	{
		SIZE_CHECK(ndim() == 0 || ndim() == y.ndim())

		if (ndim() == 0)
		{
			return y;
		}
		else
		{
			unsigned i, j, xi, yi, zi;
			//number of last column
			unsigned xlast = dim(ndim() - 1);
			//number of last columns in y
			unsigned ylast = y.dim(y.ndim() - 1);
			unsigned num;
			//new array
			Array< T > z;
			//stores new dimensions
			Array< unsigned > da(dimarr());
			//new number of columns
			da(ndim() - 1) = xlast + ylast;
			//resize new array according to da
			z.resize_i(da.elemvec(), da.nelem(), false);

			num = z.nelem() / (xlast + ylast);//number of rows
			for (i = xi = yi = zi = 0; i < num; ++i)
			{
			  //copy all elements from old array
				for (j = 0; j < xlast; ++j)
				{
					z.elem(zi++) = elem(xi++);
				}
			  //copy all elements from y
				for (j = 0; j < ylast; ++j)
				{
					z.elem(zi++) = y.elem(yi++);
				}
			}
			//return new array
			return Array< T > (z, true);
		}
	}




	//========================================================================
	/*!
	 *  \ingroup Extract
	 *  \brief Returns a subarray from position "from" to position "to"
	 *         of the current array.
	 * 
	 *  If the first dimension of the current array \f$A\f$ is 
	 *  \f$d1\f$ and \f$from \leq to < d1\f$, then an array containing
	 *  the subarrays \f$A[ \mbox{from} ] \dots A[ \mbox{to} ]\f$ is returned.
	 *
	 *  \param  from the first position of the subarray, must be less than
	 *               the size of the first dimension
	 *  \param  to   the last position of the subarray, must be less than
	 *               the size of the first dimension
	 *  \return the subarray
	 *  \throw  SharkException the type of the exception will be
	 *          "size mismatch" if the current array has no dimensions
	 *          and "range check error" if \em from is greater than
	 *          \em to or at least one of the both values exceeds the
	 *          size of the first dimension of the current array
	 *
	 *  \par Example
	 * 
	 *  Given the following \f$4 \times 2\f$ array <br>
	 *
	 *  \f$
	 *      \left(\begin{array}{ll}
	 *          1. & 2.\\
	 *          3. & 4.\\
	 *          5. & 6.\\
	 *          7. & 8.
	 *      \end{array}\right)
	 *  \f$ 
	 *
	 *  and \f$\mbox{from} = 1 < 4\f$ and \f$\mbox{to} = 2 < 4\f$, the
	 *  subarray <br>
	 *
	 *  \f$
	 *      \left(\begin{array}{ll}
	 *          3. & 4.\\
	 *          5. & 6.\\
	 *      \end{array}\right)
	 *  \f$ 
	 *
	 *  is returned.
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	Array< T > subarr(unsigned from, unsigned to) const
	{
		SIZE_CHECK(ndim() > 0)
		RANGE_CHECK(from <= to && from < dim(0) && to < dim(0))

		unsigned i, j;
		//number of first element
		unsigned fromi = from   * nelem() / dim(0);
		//number of last element
		unsigned toi   = (to + 1) * nelem() / dim(0);
		//subarray
		Array< T > z;
		//stores dimensions of subarray
		Array< unsigned > da(dimarr());
		// number of rows in subarray
		da(0) = to - from + 1;
		//resize subarray
		z.resize_i(da.elemvec(), da.nelem(), false);
		//copy elements
		for (i = fromi, j = 0; i < toi; z.elem(j++) = elem(i++));
		//return subarray
		return Array< T > (z, true);
	}




	//========================================================================
	/*!
	 *  \ingroup Extract
	 *  \brief Given an absolute position "p" in the element vector
	 *         Array::e,
	 *         the positions relative to the single dimensions
	 *         of the array are returned.
	 * 
	 *  \param  p the absolute position
	 *  \return an array with the positions due to the single
	 *          dimensions of the current Array object
	 *  \throw SharkException the type of the exception will
	 *         be "range check error" and indicates that \em p
	 *         exceeds the number of elements of the current array
	 *
	 *  \par Example
	 *  Guess you have an \f$5 \times 4\f$ array, i.e. an array
	 *  with a total number of 20 elements. You want to extract
	 *  the element at the absolute position 13 of the element
	 *  vector (containing elements with indices 0 to 19). <br>
	 *  Transformed to the array as seen by the user, you will
	 *  get the relative positions \f$(3,1)\f$, because
	 *  \f$3 \ast 4 + 1 = 13\f$.
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	Array< unsigned > pos2idx(unsigned p)
	{
		RANGE_CHECK(p < ne)

		Array< unsigned > idx(ndim());
		for (unsigned i = 0, j = nelem(); i < ndim(); ++i)
		{
			p %= j;
			j /= dim(i);
			idx(i) = p / j;
		}
		return idx;
	}



	//========================================================================
	/*!
	 *  \brief Given an absolute position "p" in the element vector
	 *         Array::e,
	 *         the positions relative to the single dimensions
	 *         are stored in "idx".
	 * 
	 *  \param p   the absolute position
	 *  \param idx an array with the relative positions
	 *  \return none
	 *  \throw SharkException the type of the exception will
	 *         be "range check error" and indicates that \em p
	 *         exceeds the number of elements of the current array
	 *
	 *  \par Example
	 *  Guess you have an \f$5 \times 4\f$ array, i.e. an array
	 *  with a total number of 20 elements. You want to extract
	 *  the element at the absolute position 13 of the element
	 *  vector (containing elements with indices 0 to 19). <br>
	 *  Transformed to the array as seen by the user, you will
	 *  get the relative positions \f$(3,1)\f$, because
	 *  \f$3 \ast 4 + 1 = 13\f$.
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	void pos2idx(unsigned p, Array<unsigned>& idx)
	{
		RANGE_CHECK(p < ne)

		idx.resize(ndim());
		for (unsigned i = 0, j = nelem(); i < ndim(); ++i)
		{
			p %= j;
			j /= dim(i);
			idx(i) = p / j;
		}
	}


	//========================================================================
	/*!
	 *  \ingroup Extract
	 *  \brief Returns the row number "i" of the current array.
	 *
	 *  Row here always refers to the first dimension of the array.
	 *
	 *  \param i the number of the row that will be returned, must be
	 *           less than the size of dimension one.
	 *  \return the row number \em i
	 *  \throw SharkException the type of the exception will be
	 *         "size mismatch" if the current array has no dimensions
	 *         and "range check error" if \em i exceeds the size of
	 *         the array's first dimension
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	Array< T > row(unsigned i) const
	{
		return (*this)[ i ];
	}


	//========================================================================
	/*!
	 *  \ingroup Extract
	 *  \brief Returns the array rows with the numbers defined in array
	 *         "idx" from the current non-empty array.
	 *
	 *  Row here always refers to the first dimension of the array.
	 *
	 *  \param idx one-dimensional or empty array with the numbers of the
	 *             rows that will be returned
	 *  \return an array containing all the chosen rows
	 *  \throw SharkException the type of the exception will
	 *         be "size mismatch" and if the current array
	 *         has no dimensions or \em idx is more than
	 *         one-dimensional and "range check error" if
	 *         at least one of the values in \em idx exceeds the
	 *         size of the first dimension of the current array
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	Array< T > rows(const Array< unsigned >& idx) const
	{
		SIZE_CHECK(ndim() > 0)
		SIZE_CHECK(idx.ndim() <= 1)

		unsigned i, j, k, l;
		//number of rows
		unsigned n = nelem() / dim(0);
		//subarray
		Array< T > z;
		//stores dimensions of subarray
		Array< unsigned > da(dimarr());

		da(0) = idx.nelem();
		//resize subarray
		z.resize_i(da.elemvec(), da.nelem(), false);
		//copy all elements mentioned in idx 
		for (k = i = 0; i < idx.nelem(); ++i)
		{
			for (l = idx(i) * n, j = 0; j < n; ++j)
			{
				z.elem(k++) = elem(l++);
			}
		}
		//return subarray
		return Array< T > (z, true);
	}



	//========================================================================
	/*!
	 *  \ingroup Extract
	 *  \brief Returns the array row number "i" from the current non-empty
	 *         array. 
	 *
	 *  Row here always refers to the first dimension of the array.
	 *
	 *  \param i number of the row that will be returned, must be less than
	 *           the size of dimension one.
	 *  \return an array containing row no. \em i
	 *  \throw SharkException the type of the exception will
	 *         be "size mismatch" and if the current array
	 *         has no dimensions or \em idx is more than
	 *         one-dimensional and "range check error" if \em i exceeds the
	 *         size of the first dimension of the current array
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 *  \sa #rows(const Array< unsigned >&)
	 *
	 */
	Array< T > rows(unsigned i) const
	{
		Array< unsigned > idx(1);

		idx(0) = i;

		return rows(idx);
	}



	//========================================================================
	/*!
	 *  \ingroup Extract
	 *  \brief Returns the array rows with the numbers "i" and "j" from the 
	 *         current non-empty array. 
	 *
	 *  Row here always refers to the first dimension of the array.
	 *
	 *  \param i number of the first row that will be returned, must be less 
	 *           than the size of dimension one.
	 *  \param j number of the second row that will be returned, must be less 
	 *           than the size of dimension one.
	 *  \return an array containing rows no. \em i and \em j
	 *  \throw SharkException the type of the exception will
	 *         be "size mismatch" and if the current array
	 *         has no dimensions or \em idx is more than
	 *         one-dimensional and "range check error" if \em i or 
	 *         \em j exceed the size of the first dimension of the 
	 *         current array
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 *  \sa #rows(const Array< unsigned >&)
	 *
	 */
	Array< T > rows(unsigned i, unsigned j) const
	{
		Array< unsigned > idx(2);

		idx(0) = i;
		idx(1) = j;

		return rows(idx);
	}



	//========================================================================
	/*!
	 *  \ingroup Extract
	 *  \brief Returns the array rows with the numbers "i", "j" and "k" 
	 *         from the current non-empty array. 
	 *
	 *  Row here always refers to the first dimension of the array.
	 *
	 *  \param i number of the first row that will be returned, must be less 
	 *           than the size of dimension one.
	 *  \param j number of the second row that will be returned, must be less 
	 *           than the size of dimension one.
	 *  \param k number of the third row that will be returned, must be less 
	 *           than the size of dimension one.
	 *  \return an array containing rows no. \em i, \em j and \em k
	 *  \throw SharkException the type of the exception will
	 *         be "size mismatch" and if the current array
	 *         has no dimensions or \em idx is more than
	 *         one-dimensional and "range check error" if \em i or 
	 *         \em j or \em k exceed the size of the first dimension of the 
	 *         current array
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 *  \sa #rows(const Array< unsigned >&)
	 *
	 */
	Array< T > rows(unsigned i, unsigned j, unsigned k) const
	{
		Array< unsigned > idx(3);

		idx(0) = i;
		idx(1) = j;
		idx(2) = k;

		return rows(idx);
	}



	//========================================================================
	/*!
	 *  \ingroup Extract
	 *  \brief Returns the array rows with the numbers "i", "j", "k" and "l"
	 *         from the current non-empty array. 
	 *
	 *  Row here always refers to the first dimension of the array.
	 *
	 *  \param i number of the first row that will be returned, must be less 
	 *           than the size of dimension one.
	 *  \param j number of the second row that will be returned, must be less 
	 *           than the size of dimension one.
	 *  \param k number of the third row that will be returned, must be less 
	 *           than the size of dimension one.
	 *  \param l number of the 4-th row that will be returned, must be less 
	 *           than the size of dimension one.
	 *  \return an array containing rows no. \em i, \em j, \em k and \em l
	 *  \throw SharkException the type of the exception will
	 *         be "size mismatch" and if the current array
	 *         has no dimensions or \em idx is more than
	 *         one-dimensional and "range check error" if \em i,
	 *         \em j, \em k or \em l exceed the size of the first 
	 *         dimension of the current array
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 *  \sa #rows(const Array< unsigned >&)
	 *
	 */
	Array< T > rows(unsigned i, unsigned j, unsigned k,
					unsigned l) const
	{
		Array< unsigned > idx(4);

		idx(0) = i;
		idx(1) = j;
		idx(2) = k;
		idx(3) = l;

		return rows(idx);
	}



	//========================================================================
	/*!
	 *  \ingroup Extract
	 *  \brief Returns the array rows with the numbers "i", "j", "k", "l"
	 *         and "m" from the current non-empty array. 
	 *
	 *  Row here always refers to the first dimension of the array.
	 *
	 *  \param i number of the first row that will be returned, must be less 
	 *           than the size of dimension one.
	 *  \param j number of the second row that will be returned, must be less 
	 *           than the size of dimension one.
	 *  \param k number of the third row that will be returned, must be less 
	 *           than the size of dimension one.
	 *  \param l number of the 4-th row that will be returned, must be less 
	 *           than the size of dimension one.
	 *  \param m number of the 5-th row that will be returned, must be less 
	 *           than the size of dimension one.
	 *  \return an array containing rows no. \em i, \em j, \em k, \em l
	 *          and \em m
	 *  \throw SharkException the type of the exception will
	 *         be "size mismatch" and if the current array
	 *         has no dimensions or \em idx is more than
	 *         one-dimensional and "range check error" if \em i,
	 *         \em j, \em k, \em l or \em m exceed the size of the first 
	 *         dimension of the current array
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 *  \sa #rows(const Array< unsigned >&)
	 *
	 */
	Array< T > rows(unsigned i, unsigned j, unsigned k,
					unsigned l, unsigned m) const
	{
		Array< unsigned > idx(5);

		idx(0) = i;
		idx(1) = j;
		idx(2) = k;
		idx(3) = l;
		idx(4) = m;

		return rows(idx);
	}



	//========================================================================
	/*!
	 *  \ingroup Extract
	 *  \brief Returns the array rows with the numbers "i", "j", "k", "l",
	 *         "m" and "n" from the current non-empty array. 
	 *
	 *  Row here always refers to the first dimension of the array.
	 *
	 *  \param i number of the first row that will be returned, must be less 
	 *           than the size of dimension one.
	 *  \param j number of the second row that will be returned, must be less 
	 *           than the size of dimension one.
	 *  \param k number of the third row that will be returned, must be less 
	 *           than the size of dimension one.
	 *  \param l number of the 4-th row that will be returned, must be less 
	 *           than the size of dimension one.
	 *  \param m number of the 5-th row that will be returned, must be less 
	 *           than the size of dimension one.
	 *  \param n number of the 6-th row that will be returned, must be less 
	 *           than the size of dimension one.
	 *  \return an array containing rows no. \em i, \em j, \em k, \em l,
	 *          \em m and \em n
	 *  \throw SharkException the type of the exception will
	 *         be "size mismatch" and if the current array
	 *         has no dimensions or \em idx is more than
	 *         one-dimensional and "range check error" if \em i,
	 *         \em j, \em k, \em l, \em m or \em n exceed the size of the first 
	 *         dimension of the current array
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 *  \sa #rows(const Array< unsigned >&)
	 *
	 */
	Array< T > rows(unsigned i, unsigned j, unsigned k,
					unsigned l, unsigned m, unsigned n) const
	{
		Array< unsigned > idx(6);

		idx(0) = i;
		idx(1) = j;
		idx(2) = k;
		idx(3) = l;
		idx(4) = m;
		idx(5) = n;

		return rows(idx);
	}



	//========================================================================
	/*!
	 *  \ingroup Extract
	 *  \brief Returns the array rows with the numbers "i", "j", "k", "l",
	 *         "m", "n" and "o" from the current non-empty array. 
	 *
	 *  Row here always refers to the first dimension of the array.
	 *
	 *  \param i number of the first row that will be returned, must be less 
	 *           than the size of dimension one.
	 *  \param j number of the second row that will be returned, must be less 
	 *           than the size of dimension one.
	 *  \param k number of the third row that will be returned, must be less 
	 *           than the size of dimension one.
	 *  \param l number of the 4-th row that will be returned, must be less 
	 *           than the size of dimension one.
	 *  \param m number of the 5-th row that will be returned, must be less 
	 *           than the size of dimension one.
	 *  \param n number of the 6-th row that will be returned, must be less 
	 *           than the size of dimension one.
	 *  \param o number of the 7-th row that will be returned, must be less 
	 *           than the size of dimension one.
	 *  \return an array containing rows no. \em i, \em j, \em k, \em l,
	 *          \em m, \em n and \em o
	 *  \throw SharkException the type of the exception will
	 *         be "size mismatch" and if the current array
	 *         has no dimensions or \em idx is more than
	 *         one-dimensional and "range check error" if \em i,
	 *         \em j, \em k, \em l, \em m, \em n or \em p exceed the size 
	 *         of the first dimension of the current array
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 *  \sa #rows(const Array< unsigned >&)
	 *
	 */
	Array< T > rows(unsigned i, unsigned j, unsigned k,
					unsigned l, unsigned m, unsigned n,
					unsigned o) const
	{
		Array< unsigned > idx(7);

		idx(0) = i;
		idx(1) = j;
		idx(2) = k;
		idx(3) = l;
		idx(4) = m;
		idx(5) = n;
		idx(6) = o;

		return rows(idx);
	}



	//========================================================================
	/*!
	 *  \ingroup Extract
	 *  \brief Returns the array rows with the numbers "i", "j", "k", "l",
	 *         "m", "n", "o" and "p" from the current non-empty array. 
	 *
	 *  Row here always refers to the first dimension of the array.
	 *
	 *  \param i number of the first row that will be returned, must be less 
	 *           than the size of dimension one.
	 *  \param j number of the second row that will be returned, must be less 
	 *           than the size of dimension one.
	 *  \param k number of the third row that will be returned, must be less 
	 *           than the size of dimension one.
	 *  \param l number of the 4-th row that will be returned, must be less 
	 *           than the size of dimension one.
	 *  \param m number of the 5-th row that will be returned, must be less 
	 *           than the size of dimension one.
	 *  \param n number of the 6-th row that will be returned, must be less 
	 *           than the size of dimension one.
	 *  \param o number of the 7-th row that will be returned, must be less 
	 *           than the size of dimension one.
	 *  \param p number of the 8-th row that will be returned, must be less 
	 *           than the size of dimension one.
	 *  \return an array containing rows no. \em i, \em j, \em k, \em l,
	 *          \em m, \em n, \em o and \em p
	 *  \throw SharkException the type of the exception will
	 *         be "size mismatch" and if the current array
	 *         has no dimensions or \em idx is more than
	 *         one-dimensional and "range check error" if \em i,
	 *         \em j, \em k, \em l, \em m, \em n, \em p or \em p exceed 
	 *         the size of the first dimension of the current array
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 *  \sa #rows(const Array< unsigned >&)
	 *
	 */
	Array< T > rows(unsigned i, unsigned j, unsigned k,
					unsigned l, unsigned m, unsigned n,
					unsigned o, unsigned p) const
	{
		Array< unsigned > idx(8);

		idx(0) = i;
		idx(1) = j;
		idx(2) = k;
		idx(3) = l;
		idx(4) = m;
		idx(5) = n;
		idx(6) = o;
		idx(7) = p;

		return rows(idx);
	}




	//========================================================================
	/*!
	 *  \ingroup Extract
	 *  \brief Returns the column number "i" of the current array.
	 *
	 *  Column here always refers to the last dimension of the array.
	 *
	 *  \param i the number of the column that will be returned, must be
	 *           less than the size of the last dimension
	 *  \return the column number \em i
	 *  \throw SharkException the type of the exception will be
	 *         "size mismatch" if the current array has no dimensions
	 *         or "range check error" if \em i exceeds the size of the
	 *         last dimension of the current array 
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	Array< T > col(unsigned i) const
	{
		SIZE_CHECK(ndim() > 0)
		RANGE_CHECK(i < dim(ndim() - 1))
		//subarray
		Array< T > z;
		//stores dimensions of subarray
		Array< unsigned > da(dimarr());
		//resize subarray
		z.resize_i(da.elemvec(), da.nelem() - 1, false);
		//copy i'th  col
		for (unsigned k = 0; k < z.nelem(); ++k, i += dim(ndim() - 1))
		{
			z.elem(k) = elem(i);
		}
		//return subarry
		return z;
	}



	//========================================================================
	/*!
	 *  \ingroup Extract
	 *  \brief Returns the array columns with the numbers defined in array
	 *         "idx" from the current non-empty array.
	 *
	 *  Column here always refers to the last dimension of the array.
	 *
	 *  \param idx one-dimensional or empty array with the numbers of the
	 *             columns that will be returned
	 *  \return an array containing all the chosen columns
	 *  \throw SharkException the type of the exception will be
	 *         "size mismatch" if the current array has no dimensions
	 *         or "range check error" if at least one element of 
	 *         \em idx  exceeds the size of the
	 *         last dimension of the current array 
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	Array< T > cols(const Array< unsigned >& idx) const
	{
		SIZE_CHECK(ndim() > 0)
		SIZE_CHECK(idx.ndim() == 1)

		unsigned j, k, m;
		//number of columns
		unsigned l = dim(ndim() - 1);
		//number of rows
		unsigned n = nelem() / l;
		//subarray
		Array< T > z;
		Array< unsigned > ix(idx);
		//stores dimensions of subarray
		Array< unsigned > da(dimarr());

		da(ndim() - 1) = ix.nelem();
		//resize subarray
		z.resize_i(da.elemvec(), da.nelem(), false);
		//copy all elements mentioned in idx
		for (k = m = 0; k < n; ++k)
		{
			for (j = 0; j < ix.nelem(); ++j, ++m)
			{
				z.elem(m) = elem(ix(j));
				ix(j) += l;//switch to next row
			}
		}
		//return subarray
		return z;
	}



	//========================================================================
	/*!
	 *  \ingroup Extract
	 *  \brief Returns the array column number "i" from the current non-empty
	 *         array. 
	 *
	 *  Column here always refers to the last dimension of the array.
	 *
	 *  \param i number of the column that will be returned, must be less than
	 *           the size of the last dimension
	 *  \return an array containing column no. \em i
	 *  \throw SharkException the type of the exception will be
	 *         "size mismatch" if the current array has no dimensions
	 *         or "range check error" if \em i exceeds the size of the
	 *         last dimension of the current array 
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 *  \sa #cols(const Array< unsigned >&)
	 *
	 */
	Array< T > cols(unsigned i) const
	{
		Array< unsigned > idx(1);

		idx(0) = i;

		return cols(idx);
	}



	//========================================================================
	/*!
	 *  \ingroup Extract
	 *  \brief Returns the array columns with the numbers "i" and "j" from the 
	 *         current non-empty array. 
	 *
	 *  Column here always refers to the last dimension of the array.
	 *
	 *  \param i number of the first column that will be returned, must be less 
	 *           than the size of the last dimension
	 *  \param j number of the second column that will be returned, must be 
	 *           less than the size of the last dimension
	 *  \return an array containing columns no. \em i and \em j
	 *  \throw SharkException the type of the exception will be
	 *         "size mismatch" if the current array has no dimensions
	 *         or "range check error" if \em i or \em j exceed the size of the
	 *         last dimension of the current array 
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 *  \sa #cols(const Array< unsigned >&)
	 *
	 */
	Array< T > cols(unsigned i, unsigned j) const
	{
		Array< unsigned > idx(2);

		idx(0) = i;
		idx(1) = j;

		return cols(idx);
	}



	//========================================================================
	/*!
	 *  \ingroup Extract
	 *  \brief Returns the array columns with the numbers "i", "j" and "k" 
	 *         from the current non-empty array. 
	 *
	 *  Column here always refers to the last dimension of the array.
	 *
	 *  \param i number of the first column that will be returned, must be less 
	 *           than the size of the last dimension
	 *  \param j number of the second column that will be returned, must be 
	 *           less than the size of the last dimension
	 *  \param k number of the third column that will be returned, must be 
	 *           less than the size of the last dimension
	 *  \return an array containing columns no. \em i, \em j and \em k
	 *  \throw SharkException the type of the exception will be
	 *         "size mismatch" if the current array has no dimensions
	 *         or "range check error" if \em i or \em j or \em k 
	 *         exceed the size of the last dimension of the current array 
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 *  \sa #cols(const Array< unsigned >&)
	 *
	 */
	Array< T > cols(unsigned i, unsigned j, unsigned k) const
	{
		Array< unsigned > idx(3);

		idx(0) = i;
		idx(1) = j;
		idx(2) = k;

		return cols(idx);
	}



	//========================================================================
	/*!
	 *  \ingroup Extract
	 *  \brief Returns the array columns with the numbers "i", "j", "k" 
	 *         and "l" from the current non-empty array. 
	 *
	 *  Column here always refers to the last dimension of the array.
	 *
	 *  \param i number of the first column that will be returned, must be less 
	 *           than the size of the last dimension
	 *  \param j number of the second column that will be returned, must be 
	 *           less than the size of the last dimension
	 *  \param k number of the third column that will be returned, must be 
	 *           less than the size of the last dimension
	 *  \param l number of the 4-th column that will be returned, must be 
	 *           less than the size of the last dimension
	 *  \return an array containing columns no. \em i, \em j, \em k and \em l
	 *  \throw SharkException the type of the exception will be
	 *         "size mismatch" if the current array has no dimensions
	 *         or "range check error" if \em i, \em j, \em k or \em l 
	 *         exceed the size of the last dimension of the current array 
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 *  \sa #cols(const Array< unsigned >&)
	 *
	 */
	Array< T > cols(unsigned i, unsigned j, unsigned k,
					unsigned l) const
	{
		Array< unsigned > idx(4);

		idx(0) = i;
		idx(1) = j;
		idx(2) = k;
		idx(3) = l;

		return cols(idx);
	}



	//========================================================================
	/*!
	 *  \ingroup Extract
	 *  \brief Returns the array columns with the numbers "i", "j", "k", 
	 *         "l" and "m" from the current non-empty array. 
	 *
	 *  Column here always refers to the last dimension of the array.
	 *
	 *  \param i number of the first column that will be returned, must be less 
	 *           than the size of the last dimension
	 *  \param j number of the second column that will be returned, must be 
	 *           less than the size of the last dimension
	 *  \param k number of the third column that will be returned, must be 
	 *           less than the size of the last dimension
	 *  \param l number of the 4-th column that will be returned, must be 
	 *           less than the size of the last dimension
	 *  \param m number of the 5-th column that will be returned, must be 
	 *           less than the size of the last dimension
	 *  \return an array containing columns no. \em i, \em j, \em k, \em l
	 *          and \em m
	 *  \throw SharkException the type of the exception will be
	 *         "size mismatch" if the current array has no dimensions
	 *         or "range check error" if \em i, \em j, \em k, \em l 
	 *         or \em m exceed the size of the last dimension of the 
	 *         current array 
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 *  \sa #cols(const Array< unsigned >&)
	 *
	 */
	Array< T > cols(unsigned i, unsigned j, unsigned k,
					unsigned l, unsigned m) const
	{
		Array< unsigned > idx(5);

		idx(0) = i;
		idx(1) = j;
		idx(2) = k;
		idx(3) = l;
		idx(4) = m;

		return cols(idx);
	}



	//========================================================================
	/*!
	 *  \ingroup Extract
	 *  \brief Returns the array columns with the numbers "i", "j", "k", 
	 *         "l", "m" and "n" from the current non-empty array. 
	 *
	 *  Column here always refers to the last dimension of the array.
	 *
	 *  \param i number of the first column that will be returned, must be less 
	 *           than the size of the last dimension
	 *  \param j number of the second column that will be returned, must be 
	 *           less than the size of the last dimension
	 *  \param k number of the third column that will be returned, must be 
	 *           less than the size of the last dimension
	 *  \param l number of the 4-th column that will be returned, must be 
	 *           less than the size of the last dimension
	 *  \param m number of the 5-th column that will be returned, must be 
	 *           less than the size of the last dimension
	 *  \param n number of the 6-th column that will be returned, must be 
	 *           less than the size of the last dimension
	 *  \return an array containing columns no. \em i, \em j, \em k, \em l,
	 *          \em m and \em n
	 *  \throw SharkException the type of the exception will be
	 *         "size mismatch" if the current array has no dimensions
	 *         or "range check error" if \em i, \em j, \em k, \em l, 
	 *         \em m or \em n exceed the size of the last dimension of the 
	 *         current array 
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 *  \sa #cols(const Array< unsigned >&)
	 *
	 */
	Array< T > cols(unsigned i, unsigned j, unsigned k,
					unsigned l, unsigned m, unsigned n) const
	{
		Array< unsigned > idx(6);

		idx(0) = i;
		idx(1) = j;
		idx(2) = k;
		idx(3) = l;
		idx(4) = m;
		idx(5) = n;

		return cols(idx);
	}



	//========================================================================
	/*!
	 *  \ingroup Extract
	 *  \brief Returns the array columns with the numbers "i", "j", "k", 
	 *         "l", "m", "n" and "o" from the current non-empty array. 
	 *
	 *  Column here always refers to the last dimension of the array.
	 *
	 *  \param i number of the first column that will be returned, must be less 
	 *           than the size of the last dimension
	 *  \param j number of the second column that will be returned, must be 
	 *           less than the size of the last dimension
	 *  \param k number of the third column that will be returned, must be 
	 *           less than the size of the last dimension
	 *  \param l number of the 4-th column that will be returned, must be 
	 *           less than the size of the last dimension
	 *  \param m number of the 5-th column that will be returned, must be 
	 *           less than the size of the last dimension
	 *  \param n number of the 6-th column that will be returned, must be 
	 *           less than the size of the last dimension
	 *  \param o number of the 7-th column that will be returned, must be 
	 *           less than the size of the last dimension
	 *  \return an array containing columns no. \em i, \em j, \em k, \em l,
	 *          \em m, \em n and \em o
	 *  \throw SharkException the type of the exception will be
	 *         "size mismatch" if the current array has no dimensions
	 *         or "range check error" if \em i, \em j, \em k, \em l, 
	 *         \em m, \em n or \em o exceed the size of the last dimension 
	 *         of the current array 
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 *  \sa #cols(const Array< unsigned >&)
	 *
	 */
	Array< T > cols(unsigned i, unsigned j, unsigned k,
					unsigned l, unsigned m, unsigned n,
					unsigned o) const
	{
		Array< unsigned > idx(7);

		idx(0) = i;
		idx(1) = j;
		idx(2) = k;
		idx(3) = l;
		idx(4) = m;
		idx(5) = n;
		idx(6) = o;

		return cols(idx);
	}



	//========================================================================
	/*!
	 *  \ingroup Extract
	 *  \brief Returns the array columns with the numbers "i", "j", "k", 
	 *         "l", "m", "n", "o" and "p" from the current non-empty array. 
	 *
	 *  Column here always refers to the last dimension of the array.
	 *
	 *  \param i number of the first column that will be returned, must be less 
	 *           than the size of the last dimension
	 *  \param j number of the second column that will be returned, must be 
	 *           less than the size of the last dimension
	 *  \param k number of the third column that will be returned, must be 
	 *           less than the size of the last dimension
	 *  \param l number of the 4-th column that will be returned, must be 
	 *           less than the size of the last dimension
	 *  \param m number of the 5-th column that will be returned, must be 
	 *           less than the size of the last dimension
	 *  \param n number of the 6-th column that will be returned, must be 
	 *           less than the size of the last dimension
	 *  \param o number of the 7-th column that will be returned, must be 
	 *           less than the size of the last dimension
	 *  \param p number of the 8-th column that will be returned, must be 
	 *           less than the size of the last dimension
	 *  \return an array containing columns no. \em i, \em j, \em k, \em l,
	 *          \em m, \em n, \em o and \em p
	 *  \throw SharkException the type of the exception will be
	 *         "size mismatch" if the current array has no dimensions
	 *         or "range check error" if \em i, \em j, \em k, \em l, 
	 *         \em m, \em n, \em o or \em p exceed the size of the last 
	 *         dimension of the current array 
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 *  \sa #cols(const Array< unsigned >&)
	 *
	 */
	Array< T > cols(unsigned i, unsigned j, unsigned k,
					unsigned l, unsigned m, unsigned n,
					unsigned o, unsigned p) const
	{
		Array< unsigned > idx(8);

		idx(0) = i;
		idx(1) = j;
		idx(2) = k;
		idx(3) = l;
		idx(4) = m;
		idx(5) = n;
		idx(6) = o;
		idx(7) = p;

		return cols(idx);
	}



	//========================================================================
	/*!
	 *  \ingroup Math
	 *  \brief Returns the transposition of the current array.
	 *
	 *  See the current array as matrix \f$A\f$, then the transposed
	 *  matrix \f$A^T\f$ is returned.
	 *
	 *  \return the transposed array
	 *
	 *  \par Example
	 *  Given an array with the content <br>
	 *
	 *  \f$
	 *  \left(\begin{array}{lll}
	 *      1 & 2 & 3\\
	 *      4 & 5 & 6\\
	 *  \end{array}\right)
	 *  \f$
	 *
	 *  then the transposed array will be <br>
	 *
	 *  \f$
	 *  \left(\begin{array}{ll}
	 *      1 & 4\\
	 *      2 & 5\\
	 *      3 & 6
	 *  \end{array}\right)
	 *  \f$
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	Array< T >& transpose()
	{
		switch (ndim())
		{
		case 0 :
		case 1 :
			break;
		case 2 :
		{
			int d0 = d[1];
			int d1 = d[0];
			int i, j, k, s;
			for (i = 0, s = d0 * d1; s > 0; i++)
			{
				for (j = (i % d1) * d0 + i / d1; j > i; j = (j % d1) * d0 + j / d1);
				if (j < i) continue;
				for (k = i, j = (i % d1) * d0 + i / d1; j != i; k = j, j = (j % d1) * d0 + j / d1)
				{
					std::swap(e[k], e[j]);
					s--;
				}
				s--;
			}
			d[0] = d0;
			d[1] = d1;
			break;
		}
		default :
		{
			int d0 = d[1];
			int d1 = d[0];
			int i, j, k, s;
			unsigned a, b, c, size = nelem() / (d0 * d1);
			for (i = 0, s = d0 * d1; s > 0; i++)
			{
				for (j = (i % d1) * d0 + i / d1; j > i; j = (j % d1) * d0 + j / d1);
				if (j < i) continue;
				for (k = i, j = (i % d1) * d0 + i / d1; j != i; k = j, j = (j % d1) * d0 + j / d1)
				{
					a = k * size;
					b = j * size;
					for (c = 0; c < size; c++)
					{
						std::swap(e[a], e[b]);
						a++;
						b++;
					}
					s--;
				}
				s--;
			}
			d[0] = d0;
			d[1] = d1;
			break;
		}
		}

		return *this;
	}

	//========================================================================
	/*!
	 *  \brief The last "n" rows of the array will become the first rows.
	 *
	 *  \param n number of last rows that will become the \em n first rows
	 *  \return the array with the rotated rows
	 *  \throw SharkException the type of the exception will be
	 *         "size mismatch" and indicates that the current array
	 *         has no dimensions
	 *
	 *  \par Example
	 *  Given an array with the content <br>
	 *
	 *  \f$
	 *  \left(\begin{array}{lll}
	 *      1 & 2 & 3\\
	 *      4 & 5 & 6\\
	 *      7 & 8 & 9\\
	 *     10 & 11 & 12
	 *  \end{array}\right)
	 *  \f$
	 *
	 *  and \f$n = 2\f$ then the returned array will be <br>
	 *
	 *  \f$
	 *  \left(\begin{array}{lll}
	 *      7 & 8 & 9\\
	 *      10 & 11 & 12\\
	 *      1 & 2 & 3\\
	 *      4 & 5 & 6
	 *  \end{array}\right)
	 *  \f$
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 */
	Array<T>& rotate_rows(int n)
	{
		SIZE_CHECK(ndim() > 0);

		int f, fc = d[0];
		int i, ic = ne / fc;
		std::vector<T> tmp(fc);

		n = n % fc;
		if (n < 0) n += fc;
		if (n == 0) return *this;
		RANGE_CHECK(0 <= n && n < fc);

		T* data = e;
		for (i=0; i<ic; i++)
		{
			for (f=0; f<fc; f++) tmp[f] = data[ic * f];
			for (f=0; f<n; f++) data[ic * f] = tmp[f - n + fc];
			for (; f<fc; f++) data[ic * f] = tmp[f - n];
			data++;
		}

		return *this;
	}


	//========================================================================
	/*!
	 *  \brief The last "n" columns of the array will become the first columns.
	 *
	 *  \param n number of last columns that will become the \em n first columns
	 *  \return the array with the rotated columns
	 *  \throw SharkException the type of the exception will be
	 *         "size mismatch" and indicates that the current array
	 *         has no dimensions
	 *
	 *  \par Example
	 *  Given an array with the content <br>
	 *
	 *  \f$
	 *  \left(\begin{array}{lll}
	 *      1 & 2 & 3\\
	 *      4 & 5 & 6\\
	 *      7 & 8 & 9\\
	 *     10 & 11 & 12
	 *  \end{array}\right)
	 *  \f$
	 *
	 *  and \f$n = 2\f$ then the returned array will be <br>
	 *
	 *  \f$
	 *  \left(\begin{array}{lll}
	 *      2 & 3 & 1\\
	 *      5 & 6 & 4\\
	 *      8 & 9 & 7\\
	 *      11 & 12 & 10
	 *  \end{array}\right)
	 *  \f$
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	Array<T>& rotate_cols(int n)
	{
		SIZE_CHECK(ndim() > 0);

		int lastDim = nd - 1;
		int f, fc = d[lastDim];
		int i, ic = ne / fc;
		std::vector<T> tmp(fc);

		n = n % fc;
		if (n < 0) n += fc;
		if (n == 0) return *this;
		RANGE_CHECK(0 < n && n < fc);

		T* data = e;
		for (i=0; i<ic; i++)
		{
			for (f=0; f<fc; f++) tmp[f] = data[f];
			for (f=0; f<n; f++) data[f] = tmp[f - n + fc];
			for (; f<fc; f++) data[f] = tmp[f - n];
			data += fc;
		}

		return *this;
	}


	//========================================================================
	/*!
	 *  \ingroup Copy
	 *  \brief Returns an array that has the same content as the
	 *         dimension vector ArrayBase::d of the current Array object.
	 * 
	 *  \return the dimension vector as array
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 *  \sa ArrayBase::clone
	 */
	Array< unsigned > dimarr() const
	{
		Array< unsigned > dim(nd);
		for (unsigned i = nd; i--; dim(i) = d[ i ]);
		return dim;
	}



	//========================================================================
	/*!
	 *  \ingroup Copy
	 *  \brief Returns an identical copy of this Array object.
	 * 
	 *  Creates a new array that is identical to the current one
	 *  and returns a pointer to this clone.
	 *
	 *  \return a copy of the current array
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 *  \sa ArrayBase::clone
	 */
	ArrayBase* clone() const
	{
		return new Array< T > (*this);
	}


	//========================================================================
	/*!
	 *  \ingroup Constructor
	 *  \brief Returns an empty array with the same type as this array object.
	 * 
	 *  A new empty Array object with the same type as the current array is
	 *  created and a pointer to this new object returned.
	 *
	 *  \return an empty copy of the current array
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 *  \sa ArrayBase::empty
	 *
	 */
	ArrayBase* empty() const
	{
		return new Array< T >();
	}

#ifdef _WIN32
	//! Just a dummy method, necessary for Visual C++.
	bool operator == (const Array< T >&) const
	{
		return false;
	}

	//! Just a dummy method, necessary for Visual C++.
	bool operator != (const Array< T >&) const
	{
		return false;
	}

	//! Just a dummy method, necessary for Visual C++.
	bool operator < (const Array< T >&) const
	{
		return false;
	}

	//! Just a dummy method, necessary for Visual C++.
	bool operator > (const Array< T >&) const
	{
		return false;
	}
#endif


	//=======================================================================
	/*!
	 *  \brief Returns an iterator to the first array element.
	 *
	 *  A pointer to the first element of the element vector Array::e
	 *  of the current array is returned as iterator. <br>
	 *  This method is used for compatibility with the other
	 *  stdlib structures.
	 *
	 *  \return Iterator to the first array element.
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	inline iterator begin()
	{
		return e;
	}


	//=======================================================================
	/*!
	 *  \brief Returns a constant iterator to the first array element.
	 *
	 *  A constant pointer to the first element of the element vector
	 *  Array::e of the current array is returned as iterator. <br>
	 *  This method is used for compatibility with the other
	 *  stdlib structures.
	 *
	 *  \return Constant iterator to the first array element.
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	inline const const_iterator begin() const
	{
		return e;
	}

	//=======================================================================
	/*!
	 *  \brief Returns an iterator to the last array element.
	 *
	 *  A constant pointer to the last element of the element vector
	 *  Array::e of the current array is returned as iterator. <br>
	 *  This method is used for compatibility with the other
	 *  stdlib structures.
	 *
	 *  \return Iterator to the last array element.
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	inline iterator end()
	{
		return eEnd;
	}

	//=======================================================================
	/*!
	 *  \brief Returns a constant iterator to the last array element.
	 *
	 *  A constant pointer to the last element of the element vector
	 *  Array::e of the current array is returned as iterator. <br>
	 *  This method is used for compatibility with the other
	 *  stdlib structures.
	 *
	 *  \return Constant iterator to the last array element.
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	inline const const_iterator end()   const
	{
		return eEnd;
	}

protected:

	//=======================================================================
	/*!
	 *  \brief The element vector of the current array.
	 *
	 *  Internally, the major information about an array is stored in
	 *  two vectors, the element vector and the dimension vector
	 *  ArrayBase::d. <br>
	 *  The element vector stores all single elements of an array
	 *  successively. Beginning with the first position in the first
	 *  dimension of an array as seen by the user, go recursively
	 *  down to the last dimension and list all elements there. <br>
	 *  This will give you the order of the elements in the element
	 *  vector.
	 *
	 *  \par Example
	 *  \code
	 *  #include "Array/Array.h"
	 *
	 *  void main()
	 *  {
	 *      Array< unsigned >  test( 2, 2, 4 );  // 3-dimensional test array
	 *      unsigned          *evec,             // element vector
	 *                         no_elems,         // total number of array 
	 *                                           // elements
	 *                         curr_no;          // number of current element 
	 *
	 *      // Fill test array with cotent:
	 *      test( 0, 0, 0 ) = 1;
	 *      test( 0, 0, 1 ) = 2;
	 *      test( 0, 0, 2 ) = 3;
	 *      test( 0, 0, 3 ) = 4;
	 *      test( 0, 1, 0 ) = 5;
	 *      test( 0, 1, 1 ) = 6;
	 *      test( 0, 1, 2 ) = 7;
	 *      test( 0, 1, 3 ) = 8;
	 *      test( 1, 0, 0 ) = 9;
	 *      test( 1, 0, 1 ) = 10;
	 *      test( 1, 0, 2 ) = 11;
	 *      test( 1, 0, 3 ) = 12;
	 *      test( 1, 1, 0 ) = 13;
	 *      test( 1, 1, 1 ) = 14;
	 *      test( 1, 1, 2 ) = 15;
	 *      test( 1, 1, 3 ) = 16;
	 *
	 *      // Get total number of array elements:
	 *      no_elems = test.nelem( );
	 *
	 *      // Get element vector, output of each vector element:
	 *      for ( curr_no = 0, evec = test.elemvec( ); 
	 *            curr_no < no_elems; curr_no++, evec++ )
	 *      {
	 *          cout << *evec << ' ';
	 *      }
	 *  }
	 *  \endcode
	 *
	 *  This program will produce the output:
	 *
	 *  \f$1\mbox{\ }2\mbox{\ }3\mbox{\ }4\mbox{\ }5\mbox{\ }6\mbox{\ }7\mbox{\ }8\mbox{\ }9\mbox{\ }10\mbox{\ }11\mbox{\ }12\mbox{\ }13\mbox{\ }14\mbox{\ }15\mbox{\ }16\f$
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 *  \sa ArrayBase::resize_i(unsigned*, unsigned, bool)
	 *
	 */
	T*        e;

	//! The last element of the element vector Array::e.
	T*        eEnd;


	//=======================================================================
	/*!
	 *  \brief Handles memory allocation for the other resize methods.
	 *
	 *  This function handles memory allocation of the
	 *  respective template type in case of resizing and is only for
	 *  internal use.
	 *
	 *  \param _d one-dimensional vector of array-dimensions
	 *  \param _nd number of array-dimensions
	 *  \param copy flag which indicates whether existing elements of
	 *             an array should be copied in case of resizing (as long
	 *             as possible)
	 *  \return none
	 *  \throw SharkException the type of the exception is "size mismatch"
	 *         and indicates, that the current array is a static array 
	 *         reference. You can't change the size of an array reference,
	 *         so it is checked, if the current number of dimensions
	 *         is not equal to the new defined number of dimensions \em _nd
	 *         and if the current total number of array elements
	 *         is not equal to the new number of array elements. 
	 *         If the preprocessor directive "-DNDEBUG" is not used, then
	 *         it is also checked, if at least one of the sizes of the 
	 *         single dimensions of the current array (reference) is not
	 *         equal to the corresponding new defined size in \em _d. <br>
	 *         Both cases are illegal for static array references,
	 *         so the exception is thrown. <br>
	 *         So the only chance to use this method on static array
	 *         references is to change nothing, i.e. the new number
	 *         of dimensions and dimension sizes are equal to the
	 *         old ones (what makes not a real sense at all).
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 *  \sa ArrayBase::resize_i(unsigned*, unsigned, bool)
	 *
	 */
	void resize_i(unsigned* _d, unsigned _nd, bool copy)
	{
		unsigned j, _ne;

		// Adopt dimension vector if necessary:
		if (nd != _nd)
		{
			// Can only resize non-static arrays:
			SIZE_CHECK(! stat)

			// Delete dimension vector if present:
			if (nd)
			{
				delete[ ] d;
			}

			// Create new dimension vector:
			if ((nd = _nd) > 0)
			{
				d = new unsigned[ _nd ];
			}
		}

#ifndef NDEBUG
		if (stat)
		{
			for (j = _nd; j--;)
			{
				SIZE_CHECK(d[ j ] == _d[ j ])
			}
		}
#endif

		//calculate number of elements
		for (j = _nd, _ne = 1; j--; _ne *= (d[ j ] = _d[ j ]));
		if (_nd == 0)
		{
			_ne = 0;
		}

		if (ne != _ne)	//add or delete elements
		{
			SIZE_CHECK(! stat);
#ifndef NDEBUG
			if (nd != _nd) throw ("Array resize warning: copying elements will not arrange elements correctly");
			else
			{
				for (j = 1; j < _nd; j++)
				{
					if (d[j] != _d[j])
						throw ("Array resize warning: copying elements will not arrange elements correctly");
				}
			}
#endif

			if (copy)	//does not consider structure of elements
			{
				T* _e = e;	//create new pointer to old element vector
				if (_ne)	//create new element vector
				{
					e = new T[ _ne ];
				}
				for (j = ne < _ne ? ne : _ne; j--; e[ j ] = _e[ j ]);//copy existing elements, j=min(ne,_ne)
				if (ne)//delete elements, array is empty now
				{
					delete[ ] _e;
				}
			}
			else//without copying
			{
				if (ne)//delete element vector
				{
					delete[ ] e;
				}
				if (_ne)//create new element vector
				{
					e = new T[ _ne ];
				}
			}
			ne = _ne;//update number of elements in array
		}
		eEnd = e + ne;//update pointer to end of array
	}


	//=======================================================================
	/*!
	 *  \brief Creates a new Array object with the structure and content
	 *         given by the parameters. 
	 *
	 *  A new Array is created by using the parameterized values
	 *  for the internal structure and content. <br>
	 *  This method is for internal use only.
	 *
	 *  \param _d one-dimensional vector of array-dimensions
	 *  \param _e one-dimensional vector of array elements for all dimensions
	 *  \param _nd number of array-dimensions
	 *  \param _ne total number of array elements
	 *  \param _stat defines, whether the array object is a static reference
	 *               or not
	 *  \return none
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	Array(unsigned* _d, T* _e, unsigned _nd, unsigned _ne, bool _stat)
	{
		d    = _d;
		e    = _e;
		nd   = _nd;
		ne   = _ne;
		stat = _stat;
		eEnd = e + ne;
	}

#ifndef __ARRAY_NO_GENERIC_IOSTREAM


	//=======================================================================
	/*!
	 *  \ingroup InOut 
	 *  \brief Replaces the structure and content of the current array by
	 *         the information read from the input stream "is".
	 *
	 *  An array object that is read from \em is must have the following
	 *  format: <br>
	 *
	 *  Array<Type>(list of dimension sizes) <br>
	 *  list of all array elements <br>
	 *  Please notice that there are no whitespaces allowed in the
	 *  list of dimension sizes and between the parentheses and the
	 *  list, otherwise the method will exit with an exception. <br>
	 *  This method is only available, when the flag
	 *  #__ARRAY_NO_GENERIC_IOSTREAM is undefined at the beginning
	 *  of Array.h.
	 *
	 *  \par Example
	 *  From \em is the following data is read: <br>
	 *  Array<double>(3,2) <br>
	 *  1.     2.     3.     4.     5.     6. <br>
	 *
	 *  So the current array will be resized to a \f$3 \times 2\f$ 
	 *  array of type "double" with content 
	 *  \f$
	 *      \left(\begin{array}{ll}
	 *          1. & 2.\\
	 *          3. & 4.\\
	 *          5. & 6.
	 *      \end{array}\right)
	 *  \f$ 
	 *
	 *  \param is input stream from which the array content is read.
	 *  \return none
	 *  \throw SharkException the type of the exception is "type mismatch"
	 *         and indicates, that the data read from \em is has not the
	 *         right format.
	 *
	 *  \par Example
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	void readFrom(std::istream& is)
	{
		const unsigned     MaxLen = 1024;
		char               s[ MaxLen ];
		char*              p;
		unsigned           i;
		std::vector< unsigned > idx; // dimension vector

		// Skip leading whitespaces
		is >> std::ws;
		is.getline(s, MaxLen);

		// Get end of type definition:
		p = strchr(s, '>');
		// Go to begin of dimension sizes:
		if (p && *p) p++;
		if (p && *p) p++;

		// Get single dimension sizes, store them in vector
		// and resize current array:
		TYPE_CHECK(p && *p)
		if (p && *p)
		{
			do
			{
				if (*p == ',')
				{
					p++;
				}
				idx.push_back(unsigned(strtol(p, &p, 10)));
			}
			while (p && *p == ',');

			TYPE_CHECK(p && *p == ')')

			resize(idx);

			// Get elements of the array:
			for (i = 0; i < ne && is.good(); i++)
			{
				is >> e[ i ];
			}
		}

		is >> std::ws;
	}



	//=======================================================================
	/*!
	 *  \ingroup InOut
	 *  \brief Writes the structure and content of the current array to 
	 *         output stream "os".
	 *
	 *  An array object is written to \em os in the following format: <br>
	 *
	 *  Array<Type>(list of dimension sizes) <br>
	 *  list of all array elements <br>
	 *  This method is only available, when the flag
	 *  #__ARRAY_NO_GENERIC_IOSTREAM is undefined at the beginning
	 *  of Array.h.
	 *
	 *  \param os The output stream to where the array is written to
	 *  \return none
	 *
	 *  \par Example
	 *  Given an \f$3 \times 2\f$ array of type "double" with
	 *  content 
	 *  \f$
	 *      \left(\begin{array}{ll}
	 *          1. & 2.\\
	 *          3. & 4.\\
	 *          5. & 6.
	 *      \end{array}\right)
	 *  \f$ 
	 *  this method will produce the output: <br>
	 *  Array<double>(3,2) <br>
	 *  1.     2.     3.     4.     5.     6.
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	void writeTo(std::ostream& os) const
	{
		unsigned i;

		os << "Array<" << typeid(T).name() << ">(";
		for (i = 0; i < nd; i++)
		{
			if (i) os << ',';
			os << d[ i ];
		}
		os << ")\n";

		for (i = 0; i < ne; i++)
		{
			if (i) os << '\t';
			os << e[ i ];
		}
		os << std::endl;
	}
#endif // !__ARRAY_NO_GENERIC_IOSTREAM
};


//===========================================================================
/*!
 *  \brief Class that implements references to objects of the
 *         Array template class.
 *
 *  A reference is an Array object, that is identified
 *  by the flag "stat", which is set to "true".
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
template< class T >
class ArrayReference : public Array< T >
{
public:

	//========================================================================
	/*!
	 *  \ingroup Create
	 *  \brief Creates a new empty Array Reference object.
	 * 
	 *  \return none
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	ArrayReference()
			: Array< T >()
	{
		this->stat = true;
		this->d = NULL;
		this->e = NULL;
		this->nd = 0;
		this->ne = 0;
	}


	//========================================================================
	/*!
	 *  \ingroup Create
	 *  \brief Creates a new Array Reference object with the structure
	 *         and content given by the parameters.
	 * 
	 *  This constructor is for internal usage only. 
	 *
	 *  \param _d the one-dimensional dimension vector ArrayBase::d
	 *  \param _e the one-dimensional element vector Array::e
	 *  \param _nd the number of array dimensions ArrayBase::nd
	 *  \param _ne the number of array elements ArrayBase::ne
	 *  \return none
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	ArrayReference(unsigned* _d, T* _e, unsigned _nd, unsigned _ne)
			: Array< T >()
	{
		this->d    = _d;
		this->e    = _e;
		this->nd   = _nd;
		this->ne   = _ne;
		this->stat = true;
		this->eEnd = this->e + this->ne;
	}


	//========================================================================
	/*!
	 *  \brief Destructs an Array Reference object.
	 *
	 *  This destructor is important to define, because otherwise
	 *  Visual C++ wouldn't call destructors of the base classes
	 *  Array and ArrayBase.
	 *
	 *  \return none
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	~ArrayReference()
	{ }



	//========================================================================
	/*!
	 *  \ingroup Assign
	 *  \brief Assigns the value "v" to all positions of the current array
	 *         reference.
	 *
	 *  \param v the new value for all array reference elements
	 *  \return the array reference with the new values
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	ArrayReference< T > operator = (const T& v)
	{
		Array< T >::operator = (v);
		return *this;
	}



	//========================================================================
	/*!
	 *  \ingroup Assign
	 *  \brief Assigns the values of vector "v" to the current array reference.
	 *
	 *  The size of the array reference is adopted to the size of the 
	 *  vector \em v.
	 *
	 *  \param v vector with the new values for the array reference.
	 *  \return the array reference with the new values
	 *  \throw SharkException the type of the exception will be
	 *         "size mismatch" and indicates that the current array
	 *         is a static array reference whose size is different
	 *         to the size of \em v
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	ArrayReference< T > operator = (const std::vector< T >& v)
	{
		Array< T >::operator = (v);
		return *this;
	}



	//========================================================================
	/*!
	 *  \ingroup Assign
	 *  \brief Assigns the values of array "v" to the current array reference.
	 *
	 *  The size of the current array reference is adopted to the size of 
	 *  array \em v.
	 *
	 *  \param v array with the new values for the current array reference
	 *  \return the current array reference with the new values
	 *  \throw SharkException the type of the exception will be
	 *         "size mismatch" and indicates that the current array
	 *         is a static array reference whose size is different
	 *         to the size of \em v
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	ArrayReference< T > operator = (const Array< T >& v)
	{
		Array< T >::operator = (v);
		return *this;
	}


	//========================================================================
	/*!
	 *  \ingroup Copy
	 *  \brief Copies array "v" to the current array reference.
	 *
	 *  \param v array which will be copied to the current array reference
	 *  \return the current array reference with the new values
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	ArrayReference< T > copyReference(Array< T >& v)
	{
		this->d    = v.d;
		this->e    = v.e;
		this->nd   = v.nd;
		this->ne   = v.ne;
		this->stat = true;
		this->eEnd = this->e + this->ne;
		return *this;
	}

	//========================================================================
	/*!
	 *  \ingroup Copy
	 *  \brief Copies array reference "v" to the current array reference.
	 *
	 *  \param v array reference which will be copied to the current array 
	 *           reference
	 *  \return the current array reference with the new values
	 *
	 *  \author  M. Kreutz
	 *  \date    1995-01-01
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	ArrayReference< T > copyReference(ArrayReference< T > v)
	{
		this->d    = v.d;
		this->e    = v.e;
		this->nd   = v.nd;
		this->ne   = v.ne;
		this->stat = true;
		this->eEnd = this->e + this->ne;
		return *this;
	}
};

//========================================================================
/*!
 *  \ingroup Extract
 *  \brief Returns a reference to subarray "i" of the current array.
 *
 *  A subarray is identified here by a position in the first dimension.
 *  The subarray will then include the element at this position and
 *  all elements of the subdimensions, starting from the mentioned position.
 *
 *  \param i index of the subarray that will be returned
 *  \return reference to the subarray with index \em i
 *  \throw SharkException if the type of the exception is "size mismatch",
 *         then the array has no dimensions. If the type is
 *         "range check error", then the value of \em i exceeds the
 *         size of the array's first dimension.
 *
 *  \par Example
 *  Given the array \f$A\f$ with content
 *
 *  \f$
 *      \left(\begin{array}{ll}
 *          1. & 2.\\
 *          3. & 4.\\
 *          5. & 6.\\
 *          7. & 8.
 *      \end{array}\right)
 *  \f$
 *
 *  then \f$A[ 2 ]\f$ will return the subarray \f$(5. \mbox{\ \ } 6.)\f$.
 *
 *  \author  M. Kreutz
 *  \date    1995-01-01
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
template< class T >
inline ArrayReference< T > Array< T >::operator [ ](unsigned i)
{
	SIZE_CHECK(nd > 0)
	RANGE_CHECK(i < d[ 0 ])
	return ArrayReference< T > (d + 1, e + i*(ne / d[ 0 ]), nd - 1, ne / d[ 0 ]);
}


//========================================================================
/*!
 *  \ingroup Extract
 *  \brief Returns a constant reference to subarray "i" of the current array.
 *
 *  A subarray is identified here by a position in the first dimension.
 *  The subarray will then include the element at this position and
 *  all elements of the subdimensions, starting from the mentioned position.
 *
 *  \param i index of the subarray that will be returned
 *  \return constant reference to the subarray with index \em i
 *  \throw SharkException if the type of the exception is "size mismatch",
 *         then the array has no dimensions. If the type is
 *         "range check error", then the value of \em i exceeds the
 *         size of the array's first dimension
 *
 *  \par Example
 *  Given the array \f$A\f$ with content
 *
 *  \f$
 *      \left(\begin{array}{ll}
 *          1. & 2.\\
 *          3. & 4.\\
 *          5. & 6.\\
 *          7. & 8.
 *      \end{array}\right)
 *  \f$
 *
 *  then \f$A[ 2 ]\f$ will return the subarray \f$(5. \mbox{\ \ } 6.)\f$.
 *
 *  \author  M. Kreutz
 *  \date    1995-01-01
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
template< class T >
inline const ArrayReference< T > Array< T >::operator [ ](unsigned i) const
{
	SIZE_CHECK(nd > 0)
	RANGE_CHECK(i < d[ 0 ])
	return ArrayReference< T > (d + 1, e + i*(ne / d[ 0 ]), nd - 1, ne / d[ 0 ]);
}


#endif //__ARRAY_H

