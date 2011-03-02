//===========================================================================
/*!
 *  \file SharkDefs.h
 *
 *  \brief basic definitions
 *
 *  \par
 *  This file serves as a minimal abstraction layer.
 *  Inclusion of this file makes some frequently used
 *  functions, constants, and header file inclusions
 *  OS-, compiler-, and version-independent.
 *
 *
 *  \par Copyright (c) 1998-2008:
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

#define NOMINMAX

#ifndef SHARK_DEFS_H
#define SHARK_DEFS_H


#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string.h>

#define _USE_MATH_DEFINES
#include <math.h>

// Disable some warnings
#ifdef WIN32
#pragma warning( disable : 4507 34 )
#endif

////////////////////////////////////////////////////////////
// 64bit integer type
#ifdef _WIN32
#define finite(x) _finite(x)
#ifdef _MSC_VER
#define isnan(x) _isnan(x)
#endif
#define hypot(x,y) _hypot(x,y)
typedef __int64 SharkInt64;
#else
typedef long SharkInt64;
#endif


////////////////////////////////////////////////////////////
// constants

#ifdef __SOLARIS__
#include <values.h>
#include <float.h>
#else
#include <limits>
#endif

// largest double value
#ifndef MAXDOUBLE
#define MAXDOUBLE (std::numeric_limits< double >::max())
#endif

// smallest double value
#ifndef MINDOUBLE
#define MINDOUBLE (std::numeric_limits< double >::min())
#endif

// sqrt(pi)
#define SqrtPI     1.7724538509055160273

// sqrt(2*pi)
#define Sqrt2PI    2.50662827463100050242

// pi
#ifndef M_PI
#define M_PI       3.141592653589793238
#endif

// 2*pi
#ifndef M_2PI
#define M_2PI      6.28318530717958647692
#endif

// pi/2
#ifndef M_PI_2
#define M_PI_2     1.57079632679489661923
#endif

// e
#ifndef M_E
#define M_E        2.718281828459045235
#endif

// sqrt(e)
#define SqrtE      1.6487212707001281468



////////////////////////////////////////////////////////////
// mathematical functions are kept inside a namespace
// in order to avoid ambiguities. This proceeding is
// a sad workaround, but it proves to be very robust.


#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
// #ifdef sqr
// #undef sqr
// #endif


namespace Shark
{


template <class T> inline const T& min(const T& a, const T& b)
{
	return b < a ? b : a;
}

template <class T> inline const T& max(const T& a, const T& b)
{
	return a < b ? b : a;
}

template <class T, class Compare> inline const T& min(const T& a, const T& b, Compare comp)
{
	return compare(b, a) ? b : a;
}

template <class T, class Compare> inline const T& max(const T& a, const T& b, Compare comp)
{
	return compare(a, b) ? b : a;
}

template <class T> inline T sqr(T value)
{
	return (value * value);
}

double round(double x);
double erf(double x);
double erfc(double x);
double gamma(double x);

};



////////////////////////////////////////////////////////////
// encapsulation of cosnt char* exceptions


class SharkException
{
public:
	SharkException(const char* file, int line, const char* message);

	inline const char* what() const
	{
		return msg;
	}

protected:
	char msg[1024];
};


#define SHARKEXCEPTION(message) SharkException(__FILE__, __LINE__, message)


// some handy macros for special types of checks,
// throwing standard error messages
#ifdef DEBUG
	#define RANGE_CHECK(cond) { if (!(cond)) throw SHARKEXCEPTION("range check error"); }
	#define SIZE_CHECK(cond) { if (!(cond)) throw SHARKEXCEPTION("size mismatch"); }
	#define TYPE_CHECK(cond) { if (!(cond)) throw SHARKEXCEPTION("type mismatch"); }
	#define IO_CHECK(cond) { if (!(cond)) throw SHARKEXCEPTION("I/O error"); }
	#define UNDEFINED { throw SHARKEXCEPTION("undefined operator"); }
	#ifndef ASSERT
		#define ASSERT(cond) { if (!(cond)) throw SHARKEXCEPTION("assertion failed"); }
	#endif
#else
	#define RANGE_CHECK(cond) { }
	#define SIZE_CHECK(cond) { }
	#define TYPE_CHECK(cond) { }
	#define IO_CHECK(cond) { }
	#define UNDEFINED { }
	#ifdef ASSERT
		#undef ASSERT
	#endif
	#define ASSERT(cond) { }
#endif


#endif
