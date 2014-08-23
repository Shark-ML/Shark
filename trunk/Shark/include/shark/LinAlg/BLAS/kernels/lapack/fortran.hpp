/*!
 * \brief       Defines Fortran naming conventions when binding to lapack routines
 *
 * \author      O. Krause
 * \date        2012
 *
 *
 * \par Copyright 1995-2014 Shark Development Team
 *
 * This is based on boost::numeric::bindings, written by Toon Knapen and Kresimir Fresl 
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://image.diku.dk/shark/>
 * 
 * Shark is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Shark is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with Shark.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef SHARK_LINALG_BLAS_KERNELS_LAPACK_FORTRAN_H
#define SHARK_LINALG_BLAS_KERNELS_LAPACK_FORTRAN_H

#if defined(BIND_FORTRAN_LOWERCASE_UNDERSCORE) || defined(BIND_FORTRAN_LOWERCASE)
// Allow manual override of the defaults, e.g. if you want to use a fortran
// lib compiled with gcc from MSVC
#else

// First we need to know what the conventions for linking
// C with Fortran is on this platform/toolset
#if defined(__GNUC__) || defined(__ICC) || defined(__sgi) || defined(__COMO__) || defined(__KCC)
#define BIND_FORTRAN_LOWERCASE_UNDERSCORE
#elif defined(__IBMCPP__) || defined(_MSC_VER)
#define BIND_FORTRAN_LOWERCASE
#else
	#error do not know how to link with fortran for the given platform
#endif

#endif

// Next we define macro's to convert our symbols to 
// the current convention
#if defined(BIND_FORTRAN_LOWERCASE_UNDERSCORE)
#define FORTRAN_ID( id ) id##_
#elif defined(BIND_FORTRAN_LOWERCASE)
#define FORTRAN_ID( id ) id
#else
#error do not know how to bind to fortran calling convention
#endif

#endif
