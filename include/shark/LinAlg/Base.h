/**
*
*  \brief Entry Point for all Basic Linear Algebra(BLAS) in shark
*
*  \author O.Krause, T.Glasmachers, T. Voss
*  \date 2010-2011
*
*
*  <BR><HR>
*  This file is part of Shark. This library is free software;
*  you can redistribute it and/or modify it under the terms of the
*  GNU General Public License as published by the Free Software
*  Foundation; either version 3, or (at your option) any later version.
*
*  This library is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with this library; if not, see <http://www.gnu.org/licenses/>.
*
*/
#ifndef SHARK_LINALG_BASE_H
#define SHARK_LINALG_BASE_H

/**
* \brief Shark linear algebra definitions
*
* \par
* This file provides all basic definitions for linear algebra.
* If defines objects and views for vectors and matrices over
* several base types as well as a lot of usefull functions.
*/

//for debug error handling of linear algebra
#include <shark/Core/Exception.h>
#include <shark/LinAlg/BLAS/ublas.h>
//All relevant Linear Algebra introduced by SHARK
#include <shark/LinAlg/BLAS/VectorMatrixType.h>
#include <shark/LinAlg/BLAS/fastOperations.h>
#include <shark/LinAlg/BLAS/Permutation.h>
#include <shark/LinAlg/BLAS/Initialize.h>
#include <shark/LinAlg/BLAS/StorageAdaptors.h>
//also includes BLAS/VectorTransformations.h
#include <shark/LinAlg/BLAS/Tools.h>
#include <shark/LinAlg/BLAS/Metrics.h>

#include <boost/serialization/deque.hpp>
#include <deque>

//this ensures, that Sequence is serializable

namespace shark{
///Type of Data sequences.
typedef std::deque<RealVector> Sequence;
}





#endif
