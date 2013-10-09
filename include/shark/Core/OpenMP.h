/*!
 *  \brief Set of macros to help usage of OpenMP with Shark
 *
 *
 *  \author  Oswin Krause
 *  \date    2012
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
#ifndef SHARK_CORE_OPENMP_H
#define SHARK_CORE_OPENMP_H

#ifdef SHARK_USE_OPENMP
#include <omp.h>
#include <boost/config.hpp>

#ifdef BOOST_MSVC
#define SHARK_PARALLEL_FOR __pragma(omp parallel for)\
for

#define SHARK_CRITICAL_REGION __pragma(omp critical)

#else
#define SHARK_PARALLEL_FOR \
_Pragma ( "omp parallel for" )\
for

#define SHARK_CRITICAL_REGION _Pragma("omp critical")
#endif

#define SHARK_NUM_THREADS (std::size_t)(omp_in_parallel()?omp_get_num_threads():omp_get_max_threads())
#define SHARK_THREAD_NUM (std::size_t)(omp_in_parallel()?omp_get_thread_num():0)

#else
#define SHARK_PARALLEL_FOR for
#define SHARK_CRITICAL_REGION
#define SHARK_NUM_THREADS (std::size_t)1
#define SHARK_THREAD_NUM (std::size_t)0
#endif

#endif
