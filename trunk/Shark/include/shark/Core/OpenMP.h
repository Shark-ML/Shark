/*!
 * 
 * \file        OpenMP.h
 *
 * \brief       Set of macros to help usage of OpenMP with Shark
 * 
 * 
 * 
 *
 * \author      Oswin Krause
 * \date        2012
 *
 *
 * \par Copyright 1995-2014 Shark Development Team
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
