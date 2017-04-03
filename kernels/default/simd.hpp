/*!
 *
 *
 * \brief       Some Macros and basic definitions for the use of SIMD block storage
 *
 * \author      O. Krause
 * \date        2016
 *
 *
 * \par Copyright 1995-2015 Shark Development Team
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

#ifndef REMORA_KERNELS_DEFAULT_SIMD_HPP
#define REMORA_KERNELS_DEFAULT_SIMD_HPP

#include <boost/version.hpp>
#include <cstddef>

//older boost versions have some issues
 #if (BOOST_VERSION >= 106300)
 	#include <boost/align/assume_aligned.hpp>
 	#include <boost/align/aligned_allocator.hpp>
#else//subset of boost/align 1.63
	#include "boost_align/assume_aligned.hpp"
	#include "boost_align/aligned_allocator.hpp"
#endif




#ifdef __AVX__
	#define REMORA_VECTOR_LENGTH 32
#else
	#define REMORA_VECTOR_LENGTH 16
#endif

namespace remora{namespace bindings{namespace detail{
template<class T>
struct block{
	static const std::size_t max_vector_elements = REMORA_VECTOR_LENGTH/sizeof(T);
	#ifdef REMORA_USE_SIMD
		static const std::size_t vector_elements = REMORA_VECTOR_LENGTH/sizeof(T);
		#ifdef BOOST_COMP_CLANG_DETECTION
			typedef T type __attribute__((ext_vector_type (vector_elements)));
		#else
		    typedef T type __attribute__((vector_size (REMORA_VECTOR_LENGTH)));
		#endif
	#else
		static const std::size_t vector_elements = 1;
		typedef T type;
	#endif
	static const std::size_t align = 64;
};
}}}
#endif
