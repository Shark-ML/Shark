//===========================================================================
/*!
 *
 *
 * \brief       General functions for Tree modeling.
 *
 *
 *
 * \author      J. Wrigley
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
//===========================================================================

#ifndef SHARK_MODELS_TREES_GENERAL_H
#define SHARK_MODELS_TREES_GENERAL_H
#include <memory>
namespace shark {
namespace detail {
namespace cart {
// Helper functions
template<class T> using sink = T;

template<class T, class F>
inline T sum(std::size_t i, std::size_t n, F&& f) {
	if(i>=n) return T{};
	T out = f(i);
	for(++i;i<n;++i) out += f(i);
	return out;
}
template<class T, class F>
inline T sum(std::size_t n, F&& f) {
	return sum<T>(0,n,std::forward<F>(f));
}
//END Helper functions


}}} // namespace shark::detail::cart

#endif //SHARK_MODELS_TREES_GENERAL_H
