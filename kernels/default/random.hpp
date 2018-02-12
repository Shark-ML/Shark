/*!
 * 
 *
 * \brief       Generation of random variates on cpu
 *
 * \author      O. Krause
 * \date        2017
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
#ifndef REMORA_KERNELS_DEFAULT_RANDOM_HPP
#define REMORA_KERNELS_DEFAULT_RANDOM_HPP

#include <random>
#include <cmath>

namespace remora{ namespace bindings{
template<class V, class Rng>
void generate_normal(
	vector_expression<V, cpu_tag>& v,
	Rng& rng,
	typename V::value_type mean,
	typename V::value_type variance
) {
	std::normal_distribution<typename V::value_type> dist(mean,std::sqrt(variance));
	for(auto& val: v())
		val = dist(rng);
}

template<class M, class Rng>
void generate_normal(
	matrix_expression<M, cpu_tag>& m,
	Rng& rng,
	typename M::value_type mean,
	typename M::value_type variance
) {
	std::normal_distribution<typename M::value_type> dist(mean,std::sqrt(variance));
	std::size_t size = M::orientation::index_M(m().size1(),m().size2());
	for(std::size_t i = 0; i != size; ++i){
		auto end = m().major_end(i);
		for(auto pos = m().major_begin(i);pos != end; ++pos){
			*pos = dist(rng);
		}
	}
}

template<class V, class Rng>
void generate_uniform(
	vector_expression<V, cpu_tag>& v,
	Rng& rng,
	typename V::value_type low,
	typename V::value_type high
) {
	std::uniform_real_distribution<typename V::value_type> dist(low,high);
	for(auto& val: v())
		val = dist(rng);
}

template<class M, class Rng>
void generate_uniform(
	matrix_expression<M, cpu_tag>& m,
	Rng& rng,
	typename M::value_type low,
	typename M::value_type high
) {
	std::uniform_real_distribution<typename M::value_type> dist(low,high);
	std::size_t size = M::orientation::index_M(m().size1(),m().size2());
	for(std::size_t i = 0; i != size; ++i){
		auto end = m().major_end(i);
		for(auto pos = m().major_begin(i);pos != end; ++pos){
			*pos = dist(rng);
		}
	}
}

template<class V, class Rng>
void generate_discrete(
	vector_expression<V, cpu_tag>& v,
	Rng& rng,
	int low,
	int high
) {
	std::uniform_int_distribution<int> dist(low,high);
	for(auto& val: v())
		val = dist(rng);
}

template<class M, class Rng>
void generate_discrete(
	matrix_expression<M, cpu_tag>& m,
	Rng& rng,
	int low,
	int high
) {
	std::uniform_int_distribution<int> dist(low,high);
	std::size_t size = M::orientation::index_M(m().size1(),m().size2());
	for(std::size_t i = 0; i != size; ++i){
		auto end = m().major_end(i);
		for(auto pos = m().major_begin(i);pos != end; ++pos){
			*pos = dist(rng);
		}
	}
}

}}
#endif