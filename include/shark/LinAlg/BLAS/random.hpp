/*!
 * \brief       Defines types for matrix decompositions
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
#ifndef REMORA_RANDOM_HPP
#define REMORA_RANDOM_HPP

#include "assignment.hpp"
#include "kernels/random.hpp"
#include "detail/proxy_optimizers_fwd.hpp"

namespace remora{ namespace detail{
template<class T, class Rng>
struct RandomNormal{
	typedef T value_type;
	T mean;
	T variance;
	Rng* rng;
	
	template<class E>
	void generate(E& e, typename E::value_type alpha) const{
		kernels::generate_normal(e, *rng, mean*alpha, variance*alpha*alpha);
	}
};

template<class T, class Rng>
struct RandomUniform{
	typedef T value_type;
	T low;
	T high;
	Rng* rng;
	
	template<class E>
	void generate(E& e, typename E::value_type alpha) const{
		kernels::generate_uniform(e, *rng, low * alpha, high * alpha);
	}
};

//~ template<class T, class Rng>
//~ struct RandomDiscrete{
	//~ typedef T value_type;
	//~ T low;
	//~ T high;
	//~ Rng* rng;
	
	//~ template<class E>
	//~ void generate(E& e) const{
		//~ kernels::generate_discrete(v, *rng, low, high);
		
	//~ }
//~ };

}

template<class Distribution, class Device>
class random_vector: public vector_expression<random_vector<Distribution,Device>, Device>{
public:
	typedef typename Distribution::value_type value_type;
	typedef std::size_t size_type;
	typedef value_type const_reference;
	typedef const_reference reference;

	typedef random_vector const_closure_type;
	typedef const_closure_type closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef blockwise<dense_tag> evaluation_category;
	typedef Device device_type;

	random_vector(
		Distribution const& distribution,
		typename device_traits<device_type>::queue_type& queue,
		std::size_t size,
		value_type alpha = value_type(1)
	):m_distribution(distribution), m_queue(&queue), m_size(size), m_alpha(alpha){};

	size_type size() const {
		return m_size;
	}
	typename device_traits<device_type>::queue_type& queue() const{
		return *m_queue;
	}
	
	Distribution const& dist() const{
		return m_distribution;
	}
	
	value_type alpha() const{
		return m_alpha;
	}
	
	typedef no_iterator iterator;
	typedef iterator const_iterator;
	
	//dispatcher to computation kernels
	template<class VecX>
	void assign_to(vector_expression<VecX, device_type>& x)const{
		m_distribution.generate(x(), m_alpha);
	}
	template<class VecX>
	void plus_assign_to(vector_expression<VecX, device_type>& x)const{
		plus_assign(x,typename vector_temporary<VecX>::type(*this));
	}
private:
	Distribution m_distribution;
	typename device_traits<device_type>::queue_type* m_queue;
	std::size_t m_size;
	value_type m_alpha;
};

template<class Distribution, class Device>
class random_matrix: public matrix_expression<random_matrix<Distribution,Device>, Device>{
public:
	typedef typename Distribution::value_type value_type;
	typedef std::size_t size_type;
	typedef value_type const_reference;
	typedef const_reference reference;

	typedef random_matrix const_closure_type;
	typedef const_closure_type closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef blockwise<dense_tag> evaluation_category;
	typedef unknown_orientation orientation;
	typedef Device device_type;

	random_matrix(
		Distribution const& distribution,
		typename device_traits<device_type>::queue_type& queue,
		std::size_t size1, std::size_t size2,
		value_type alpha = value_type(1)
	):m_distribution(distribution), m_queue(&queue), m_size1(size1), m_size2(size2), m_alpha(alpha){};

	size_type size1() const {
		return m_size1;
	}
	size_type size2() const {
		return m_size2;
	}
	typename device_traits<device_type>::queue_type& queue() const{
		return *m_queue;
	}
	
	Distribution const& dist() const{
		return m_distribution;
	}
	
	value_type alpha() const{
		return m_alpha;
	}
	
	typedef no_iterator major_iterator;
	typedef no_iterator const_major_iterator;
	
	//dispatcher to computation kernels
	template<class MatX>
	void assign_to(matrix_expression<MatX, device_type>& x)const{
		m_distribution.generate(x(),m_alpha);
	}
	template<class MatX>
	void plus_assign_to(matrix_expression<MatX, device_type>& x)const{
		plus_assign(x,typename matrix_temporary<MatX>::type(*this));
	}
private:
	Distribution m_distribution;
	typename device_traits<device_type>::queue_type* m_queue;
	std::size_t m_size1;
	std::size_t m_size2;
	value_type m_alpha;
};

namespace detail{
template<class Distribution, class Device>
struct vector_scalar_multiply_optimizer<random_vector<Distribution, Device> >{
	typedef random_vector<Distribution, Device> type;
	static type create(type const& v, typename type::value_type alpha){
		return type(v.dist(), v.queue(), v.size(), v.alpha() * alpha);
	}
};

template<class Distribution, class Device>
struct vector_range_optimizer<random_vector<Distribution, Device> >{
	typedef random_vector<Distribution, Device> type;
	static type create(type const& v, std::size_t start, std::size_t end){
		return type(v.dist(), v.queue(), end - start, v.alpha());
	}
};


template<class Distribution, class Device>
struct matrix_scalar_multiply_optimizer<random_matrix<Distribution, Device> >{
	typedef random_matrix<Distribution, Device> type;
	static type create(type const& m, typename type::value_type alpha){
		return type(m.dist(), m.queue(), m.size1(), m.size2(), m.alpha() * alpha);
	}
};

template<class Distribution, class Device>
struct matrix_transpose_optimizer<random_matrix<Distribution, Device> >{
	typedef random_matrix<Distribution, Device> type;
	static type create(type const& m){
		return type(m.dist(), m.queue(), m.size2(), m.size1(), m.alpha());
	}
};

template<class Distribution, class Device>
struct matrix_row_optimizer<random_matrix<Distribution, Device> >{
	typedef random_vector<Distribution, Device> type;
	static type create(random_matrix<Distribution, Device> const& m){
		return type(m.dist(), m.queue(), m.size2(), m.alpha());
	}
};

template<class Distribution, class Device>
struct matrix_diagonal_optimizer<random_matrix<Distribution, Device> >{
	typedef random_vector<Distribution, Device> type;
	static type create(type const& m){
		return type(m.dist(), m.queue(), std::min(m.size1(),m.size2()), m.alpha());
	}
};

template<class Distribution, class Device>
struct matrix_range_optimizer<random_matrix<Distribution, Device> >{
	typedef random_matrix<Distribution, Device> type;
	
	static type create(type const& m,
		std::size_t start1, std::size_t end1, std::size_t start2, std::size_t end2
	){
		return type(m.dist(), m.queue(), end1 - start1, end2-start2, m.alpha());
	}
};

template<class Distribution, class Device>
struct matrix_rows_optimizer<random_matrix<Distribution, Device> >{
	typedef random_matrix<Distribution, Device> type;
	static type create(type const& m, std::size_t start, std::size_t end
	){
		return type(m.dist(), m.queue(), end - start, m.size2(), m.alpha());
	}
};

}


//////////////////////////////////////////////////////
////////Expressions
//////////////////////////////////////////////////////


//solvers for vector rhs
template<class Device, class T, class Rng>
random_vector<detail::RandomNormal<T,Rng>, Device> normal(Rng& rng, std::size_t size, T mean, T variance, Device){
	return {{mean,variance, &rng},device_traits<Device>::default_queue(),size};
}

template<class Device, class T, class Rng>
random_matrix<detail::RandomNormal<T,Rng>, Device> normal(Rng& rng, std::size_t size1, std::size_t size2, T mean, T variance, Device){
	return {{mean,variance, &rng},device_traits<Device>::default_queue(),size1,size2};
}

template<class Device, class T, class Rng>
random_vector<detail::RandomUniform<T,Rng>, Device> uniform(Rng& rng, std::size_t size, T low, T high, Device){
	return {{low,high, &rng},device_traits<Device>::default_queue(),size};
}

template<class Device, class T, class Rng>
random_matrix<detail::RandomUniform<T,Rng>, Device> uniform(Rng& rng, std::size_t size1, std::size_t size2, T low, T high, Device){
	return {{low,high, &rng},device_traits<Device>::default_queue(),size1,size2};
}


}
#endif