//===========================================================================
/*!
 * 
 *
 * \brief       Traits of gpu expressions
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
//===========================================================================

#ifndef SHARK_LINALG_BLAS_GPU_DETAIL_TRAITS_HPP
#define SHARK_LINALG_BLAS_GPU_DETAIL_TRAITS_HPP

#include "../detail/traits.hpp"
#include "../detail/functional.hpp"
#include <boost/compute/core.hpp>
#include <boost/compute/functional/operator.hpp>
#include <tuple>

namespace shark{namespace blas{namespace gpu{
	
template<class T>
struct dense_vector_storage{
	typedef dense_tag storage_tag;
	boost::compute::vector<T> const& buffer;
	std::size_t offset;
	std::size_t stride;
	
	dense_vector_storage<T> sub_region(std::size_t offset){
		return {buffer, this->offset+offset, stride};
	}
};

template<class T>
struct dense_matrix_storage{
	typedef dense_tag storage_tag;
	typedef dense_vector_storage<T> row_storage;
	boost::compute::vector<T> const& buffer;
	std::size_t offset;
	std::size_t leading_dimension;
	
	template<class Orientation>
	dense_matrix_storage<T> sub_region(std::size_t offset1, std::size_t offset2, Orientation){
		std::size_t offset_major = Orientation::index_M(offset1,offset2);
		std::size_t offset_minor = Orientation::index_m(offset1,offset2);
		return {buffer, offset + offset_major*leading_dimension+offset_minor, leading_dimension};
	}
	
	template<class Orientation>
	row_storage row(std::size_t i, Orientation){
		return {buffer, offset + i * Orientation::index_M(leading_dimension,1), Orientation::index_m(leading_dimension,1)};
	}
	template<class Orientation>
	row_storage diag(){
		return {buffer, offset, leading_dimension+1};
	}
};

}


namespace detail{
template<class Arg1, class T>
struct invoked_multiply_scalar{
	typedef T result_type;
	Arg1 arg1;
	T m_scalar;
};

template<class Arg1, class T>
struct invoked_soft_plus{
	typedef T result_type;
	Arg1 arg1;
};
template<class Arg1, class T>
struct invoked_sigmoid{
	typedef T result_type;
	Arg1 arg1;
};

template<class Arg1, class T>
struct invoked_sqr{
	typedef T result_type;
	Arg1 arg1;
};

template<class Arg1, class T>
struct invoked_inv{
	typedef T result_type;
	Arg1 arg1;
};

template<class Arg1, class Arg2, class T>
struct invoked_safe_div{
	typedef T result_type;
	Arg1 arg1;
	Arg2 arg2;
};


template<class Arg1, class T>
boost::compute::detail::meta_kernel& operator<<(boost::compute::detail::meta_kernel& k, invoked_multiply_scalar<Arg1,T> const& e){
	return k << '('<<e.m_scalar << '*'<< e.arg1<<')';
}
template<class Arg1, class T>
boost::compute::detail::meta_kernel& operator<<(boost::compute::detail::meta_kernel& k, invoked_soft_plus<Arg1,T> const& e){
	return k << "(log(1+exp("<< e.arg1<<")))";
}
template<class Arg1, class T>
boost::compute::detail::meta_kernel& operator<<(boost::compute::detail::meta_kernel& k, invoked_sigmoid<Arg1,T> const& e){
	return k << "(1/(1+exp(-"<< e.arg1<<")))";
}
template<class Arg1, class T>
boost::compute::detail::meta_kernel& operator<<(boost::compute::detail::meta_kernel& k, invoked_sqr<Arg1,T> const& e){
	return k << '('<<e.arg1<<'*'<<e.arg1<<')';
}
template<class Arg1, class T>
boost::compute::detail::meta_kernel& operator<<(boost::compute::detail::meta_kernel& k, invoked_inv<Arg1,T> const& e){
	return k << "1/("<<e.arg1<<')';
}

template<class Arg1, class Arg2, class T>
boost::compute::detail::meta_kernel& operator<<(boost::compute::detail::meta_kernel& k, invoked_safe_div<Arg1,Arg2,T> const& e){
	return k << "(("<<e.arg2<<"!=0)?"<<e.arg1/e.arg2<<':'<<e.default_value<<')';
}
}//End namespace detail

template<>
struct device_traits<gpu_tag>{
	template <class Iterator, class Functor>
	using transform_iterator = boost::compute::transform_iterator<Iterator, Functor>;

	//~ template <class Iterator>
	//~ using subrange_iterator = shark::blas::iterators::subrange_iterator<Iterator>;
	
	//~ template<class Iterator1, class Iterator2, class Functor>
	//~ using binary_transform_iterator = shark::blas::iterators::binary_transform_iterator<Iterator1,Iterator2, Functor>;
	
	//~ template<class T>
	//~ using constant_iterator = shark::blas::iterators::constant_iterator<T>;
	
	//~ template<class T>
	//~ using one_hot_iterator = shark::blas::iterators::one_hot_iterator<T>;
	
	//~ template<class Closure>
	//~ using indexed_iterator = shark::blas::iterators::indexed_iterator<Closure>;
	
	//functors
	
	//basic arithmetic
	template<class T>
	using add = boost::compute::plus<T>;
	template<class T>
	using subtract = boost::compute::minus<T>;
	template<class T>
	using multiply = boost::compute::multiplies<T>;
	template<class T>
	using divide = boost::compute::divides<T>;
	template<class T>
	using pow = boost::compute::pow<T>;
	template<class T>
	struct safe_divide : public boost::compute::function<T (T, T)>{
		typedef T result_type;
		safe_divide(T default_value) : boost::compute::function<T (T, T)>("safe_divide"), default_value(default_value) { }
		
		template<class Arg1, class Arg2>
		detail::invoked_safe_div<Arg1,Arg2, T> operator()(const Arg1 &x, const Arg2& y) const
		{
			return {x,y};
		}
		T default_value;
	};
	template<class T>
	struct multiply_scalar : public boost::compute::function<T (T)>{
		typedef T result_type;
		multiply_scalar(T scalar) : boost::compute::function<T (T)>("multiply_scalar"), m_scalar(scalar) { }
		
		template<class Arg1>
		detail::invoked_multiply_scalar<Arg1,T> operator()(const Arg1 &x) const
		{
			return {x, m_scalar};
		}
	private:
		T m_scalar;
	};
	
	//math unary functions
	template<class T>
	using log = boost::compute::log<T>;
	template<class T>
	using exp = boost::compute::exp<T>;
	template<class T>
	using tanh = boost::compute::tanh<T>;
	template<class T>
	using sqrt = boost::compute::sqrt<T>;
	template<class T>
	using abs = boost::compute::fabs<T>;
	
	template<class T>
	struct sqr : public boost::compute::function<T (T)>{
		typedef T result_type;
		sqr() : boost::compute::function<T (T)>("sqr") { }
		
		template<class Arg1>
		detail::invoked_sqr<Arg1,T> operator()(const Arg1 &x) const
		{
			return {x};
		}
	};
	template<class T>
	struct soft_plus : public boost::compute::function<T (T)>{
		typedef T result_type;
		soft_plus() : boost::compute::function<T (T)>("soft_plus") { }
		
		template<class Arg1>
		detail::invoked_soft_plus<Arg1,T> operator()(const Arg1 &x) const
		{
			return {x};
		}
	};
	template<class T>
	struct sigmoid : public boost::compute::function<T (T)>{
		typedef T result_type;
		sigmoid() : boost::compute::function<T (T)>("sigmoid") { }
		
		template<class Arg1>
		detail::invoked_sigmoid<Arg1,T> operator()(const Arg1 &x) const
		{
			return {x};
		}
	};
	template<class T>
	struct inv : public boost::compute::function<T (T)>{
		typedef T result_type;
		inv() : boost::compute::function<T (T)>("inv") { }
		
		template<class Arg1>
		detail::invoked_inv<Arg1,T> operator()(const Arg1 &x) const
		{
			return {x};
		}
	};
	
	//min/max
	template<class T>
	using min = boost::compute::fmin<T>;
	template<class T>
	using max = boost::compute::fmax<T>;
	
	//math binary functions
	
	
};

}}

#endif