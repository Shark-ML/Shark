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

#ifndef REMORA_GPU_TRAITS_HPP
#define REMORA_GPU_TRAITS_HPP

#include <boost/compute/command_queue.hpp>
#include <boost/compute/core.hpp>
#include <boost/compute/functional/detail/unpack.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/functional/operator.hpp>
#include <boost/compute/iterator/zip_iterator.hpp>
#include <boost/compute/iterator/constant_iterator.hpp>
#include <boost/compute/iterator/transform_iterator.hpp>
#include <boost/compute/functional.hpp>
#include <tuple>

namespace remora{namespace gpu{
	
template<class Device>
struct device_traits;
	
template<class T, class Tag>
struct dense_vector_storage{
	typedef Tag storage_tag;
	
	boost::compute::buffer buffer;
	std::size_t offset;
	std::size_t stride;
	
	dense_vector_storage(){}
	dense_vector_storage(boost::compute::buffer const& buffer, std::size_t offset, std::size_t stride)
	:buffer(buffer), offset(offset), stride(stride){}
	template<class U, class Tag2>
	dense_vector_storage(dense_vector_storage<U, Tag2> const& storage):
	buffer(storage.buffer), offset(storage.offset), stride(storage.stride){
		static_assert(!(std::is_same<Tag,continuous_dense_tag>::value && std::is_same<Tag2,dense_tag>::value), "Trying to assign dense to continuous dense storage");
	}
	
	dense_vector_storage<T,Tag> sub_region(std::size_t offset){
		return {buffer, this->offset+offset * stride, stride};
	}
};

template<class T, class Tag>
struct dense_matrix_storage{
	typedef Tag storage_tag;
	template<class O>
	struct row_storage: public std::conditional<
		std::is_same<O,row_major>::value,
		dense_vector_storage<T, Tag>,
		dense_vector_storage<T, dense_tag>
	>{};
	
	typedef dense_vector_storage<T,Tag> diag_storage;
	typedef dense_matrix_storage<T,dense_tag> sub_region_storage;
	
	boost::compute::buffer buffer;
	std::size_t offset;
	std::size_t leading_dimension;
	
	dense_matrix_storage(){}
	dense_matrix_storage(boost::compute::buffer const& buffer, std::size_t offset, std::size_t leading_dimension)
	:buffer(buffer), offset(offset), leading_dimension(leading_dimension){}
	template<class U, class Tag2>
	dense_matrix_storage(dense_matrix_storage<U, Tag2> const& storage):
	buffer(storage.buffer), offset(storage.offset), leading_dimension(storage.leading_dimension){
		static_assert(!(std::is_same<Tag,continuous_dense_tag>::value && std::is_same<Tag2,dense_tag>::value), "Trying to assign dense to continuous dense storage");
	}
	
	template<class Orientation>
	sub_region_storage sub_region(std::size_t offset1, std::size_t offset2, Orientation){
		std::size_t offset_major = Orientation::index_M(offset1,offset2);
		std::size_t offset_minor = Orientation::index_m(offset1,offset2);
		return {buffer, offset + offset_major*leading_dimension+offset_minor, leading_dimension};
	}
	
	template<class Orientation>
	typename row_storage<Orientation>::type row(std::size_t i, Orientation){
		return {buffer, offset + i * Orientation::index_M(leading_dimension,std::size_t(1)), Orientation::index_m(leading_dimension,std::size_t(1))};
	}
	diag_storage diag(){
		return {buffer, offset, leading_dimension+1};
	}
};


namespace detail{
	
template<class Arg1, class T>
struct invoked_multiply_scalar{
	typedef T result_type;
	Arg1 arg1;
	T m_scalar;
};

template<class Arg1, class Arg2, class T>
struct invoked_multiply_and_add{
	typedef T result_type;
	Arg1 arg1;
	Arg2 arg2;
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
	T default_value;
};


template<class Arg1, class T>
boost::compute::detail::meta_kernel& operator<<(boost::compute::detail::meta_kernel& k, invoked_multiply_scalar<Arg1,T> const& e){
	return k << '('<<e.m_scalar << '*'<< e.arg1<<')';
}
template<class Arg1, class Arg2, class T>
boost::compute::detail::meta_kernel& operator<<(boost::compute::detail::meta_kernel& k, invoked_multiply_and_add<Arg1,Arg2,T> const& e){
	return k << '('<<e.arg1<<'+'<<e.m_scalar << '*'<< e.arg2<<')';
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
	return k << "(("<<e.arg2<<"!=0)?"<<e.arg1<<'/'<<e.arg2<<':'<<e.default_value<<')';
}

template<class M, class Index, class Orientation>
struct invoked_linearized_matrix{
	typedef typename M::value_type result_type;
	M mat_closure;
	Index index;
};

template<class M, class Index>
boost::compute::detail::meta_kernel& operator<<(boost::compute::detail::meta_kernel& k, invoked_linearized_matrix<M, Index, row_major> const& e){
	std::string index_str = k.expr_to_string(e.index);
	std::string i1 = "(("+index_str+")/"+std::to_string(e.mat_closure.size2())+")";
	std::string i2 = "(("+index_str+")%"+std::to_string(e.mat_closure.size2())+")";
	return k << e.mat_closure(k.expr<std::size_t>(i1),k.expr<std::size_t>(i2));
}

template<class M, class Index>
boost::compute::detail::meta_kernel& operator<<(boost::compute::detail::meta_kernel& k, invoked_linearized_matrix<M, Index, column_major> const& e){
	std::string index_str = k.expr_to_string(e.index);
	std::string i1 = "(("+index_str+")/"+std::to_string(e.mat_closure.size1())+")";
	std::string i2 = "(("+index_str+")%"+std::to_string(e.mat_closure.size1())+")";
	return k << '('<<e.mat_closure(k.expr<std::size_t>(i2),k.expr<std::size_t>(i1)) << ')';
}

template<class Iterator1, class Iterator2, class Functor>
struct binary_transform_iterator
: public boost::compute::transform_iterator<
	boost::compute::zip_iterator<boost::tuple<Iterator1, Iterator2> >,
	boost::compute::detail::unpacked<Functor>
>{
	typedef boost::compute::transform_iterator<
		boost::compute::zip_iterator<boost::tuple<Iterator1, Iterator2> >,
		boost::compute::detail::unpacked<Functor>
	> self_type;
	binary_transform_iterator(){}
	binary_transform_iterator(
		Functor const& f,
		Iterator1 const& iter1, Iterator1 const& iter1_end,
		Iterator2 const& iter2, Iterator2 const& iter2_end
	): self_type(boost::compute::make_zip_iterator(boost::make_tuple(iter1,iter2)), boost::compute::detail::unpack(f)){}
};

template<class Closure>
class indexed_iterator : public boost::iterator_facade<
	indexed_iterator<Closure>,
        typename Closure::value_type,
        std::random_access_iterator_tag,
	typename Closure::value_type
>{
public:
	indexed_iterator() = default;
	indexed_iterator(Closure const& closure, std::size_t index)
	: m_closure(closure)
	, m_index(index){}
		
	template<class C>
	indexed_iterator(indexed_iterator<C> const& other)
	: m_closure (other.m_closure)
	, m_index(other.m_index){}

	template<class C>
	indexed_iterator& operator=(indexed_iterator<C> const& other){
		m_closure = other.m_closure;
		m_index = other.m_index;
		return *this;
	}

	size_t get_index() const{
		return m_index;
	}

	/// \internal_
	template<class Expr>
	auto operator[](Expr const& expr) const-> decltype(std::declval<Closure>()(expr)){
		return m_closure(expr);
	}

private:
	friend class ::boost::iterator_core_access;

	/// \internal_
	typename Closure::value_type dereference() const
	{
		return typename Closure::value_type();
	}

	/// \internal_
	template<class C>
	bool equal(indexed_iterator<C> const& other) const
	{
		return m_index == other.m_index;
	}

	/// \internal_
	void increment()
	{
		m_index++;
	}

	/// \internal_
	void decrement()
	{
		m_index--;
	}

	/// \internal_
	void advance(std::ptrdiff_t n)
	{
		m_index = static_cast<size_t>(static_cast<std::ptrdiff_t>(m_index) + n);
	}

	/// \internal_
	template<class C>
	std::ptrdiff_t distance_to(indexed_iterator<C> const& other) const
	{
		return static_cast<std::ptrdiff_t>(other.m_index - m_index);
	}

private:
	Closure m_closure;
	std::size_t m_index;
	template<class> friend class indexed_iterator;
};


template<class Iterator>
class subrange_iterator : public boost::iterator_facade<
	subrange_iterator<Iterator>,
        typename Iterator::value_type,
        std::random_access_iterator_tag,
	typename Iterator::value_type
>{
public:
	subrange_iterator() = default;
	subrange_iterator(Iterator const &it, Iterator const& /*end*/, std::size_t startIterIndex,std::size_t /*startIndex*/)
	: m_iterator(it+startIterIndex){}
		
	template<class I>
	subrange_iterator(subrange_iterator<I> other):m_iterator(other.m_iterator){}

	template<class I>
	subrange_iterator& operator=(subrange_iterator<I> const& other){
		m_iterator = other.m_iterator;
		return *this;
	}

	size_t get_index() const{
		return m_iterator.index();
	}

	/// \internal_
	template<class Expr>
	auto operator[](Expr const& expr) const-> decltype(std::declval<Iterator>()[expr]){
		return m_iterator[expr];
	}

private:
	friend class ::boost::iterator_core_access;

	/// \internal_
	typename Iterator::value_type dereference() const
	{
		return typename Iterator::value_type();
	}

	/// \internal_
	template<class I>
	bool equal(subrange_iterator<I> const& other) const
	{
		return m_iterator == other.m_iterator;
	}

	/// \internal_
	void increment()
	{
		++m_iterator;
	}

	/// \internal_
	void decrement()
	{
		--m_iterator;
	}

	/// \internal_
	void advance(std::ptrdiff_t n)
	{
		m_iterator +=n;
	}

	/// \internal_
	template<class I>
	std::ptrdiff_t distance_to(subrange_iterator<I> const& other) const
	{
		return static_cast<std::ptrdiff_t>(other.m_iterator - m_iterator);
	}

private:
	Iterator m_iterator;
	template<class> friend class subrange_iterator;
};

}//End namespace detail
}//End namespace gpu

template<>
struct device_traits<gpu_tag>{
	typedef boost::compute::command_queue queue_type;
	
	static queue_type& default_queue(){
		return boost::compute::system::default_queue();
	}
	
	//adding of indices
	template<class Expr1, class Expr2>
	static auto index_add(Expr1 const& expr1, Expr2 const& expr2) -> decltype(boost::compute::plus<std::size_t>()(expr1,expr2)){
		return boost::compute::plus<std::size_t>()(expr1,expr2);
	}
	
	template<class E, class Index>
	static gpu::detail::invoked_linearized_matrix<typename E::const_closure_type,Index, typename E::orientation>
	linearized_matrix_element(matrix_expression<E, gpu_tag> const& e, Index const& index){
		return {e(),index};
	}
	
	
	template <class Iterator, class Functor>
	struct transform_iterator{
		typedef boost::compute::transform_iterator<Iterator, Functor> type;
	};
	
	template <class Iterator>
	struct subrange_iterator{
		typedef gpu::detail::subrange_iterator<Iterator> type;
	};
	
	template <class Iterator1, class Iterator2, class Functor>
	struct binary_transform_iterator{
		typedef gpu::detail::binary_transform_iterator<Iterator1,Iterator2, Functor> type;
	};
	
	template<class T>
	struct constant_iterator{
		typedef boost::compute::constant_iterator<T> type;
	};
	
	template<class T>
	struct one_hot_iterator{
		typedef iterators::one_hot_iterator<T> type;
	};
	
	template<class Closure>
	struct indexed_iterator{
		typedef gpu::detail::indexed_iterator<Closure> type;
	};
	
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
		gpu::detail::invoked_safe_div<Arg1,Arg2, T> operator()(const Arg1 &x, const Arg2& y) const
		{
			return {x,y,default_value};
		}
		T default_value;
	};
	template<class T>
	struct multiply_and_add : public boost::compute::function<T (T,T)>{
		typedef T result_type;
		multiply_and_add(T scalar) : boost::compute::function<T (T,T)>("multiply_and_add"), m_scalar(scalar) { }
		
		template<class Arg1, class Arg2>
		gpu::detail::invoked_multiply_and_add<Arg1,Arg2,T> operator()(const Arg1 &x, const Arg2& y) const
		{
			return {x,y, m_scalar};
		}
	private:
		T m_scalar;
	};
	template<class T>
	struct multiply_scalar : public boost::compute::function<T (T)>{
		typedef T result_type;
		multiply_scalar(T scalar) : boost::compute::function<T (T)>("multiply_scalar"), m_scalar(scalar) { }
		
		template<class Arg1>
		gpu::detail::invoked_multiply_scalar<Arg1,T> operator()(const Arg1 &x) const
		{
			return {x, m_scalar};
		}
	private:
		T m_scalar;
	};
	
	template<class T>
	struct multiply_assign : public boost::compute::function<T (T,T)>{
		typedef T result_type;
		multiply_assign(T scalar) : boost::compute::function<T (T,T)>("multiply_assign"), m_scalar(scalar) { }
		
		template<class Arg1, class Arg2>
		gpu::detail::invoked_multiply_scalar<Arg2,T> operator()(const Arg1&, const Arg2& y) const
		{
			return {y, m_scalar};
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
	using sin = boost::compute::sin<T>;
	template<class T>
	using cos = boost::compute::cos<T>;
	template<class T>
	using tan = boost::compute::tan<T>;
	template<class T>
	using asin = boost::compute::asin<T>;
	template<class T>
	using acos = boost::compute::acos<T>;
	template<class T>
	using atan = boost::compute::atan<T>;
	template<class T>
	using tanh = boost::compute::tanh<T>;
	template<class T>
	using sqrt = boost::compute::sqrt<T>;
	template<class T>
	using cbrt = boost::compute::cbrt<T>;
	template<class T>
	using abs = boost::compute::fabs<T>;
	
	template<class T>
	using erf = boost::compute::erf<T>;
	template<class T>
	using erfc = boost::compute::erfc<T>;
	
	template<class T>
	struct sqr : public boost::compute::function<T (T)>{
		typedef T result_type;
		sqr() : boost::compute::function<T (T)>("sqr") { }
		
		template<class Arg1>
		gpu::detail::invoked_sqr<Arg1,T> operator()(const Arg1 &x) const
		{
			return {x};
		}
	};
	template<class T>
	struct soft_plus : public boost::compute::function<T (T)>{
		typedef T result_type;
		soft_plus() : boost::compute::function<T (T)>("soft_plus") { }
		
		template<class Arg1>
		gpu::detail::invoked_soft_plus<Arg1,T> operator()(const Arg1 &x) const
		{
			return {x};
		}
	};
	template<class T>
	struct sigmoid : public boost::compute::function<T (T)>{
		typedef T result_type;
		sigmoid() : boost::compute::function<T (T)>("sigmoid") { }
		
		template<class Arg1>
		gpu::detail::invoked_sigmoid<Arg1,T> operator()(const Arg1 &x) const
		{
			return {x};
		}
	};
	template<class T>
	struct inv : public boost::compute::function<T (T)>{
		typedef T result_type;
		inv() : boost::compute::function<T (T)>("inv") { }
		
		template<class Arg1>
		gpu::detail::invoked_inv<Arg1,T> operator()(const Arg1 &x) const
		{
			return {x};
		}
	};
	
	//min/max
	template<class T>
	using min = boost::compute::fmin<T>;
	template<class T>
	using max = boost::compute::fmax<T>;
	
	//comparison
	template<class T>
	using less = boost::compute::less<T>;
	template<class T>
	using less_equal  = boost::compute::less_equal<T>;
	template<class T>
	using greater = boost::compute::greater<T>;
	template<class T>
	using greater_equal  = boost::compute::greater_equal<T>;
	template<class T>
	using equal = boost::compute::equal_to<T>;
	template<class T>
	using not_equal  = boost::compute::not_equal_to<T>;
	
};

}

namespace boost{namespace compute{
template<class I1, class I2, class F>
struct is_device_iterator<remora::gpu::detail::binary_transform_iterator<I1,I2, F> > : boost::true_type {};
template<class Closure>
struct is_device_iterator<remora::gpu::detail::indexed_iterator<Closure> > : boost::true_type {};
template<class Iterator>
struct is_device_iterator<remora::gpu::detail::subrange_iterator<Iterator> > : boost::true_type {};
}}

#endif