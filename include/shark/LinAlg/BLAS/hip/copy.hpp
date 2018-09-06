/*!
 * \brief       Implements operations to copy data from cpu to HIP and back
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
#ifndef REMORA_HIP_COPY_HPP
#define REMORA_HIP_COPY_HPP

#include "traits.hpp"
#include "../dense.hpp" //required for vector proxy on cpu
#include "../assignment.hpp"

namespace remora{namespace hip{

///////////////////////////////////////
//////// Vector Transport
///////////////////////////////////////	

template<class E>
class vector_transport_to_host: public vector_expression<vector_transport_to_host<E>, cpu_tag>{
public:
	typedef typename E::const_closure_type expression_closure_type;

	typedef typename E::value_type value_type;
	typedef typename E::size_type size_type;
	typedef value_type const& const_reference;
	typedef const_reference reference;

	typedef vector_transport_to_host const_closure_type;
	typedef vector_transport_to_host closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef blockwise<typename E::evaluation_category::tag> evaluation_category;


	//FIXME: This is required even though iterators for block expressions are meaningless
	typedef typename E::const_iterator const_iterator;
	typedef const_iterator iterator;

	// Construction and destruction
	explicit vector_transport_to_host(
		expression_closure_type const& expression
	):m_expression(expression){}

	size_type size() const {
		return m_expression.size();
	}
	
	expression_closure_type const& expression() const {
		return m_expression;
	}
	hip::device& queue() const{
		return m_expression.queue();
	}
	
	//dispatcher to computation kernels
	template<class VecX>
	void assign_to(vector_expression<VecX, cpu_tag>& x)const{
		typedef typename VecX::value_type Xvalue_type;
		//map x to device memory
		auto x_host = x().raw_storage().values;
		auto x_stride = x().raw_storage().stride;
		std::size_t bytes = x_stride * size() * sizeof(*x_host);
		hipHostRegister(x_host, bytes, 0);
		
		//acquire device ptr
		Xvalue_type* x_device;
		check_hip(hipHostGetDevicePointer( (void**)&x_device, x_host, 0));
		
		//compute result and directly store in x
		dense_vector_adaptor<Xvalue_type, dense_tag, hip_tag> adaptX({x_device, x_stride}, queue(), size());
		assign(adaptX, m_expression);
		
		//wait until the device is done
		synchronize_stream(queue());
		//unmap
		check_hip(hipHostUnregister(x_host));
	}
	template<class VecX>
	void plus_assign_to(vector_expression<VecX, cpu_tag>& x)const{
		typedef typename VecX::value_type Xvalue_type;
		//map x to device memory
		auto x_host = x().raw_storage().values;
		auto x_stride = x().raw_storage().stride;
		std::size_t bytes = x_stride * size() * sizeof(*x_host);
		hipHostRegister(x_host, bytes, 0);
		
		//acquire device ptr
		Xvalue_type* x_device;
		check_hip(hipHostGetDevicePointer( (void**)&x_device, x_host, 0));
		
		//compute result and directly store in x
		dense_vector_adaptor<Xvalue_type, dense_tag, hip_tag> adaptX({x_device, x_stride}, queue(), size());
		plus_assign(adaptX, m_expression);
		
		//wait until the device is done
		synchronize_stream(queue());
		//unmap
		check_hip(hipHostUnregister(x_host));
	}
	
private:
	expression_closure_type m_expression;
};

template<class E>
class vector_transport_to_device: public vector_expression<vector_transport_to_device<E>, hip_tag>{
public:
	typedef typename E::const_closure_type expression_closure_type;

	typedef typename E::value_type value_type;
	typedef typename E::size_type size_type;
	typedef value_type const& const_reference;
	typedef const_reference reference;

	typedef vector_transport_to_device const_closure_type;
	typedef vector_transport_to_device closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef blockwise<typename E::evaluation_category::tag> evaluation_category;


	//FIXME: This is required even though iterators for block expressions are meaningless
	typedef typename E::const_iterator const_iterator;
	typedef const_iterator iterator;

	// Construction and destruction
	explicit vector_transport_to_device(
		expression_closure_type const& expression,
		device& queue
	):m_expression(expression), m_queue(&queue){}

	size_type size() const {
		return m_expression.size();
	}
	expression_closure_type const& expression() const {
		return m_expression;
	}
	hip::device& queue() const{
		return *m_queue;
	}
	
	//dispatcher to computation kernels
	template<class VecX>
	void assign_to(vector_expression<VecX, hip_tag>& x)const{
		//evaluate expression is needed so we have host memory we can map to the device
		auto e_eval = eval_expression(m_expression);
		
		//map e to device memory
		auto e_host = e_eval.raw_storage().values;
		auto e_stride = e_eval.raw_storage().stride;
		std::size_t bytes = e_stride * size() * sizeof(*e_host);
		check_hip(hipHostRegister((void*)e_host, bytes, 0));
		
		//acquire device ptr
		value_type const* e_device;
		check_hip(hipHostGetDevicePointer( (void**)&e_device, (void*)e_host, 0));
		
		//compute result and directly store in x
		dense_vector_adaptor<value_type const, dense_tag, hip_tag> adaptE({e_device, e_stride}, x().queue(), size());
		assign(x, adaptE);
		
		//wait until the device is done
		synchronize_stream(x().queue());
		//unmap
		check_hip(hipHostUnregister((void*)e_host));
	}
	template<class VecX>
	void plus_assign_to(vector_expression<VecX, hip_tag>& x)const{
		//evaluate expression is needed so we have host memory we can map to the device
		auto e_eval = eval_expression(m_expression);
		
		//map e to device memory
		auto e_host = e_eval.raw_storage().values;
		auto e_stride = e_eval.raw_storage().stride;
		std::size_t bytes = e_stride * size() * sizeof(*e_host);
		hipHostRegister((void*)e_host, bytes, 0);
		
		//acquire device ptr
		value_type const* e_device;
		check_hip(hipHostGetDevicePointer( (void**)&e_device, (void*)e_host, 0));
		
		//compute result and directly store in x
		dense_vector_adaptor<value_type const, dense_tag, hip_tag> adaptE({e_device, e_stride}, x().queue(), size());
		plus_assign(x, adaptE);
		
		//wait until the device is done
		synchronize_stream(x().queue());
		//unmap
		check_hip(hipHostUnregister((void*)e_host));
	}
	
private:
	expression_closure_type m_expression;
	hip::device* m_queue;
};


///////////////////////////////////////
//////// Matrix Transport
///////////////////////////////////////	

template<class E>
class matrix_transport_to_host: public matrix_expression<matrix_transport_to_host<E>, cpu_tag>{
public:
	typedef typename E::const_closure_type expression_closure_type;

	typedef typename E::value_type value_type;
	typedef typename E::size_type size_type;
	typedef value_type const& const_reference;
	typedef const_reference reference;

	typedef matrix_transport_to_host const_closure_type;
	typedef matrix_transport_to_host closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef blockwise<typename E::evaluation_category::tag> evaluation_category;
	typedef typename E::orientation orientation;

	typedef no_iterator const_major_iterator;
	typedef no_iterator major_iterator;

	// Construction and destruction
	explicit matrix_transport_to_host(
		expression_closure_type const& expression
	):m_expression(expression){}

	size_type size1() const {
		return m_expression.size1();
	}
	size_type size2() const {
		return m_expression.size2();
	}
	expression_closure_type const& expression() const {
		return m_expression;
	}
	hip::device& queue() const{
		return m_expression.queue();
	}
	
	//dispatcher to computation kernels
	template<class MatX>
	void assign_to(matrix_expression<MatX, cpu_tag>& X) const{
		typedef typename MatX::value_type Xvalue_type;
		//map X to device memory
		auto X_host = X().raw_storage().values;
		std::size_t X_ld = X().raw_storage().leading_dimension;
		std::size_t X_stride1 = MatX::orientation::stride1(X_ld);
		std::size_t X_stride2 = MatX::orientation::stride2(X_ld);
		std::size_t bytes = (size1() * X_stride1 + size2() * X_stride2) * sizeof(*X_host);
		hipHostRegister(X_host, bytes, 0);
		
		//acquire device ptr
		Xvalue_type* X_device;
		check_hip(hipHostGetDevicePointer( (void**)&X_device, X_host, 0));
		
		//compute result and directly store in x
		typedef dense_matrix_adaptor<typename MatX::value_type, typename MatX::orientation, dense_tag, hip_tag> AdaptX;
		AdaptX adaptX({X_device, X_ld}, queue(), size1(), size2());
		assign(adaptX, m_expression);
		
		//wait until the device is done
		synchronize_stream(queue());
		//unmap
		check_hip(hipHostUnregister(X_host));
	}
	template<class MatX>
	void plus_assign_to(matrix_expression<MatX, cpu_tag>& X)const{
		typedef typename MatX::value_type Xvalue_type;
		//map X to device memory
		auto X_host = X().raw_storage().values;
		std::size_t X_ld = X().raw_storage().leading_dimension;
		std::size_t X_stride1 = MatX::orientation::stride1(X_ld);
		std::size_t X_stride2 = MatX::orientation::stride2(X_ld);
		std::size_t bytes = (size1() * X_stride1 + size2() * X_stride2) * sizeof(*X_host);
		hipHostRegister(X_host, bytes, 0);	
		
		//acquire device ptr
		Xvalue_type* X_device;
		check_hip(hipHostGetDevicePointer( (void**)&X_device, X_host, 0));
		
		//compute result and directly store in x
		typedef dense_matrix_adaptor<typename MatX::value_type, typename MatX::orientation, dense_tag, hip_tag> AdaptX;
		AdaptX adaptX({X_device, X_ld}, queue(), size1(), size2());
		plus_assign(adaptX, m_expression);
		
		//wait until the device is done
		synchronize_stream(queue());
		//unmap
		check_hip(hipHostUnregister(X_host));
	}
private:
	expression_closure_type m_expression;
};

template<class E>
class matrix_transport_to_device: public matrix_expression<matrix_transport_to_device<E>, hip_tag>{
public:
	typedef typename E::const_closure_type expression_closure_type;

	typedef typename E::value_type value_type;
	typedef typename E::size_type size_type;
	typedef value_type const& const_reference;
	typedef const_reference reference;

	typedef matrix_transport_to_device const_closure_type;
	typedef matrix_transport_to_device closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef blockwise<typename E::evaluation_category::tag> evaluation_category;
	typedef typename E::orientation orientation;

	typedef no_iterator const_major_iterator;
	typedef no_iterator major_iterator;

	// Construction and destruction
	explicit matrix_transport_to_device(
		expression_closure_type const& expression,
		hip::device& queue
	):m_expression(expression), m_queue(&queue){}

	size_type size1() const {
		return m_expression.size1();
	}
	size_type size2() const {
		return m_expression.size2();
	}
	expression_closure_type const& expression() const {
		return m_expression;
	}
	hip::device& queue() const{
		return *m_queue;
	}
	
	//dispatcher to computation kernels
	template<class MatX>
	void assign_to(matrix_expression<MatX, hip_tag>& X)const{
		//evaluate expression is needed so we have host memory we can map to the device
		auto e_eval = eval_expression(m_expression);
		
		//map X to device memory
		auto e_host = e_eval.raw_storage().values;
		std::size_t e_ld = e_eval.raw_storage().leading_dimension;
		std::size_t e_stride1 = orientation::stride1(e_ld);
		std::size_t e_stride2 = orientation::stride2(e_ld);
		std::size_t bytes = (size1() * e_stride1 + size2() * e_stride2) * sizeof(value_type);
		hipHostRegister((void*)e_host, bytes, 0);
		
		//acquire device ptr
		value_type* e_device;
		check_hip(hipHostGetDevicePointer( (void**)&e_device, (void*)e_host, 0));
		
		//compute result and directly store in x
		typedef dense_matrix_adaptor<value_type const, orientation, dense_tag, hip_tag> AdaptE;
		AdaptE adaptE({e_device, e_ld}, X().queue(), size1(), size2());
		assign(X, adaptE);
		
		//wait until the device is done
		synchronize_stream(X().queue());
		//unmap
		check_hip(hipHostUnregister((void*)e_host));
	}
	template<class MatX>
	void plus_assign_to(matrix_expression<MatX, hip_tag>& X) const{
		//evaluate expression is needed so we have host memory we can map to the device
		auto e_eval = eval_expression(m_expression);
		
		//map X to device memory
		auto e_host = e_eval.raw_storage().values;
		std::size_t e_ld = e_eval.raw_storage().leading_dimension;
		std::size_t e_stride1 = orientation::stride1(e_ld);
		std::size_t e_stride2 = orientation::stride2(e_ld);
		std::size_t bytes = (size1() * e_stride1 + size2() * e_stride2) * sizeof(value_type);
		hipHostRegister((void*)e_host, bytes, 0);
		
		//acquire device ptr
		value_type* e_device;
		check_hip(hipHostGetDevicePointer( (void**)&e_device, (void*)e_host, 0));
		
		//compute result and directly store in x
		typedef dense_matrix_adaptor<value_type const, orientation, dense_tag, hip_tag> AdaptE;
		AdaptE adaptE({e_device, e_ld}, X().queue(), size1(), size2());
		plus_assign(X, adaptE);
		
		//wait until the device is done
		synchronize_stream(X().queue());
		//unmap
		check_hip(hipHostUnregister((void*)e_host));
	}

private:
	expression_closure_type m_expression;
	hip::device* m_queue;
};

}

///////////////////////////////////////////////
//////// Expression Optimizers
///////////////////////////////////////////////

namespace detail{
template<class E>
struct matrix_scalar_multiply_optimizer<hip::vector_transport_to_device<E> >{
	typedef vector_scalar_multiply_optimizer<E> opt;
	typedef hip::vector_transport_to_device<typename opt::type> type;
	static type create(hip::vector_transport_to_device<E> const& v, typename type::value_type alpha){
		return type(opt::create(v.expression(), alpha), v.queue());
	}
};
template<class E>
struct matrix_scalar_multiply_optimizer<hip::vector_transport_to_host<E> >{
	typedef vector_scalar_multiply_optimizer<E> opt;
	typedef hip::vector_transport_to_host<typename opt::type> type;
	static type create(hip::vector_transport_to_host<E> const& v, typename type::value_type alpha){
		return type(opt::create(v.expression(), alpha));
	}
};

template<class E>
struct matrix_scalar_multiply_optimizer<hip::matrix_transport_to_device<E> >{
	typedef matrix_scalar_multiply_optimizer<E> opt;
	typedef hip::matrix_transport_to_device<typename opt::type> type;
	static type create(hip::matrix_transport_to_device<E> const& m, typename type::value_type alpha){
		return type(opt::create(m.expression(), alpha), m.queue());
	}
};

template<class E>
struct matrix_scalar_multiply_optimizer<hip::matrix_transport_to_host<E> >{
	typedef matrix_scalar_multiply_optimizer<E> opt;
	typedef hip::matrix_transport_to_host<typename opt::type> type;
	static type create(hip::matrix_transport_to_host<E> const& m, typename type::value_type alpha){
		return type(opt::create(m.expression(), alpha));
	}
};
}

//TODO: proxy(copy_to_device) should be possible...

///////////////////////////////////////////////
//////// Expressions
///////////////////////////////////////////////

template<class E>
hip::vector_transport_to_host<E> copy_to_cpu(vector_expression<E, hip_tag> const& e){
	return hip::vector_transport_to_host<E>(e());
}

template<class E>
hip::matrix_transport_to_host<E> copy_to_cpu(matrix_expression<E, hip_tag> const& e){
	return hip::matrix_transport_to_host<E>(e());
}


template<class E>
hip::vector_transport_to_device<E> copy_to_device(
	vector_expression<E, cpu_tag> const& e, hip_tag, 
	hip::device& queue = device_traits<hip_tag>::default_queue()
){
	return hip::vector_transport_to_device<E>(e(), queue);
}


template<class E>
hip::matrix_transport_to_device<E> copy_to_device(
	matrix_expression<E, cpu_tag> const& e, hip_tag,
	hip::device& queue = device_traits<hip_tag>::default_queue()
){
	return hip::matrix_transport_to_device<E>(e(), queue);
}
//moving hip->hip is for free
template<class E>
E const& copy_to_device(vector_expression<E, hip_tag> const& e, hip_tag){
	e();
}
template<class E>
E const& copy_to_device(matrix_expression<E, hip_tag> const& e, hip_tag){
	e();
}
	

}

#endif
