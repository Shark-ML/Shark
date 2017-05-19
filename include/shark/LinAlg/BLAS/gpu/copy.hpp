/*!
 * \brief       Implements operations to copy data from cpu to gpu and back
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
#ifndef REMORA_GPU_COPY_HPP
#define REMORA_GPU_COPY_HPP

#include "../detail/traits.hpp"
//includes required for storage->vector/matrix and for scalar multiplication
#include "../detail/vector_proxy_classes.hpp"
#include "../detail/vector_expression_classes.hpp"
#include "../detail/matrix_proxy_classes.hpp"
#include "../detail/matrix_expression_classes.hpp"

namespace remora{

///////////////////////////////////////
//////// Vector Transport
///////////////////////////////////////	

template<class E>
class vector_transport_to_cpu: public vector_expression<vector_transport_to_cpu<E>, cpu_tag>{
public:
	typedef typename E::const_closure_type expression_closure_type;

	typedef typename E::value_type value_type;
	typedef typename E::size_type size_type;
	typedef value_type const& const_reference;
	typedef const_reference reference;

	typedef vector_transport_to_cpu const_closure_type;
	typedef vector_transport_to_cpu closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef blockwise<typename E::evaluation_category::tag> evaluation_category;


	//FIXME: This is required even though iterators for block expressions are meaningless
	typedef typename E::const_iterator const_iterator;
	typedef const_iterator iterator;

	// Construction and destruction
	explicit vector_transport_to_cpu(
		expression_closure_type const& expression
	):m_expression(expression){}

	size_type size() const {
		return m_expression.size();
	}
	
	expression_closure_type const& expression() const {
		return m_expression;
	}
	boost::compute::command_queue& queue() const{
		return m_expression.queue();
	}
	
	//dispatcher to computation kernels
	template<class VecX>
	void assign_to(vector_expression<VecX, cpu_tag>& x, value_type const& alpha = value_type(1) )const{
		assign_to(x, m_expression, alpha, typename E::storage_type::storage_tag());
	}
	template<class VecX>
	void plus_assign_to(vector_expression<VecX, cpu_tag>& x, value_type const& alpha = value_type(1) )const{
		plus_assign_to(x, m_expression, alpha, typename E::storage_type::storage_tag());
	}
	
	template<class VecX>
	void minus_assign_to(vector_expression<VecX, cpu_tag>& x, value_type alpha = value_type(1) )const{
		plus_assign_to(x,-alpha);
	}
	
private:
	//result is represented as dense storage on gpu, i.e. does not need to be calculated
	template<class VecX, class VecE>
	void assign_to(
		vector_expression<VecX, cpu_tag>& x, vector_expression<VecE, gpu_tag> const& e,
		value_type const& alpha, dense_tag
	)const{
		auto storageE = e().raw_storage();
		auto& buffer = storageE.buffer;
		//map buffer to host memory
		typename VecE::value_type* p = (typename VecE::value_type*) e().queue().enqueue_map_buffer(
			buffer, CL_MAP_READ, 0, buffer.size()
		);
		//adapt host memory buffer to vector and assign
		typedef dense_vector_adaptor<typename VecE::value_type> AdaptE;
		AdaptE adaptE(p + storageE.offset,size(), storageE.stride);
		assign(x, vector_scalar_multiply<AdaptE >( adaptE, alpha));
		
		//unmap memory
		e().queue().enqueue_unmap_buffer(buffer,p);
	}
	
	template<class VecX, class VecE>
	void plus_assign_to(
		vector_expression<VecX, cpu_tag>& x, vector_expression<VecE, gpu_tag> const& e,
		value_type const& alpha, dense_tag
	)const{
		auto storageE = e().raw_storage();
		auto& buffer = storageE.buffer;
		//map buffer to host memory
		typename VecE::value_type* p = (typename VecE::value_type*) e().queue().enqueue_map_buffer(
			buffer, CL_MAP_READ, 0, buffer.size()
		);
		//adapt host memory buffer to vector and assign
		typedef dense_vector_adaptor<typename VecE::value_type> AdaptE;
		AdaptE adaptE(p + storageE.offset,size(), storageE.stride);
		
		plus_assign(x,vector_scalar_multiply<AdaptE >( adaptE, alpha));
		
		//unmap memory
		e().queue().enqueue_unmap_buffer(buffer,p);
	}
	
	//result has unknown storage, so evaluate into temporary on gpu and assign that to host
	template<class VecX, class VecE>
	void assign_to(
		vector_expression<VecX, cpu_tag>& x, vector_expression<VecE, gpu_tag> const& e,
		value_type const& alpha, unknown_tag
	)const{
		//evaluate expression on gpu and assign that to host
		typedef typename vector_temporary<E>::type result_type;
		result_type result = m_expression;
		
		assign_to(x, result, alpha, typename result_type::storage_type::storage_tag());
	}
	
	template<class VecX, class VecE>
	void plus_assign_to(
		vector_expression<VecX, cpu_tag>& x, vector_expression<VecE, gpu_tag> const& e,
		value_type const& alpha, unknown_tag
	)const{
		//evaluate expression on gpu and assign that to host
		typedef typename vector_temporary<E>::type result_type;
		result_type result = m_expression;
		
		plus_assign_to(x, result, alpha, typename result_type::storage_type::storage_tag());
	}
	expression_closure_type m_expression;
};

template<class E>
class vector_transport_to_gpu: public vector_expression<vector_transport_to_gpu<E>, gpu_tag>{
public:
	typedef typename E::const_closure_type expression_closure_type;

	typedef typename E::value_type value_type;
	typedef typename E::size_type size_type;
	typedef value_type const& const_reference;
	typedef const_reference reference;

	typedef vector_transport_to_gpu const_closure_type;
	typedef vector_transport_to_gpu closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef blockwise<typename E::evaluation_category::tag> evaluation_category;


	//FIXME: This is required even though iterators for block expressions are meaningless
	typedef typename E::const_iterator const_iterator;
	typedef const_iterator iterator;

	// Construction and destruction
	explicit vector_transport_to_gpu(
		expression_closure_type const& expression,
		boost::compute::command_queue& queue
	):m_expression(expression), m_queue(&queue){}

	size_type size() const {
		return m_expression.size();
	}
	expression_closure_type const& expression() const {
		return m_expression;
	}
	boost::compute::command_queue& queue() const{
		return *m_queue;
	}
	
	//dispatcher to computation kernels
	template<class VecX>
	void assign_to(vector_expression<VecX, gpu_tag>& x, value_type const& alpha = value_type(1) )const{
		assign_to(x, m_expression, alpha, typename E::storage_type::storage_tag());
	}
	template<class VecX>
	void plus_assign_to(vector_expression<VecX, gpu_tag>& x, value_type const& alpha = value_type(1) )const{
		plus_assign_to(x, m_expression, alpha, typename E::storage_type::storage_tag());
	}
	
	template<class VecX>
	void minus_assign_to(vector_expression<VecX, gpu_tag>& x, value_type alpha = value_type(1) )const{
		plus_assign_to(x,-alpha);
	}
	
private:
	//result is represented as dense storage on cpu, i.e. does not need to be calculated
	template<class VecX, class VecE>
	void assign_to(
		vector_expression<VecX, gpu_tag>& x, vector_expression<VecE, cpu_tag> const& e,
		value_type const& alpha, dense_tag
	)const{
		auto storagex = x().raw_storage();
		auto& buffer = storagex.buffer;
		//map buffer to host memory
		typename VecX::value_type* p = (typename VecX::value_type*) x().queue().enqueue_map_buffer(
			buffer, CL_MAP_WRITE, 0, buffer.size()
		);
		//adapt host memory buffer to vector and assign
		dense_vector_adaptor<typename VecX::value_type> adaptX(p + storagex.offset,size(), storagex.stride);
		assign(adaptX,vector_scalar_multiply<expression_closure_type>(m_expression,alpha));
		
		//unmap memory
		x().queue().enqueue_unmap_buffer(buffer,p);
	}
	
	template<class VecX, class VecE>
	void plus_assign_to(
		vector_expression<VecX, gpu_tag>& x, vector_expression<VecE, cpu_tag> const& e,
		value_type const& alpha, dense_tag
	)const{
		auto storagex = x().raw_storage();
		auto& buffer = storagex.buffer;
		//map buffer to host memory
		typename VecX::value_type* p = (typename VecX::value_type*) x().queue().enqueue_map_buffer(
			buffer, CL_MAP_WRITE, storagex.offset, buffer.size() - storagex.offset
		);
		//adapt host memory buffer to vector and assign
		dense_vector_adaptor<typename VecX::value_type> adaptX(p,size(), storagex.stride);
		plus_assign(adaptX,vector_scalar_multiply<expression_closure_type>(m_expression,alpha));
		
		//unmap memory
		x().queue().enqueue_unmap_buffer(buffer,p);
	}

	expression_closure_type m_expression;
	boost::compute::command_queue* m_queue;
};


///////////////////////////////////////
//////// Matrix Transport
///////////////////////////////////////	

template<class E>
class matrix_transport_to_cpu: public matrix_expression<matrix_transport_to_cpu<E>, cpu_tag>{
public:
	typedef typename E::const_closure_type expression_closure_type;

	typedef typename E::value_type value_type;
	typedef typename E::size_type size_type;
	typedef value_type const& const_reference;
	typedef const_reference reference;

	typedef matrix_transport_to_cpu const_closure_type;
	typedef matrix_transport_to_cpu closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef blockwise<typename E::evaluation_category::tag> evaluation_category;
	typedef typename E::orientation orientation;

	//FIXME: This is required even though iterators for block expressions are meaningless
	typedef typename E::const_row_iterator const_row_iterator;
	typedef typename E::const_column_iterator const_column_iterator;
	typedef const_row_iterator row_iterator;
	typedef const_column_iterator column_iterator;

	// Construction and destruction
	explicit matrix_transport_to_cpu(
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
	boost::compute::command_queue& queue() const{
		return m_expression.queue();
	}
	
	//dispatcher to computation kernels
	template<class MatX>
	void assign_to(matrix_expression<MatX, cpu_tag>& X, value_type const& alpha = value_type(1) )const{
		assign_to(X, m_expression, alpha, typename E::storage_type::storage_tag());
	}
	template<class MatX>
	void plus_assign_to(matrix_expression<MatX, cpu_tag>& X, value_type const& alpha = value_type(1) )const{
		plus_assign_to(X, m_expression, alpha, typename E::storage_type::storage_tag());
	}
	
	template<class MatX>
	void minus_assign_to(matrix_expression<MatX, cpu_tag>& X, value_type alpha = value_type(1) )const{
		plus_assign_to(X,-alpha);
	}
	
private:
	//result is represented as dense storage on gpu, i.e. does not need to be calculated
	template<class MatX, class MatE>
	void assign_to(
		matrix_expression<MatX, cpu_tag>& X, matrix_expression<MatE, gpu_tag>const& e,
		value_type const& alpha, dense_tag
	)const{
		auto storageE = e().raw_storage();
		auto& buffer = storageE.buffer;
		//map buffer to host memory
		typename MatE::value_type* p = (typename MatE::value_type*) e().queue().enqueue_map_buffer(
			buffer, CL_MAP_READ, 0, buffer.size()
		);
		//adapt host memory buffer to matrix and assign
		typedef typename MatE::orientation EOrientation;
		std::size_t stride1 = EOrientation::index_M(storageE.leading_dimension,1);
		std::size_t stride2 = EOrientation::index_m(storageE.leading_dimension,1);
		typedef dense_matrix_adaptor<typename MatE::value_type, EOrientation> AdaptE;
		AdaptE adaptE(p + storageE.offset,size1(), size2(), stride1,stride2);
		
		assign(X, matrix_scalar_multiply<AdaptE >( adaptE, alpha));
		
		//unmap memory
		e().queue().enqueue_unmap_buffer(buffer,p);
	}
	
	template<class MatX, class MatE>
	void plus_assign_to(
		matrix_expression<MatX, cpu_tag>& X, matrix_expression<MatE, gpu_tag> const& e,
		value_type const& alpha, dense_tag
	)const{
		auto storageE = e().raw_storage();
		auto& buffer = storageE.buffer;
		//map buffer to host memory
		typename MatE::value_type* p = (typename MatE::value_type*) e().queue().enqueue_map_buffer(
			buffer, CL_MAP_READ, 0, buffer.size()
		);
		//adapt host memory buffer to matrix and assign
		typedef typename MatE::orientation EOrientation;
		std::size_t stride1 = EOrientation::index_M(storageE.leading_dimension,1);
		std::size_t stride2 = EOrientation::index_m(storageE.leading_dimension,1);
		typedef dense_matrix_adaptor<typename MatE::value_type, EOrientation> AdaptE;
		AdaptE adaptE(p + storageE.offset, size1(), size2(), stride1,stride2);
		
		plus_assign(X,matrix_scalar_multiply<AdaptE >( adaptE, alpha));
		
		//unmap memory
		e().queue().enqueue_unmap_buffer(buffer,p);
	}
	
	//result has unknown storage, so evaluate into temporary on gpu and assign that to host
	template<class MatX, class MatE>
	void assign_to(
		matrix_expression<MatX, cpu_tag>& X, matrix_expression<MatE, gpu_tag>const& e,
		value_type const& alpha, unknown_tag
	)const{
		//evaluate expression on gpu and assign that to host
		typedef typename matrix_temporary<E>::type result_type;
		result_type result = m_expression;
		
		assign_to(X, result, alpha, typename result_type::storage_type::storage_tag());
	}
	
	template<class MatX, class MatE>
	void plus_assign_to(
		matrix_expression<MatX, cpu_tag>& X, matrix_expression<MatE, gpu_tag>const& e,
		value_type const& alpha, unknown_tag
	)const{
		//evaluate expression on gpu and assign that to host
		typedef typename matrix_temporary<E>::type result_type;
		result_type result = m_expression;
		
		plus_assign_to(X, result, alpha, typename result_type::storage_type::storage_tag());
	}

	expression_closure_type m_expression;
};

template<class E>
class matrix_transport_to_gpu: public matrix_expression<matrix_transport_to_gpu<E>, gpu_tag>{
public:
	typedef typename E::const_closure_type expression_closure_type;

	typedef typename E::value_type value_type;
	typedef typename E::size_type size_type;
	typedef value_type const& const_reference;
	typedef const_reference reference;

	typedef matrix_transport_to_gpu const_closure_type;
	typedef matrix_transport_to_gpu closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef blockwise<typename E::evaluation_category::tag> evaluation_category;
	typedef typename E::orientation orientation;

	//FIXME: This is required even though iterators for block expressions are meaningless
	typedef typename E::const_row_iterator const_row_iterator;
	typedef typename E::const_column_iterator const_column_iterator;
	typedef const_row_iterator row_iterator;
	typedef const_column_iterator column_iterator;

	// Construction and destruction
	explicit matrix_transport_to_gpu(
		expression_closure_type const& expression,
		boost::compute::command_queue& queue
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
	boost::compute::command_queue& queue() const{
		return *m_queue;
	}
	
	//dispatcher to computation kernels
	template<class MatX>
	void assign_to(matrix_expression<MatX, gpu_tag>& X, value_type const& alpha = value_type(1) )const{
		assign_to(X, m_expression, alpha, typename E::storage_type::storage_tag());
	}
	template<class MatX>
	void plus_assign_to(matrix_expression<MatX, gpu_tag>& X, value_type const& alpha = value_type(1) )const{
		plus_assign_to(X, m_expression, alpha, typename E::storage_type::storage_tag());
	}
	
	template<class MatX>
	void minus_assign_to(matrix_expression<MatX, cpu_tag>& X, value_type alpha = value_type(1) )const{
		plus_assign_to(X,-alpha);
	}
	
private:
	//result is represented as dense storage on gpu, i.e. does not need to be calculated
	template<class MatX, class MatE>
	void assign_to(
		matrix_expression<MatX, gpu_tag>& X, matrix_expression<MatE, cpu_tag> const& e,
		value_type const& alpha, dense_tag
	)const{
		auto storageX = X().raw_storage();
		auto& buffer = storageX.buffer;
		//map buffer to host memory
		typename MatX::value_type* p = (typename MatX::value_type*) X().queue().enqueue_map_buffer(
			buffer, CL_MAP_WRITE, 0, buffer.size()
		);
		//adapt host memory buffer to vector and assign
		typedef typename MatX::orientation XOrientation;
		std::size_t stride1 = XOrientation::index_M(storageX.leading_dimension, 1);
		std::size_t stride2 = XOrientation::index_m(storageX.leading_dimension, 1);
		dense_matrix_adaptor<typename MatX::value_type, XOrientation> adaptX(p, size1(), size2(), stride1, stride2);
		assign(adaptX,matrix_scalar_multiply<MatE>(e(),alpha));
		
		//unmap memory
		X().queue().enqueue_unmap_buffer(buffer,p);
	}
	
	template<class MatX, class MatE>
	void plus_assign_to(
		matrix_expression<MatX, gpu_tag>& X, matrix_expression<MatE, cpu_tag> const& e,
		value_type const& alpha, dense_tag
	)const{
		auto storageX = X().raw_storage();
		auto& buffer = storageX.buffer;
		//map buffer to host memory
		typename MatX::value_type* p = (typename MatX::value_type*) X().queue().enqueue_map_buffer(
			buffer, CL_MAP_WRITE, 0, buffer.size()
		);
		//adapt host memory buffer to matrix and assign
		typedef typename MatX::orientation XOrientation;
		std::size_t stride1 = XOrientation::index_M(storageX.leading_dimension, 1);
		std::size_t stride2 = XOrientation::index_m(storageX.leading_dimension, 1);
		typedef dense_matrix_adaptor<typename MatX::value_type, XOrientation> AdaptX;
		AdaptX adaptX(p + storageX.offset, size1(), size2(), stride1, stride2);
		
		plus_assign(adaptX,matrix_scalar_multiply<MatE >( e(), alpha));
		
		//unmap memory
		X().queue().enqueue_unmap_buffer(buffer,p);
	}

	expression_closure_type m_expression;
	boost::compute::command_queue* m_queue;
};


///////////////////////////////////////////////
////////Proxy expressions
///////////////////////////////////////////////

template<class E>
vector_transport_to_cpu<E> copy_to_cpu(vector_expression<E, gpu_tag> const& e){
	return vector_transport_to_cpu<E>(e());
}

template<class E>
matrix_transport_to_cpu<E> copy_to_cpu(matrix_expression<E, gpu_tag> const& e){
	return matrix_transport_to_cpu<E>(e());
}
template<class E>
vector_transport_to_gpu<E> copy_to_gpu(
	vector_expression<E, cpu_tag> const& e,
	boost::compute::command_queue& queue = boost::compute::system::default_queue()
){
	return vector_transport_to_gpu<E>(e(), queue);
}

template<class E>
matrix_transport_to_gpu<E> copy_to_gpu(
	matrix_expression<E, cpu_tag> const& e,
	boost::compute::command_queue& queue = boost::compute::system::default_queue()
){
	return matrix_transport_to_gpu<E>(e(),queue);
}


//moving gpu->gpu is for free
template<class E>
E const& copy_to_gpu(
	vector_expression<E, gpu_tag> const& e,
	boost::compute::command_queue& queue = boost::compute::system::default_queue()
){
	return e();
}

template<class E>
E const& copy_to_gpu(
	matrix_expression<E, gpu_tag> const& e,
	boost::compute::command_queue& queue = boost::compute::system::default_queue()
){
	return e();
}
	
}

#endif
