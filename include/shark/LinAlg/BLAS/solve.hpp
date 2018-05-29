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
#ifndef REMORA_SOLVE_HPP
#define REMORA_SOLVE_HPP

#include "decompositions.hpp"

namespace remora{


//////////////////////////////////////////////////////
////////Expression Types for Solving and inverse
//////////////////////////////////////////////////////

template<class MatA, class VecV, class SystemType, class Side>
class matrix_vector_solve: public vector_expression<
	matrix_vector_solve<MatA, VecV,SystemType, Side>,
	typename MatA::device_type
>{
public:
	typedef typename MatA::const_closure_type matrix_closure_type;
	typedef typename VecV::const_closure_type vector_closure_type;
	typedef decltype(
		typename MatA::value_type() * typename VecV::value_type()
	) value_type;
	typedef typename MatA::size_type size_type;
	typedef value_type const_reference;
	typedef const_reference reference;

	typedef matrix_vector_solve const_closure_type;
	typedef const_closure_type closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef blockwise<dense_tag> evaluation_category;
	typedef typename MatA::device_type device_type;

	size_type size() const {
		return m_rhs.size();
	}
	typename device_traits<device_type>::queue_type& queue()const{
		return m_matrix.queue();
	}
	matrix_vector_solve(
		matrix_closure_type const& matrix, vector_closure_type const&rhs, 
		SystemType system_type = SystemType()
	):m_matrix(matrix), m_rhs(rhs), m_system_type(system_type){}
	
	matrix_closure_type const& lhs()const{
		return m_matrix;
	}
	
	vector_closure_type const& rhs()const{
		return m_rhs;
	}
	
	SystemType const& system_type()const{
		return m_system_type;
	}
	
	typedef no_iterator iterator;
	typedef iterator const_iterator;
	
	//dispatcher to computation kernels
	template<class VecX>
	void assign_to(vector_expression<VecX, device_type>& x)const{
		assign(x,m_rhs);
		solver<MatA,SystemType> alg(m_matrix, m_system_type);
		alg.solve(x,Side());
	}
	template<class VecX>
	void plus_assign_to(vector_expression<VecX, device_type>& x)const{
		typename vector_temporary<VecX>::type temp(m_rhs);
		solver<MatA,SystemType> alg(m_matrix, m_system_type);
		alg.solve(temp,Side());
		plus_assign(x,temp);
	}
private:
	matrix_closure_type m_matrix;
	vector_closure_type m_rhs;
	SystemType m_system_type;
};


template<class MatA, class MatB, class SystemType, class Side>
class matrix_matrix_solve: public matrix_expression<
	matrix_matrix_solve<MatA, MatB, SystemType, Side>,
	typename MatA::device_type
>{
public:
	typedef typename MatA::const_closure_type matrixA_closure_type;
	typedef typename MatB::const_closure_type matrixB_closure_type;
	typedef decltype(
		typename MatA::value_type() * typename MatB::value_type()
	) value_type;
	typedef typename MatA::size_type size_type;
	typedef value_type const_reference;
	typedef const_reference reference;

	typedef matrix_matrix_solve const_closure_type;
	typedef const_closure_type closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef blockwise<dense_tag> evaluation_category;
	typedef typename MatA::device_type device_type;
	typedef unknown_orientation orientation;
	
	matrix_matrix_solve(
		matrixA_closure_type const& matrix, matrixB_closure_type const& rhs,
		SystemType const& system_type =SystemType()
	):m_matrix(matrix), m_rhs(rhs), m_system_type(system_type){}

	size_type size1() const {
		return m_rhs.size1();
	}
	size_type size2() const {
		return m_rhs.size2();
	}
	typename device_traits<device_type>::queue_type& queue()const{
		return m_matrix.queue();
	}
	
	matrixA_closure_type const& lhs()const{
		return m_matrix;
	}
	
	matrixB_closure_type const& rhs()const{
		return m_rhs;
	}
	
	SystemType const& system_type()const{
			return m_system_type;
	}
	
	typedef no_iterator major_iterator;
	typedef major_iterator const_major_iterator;
	
	//dispatcher to computation kernels
	template<class MatX>
	void assign_to(matrix_expression<MatX, device_type>& X)const{
		assign(X,m_rhs);
		solver<MatA,SystemType> alg(m_matrix,m_system_type);
		alg.solve(X,Side());
	}
	template<class MatX>
	void plus_assign_to(matrix_expression<MatX, device_type>& X)const{
		typename matrix_temporary<MatX>::type temp(m_rhs);
		solver<MatA,SystemType> alg(m_matrix,m_system_type);
		alg.solve(temp,Side());
		plus_assign(X,temp);
	}
private:
	matrixA_closure_type m_matrix;
	matrixB_closure_type m_rhs;
	SystemType m_system_type;
};

template<class MatA, class SystemType>
class matrix_inverse: public matrix_expression<
	matrix_inverse<MatA, SystemType>,
	typename MatA::device_type
>{
public:
	typedef typename MatA::const_closure_type matrix_closure_type;
	typedef typename MatA::value_type value_type;
	typedef typename MatA::size_type size_type;
	typedef value_type const_reference;
	typedef const_reference reference;

	typedef matrix_inverse const_closure_type;
	typedef const_closure_type closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef blockwise<dense_tag> evaluation_category;
	typedef typename MatA::device_type device_type;
	typedef typename MatA::orientation orientation;

	matrix_inverse(matrix_closure_type const& matrix, SystemType system_type)
	:m_matrix(matrix), m_system_type(system_type){}
	
	size_type size1() const {
		return m_matrix.size1();
	}
	size_type size2() const {
		return m_matrix.size1();
	}

	typename device_traits<device_type>::queue_type& queue()const{
		return m_matrix.queue();
	}
	
	matrix_closure_type const& matrix()const{
		return m_matrix;
	}
	
	SystemType const& system_type()const{
			return m_system_type;
	}
	
	typedef no_iterator major_iterator;
	typedef major_iterator const_major_iterator;
	
	//dispatcher to computation kernels
	template<class MatX>
	void assign_to(matrix_expression<MatX, device_type>& X)const{
		typedef scalar_vector<value_type, device_type> diag_vec;
		assign(X,diagonal_matrix<diag_vec>(diag_vec(size1(),value_type(1))));
		solver<MatA,SystemType> alg(m_matrix,m_system_type);
		alg.solve(X,left());
	}
	template<class MatX>
	void plus_assign_to(matrix_expression<MatX, device_type>& X)const{
		typedef scalar_vector<value_type, device_type> diag_vec;
		typename matrix_temporary<MatX>::type temp = diagonal_matrix<diag_vec>(diag_vec(size1(),value_type(1)));
		solver<MatA,SystemType> alg(m_matrix,m_system_type);
		alg.solve(temp,left());
		plus_assign(X,temp);
	}
private:
	matrix_closure_type m_matrix;
	SystemType m_system_type;
};

//////////////////////////////////////////////////////
////////Expression Optimizations
//////////////////////////////////////////////////////

namespace detail{
	
////////////////////////////////////
//// Vector Solve
////////////////////////////////////
template<class M, class V, class Tag, class Side>
struct matrix_vector_solve_optimizer{
	typedef matrix_vector_solve<M,V, Tag, Side> type;
	
	static type create(
		typename M::const_closure_type const& m, 
		typename V::const_closure_type const& v,
		Tag t
	){
		return type(m,v,t);
	}
};

////////////////////////////////////
//// Matrix Solve
////////////////////////////////////
template<class M1, class M2, class Tag, class Side>
struct matrix_matrix_solve_optimizer{
	typedef matrix_matrix_solve<M1,M2, Tag, Side> type;
	
	static type create(
		typename M1::const_closure_type const& lhs, 
		typename M2::const_closure_type const& rhs,
		Tag t
	){
		return type(lhs,rhs,t);
	}
};

////////////////////////////////////
//// Matrix Inverse
////////////////////////////////////
template<class M, class Tag>
struct matrix_inverse_optimizer{
	typedef matrix_inverse<M, Tag> type;
	
	static type create(typename M::const_closure_type const& m, Tag t){
		return type(m,t);
	}
};

//////////////////////////////////
/////Interactions with other expressions
//////////////////////////////////

//small helper needed for transpose
template<class T>
struct solve_tag_transpose_helper{
	static T transpose(T t){return t;}
};
template<bool Upper, bool Unit>
struct solve_tag_transpose_helper<triangular_tag<Upper,Unit> >{
	static triangular_tag<!Upper,Unit> transpose(triangular_tag<Upper,Unit>){return triangular_tag<!Upper,Unit>();}
};

//trans(solve(A,B,left)) = solve(trans(A),trans(B),right)
//trans(solve(A,B,right)) = solve(trans(A),trans(B),left)
template<class M1, class M2, bool Left, class Tag>
struct matrix_transpose_optimizer<matrix_matrix_solve<M1,M2, Tag, system_tag<Left> > >{
	typedef matrix_transpose_optimizer<typename M2::const_closure_type> lhs_opt;
	typedef matrix_transpose_optimizer<typename M2::const_closure_type> rhs_opt;
	typedef matrix_matrix_solve_optimizer<
		typename lhs_opt::type,typename rhs_opt::type,
		typename Tag::transposed_tag, system_tag<!Left>
	> opt;
	typedef typename opt::type type;
	
	static type create(matrix_matrix_solve<M1,M2, Tag, system_tag<Left> > const& m){
		return opt::create(
			lhs_opt::create(m.rhs()),rhs_opt::create(m.lhs()), 
			solve_tag_transpose_helper<Tag>::transpose(m.system_type())
		);
	}
};


template<class M, class Tag>
struct matrix_transpose_optimizer<matrix_inverse<M, Tag> >{
	typedef matrix_transpose_optimizer<typename M::const_closure_type> mat_opt;
	typedef matrix_inverse_optimizer<
		typename mat_opt::type, typename Tag::transposed_orientation
	> opt;
	typedef typename opt::type type;
	
	static type create(matrix_inverse<M, Tag> const& m){
		return opt::create(
			mat_opt::create(m.matrix()), 
			solve_tag_transpose_helper<Tag>::transpose(m.system_type())
		);
	}
};



//prod(inv(A),b) = solve(A,b,left)
template<class M, class V, class Tag>
struct matrix_vector_prod_optimizer<matrix_inverse<M,Tag>, V>{
	typedef matrix_vector_solve_optimizer<M,V, Tag, left> opt;
	typedef typename opt::type type;
	
	static type create(matrix_inverse<M,Tag> const& inv, typename V::const_closure_type const& v){
		return opt::create(inv.matrix(),v,inv.system_type());
	}
};

//prod(solve(A,B,left),c) = solve(A, prod(B,c),right) 
template<class M1, class M2,class V, class Tag>
struct matrix_vector_prod_optimizer<matrix_matrix_solve<M1,M2, Tag, left >, V >{
	typedef matrix_vector_prod_optimizer<M2,V> prod_opt;
	typedef matrix_vector_solve_optimizer<M1, typename prod_opt::type, Tag, left> opt;
	typedef typename opt::type type;
	
	static type create(matrix_matrix_solve<M1,M2, Tag, left> const& m,  typename V::const_closure_type const& v){
		return opt::create(
			m.lhs(), prod_opt::create(m.rhs(),v), m.system_type()
		);
	}
};

//prod(solve(A,B,right),c) = prod(B,solve(A,c, left)) 
template<class M1, class M2, class V, class Tag>
struct matrix_vector_prod_optimizer<matrix_matrix_solve<M1,M2, Tag, right >, V >{
	typedef matrix_vector_solve_optimizer<M1, V, Tag, left> solve_opt;
	typedef matrix_vector_prod_optimizer<M2,typename solve_opt::type> opt;
	typedef typename opt::type type;
	static type create(matrix_matrix_solve<M1,M2, Tag, right> const& m,  typename V::const_closure_type const& v){
		return opt::create(
			m.rhs(), solve_opt::create(m.lhs(),v,m.system_type()) 
		);
	}
};

//row(solve(A,B,left),i) = prod(solve(A,e_i,right),B) = prod(trans(B),solve(A,e_i,right)) 
template<class M1, class M2,class Tag>
struct matrix_row_optimizer<matrix_matrix_solve<M1,M2, Tag, left > >{
	typedef matrix_transpose_optimizer<typename M2::const_closure_type> rhs_opt;
	typedef unit_vector<typename M1::value_type, typename M1::device_type> unit;
	typedef matrix_vector_solve_optimizer<M1, unit, Tag, right> solve_opt;
	typedef matrix_vector_prod_optimizer<typename rhs_opt::type,typename solve_opt::type> opt;
	typedef typename opt::type type;
	
	static type create(matrix_matrix_solve<M1,M2, Tag, left> const& m, std::size_t i){
		return opt::create(
			rhs_opt::create(m.rhs()),
			solve_opt::create(m.lhs(),unit(m.lhs().size2(),i), m.system_type())
		);
	}
};

//row(solve(A,B,right),i) = solve(A,row(B,i),right) 
template<class M1, class M2, class Tag>
struct matrix_row_optimizer<matrix_matrix_solve<M1,M2, Tag, right > >{
	typedef matrix_row_optimizer<typename M2::const_closure_type> rhs_opt;
	typedef matrix_vector_solve_optimizer<M1, typename rhs_opt::type, Tag, right> opt;
	typedef typename opt::type type;
	
	static type create(matrix_matrix_solve<M1,M2, Tag, right> const& m, std::size_t i){
		return opt::create(m.lhs(),rhs_opt::create(m.rhs(),i), m.system_type());
	}
};

//row(inv(A),i) = solve(A,e_1,right) 
template<class M, class Tag>
struct matrix_row_optimizer<matrix_inverse<M, Tag > >{
	typedef unit_vector<typename M::value_type, typename M::device_type> unit;
	typedef matrix_vector_solve_optimizer<M, unit, Tag, right> opt;
	typedef typename opt::type type;
	
	static type create(matrix_inverse<M, Tag > const& m, std::size_t i){
		return opt::create(m.matrix(), unit(m.lhs().size2(),i), m.system_type());
	}
};

//prod(inv(A),B) = solve(A,B,left)
template<class M1, class M2, class Tag>
struct matrix_matrix_prod_optimizer<matrix_inverse<M1,Tag>, M2>{
	typedef matrix_matrix_solve_optimizer<M1,M2, Tag, left> opt;
	typedef typename opt::type type;
	
	static type create(matrix_inverse<M1,Tag> const& inv, typename M2::const_closure_type const& m){
		return opt::create(inv.matrix(),m,inv.system_type());
	}
};

//prod(B,inv(A)) = solve(A,B,right)
template<class M1, class M2, class Tag>
struct matrix_matrix_prod_optimizer<M1, matrix_inverse<M2,Tag> >{
	typedef matrix_matrix_solve_optimizer<M2,M1, Tag, right> opt;
	typedef typename opt::type type;
	
	static type create(typename M1::const_closure_type const& m, matrix_inverse<M2,Tag> const& inv){
		return opt::create(inv.matrix(),m,inv.system_type());
	}
};


}


//solvers for vector rhs
template<class MatA, class VecB, bool Left, class Device, class SystemType>
typename detail::matrix_vector_solve_optimizer<MatA,VecB, SystemType, system_tag<Left> >::type
solve(
	matrix_expression<MatA, Device> const& A,
	vector_expression<VecB, Device> const& b,
	SystemType t, system_tag<Left>
){
	REMORA_SIZE_CHECK(A().size1() ==  A().size2());
	REMORA_SIZE_CHECK(A().size1() ==  b().size());
	typedef detail::matrix_vector_solve_optimizer<MatA,VecB, SystemType, system_tag<Left> > opt;
	return opt::create(A(),b(), t);
}
//solvers for matrix rhs
template<class MatA, class MatB, bool Left, class Device, class SystemType>
typename detail::matrix_matrix_solve_optimizer<MatA,MatB, SystemType, system_tag<Left> >::type
solve(
	matrix_expression<MatA, Device> const& A,
	matrix_expression<MatB, Device> const& B,
	SystemType t, system_tag<Left>
){
	REMORA_SIZE_CHECK(A().size1() ==  A().size2());
	typedef detail::matrix_matrix_solve_optimizer<MatA,MatB, SystemType, system_tag<Left> > opt;
	return opt::create(A(),B(), t);
}

//matrix inverses
template<class MatA, class Device, class SystemType>
typename detail::matrix_inverse_optimizer<MatA, SystemType>::type
inv(
	matrix_expression<MatA, Device> const& A, SystemType t
){
	REMORA_SIZE_CHECK(A().size1() ==  A().size2());
	typedef detail::matrix_inverse_optimizer<MatA,SystemType> opt;
	return opt::create(A(), t);
}


}
#endif