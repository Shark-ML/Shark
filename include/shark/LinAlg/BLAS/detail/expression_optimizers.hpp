/*!
 * \brief       Expression Optimizations
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
 #ifndef REMORA_EXPRESSION_OPTIMIZERS_HPP
#define REMORA_EXPRESSION_OPTIMIZERS_HPP

#include "vector_proxy_classes.hpp"
#include "vector_expression_classes.hpp"
#include "matrix_proxy_classes.hpp"
#include "matrix_expression_classes.hpp"

namespace remora{namespace detail{
	
//forward declarations
template<class V>
struct vector_range_optimizer;
	
template<class M>
struct matrix_transpose_optimizer;
template<class M>
struct matrix_row_optimizer;
template<class M>
struct matrix_range_optimizer;
template<class M, class V>
struct matrix_vector_prod_optimizer;
template<class M1, class M2>
struct matrix_matrix_prod_optimizer;
	
template<class M1, class M2, class Tag, class Side>
struct matrix_matrix_solve_optimizer;
template<class M, class V, class Tag, class Side>
struct matrix_vector_solve_optimizer;
	
template<class M, class Tag>
struct matrix_inverse_optimizer;

////////////////////////////////////
//// Vector Range
////////////////////////////////////
template<class V>
struct vector_range_optimizer{
	typedef vector_range<V> type;
	
	static type create(typename closure<V>::type const& m, std::size_t start, std::size_t end){
		return type(m,start,end);
	}
};

// range(Mv)= rows(M) v
template<class M, class V>
struct vector_range_optimizer<matrix_vector_prod<M,V> >{
	typedef matrix_range_optimizer<typename const_expression<M>::type> left_opt;
	typedef matrix_vector_prod_optimizer<typename left_opt::type,V const> opt;
	typedef typename opt::type type;
	
	static type create(matrix_vector_prod<M,V> const& m, std::size_t start, std::size_t end){
		return opt::create(left_opt::create(m.matrix(), start,end, 0, m.matrix().size2()),m.vector());
	}
};

//range(alpha * v) = alpha * range(v)
template<class V>
struct vector_range_optimizer<vector_scalar_multiply<V> >{
	typedef vector_range_optimizer<typename const_expression<V>::type > opt;
	typedef vector_scalar_multiply<typename opt::type > type;
	
	static type create(vector_scalar_multiply<V> const& v, std::size_t start, std::size_t end){
		return type(opt::create(v.expression(),start,end), v.scalar());
	}
};

//range(v1+v2) = range(v1) + range(v2)
template<class V1, class V2>
struct vector_range_optimizer<vector_addition<V1,V2> >{
	typedef vector_range_optimizer<typename const_expression<V1>::type > left_opt;
	typedef vector_range_optimizer<typename const_expression<V2>::type > right_opt;
	typedef vector_addition<typename left_opt::type, typename right_opt::type > type;
	
	static type create(vector_addition<V1,V2> const& v, std::size_t start, std::size_t end){
		return type(left_opt::create(v.lhs(),start,end),right_opt::create(v.rhs(),start,end));
	}
};

//range(f(v)) = f(range(v))
template<class V, class F>
struct vector_range_optimizer<vector_unary<V, F> >{
	typedef vector_range_optimizer<typename const_expression<V>::type > opt;
	typedef vector_unary<typename opt::type, F > type;
	
	static type create(vector_unary<V, F> const& v, std::size_t start, std::size_t end){
		return type(opt::create(v.expression(),start,end), v.functor());
	}
};

//range(f(v1,v2)) = f(range(v1),range(v2))
template<class V1, class V2, class F>
struct vector_range_optimizer<vector_binary<V1,V2, F> >{
	typedef vector_range_optimizer<typename const_expression<V1>::type > left_opt;
	typedef vector_range_optimizer<typename const_expression<V2>::type > right_opt;
	typedef vector_binary<typename left_opt::type, typename right_opt::type, F > type;
	
	static type create(vector_binary<V1,V2,F> const& v, std::size_t start, std::size_t end){
		return type( left_opt::create(v.lhs(),start,end), right_opt::create(v.rhs(),start,end), v.functor());
	}
};

////////////////////////////////////
//// Matrix Transpose
////////////////////////////////////
	
template<class M>
struct matrix_transpose_optimizer{
	typedef matrix_transpose<M> type;
	
	static type create(typename closure<M>::type const& m){
		return type(m);
	}
};

//(M^T)^T = M
template<class M>
struct matrix_transpose_optimizer<matrix_transpose<M> >{
	typedef typename closure<M>::type type;
	
	static type create(matrix_transpose<M> const& m){
		return m.expression();
	}
};

//(alpha M)^T = alpha M^T
template<class M>
struct matrix_transpose_optimizer<matrix_scalar_multiply<M> >{
	typedef matrix_transpose_optimizer<typename const_expression<M>::type > opt;
	typedef matrix_scalar_multiply<typename opt::type> type;
	
	static type create(matrix_scalar_multiply<M> const& m){
		return type(opt::create(m.expression()), m.scalar());
	}
};

//(M1+M2)^T=M1^T+M2^T
template<class M1, class M2>
struct matrix_transpose_optimizer<matrix_addition<M1,M2> >{
	typedef matrix_transpose_optimizer<typename const_expression<M1>::type > left_opt;
	typedef matrix_transpose_optimizer<typename const_expression<M2>::type > right_opt;
	typedef matrix_addition<typename left_opt::type,typename right_opt::type > type;
	
	static type create(matrix_addition<M1,M2> const& m){
		return type(left_opt::create(m.lhs()),right_opt::create(m.rhs()));
	}
};

//f(M)^T = f(M^T) for f(M)_ij=f(M_ij)
template<class M, class F>
struct matrix_transpose_optimizer<matrix_unary<M,F> >{
	typedef matrix_transpose_optimizer<typename const_expression<M>::type > opt;
	typedef matrix_unary<typename opt::type, F> type;
	
	static type create(matrix_unary<M,F> const& m){
		return type(opt::create(m.expression()),m.functor());
	}
};

//f(M1,M2)^T=f(M1^T,M2^T) for f(M)_ij=f(M_ij)
template<class M1, class M2, class F>
struct matrix_transpose_optimizer<matrix_binary<M1,M2, F> >{
	typedef matrix_transpose_optimizer<typename const_expression<M1>::type > left_opt;
	typedef matrix_transpose_optimizer<typename const_expression<M2>::type > right_opt;
	typedef matrix_binary<typename left_opt::type,typename right_opt::type, F > type;
	
	static type create(matrix_binary<M1,M2,F> const& m){
		return type(left_opt::create(m.lhs()),right_opt::create(m.rhs()),m.functor());
	}
};

//(v1 v2^T)^T = v2 v1^T
template<class V1, class V2>
struct matrix_transpose_optimizer<outer_product<V1,V2> >{
	typedef outer_product<V2,V1> type;
	
	static type create(outer_product<V1,V2> const& m){
		return type(m.rhs(),m.lhs());
	}
};

//vector repeater behaves as outer product to: (v 1^T)^T = (1 v^T)
template<class V, class Orientation>
struct matrix_transpose_optimizer<vector_repeater<V,Orientation> >{
	typedef vector_repeater<V,typename Orientation::transposed_orientation> type;
	
	static type create(vector_repeater<V,Orientation> const& m){
		return type(m.expression(),m.num_repetitions());
	}
};

//(M1 M2)^T = M2^T M1^T
template<class M1, class M2>
struct matrix_transpose_optimizer<matrix_matrix_prod<M1,M2> >{
	typedef matrix_transpose_optimizer<typename const_expression<M2>::type> left_opt;
	typedef matrix_transpose_optimizer<typename const_expression<M1>::type> right_opt;
	typedef matrix_matrix_prod_optimizer<typename left_opt::type,typename right_opt::type> opt;
	typedef typename opt::type type;
	
	static type create(matrix_matrix_prod<M1,M2> const& m){
		return opt::create(left_opt::create(m.rhs()),right_opt::create(m.lhs()));
	}
};

//(A | B)^T= (A^T & B^T)
//(A & B)^T= (A^T | B^T)
template<class M1, class M2, bool B>
struct matrix_transpose_optimizer<matrix_concat<M1,M2,B> >{
	typedef matrix_transpose_optimizer<typename const_expression<M2>::type> right_opt;
	typedef matrix_transpose_optimizer<typename const_expression<M1>::type> left_opt;
	typedef matrix_concat<typename left_opt::type,typename right_opt::type,!B > type;
	
	static type create(matrix_concat<M1,M2,B> const& m){
		return type(left_opt::create(m.lhs()),right_opt::create(m.rhs()));
	}
};

////////////////////////////////////
//// Matrix Row
////////////////////////////////////

template<class M>
struct matrix_row_optimizer{
	typedef matrix_row<M> type;
	
	static type create(typename closure<M>::type const& m, std::size_t i){
		return type(m,i);
	}
};

//row(alpha M,i) = alpha row(M,i)
template<class M>
struct matrix_row_optimizer<matrix_scalar_multiply<M> >{
	typedef matrix_row_optimizer<typename const_expression<M>::type > opt;
	typedef vector_scalar_multiply<typename opt::type> type;
	
	static type create(matrix_scalar_multiply<M> const& m, std::size_t i){
		return type(opt::create(m.expression(),i), m.scalar());
	}
};

// row(M1+M2,i) = row(M1,i) + row(M2,i)
template<class M1, class M2>
struct matrix_row_optimizer<matrix_addition<M1,M2> >{
	typedef matrix_row_optimizer<typename const_expression<M1>::type > left_opt;
	typedef matrix_row_optimizer<typename const_expression<M2>::type > right_opt;
	typedef vector_addition<typename left_opt::type,typename right_opt::type > type;
	
	static type create(matrix_addition<M1,M2> const& m, std::size_t i){
		return type(left_opt::create(m.lhs(),i),right_opt::create(m.rhs(),i));
	}
};

//row(f(M),i) = f(row(M,i))
template<class M, class F>
struct matrix_row_optimizer<matrix_unary<M,F> >{
	typedef matrix_row_optimizer<typename const_expression<M>::type > opt;
	typedef vector_unary<typename opt::type, F> type;
	
	static type create(matrix_unary<M,F> const& m, std::size_t i){
		return type(opt::create(m.expression(),i),m.functor());
	}
};

//row(f(M1,M2),i)=f(row(M1,i),row(M2,i))
template<class M1, class M2, class F>
struct matrix_row_optimizer<matrix_binary<M1,M2, F> >{
	typedef matrix_row_optimizer<typename const_expression<M1>::type > left_opt;
	typedef matrix_row_optimizer<typename const_expression<M2>::type > right_opt;
	typedef vector_binary<typename left_opt::type,typename right_opt::type, F > type;
	
	static type create(matrix_binary<M1,M2,F> const& m, std::size_t i){
		return type(left_opt::create(m.lhs(),i),right_opt::create(m.rhs(),i),m.functor());
	}
};

//row(prod(A,B),i) = prod(row(A),B) = prod(trans(B),row(A)) 
template<class M1, class M2>
struct matrix_row_optimizer<matrix_matrix_prod<M1,M2> >{
	typedef matrix_row_optimizer<typename const_expression<M1>::type> left_opt;
	typedef matrix_transpose_optimizer<typename const_expression<M2>::type> right_opt;
	typedef matrix_vector_prod_optimizer<typename right_opt::type, typename left_opt::type> opt;
	typedef typename opt::type type;
	
	static type create(matrix_matrix_prod<M1,M2> const& m, std::size_t i){
		return opt::create(
			right_opt::create(m.rhs()),
			left_opt::create(m.lhs(),i)
		);
	}
};

//row(v1 v2^T,i)^T = v(i) v2 
template<class V1, class V2>
struct matrix_row_optimizer<outer_product<V1,V2> >{
	typedef vector_scalar_multiply<V2> type;
	
	static type create(outer_product<V1,V2> const& m, std::size_t i){
		return type(m.rhs(),m.lhs()(i));
	}
};


//row(repeat(v),i) = v if repeat is row_major
template<class V>
struct matrix_row_optimizer<vector_repeater<V, row_major> >{
	typedef typename V::const_closure_type type;
	
	static type create(vector_repeater<V, row_major> const& m, std::size_t){
		return m.expression();
	}
};
//row(repeat(v),i) = v(i) 1^T if repeat is column_major
template<class V>
struct matrix_row_optimizer<vector_repeater<V, column_major> >{
	typedef scalar_vector<typename V::value_type, typename V::device_type> type;
	
	static type create(vector_repeater<V, column_major> const& m, std::size_t i){
		return type(m.num_repetitions(), m.expression()(i));
	}
};


////////////////////////////////////
//// Matrix Range
////////////////////////////////////
template<class M>
struct matrix_range_optimizer{
	typedef matrix_range<M> type;
	
	static type create(typename closure<M>::type const& m,
		std::size_t start1, std::size_t end1, std::size_t start2, std::size_t end2 ){
		return type(m,start1, end1, start2, end2);
	}
};

//range(alpha * M) = alpha * range(M)
template<class M>
struct matrix_range_optimizer<matrix_scalar_multiply<M> >{
	typedef matrix_range_optimizer<typename const_expression<M>::type > opt;
	typedef matrix_scalar_multiply<typename opt::type > type;
	
	static type create(matrix_scalar_multiply<M> const& m,
		std::size_t start1, std::size_t end1, std::size_t start2, std::size_t end2
	){
		return type(opt::create(m.expression(),start1,end1,start2,end2), m.scalar());
	}
};

//range(M1+M2) = range(M1) + range(M2)
template<class M1, class M2>
struct matrix_range_optimizer<matrix_addition<M1,M2> >{
	typedef matrix_range_optimizer<typename const_expression<M1>::type > left_opt;
	typedef matrix_range_optimizer<typename const_expression<M2>::type > right_opt;
	typedef matrix_addition<typename left_opt::type, typename right_opt::type > type;
	
	static type create(matrix_addition<M1,M2> const& m,
		std::size_t start1, std::size_t end1, std::size_t start2, std::size_t end2
	){
		return type(
			left_opt::create(m.lhs(),start1,end1,start2,end2),
			right_opt::create(m.rhs(),start1,end1,start2,end2)
		);
	}
};

//range(f(M)) = f(range(M))
template<class M, class F>
struct matrix_range_optimizer<matrix_unary<M, F> >{
	typedef matrix_range_optimizer<typename const_expression<M>::type > opt;
	typedef matrix_unary<typename opt::type, F > type;
	
	static type create(matrix_unary<M, F> const& m,
		std::size_t start1, std::size_t end1, std::size_t start2, std::size_t end2
	){
		return type(opt::create(m.expression(),start1,end1,start2,end2), m.functor());
	}
};

//range(f(M1,M2)) = f(range(M1),range(M2))
template<class M1, class M2, class F>
struct matrix_range_optimizer<matrix_binary<M1,M2, F> >{
	typedef matrix_range_optimizer<typename const_expression<M1>::type > left_opt;
	typedef matrix_range_optimizer<typename const_expression<M2>::type > right_opt;
	typedef matrix_binary<typename left_opt::type, typename right_opt::type, F > type;
	
	static type create(matrix_binary<M1,M2,F> const& m,
		std::size_t start1, std::size_t end1, std::size_t start2, std::size_t end2
	){
		return type(
			left_opt::create(m.lhs(),start1,end1,start2,end2),
			right_opt::create(m.rhs(),start1,end1,start2,end2),
			m.functor()
		);
	}
};

//range(uv^T) = range(u) range(v)^T
template<class V1, class V2>
struct matrix_range_optimizer<outer_product<V1,V2> >{
	typedef vector_range_optimizer<typename const_expression<V1>::type > left_opt;
	typedef vector_range_optimizer<typename const_expression<V2>::type > right_opt;
	typedef outer_product<typename left_opt::type, typename right_opt::type> type;
	
	static type create(outer_product<V1,V2> const& m,
		std::size_t start1, std::size_t end1, std::size_t start2, std::size_t end2
	){
		return type( left_opt::create(m.lhs(),start1,end1), right_opt::create(m.rhs(),start2,end2));
	}
};

//repeater behaves like outer_product
template<class V, class Orientation>
struct matrix_range_optimizer<vector_repeater<V, Orientation> >{
	typedef vector_range_optimizer<typename const_expression<V>::type > vector_opt;
	typedef vector_repeater<typename vector_opt::type, Orientation> type;
	
	static type create(
		vector_repeater<V, Orientation> const& m,
		std::size_t start1, std::size_t end1, std::size_t start2, std::size_t end2
	){
		return type( 
			vector_opt::create(m.expression(),Orientation::index_m(start1,start2),Orientation::index_m(end1,end2)), 
			Orientation::index_M(end1,end2) - Orientation::index_M(start1,start2)
		);
	}
};

//range(prod(A,B),i) = prod(range(B),range(A)) 
template<class M1, class M2>
struct matrix_range_optimizer<matrix_matrix_prod<M1,M2> >{
	typedef matrix_range_optimizer<typename const_expression<M1>::type> left_opt;
	typedef matrix_range_optimizer<typename const_expression<M2>::type> right_opt;
	typedef matrix_matrix_prod_optimizer<typename left_opt::type, typename right_opt::type> opt;
	typedef typename opt::type type;
	
	static type create(matrix_matrix_prod<M1,M2> const& m,
		std::size_t start1, std::size_t end1, std::size_t start2, std::size_t end2
	){
		return opt::create(
			left_opt::create(m.lhs(),start1,end1,0,m.lhs().size2()),
			right_opt::create(m.rhs(),0,m.rhs().size1(),start2,end2)
		);
	}
};

////////////////////////////////////
//// Matrix Vector Product
////////////////////////////////////
	
//matrix-vector multiplications
template<class M, class V>
struct matrix_vector_prod_optimizer{
	typedef matrix_vector_prod<M,V> type;
	
	static type create(typename M::const_closure_type const& m, typename V::const_closure_type const& v){
		return type(m,v);
	}
};

//(alpha M)*v = alpha (M*v)
template<class M, class V>
struct matrix_vector_prod_optimizer<matrix_scalar_multiply<M>,V >{
	typedef matrix_vector_prod_optimizer<M, V> opt;
	typedef vector_scalar_multiply<typename opt::type> type;
	
	static type create(matrix_scalar_multiply<M> const& m, typename V::const_closure_type const& v){
		return type(opt::create(m.expression(),v), m.scalar());
	}
};

//M*(alpha*v) = alpha (M*v)
template<class M, class V>
struct matrix_vector_prod_optimizer<M,vector_scalar_multiply<V> >{
	typedef matrix_vector_prod_optimizer<M, V> opt;
	typedef vector_scalar_multiply<typename opt::type> type;
	
	static type create(typename M::const_closure_type const& m, vector_scalar_multiply<V> const& v){
		return type(opt::create(m,v.expression()), v.scalar());
	}
};

//(alpha M)*(beta*v) = (alpha*beta) (M*v)
template<class M, class V>
struct matrix_vector_prod_optimizer<matrix_scalar_multiply<M>,vector_scalar_multiply<V> >{
	typedef matrix_vector_prod_optimizer<M, V> opt;
	typedef vector_scalar_multiply<typename opt::type> type;
	
	static type create(matrix_scalar_multiply<M> const& m, vector_scalar_multiply<V>const& v){
		return type(opt::create(m.expression(),v.expression()), v.scalar()*m.scalar());
	}
};

//(M1*M2)*V=M1*(M2*V)
template<class M1,class M2, class V>
struct matrix_vector_prod_optimizer<matrix_matrix_prod<M1,M2>,V>{
private:
	typedef matrix_vector_prod_optimizer<M2,V> inner_opt;
	typedef matrix_vector_prod_optimizer<M1, typename inner_opt::type> outer_opt;
public:
	typedef typename outer_opt::type type;
	
	static type create(matrix_matrix_prod<M1,M2> const& m, typename V::const_closure_type const& v){
		auto inner_result = inner_opt::create(m.rhs(),v);
		return outer_opt::create(m.lhs(),inner_result);
	}
};

//(M1+M2)*V=M1*V+M2*V
template<class M1,class M2, class V>
struct matrix_vector_prod_optimizer<matrix_addition<M1,M2>,V>{
private:
	typedef matrix_vector_prod_optimizer<M1,V> left_opt;
	typedef matrix_vector_prod_optimizer<M2,V> right_opt;
public:
	typedef vector_addition<typename left_opt::type ,typename right_opt::type> type;
	
	static type create(matrix_addition<M1,M2> const& m, typename V::const_closure_type const& v){
		auto lhs = left_opt::create(m.lhs(),v);
		auto rhs = right_opt::create(m.rhs(),v);
		return type(lhs,rhs);
	}
};

//(v1*v2^T)*v3= v1*(v2^T*v3)
template<class V1,class V2, class V3>
struct matrix_vector_prod_optimizer<outer_product<V1,V2>,V3>{
	typedef vector_scalar_multiply<V1> type;
	
	static type create(outer_product<V1,V2> const& m, typename V3::const_closure_type const& v){
		auto alpha = inner_prod(m.rhs(),v);
		return type(m.lhs(),alpha);
	}
};

template<class V1, class V2>
struct matrix_vector_prod_optimizer<vector_repeater<V1, row_major>, V2 >{
	typedef scalar_vector<typename common_value_type<V1,V2>::type, typename V1::device_type> type;
	
	static type create(vector_repeater<V1, row_major> const& m, typename V2::const_closure_type const& v){
		auto alpha = inner_prod(m.expression(),v);
		return type(alpha,m.num_repetitions());
	}
};

template<class V1, class V2>
struct matrix_vector_prod_optimizer<vector_repeater<V1, column_major>, V2 >{
	typedef vector_scalar_multiply<V1> type;
	
	static type create(vector_repeater<V1, row_major> const& m, typename V2::const_closure_type const& v){
		auto alpha = sum(m.expression(),v);
		return type(m.expression(),alpha);
	}
};



//(diag(v1) * v2) = v1 .* v2
template<class V1,class V2>
struct matrix_vector_prod_optimizer<diagonal_matrix<V1>,V2>{
	typedef typename common_value_type<V1,V2>::type value_type;
	typedef typename device_traits<typename V1::device_type>:: template multiply<value_type> functor;
	typedef vector_binary<V1, V2, functor  > type;
	static type create(diagonal_matrix<V1> const& m, typename V2::const_closure_type const& v){
		return type(m.expression(),v, functor());
	}
};


////////////////////////////////////
//// Matrix Product
////////////////////////////////////

template<class M1, class M2>
struct matrix_matrix_prod_optimizer{
	typedef matrix_matrix_prod<M1,M2> type;
	
	static type create(typename M1::const_closure_type const& lhs, typename M2::const_closure_type const& rhs){
		return type(lhs,rhs);
	}
};


//(alpha M1)*B = alpha (M1*B)
template<class M1, class M2>
struct matrix_matrix_prod_optimizer<matrix_scalar_multiply<M1>,M2 >{
	typedef matrix_matrix_prod_optimizer<M1, M2> opt;
	typedef matrix_scalar_multiply<typename opt::type> type;
	
	static type create(matrix_scalar_multiply<M1> const& A, typename M2::const_closure_type const& B){
		return type(opt::create(A.expression(),B), A.scalar());
	}
};

//M1*(alpha*B) = alpha (M1*B)
template<class M1, class M2>
struct matrix_matrix_prod_optimizer<M1,matrix_scalar_multiply<M2> >{
	typedef matrix_matrix_prod_optimizer<M1, M2> opt;
	typedef matrix_scalar_multiply<typename opt::type> type;
	
	static type create(typename M1::const_closure_type const& A, matrix_scalar_multiply<M2> const& B){
		return type(opt::create(A,B.expression()), B.scalar());
	}
};

//(alpha M1)*(beta*B) = (alpha*beta) (M1*B)
template<class M1, class M2>
struct matrix_matrix_prod_optimizer<matrix_scalar_multiply<M1>,matrix_scalar_multiply<M2> >{
	typedef matrix_matrix_prod_optimizer<M1, M2> opt;
	typedef matrix_scalar_multiply<typename opt::type> type;
	
	static type create(matrix_scalar_multiply<M1> const& A, matrix_scalar_multiply<M2>const& B){
		return type(opt::create(A.expression(),B.expression()), B.scalar()*A.scalar());
	}
};

}}
#endif
