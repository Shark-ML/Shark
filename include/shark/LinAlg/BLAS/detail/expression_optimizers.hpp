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

#include "proxy_optimizers_fwd.hpp"
#include "vector_expression_classes.hpp"
#include "matrix_expression_classes.hpp"

namespace remora{namespace detail{
	
//forward declarations
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

template<class M,  class F>
struct matrix_unary_optimizer;

template<class M,  class F>
struct vector_unary_optimizer;

////////////////////////////////////
//// Vector Range
////////////////////////////////////

// range(Mv)= rows(M) v
template<class M, class V>
struct vector_range_optimizer<matrix_vector_prod<M,V> >{
	typedef matrix_range_optimizer<typename M::const_closure_type> left_opt;
	typedef matrix_vector_prod_optimizer<typename left_opt::type,V const> opt;
	typedef typename opt::type type;
	
	static type create(matrix_vector_prod<M,V> const& m, std::size_t start, std::size_t end){
		return opt::create(left_opt::create(m.matrix(), start,end, 0, m.matrix().size2()),m.vector());
	}
};

// range(fold_rows(M, f, g),start,end) = fold_rows(columns(m,start,end), f, g)
template<class M, class F, class G>
struct vector_range_optimizer<matrix_row_transform<M, F, G> >{
	typedef matrix_range_optimizer<typename M::const_closure_type> mat_opt;
	typedef matrix_row_transform<typename mat_opt::type, F, G> type;
	
	static type create(matrix_row_transform<M, F, G> const& m, std::size_t start, std::size_t end){
		return type(mat_opt::create(m.expression(),0, m.expression().size1(), start, end), m.f(), m.g());
	}
};

//range(alpha * v) = alpha * range(v)
template<class V>
struct vector_range_optimizer<vector_scalar_multiply<V> >{
	typedef vector_range_optimizer<typename V::const_closure_type > opt;
	typedef vector_scalar_multiply<typename opt::type > type;
	
	static type create(vector_scalar_multiply<V> const& v, std::size_t start, std::size_t end){
		return type(opt::create(v.expression(),start,end), v.scalar());
	}
};

//range(constant)  -> constant (different size)
template<class T, class Device>
struct vector_range_optimizer<scalar_vector<T,Device> >{
	typedef scalar_vector<T,Device> type;
	static type create(type const& m, std::size_t start, std::size_t end){
		return type(end-start, m.scalar());
	}
};

//range(unit)  -> unit (different size, nonzero possibly outside)
template<class T, class Device>
struct vector_range_optimizer<unit_vector<T,Device> >{
	typedef unit_vector<T,Device> type;
	static type create(type const& m, std::size_t start, std::size_t end){
		return type(end-start, m.index() - start, m.scalar());
	}
};

//range(f(v)) = f(range(v))
template<class V, class F>
struct vector_range_optimizer<vector_unary<V, F> >{
	typedef vector_range_optimizer<typename V::const_closure_type> opt;
	typedef vector_unary<typename opt::type, F > type;
	
	static type create(vector_unary<V, F> const& v, std::size_t start, std::size_t end){
		return type(opt::create(v.expression(),start,end), v.functor());
	}
};

//range(v1+v2) = range(v1) + range(v2)
template<class V1, class V2>
struct vector_range_optimizer<vector_addition<V1,V2> >{
	typedef vector_range_optimizer<typename V1::const_closure_type > left_opt;
	typedef vector_range_optimizer<typename V2::const_closure_type> right_opt;
	typedef vector_addition<typename left_opt::type, typename right_opt::type > type;
	
	static type create(vector_addition<V1,V2> const& v, std::size_t start, std::size_t end){
		return type(left_opt::create(v.lhs(),start,end),right_opt::create(v.rhs(),start,end));
	}
};

//range(f(v1,v2)) = f(range(v1),range(v2))
template<class V1, class V2, class F>
struct vector_range_optimizer<vector_binary<V1,V2, F> >{
	typedef vector_range_optimizer<typename V1::const_closure_type > left_opt;
	typedef vector_range_optimizer<typename V2::const_closure_type > right_opt;
	typedef vector_binary<typename left_opt::type, typename right_opt::type, F > type;
	
	static type create(vector_binary<V1,V2,F> const& v, std::size_t start, std::size_t end){
		return type( left_opt::create(v.lhs(),start,end), right_opt::create(v.rhs(),start,end), v.functor());
	}
};

////////////////////////////////////
//// Matrix Transpose
////////////////////////////////////

//(alpha M)^T = alpha M^T
template<class M>
struct matrix_transpose_optimizer<matrix_scalar_multiply<M> >{
	typedef matrix_transpose_optimizer<typename M::const_closure_type> opt;
	typedef matrix_scalar_multiply<typename opt::type> type;
	
	static type create(matrix_scalar_multiply<M> const& m){
		return type(opt::create(m.expression()), m.scalar());
	}
};

//(M1+M2)^T=M1^T+M2^T
template<class M1, class M2>
struct matrix_transpose_optimizer<matrix_addition<M1,M2> >{
	typedef matrix_transpose_optimizer<typename M1::const_closure_type > left_opt;
	typedef matrix_transpose_optimizer<typename M2::const_closure_type > right_opt;
	typedef matrix_addition<typename left_opt::type,typename right_opt::type > type;
	
	static type create(matrix_addition<M1,M2> const& m){
		return type(left_opt::create(m.lhs()),right_opt::create(m.rhs()));
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

//trans(constant)  -> constant (swapped sizes)
template<class T, class Device, class Orientation>
struct matrix_transpose_optimizer<scalar_matrix<T,Device, Orientation> >{
	typedef scalar_matrix<T,Device, typename Orientation::transposed_orientation> type;
	static type create(scalar_matrix<T,Device, Orientation> const& m){
		return type(m.size2(), m.size1(), m.scalar());
	}
};

//f(M)^T = f(M^T) for f(M)_ij=f(M_ij)
template<class M, class F>
struct matrix_transpose_optimizer<matrix_unary<M,F> >{
	typedef matrix_transpose_optimizer<typename M::const_closure_type> opt;
	typedef matrix_unary<typename opt::type, F> type;
	
	static type create(matrix_unary<M,F> const& m){
		return type(opt::create(m.expression()),m.functor());
	}
};

//f(M1,M2)^T=f(M1^T,M2^T) for f(M)_ij=f(M_ij)
template<class M1, class M2, class F>
struct matrix_transpose_optimizer<matrix_binary<M1,M2, F> >{
	typedef matrix_transpose_optimizer<typename M1::const_closure_type> left_opt;
	typedef matrix_transpose_optimizer<typename M2::const_closure_type> right_opt;
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

//(M1 M2)^T = M2^T M1^T
template<class M1, class M2>
struct matrix_transpose_optimizer<matrix_matrix_prod<M1,M2> >{
	typedef matrix_transpose_optimizer<typename M2::const_closure_type> left_opt;
	typedef matrix_transpose_optimizer<typename M1::const_closure_type> right_opt;
	typedef matrix_matrix_prod_optimizer<typename left_opt::type,typename right_opt::type> opt;
	typedef typename opt::type type;
	
	static type create(matrix_matrix_prod<M1,M2> const& m){
		return opt::create(left_opt::create(m.rhs()),right_opt::create(m.lhs()));
	}
};

//trans(diagonal)  =  diagonal
template<class V>
struct matrix_transpose_optimizer<diagonal_matrix<V> >{
	typedef diagonal_matrix<V> type;
	static type const& create(type const& m){
		return m;
	}
};

//(A | B)^T= (A^T & B^T)
//(A & B)^T= (A^T | B^T)
template<class M1, class M2, bool B>
struct matrix_transpose_optimizer<matrix_concat<M1,M2,B> >{
	typedef matrix_transpose_optimizer<typename M2::const_closure_type> right_opt;
	typedef matrix_transpose_optimizer<typename M1::const_closure_type> left_opt;
	typedef matrix_concat<typename left_opt::type,typename right_opt::type,!B > type;
	
	static type create(matrix_concat<M1,M2,B> const& m){
		return type(left_opt::create(m.lhs()),right_opt::create(m.rhs()));
	}
};

////////////////////////////////////
//// Matrix Row
////////////////////////////////////

//row(alpha M,i) = alpha row(M,i)
template<class M>
struct matrix_row_optimizer<matrix_scalar_multiply<M> >{
	typedef matrix_row_optimizer<typename M::const_closure_type > opt;
	typedef vector_scalar_multiply<typename opt::type> type;
	
	static type create(matrix_scalar_multiply<M> const& m, std::size_t i){
		return type(opt::create(m.expression(),i), m.scalar());
	}
};

// row(M1+M2,i) = row(M1,i) + row(M2,i)
template<class M1, class M2>
struct matrix_row_optimizer<matrix_addition<M1,M2> >{
	typedef matrix_row_optimizer<typename M1::const_closure_type > left_opt;
	typedef matrix_row_optimizer<typename M2::const_closure_type > right_opt;
	typedef vector_addition<typename left_opt::type,typename right_opt::type > type;
	
	static type create(matrix_addition<M1,M2> const& m, std::size_t i){
		return type(left_opt::create(m.lhs(),i),right_opt::create(m.rhs(),i));
	}
};

//row(constant,i) = constant
template<class T, class Device, class Orientation>
struct matrix_row_optimizer<scalar_matrix<T, Device, Orientation> >{
	typedef scalar_vector<T, Device> type;
	
	static type create(scalar_matrix<T, Device, Orientation> const& m, std::size_t){
		return type(m.size2(),m.scalar());
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

//row(f(M),i) = f(row(M,i))
template<class M, class F>
struct matrix_row_optimizer<matrix_unary<M,F> >{
	typedef matrix_row_optimizer<typename M::const_closure_type > opt;
	typedef vector_unary<typename opt::type, F> type;
	
	static type create(matrix_unary<M,F> const& m, std::size_t i){
		return type(opt::create(m.expression(),i),m.functor());
	}
};

//row(f(M1,M2),i)=f(row(M1,i),row(M2,i))
template<class M1, class M2, class F>
struct matrix_row_optimizer<matrix_binary<M1,M2, F> >{
	typedef matrix_row_optimizer<typename M1::const_closure_type > left_opt;
	typedef matrix_row_optimizer<typename M2::const_closure_type > right_opt;
	typedef vector_binary<typename left_opt::type,typename right_opt::type, F > type;
	
	static type create(matrix_binary<M1,M2,F> const& m, std::size_t i){
		return type(left_opt::create(m.lhs(),i),right_opt::create(m.rhs(),i),m.functor());
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

//row(prod(A,B),i) = prod(row(A),B) = prod(trans(B),row(A)) 
template<class M1, class M2>
struct matrix_row_optimizer<matrix_matrix_prod<M1,M2> >{
	typedef matrix_row_optimizer<typename M1::const_closure_type> left_opt;
	typedef matrix_transpose_optimizer<typename M2::const_closure_type> right_opt;
	typedef matrix_vector_prod_optimizer<typename right_opt::type, typename left_opt::type> opt;
	typedef typename opt::type type;
	
	static type create(matrix_matrix_prod<M1,M2> const& m, std::size_t i){
		return opt::create(
			right_opt::create(m.rhs()),
			left_opt::create(m.lhs(),i)
		);
	}
};

//row(diagonal(V),i)  =  (0,,...,0,v_i,0,...,0)
template<class V>
struct matrix_row_optimizer<diagonal_matrix<V> >{
	typedef unit_vector<typename V::value_type, typename V::device_type> type;
	
	static type create(diagonal_matrix<V> const& m, std::size_t i){
		return type(m.size2(),i,m.expression()(i));
	}
};


////////////////////////////////////
//// Matrix-Vector Range
////////////////////////////////////

//range(alpha * M) = alpha * diag(M)
template<class M>
struct matrix_diagonal_optimizer<matrix_scalar_multiply<M> >{
	typedef matrix_diagonal_optimizer<typename M::const_closure_type > opt;
	typedef vector_scalar_multiply<typename opt::type > type;
	
	static type create(matrix_scalar_multiply<M> const& m){
		return type(opt::create(m.expression()), m.scalar());
	}
};

//diag(M1+M2) = diag(M1) + diag(M2)
template<class M1, class M2>
struct matrix_diagonal_optimizer<matrix_addition<M1,M2> >{
	typedef matrix_diagonal_optimizer<typename M1::const_closure_type > left_opt;
	typedef matrix_diagonal_optimizer<typename M2::const_closure_type > right_opt;
	typedef vector_addition<typename left_opt::type, typename right_opt::type > type;
	
	static type create(matrix_addition<M1,M2> const& m){
		return type(left_opt::create(m.lhs()),right_opt::create(m.rhs()));
	}
};

//diag(constant)  -> constant (vector)
template<class T, class Device, class Orientation>
struct matrix_diagonal_optimizer<scalar_matrix<T,Device, Orientation> >{
	typedef scalar_vector<T,Device> type;
	
	static type create(scalar_matrix<T,Device, Orientation> const& m){
		return type(m().size(), m.scalar());
	}
};

//diag(repeat(v,j)) -> range(v,0,min(v.size,j))
template<class V, class Orientation>
struct matrix_diagonal_optimizer<vector_repeater<V, Orientation> >{
	typedef vector_range_optimizer<typename V::const_closure_type > opt;
	typedef typename opt::type type;
	
	static type create(vector_repeater<V, Orientation> const& m){
		return opt::create(m.expression(),0, std::min(m.size1(),m.size2())); 

	}
};

// diag(f(M)) -> f(diag(M))
template<class M, class F>
struct matrix_diagonal_optimizer<matrix_unary<M, F> >{
	typedef matrix_diagonal_optimizer<typename M::const_closure_type > opt;
	typedef vector_unary<typename opt::type, F > type;
	
	static type create(matrix_unary<M, F> const& m){
		return type(opt::create(m.expression()), m.functor());
	}
};
// diag(f(M,M2)) -> f(diag(M1),diag(M2))
template<class M1, class M2, class F>
struct matrix_diagonal_optimizer<matrix_binary<M1,M2, F> >{
	typedef matrix_diagonal_optimizer<typename M1::const_closure_type > left_opt;
	typedef matrix_diagonal_optimizer<typename M2::const_closure_type > right_opt;
	typedef vector_binary<typename left_opt::type, typename right_opt::type, F > type;
	
	static type create(matrix_binary<M1,M2,F> const& m){
		return type(left_opt::create(m.lhs()),right_opt::create(m.rhs()),m.functor());
	}
};

//diag( u v^T) -> range(u,size) range(v,size)^T, where size=min(u.size,v.size)
template<class V1, class V2>
struct matrix_diagonal_optimizer<outer_product<V1,V2> >{
	typedef vector_range_optimizer<typename V1::const_closure_type > left_opt;
	typedef vector_range_optimizer<typename V2::const_closure_type> right_opt;
	typedef typename common_value_type<V1,V2>::type value_type;
	typedef typename device_traits<typename V1::device_type>:: template multiply<value_type> functor;
	typedef vector_binary<typename left_opt::type, typename right_opt::type, functor> type;
	
	static type create(outer_product<V1,V2> const& m){
		auto size = std::min(m.size1(),m.size2());
		return type( left_opt::create(m.lhs(),0,size), right_opt::create(m.rhs(),0,size), functor());
	}
};

//diag(prod(A,B))_i = dot(row(B,i), column(A,i)) 
//~ template<class M1, class M2>
//~ struct matrix_diagonal_optimizer<matrix_matrix_prod<M1,M2> >{
	//~ static_assert(false, "diagonal of matrix multiplication not implemented yet");
//~ };


//diag(diagonal(v))  -> v
template<class V>
struct matrix_diagonal_optimizer<diagonal_matrix<V> >{
	typedef typename V::const_closure_type type;
	static type create(diagonal_matrix<V> const& m){
		return m.expression();
	}
};

////////////////////////////////////
//// Matrix Range
////////////////////////////////////

//range(alpha * M) = alpha * range(M)
template<class M>
struct matrix_range_optimizer<matrix_scalar_multiply<M> >{
	typedef matrix_range_optimizer<typename M::const_closure_type > opt;
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
	typedef matrix_range_optimizer<typename M1::const_closure_type > left_opt;
	typedef matrix_range_optimizer<typename M2::const_closure_type > right_opt;
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

//range(constant)  -> constant (changed sizes)
template<class T, class Device, class Orientation>
struct matrix_range_optimizer<scalar_matrix<T,Device, Orientation> >{
	typedef scalar_matrix<T,Device, Orientation> type;
	static type create(type const& m,
		std::size_t start1, std::size_t end1, std::size_t start2, std::size_t end2
	){
		return type(end1 - start1, end2 - start2, m.scalar());
	}
};

//repeater behaves like outer_product
template<class V, class Orientation>
struct matrix_range_optimizer<vector_repeater<V, Orientation> >{
	typedef vector_range_optimizer<typename V::const_closure_type > vector_opt;
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

//range(f(M)) = f(range(M))
template<class M, class F>
struct matrix_range_optimizer<matrix_unary<M, F> >{
	typedef matrix_range_optimizer<typename M::const_closure_type > opt;
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
	typedef matrix_range_optimizer<typename M1::const_closure_type > left_opt;
	typedef matrix_range_optimizer<typename M2::const_closure_type > right_opt;
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

//range( u v^T) = range(u) range(v)^T
template<class V1, class V2>
struct matrix_range_optimizer<outer_product<V1,V2> >{
	typedef vector_range_optimizer<typename V1::const_closure_type > left_opt;
	typedef vector_range_optimizer<typename V2::const_closure_type> right_opt;
	typedef outer_product<typename left_opt::type, typename right_opt::type> type;
	
	static type create(outer_product<V1,V2> const& m,
		std::size_t start1, std::size_t end1, std::size_t start2, std::size_t end2
	){
		return type( left_opt::create(m.lhs(),start1,end1), right_opt::create(m.rhs(),start2,end2));
	}
};

//range(prod(A,B),i) = prod(range(B),range(A)) 
template<class M1, class M2>
struct matrix_range_optimizer<matrix_matrix_prod<M1,M2> >{
	typedef matrix_range_optimizer<typename M1::const_closure_type> left_opt;
	typedef matrix_range_optimizer<typename M2::const_closure_type> right_opt;
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


//range(diagonal  -> diagonal padded with 0
template<class V>
struct matrix_range_optimizer<diagonal_matrix<V> >{
    typedef vector_range_optimizer<typename V::const_closure_type > opt;
	typedef diagonal_matrix<typename opt::type> type;
	static type create(diagonal_matrix<V> const& m,
		std::size_t start1, std::size_t end1, std::size_t start2, std::size_t end2
	){
        REMORA_RANGE_CHECK(start1 == start2);// "unimplemented: non-diagonal subranges of diagonal matrix"
        REMORA_RANGE_CHECK(end1 == end2); //"unimplemented: non-diagonal subranges of diagonal matrix"
        std::size_t startV = std::max(start1,start2);
        std::size_t endV = std::min(end1,end2);
		return type(opt::create(m.expression(),startV, endV));
	}
};


///////////////////////////////////
//// Matrix Rows
////////////////////////////////////

//rows(alpha * M) = alpha * rows(M)
template<class M>
struct matrix_rows_optimizer<matrix_scalar_multiply<M> >{
	typedef matrix_rows_optimizer<typename M::const_closure_type > opt;
	typedef matrix_scalar_multiply<typename opt::type > type;
	
	static type create(matrix_scalar_multiply<M> const& m,
		std::size_t start, std::size_t end
	){
		return type(opt::create(m.expression(),start,end), m.scalar());
	}
};

//range(M1+M2) = range(M1) + range(M2)
template<class M1, class M2>
struct matrix_rows_optimizer<matrix_addition<M1,M2> >{
	typedef matrix_rows_optimizer<typename M1::const_closure_type > left_opt;
	typedef matrix_rows_optimizer<typename M2::const_closure_type > right_opt;
	typedef matrix_addition<typename left_opt::type, typename right_opt::type > type;
	
	static type create(matrix_addition<M1,M2> const& m,std::size_t start, std::size_t end){
		return type(left_opt::create(m.lhs(),start,end),right_opt::create(m.rhs(),start,end));
	}
};

//range(constant)  -> constant (changed sizes)
template<class T, class Device, class Orientation>
struct matrix_rows_optimizer<scalar_matrix<T,Device, Orientation> >{
	typedef scalar_matrix<T,Device, Orientation> type;
	static type create(type const& m,std::size_t start, std::size_t end){
		return type(end - start, m.size2(), m.scalar());
	}
};

//repeater behaves like outer_product
template<class V>
struct matrix_rows_optimizer<vector_repeater<V, column_major> >{
	typedef vector_range_optimizer<typename V::const_closure_type> vector_opt;
	typedef vector_repeater<typename vector_opt::type, row_major> type;
	
	static type create(vector_repeater<V, column_major> const& m,std::size_t start, std::size_t end){
		return type( vector_opt::create(m.expression(),start,end), m.size2());
	}
};
template<class V>
struct matrix_rows_optimizer<vector_repeater<V, row_major> >{
	typedef vector_repeater<V, row_major> type;
	
	static type create(type const& m,std::size_t start, std::size_t end){
		return type( m.expression(), end - start);
	}
};

//rows(f(M)) = f(rows(M))
template<class M, class F>
struct matrix_rows_optimizer<matrix_unary<M, F> >{
	typedef matrix_rows_optimizer<typename M::const_closure_type > opt;
	typedef matrix_unary<typename opt::type, F > type;
	
	static type create(matrix_unary<M, F> const& m, std::size_t start, std::size_t end){
		return type(opt::create(m.expression(),start,end), m.functor());
	}
};

//rows(f(M1,M2)) = f(rows(M1),rows(M2))
template<class M1, class M2, class F>
struct matrix_rows_optimizer<matrix_binary<M1,M2, F> >{
	typedef matrix_rows_optimizer<typename M1::const_closure_type > left_opt;
	typedef matrix_rows_optimizer<typename M2::const_closure_type > right_opt;
	typedef matrix_binary<typename left_opt::type, typename right_opt::type, F > type;
	
	static type create(matrix_binary<M1,M2,F> const& m,std::size_t start, std::size_t end){
		return type(left_opt::create(m.lhs(),start,end),right_opt::create(m.rhs(),start,end),m.functor());
	}
};

//rows( u v^T) = range(u) v^T
template<class V1, class V2>
struct matrix_rows_optimizer<outer_product<V1,V2> >{
	typedef vector_range_optimizer<typename V1::const_closure_type > left_opt;
	typedef outer_product<typename left_opt::type, V2> type;
	
	static type create(outer_product<V1,V2> const& m,std::size_t start, std::size_t end){
		return type( left_opt::create(m.lhs(),start,end), m.rhs());
	}
};

//rows(prod(A,B),i) = prod(rows(B),A) 
template<class M1, class M2>
struct matrix_rows_optimizer<matrix_matrix_prod<M1,M2> >{
	typedef matrix_range_optimizer<typename M1::const_closure_type> left_opt;
	typedef matrix_matrix_prod_optimizer<typename left_opt::type, M2> opt;
	typedef typename opt::type type;
	
	static type create(matrix_matrix_prod<M1,M2> const& m,std::size_t start, std::size_t end){
		return opt::create(left_opt::create(m.lhs(),start,end),m.rhs());
	}
};


//rows(diagonal)  -> diagonal padded with 0
//~ template<class V>
//~ struct matrix_rows_optimizer<diagonal_matrix<V> >{
    //~ typedef vector_range_optimizer<typename V::const_closure_type > opt;
	//~ typedef diagonal_matrix<typename opt::type> type;
	//~ static type create(diagonal_matrix<V> const& m,
		//~ std::size_t start, std::size_t end, std::size_t start2, std::size_t end2
	//~ ){
        //~ REMORA_RANGE_CHECK(start1 == start2);// "unimplemented: non-diagonal subranges of diagonal matrix"
        //~ REMORA_RANGE_CHECK(end1 == end2); //"unimplemented: non-diagonal subranges of diagonal matrix"
        //~ std::size_t startV = std::max(start1,start2);
        //~ std::size_t endV = std::min(end1,end2);
		//~ return type(opt::create(m.expression(),startV, endV));
	//~ }
//~ };

////////////////////////////////////
//// Vector - Scalar Product
////////////////////////////////////


//default impl for alpha * v, creates just the expression
// handles all V that can not be blockwise, e.g. : all containers, proxies, scalar_vector, unit_vector
template<class V>
struct vector_scalar_multiply_optimizer{
	typedef vector_scalar_multiply<V> type;
	
	static type create(typename V::const_closure_type const& v, typename V::value_type alpha){
		return type(v, alpha);
	}
};


// alpha * (beta * v) = (alpha * beta) * v
template<class V>
struct vector_scalar_multiply_optimizer<vector_scalar_multiply<V> >{
	typedef vector_scalar_multiply<V> type;
	
	static type create(vector_scalar_multiply<V> const& v, typename V::value_type alpha){
		return type(v.expression(), alpha * v.scalar());
	}
};


// alpha * (v + w) = alpha * v + alpha * w
template<class V1, class V2>
struct vector_scalar_multiply_optimizer<vector_addition<V1, V2> >{
	typedef typename vector_addition<V1, V2>::value_type value_type;
	typedef vector_scalar_multiply_optimizer<V1> opt1;
	typedef vector_scalar_multiply_optimizer<V2> opt2;
	typedef vector_addition<typename opt1::type, typename opt2::type> type;
	static type create(vector_addition<V1, V2> const& m, value_type alpha){
		return type(opt1::create(m.lhs(), alpha), opt2::create(m.rhs(), alpha));
	}
};

// alpha * f(v) = (alpha * f)(v)
template<class V, class F>
struct vector_scalar_multiply_optimizer<vector_unary<V, F> >{
	typedef typename V::device_type device_type;
	typedef typename vector_unary<V, F>::value_type value_type;
	typedef typename device_traits<device_type>::template multiply_scalar<value_type> Multiplier;
	typedef vector_unary_optimizer <vector_unary<V, F>, Multiplier > opt;
	typedef typename opt::type type;
	static type create(vector_unary<V, F> const& m, value_type alpha){
		return opt::create(m, Multiplier(alpha));
	}
};

// alpha * f(v, w) = (alpha * f)(v, w)
template<class V1, class V2, class F>
struct vector_scalar_multiply_optimizer<vector_binary<V1, V2, F> >{
	typedef typename V1::device_type device_type;
	typedef typename vector_binary<V1, V2, F>::value_type value_type;
	typedef typename device_traits<device_type>::template multiply_scalar<value_type> Multiplier;
	typedef vector_unary_optimizer <vector_binary<V1, V2, F>, Multiplier > opt;
	typedef typename opt::type type;
	static type create(vector_binary<V1, V2, F> const& m, value_type alpha){
		return opt::create(m, Multiplier(alpha));
	}
};


//alpha * (A * v) can be folded into matrix_vector_prod 
template<class M, class V>
struct vector_scalar_multiply_optimizer<matrix_vector_prod<M, V> >{
	typedef matrix_vector_prod<M, V> type;
	static type create(matrix_vector_prod<M, V> const& m, typename type::value_type alpha){
		return type(m.matrix(), m.vector(), alpha * m.alpha());
	}
};


// alpha * (v | w) = (alpha * v) | (alpha * w) 
template<class V1, class V2>
struct vector_scalar_multiply_optimizer<vector_concat<V1, V2> >{
	typedef vector_scalar_multiply_optimizer<V1> opt1;
	typedef vector_scalar_multiply_optimizer<V2> opt2;
	typedef vector_concat<typename opt1::type,typename  opt2::type> type;
	typedef typename type::value_type value_type;
	static type create(vector_concat<V1, V2> const& m, value_type alpha){
		return type(opt1::create(m.lhs(), alpha), opt2::create(m.rhs(), alpha));
	}
};

////////////////////////////////////
//// Matrix - Scalar Product
////////////////////////////////////


//default impl for alpha * A, creates just the expression
// handles all M that can not be blockwise, e.g. : all containers, proxies, scalar_matrix
template<class M>
struct matrix_scalar_multiply_optimizer{
	typedef matrix_scalar_multiply<M> type;
	
	static type create(typename M::const_closure_type const& m, typename M::value_type alpha){
		return type(m, alpha);
	}
};


// alpha * (beta * A) = (alpha * beta) * A
template<class M>
struct matrix_scalar_multiply_optimizer<matrix_scalar_multiply<M> >{
	typedef matrix_scalar_multiply<M> type;
	
	static type create(matrix_scalar_multiply<M> const& m, typename M::value_type alpha){
		return type(m.expression(), alpha * m.scalar());
	}
};


// alpha * (A + B) = alpha * A + alpha * B
template<class E1, class E2>
struct matrix_scalar_multiply_optimizer<matrix_addition<E1, E2> >{
	typedef typename matrix_addition<E1, E2>::value_type value_type;
	typedef matrix_scalar_multiply_optimizer<E1> opt1;
	typedef matrix_scalar_multiply_optimizer<E2> opt2;
	typedef matrix_addition<typename opt1::type, typename opt2::type> type;
	static type create(matrix_addition<E1, E2> const& m, value_type alpha){
		return type(opt1::create(m.lhs(), alpha), opt2::create(m.rhs(), alpha));
	}
};


// alpha * repeat(v,n) = repeat(alpha * v, n)
template<class V, class O>
struct matrix_scalar_multiply_optimizer<vector_repeater<V, O> >{
	typedef vector_scalar_multiply_optimizer<V> opt;
	typedef vector_repeater<typename opt::type, O> type;
	static type create(vector_repeater<V, O> const& m, typename V::value_type alpha){
		return type(opt::create(m.expression(), alpha), m.num_repetitions());
	}
};

// alpha * f(A) = (alpha * f)(A)
template<class M, class F>
struct matrix_scalar_multiply_optimizer<matrix_unary<M, F> >{
	typedef typename M::device_type device_type;
	typedef typename F::result_type value_type;
	typedef typename device_traits<device_type>::template multiply_scalar<value_type> Multiplier;
	typedef matrix_unary_optimizer <matrix_unary<M, F>, Multiplier > opt;
	typedef typename opt::type type;
	static type create(matrix_unary<M, F> const& m, value_type alpha){
		return opt::create(m, Multiplier(alpha));
	}
};

// alpha * f(A, B) = (alpha * f)(A, B)
template<class M1, class M2, class F>
struct matrix_scalar_multiply_optimizer<matrix_binary<M1, M2, F> >{
	typedef typename M1::device_type device_type;
	typedef typename F::result_type value_type;
	typedef typename device_traits<device_type>::template multiply_scalar<value_type> Multiplier;
	typedef matrix_unary_optimizer <matrix_binary<M1, M2, F>, Multiplier > opt;
	typedef typename opt::type type;
	static type create(matrix_binary<M1, M2, F> const& m, value_type alpha){
		return opt::create(m, Multiplier(alpha));
	}
};


//alpha * v * u^T = (alpha * v) * u^T 
template<class V1, class V2>
struct matrix_scalar_multiply_optimizer<outer_product<V1, V2> >{
	typedef vector_scalar_multiply_optimizer<V1> opt;
	typedef outer_product<typename opt::type, V2> type;
	typedef typename type::value_type value_type;
	static type create(outer_product<V1, V2> const& m, value_type alpha){
		return type(opt::create(m.lhs(), alpha),m.rhs());
	}
};

//alpha * (A * B) can be folded into matrix_vector_prod 
template<class M1, class M2>
struct matrix_scalar_multiply_optimizer<matrix_matrix_prod<M1, M2> >{
	typedef matrix_matrix_prod<M1, M2> type;
	typedef typename type::value_type value_type;
	static type create(matrix_matrix_prod<M1, M2> const& m, value_type alpha){
		return type(m.lhs(), m.rhs(), alpha * m.alpha());
	}
};

// alpha*(A | B) = (alpha * A) | (alpha * B) 
template<class M1, class M2, bool b>
struct matrix_scalar_multiply_optimizer<matrix_concat<M1, M2, b> >{
	typedef matrix_scalar_multiply_optimizer<M1> opt1;
	typedef matrix_scalar_multiply_optimizer<M2> opt2;
	typedef matrix_concat<typename opt1::type, typename opt2::type, b> type;
	typedef typename type::value_type value_type;
	static type create(matrix_concat<M1, M2, b> const& m, value_type alpha){
		return type(opt1::create(m.lhs(), alpha), opt2::create(m.rhs(), alpha));
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
		return type(m, v, typename type::value_type(1));
	}
};

//(alpha M)*v = alpha * (M * v)
template<class M, class V>
struct matrix_vector_prod_optimizer<matrix_scalar_multiply<M>,V >{
	typedef matrix_vector_prod_optimizer<M, V> inner_opt;
	typedef vector_scalar_multiply_optimizer<typename inner_opt::type> opt;
	typedef typename opt::type type;
	
	static type create(matrix_scalar_multiply<M> const& m, typename V::const_closure_type const& v){
		return opt::create(inner_opt::create(m.expression(), v), m.scalar());
	}
};

//M*(alpha*v) = alpha * (M * v)
template<class M, class V>
struct matrix_vector_prod_optimizer<M,vector_scalar_multiply<V> >{
	typedef matrix_vector_prod_optimizer<M, V> inner_opt;
	typedef vector_scalar_multiply_optimizer<typename inner_opt::type> opt;
	typedef typename opt::type type;
	
	static type create(typename M::const_closure_type const& m, vector_scalar_multiply<V> const& v){
		return opt::create(inner_opt::create(m, v.expression()), v.scalar());
	}
};

//(alpha M)*(beta*v) = (alpha*beta) (M*v) can be folded into matrix-vector product
template<class M, class V>
struct matrix_vector_prod_optimizer<matrix_scalar_multiply<M>,vector_scalar_multiply<V> >{
	typedef matrix_vector_prod_optimizer<M, V> inner_opt;
	typedef vector_scalar_multiply_optimizer<typename inner_opt::type> opt;
	typedef typename opt::type type;
	
	static type create(matrix_scalar_multiply<M> const& m, vector_scalar_multiply<V>const& v){
		return opt::create(inner_opt::create(m.expression(), v.expression()), v.scalar() * m.scalar());
	}
};

//(M1*M2)*V=M1*(M2*V)
template<class M1,class M2, class V>
struct matrix_vector_prod_optimizer<matrix_matrix_prod<M1,M2>,V>{
private:
	typedef matrix_vector_prod_optimizer<M2,V> inner_opt;
	typedef matrix_vector_prod_optimizer<M1, typename inner_opt::type> outer_opt;
	typedef vector_scalar_multiply_optimizer<typename outer_opt::type> scalar_opt;
public:
	typedef typename scalar_opt::type type;
	
	static type create(matrix_matrix_prod<M1,M2> const& m, typename V::const_closure_type const& v){
		auto inner_result = inner_opt::create(m.rhs(),v);
		auto outer_result = outer_opt::create(m.lhs(),inner_result);
		return scalar_opt::create(outer_result, m.alpha());
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
		return type(m.lhs(), alpha);
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

//diag(v1) * v2 = v1 .* v2
template<class V1,class V2>
struct matrix_vector_prod_optimizer<diagonal_matrix<V1>,V2>{
	typedef typename common_value_type<V1,V2>::type value_type;
	typedef typename device_traits<typename V1::device_type>:: template multiply<value_type> functor;
	typedef vector_binary<V1, V2, functor> type;
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
		return type(lhs, rhs, typename type::value_type(1));
	}
};


//(alpha M1)*B = alpha (M1*B)
template<class M1, class M2>
struct matrix_matrix_prod_optimizer<matrix_scalar_multiply<M1>,M2 >{
	typedef matrix_matrix_prod_optimizer<M1, M2> inner_opt;
	typedef matrix_scalar_multiply<typename inner_opt::type> opt;
	typedef typename opt::type type;
	
	static type create(matrix_scalar_multiply<M1> const& A, typename M2::const_closure_type const& B){
		return opt::create(inner_opt::create(A.expression(), B), A.scalar());
	}
};

//M1*(alpha*B) = alpha (M1*B)
template<class M1, class M2>
struct matrix_matrix_prod_optimizer<M1,matrix_scalar_multiply<M2> >{
	typedef matrix_matrix_prod_optimizer<M1, M2> inner_opt;
	typedef matrix_scalar_multiply<typename inner_opt::type> opt;
	typedef typename opt::type type;
	
	static type create(typename M1::const_closure_type const& A, matrix_scalar_multiply<M2> const& B){
		return opt::create(inner_opt::create(A, B.expression()), B.scalar());
	}
};

//(alpha M1)*(beta*B) = (alpha*beta) (M1*B)
template<class M1, class M2>
struct matrix_matrix_prod_optimizer<matrix_scalar_multiply<M1>,matrix_scalar_multiply<M2> >{
	typedef matrix_matrix_prod_optimizer<M1, M2> inner_opt;
	typedef matrix_scalar_multiply<typename inner_opt::type> opt;
	typedef typename opt::type type;
	
	static type create(matrix_scalar_multiply<M1> const& A, matrix_scalar_multiply<M2>const& B){
		return opt::create(inner_opt::create(A.expression(), B.expression()), A.scalar() * B.scalar());
	}
};

////////////////////////////////////
//// Matrix Unary
////////////////////////////////////

template<class M, class F>
struct matrix_unary_optimizer{
	typedef matrix_unary<M,F> type;
	
	static type create(typename M::const_closure_type const& m, F const& f){
		return type(m,f);
	}
};

//f(g(x)) = (f o g)(x)
template<class M, class F1, class F2>
struct matrix_unary_optimizer<matrix_unary<M,F1>, F2 >{
	typedef typename device_traits<typename M::device_type>::template compose<F1, F2> composed_type;
	typedef matrix_unary<M,composed_type> type;
	
	static type create(matrix_unary<M,F1> const& m, F2 const& f){
		return type(m.expression(),composed_type(m.functor(),f));
	}
};

//f(g(x,y)) = (f o g)(x,y)
template<class M1, class M2, class F1, class F2>
struct matrix_unary_optimizer<matrix_binary<M1,M2, F1>, F2 >{
	typedef typename device_traits<typename M1::device_type>::template compose<F1, F2> composed_type;
	typedef matrix_binary<M1, M2,composed_type> type;
	
	static type create(matrix_binary<M1, M2, F1> const& m, F2 const& f){
		return type(m.lhs(), m.rhs(), composed_type(m.functor(),f));
	}
};

////////////////////////////////////
//// Vector Unary
////////////////////////////////////

template<class V, class F>
struct vector_unary_optimizer{
	typedef vector_unary<V,F> type;
	
	static type create(typename V::const_closure_type const& v, F const& f){
		return type(v,f);
	}
};


//f(g(x)) = (f o g)(x)
template<class V, class F1, class F2>
struct vector_unary_optimizer<vector_unary<V,F1>, F2 >{
	typedef typename device_traits<typename V::device_type>::template compose<F1, F2> composed_type;
	typedef vector_unary<V,composed_type> type;
	
	static type create(vector_unary<V,F1> const& v, F2 const& f){
		return type(v.expression(),composed_type(v.functor(),f));
	}
};

//f(g(x,y)) = (f o g)(x,y)
template<class V1, class V2, class F1, class F2>
struct vector_unary_optimizer<vector_binary<V1,V2, F1>, F2 >{
	typedef typename device_traits<typename V1::device_type>::template compose<F1, F2> composed_type;
	typedef vector_binary<V1, V2,composed_type> type;
	
	static type create(vector_binary<V1, V2, F1> const& v, F2 const& f){
		return type(v.lhs(), v.rhs(), composed_type(v.functor(),f));
	}
};

//g2(fold(A, f, g)) = fold(A, f, g2 o g)
template<class M, class F, class G, class G2>
struct vector_unary_optimizer<matrix_row_transform<M, F, G>, G2>{
	typedef typename device_traits<typename M::device_type>::template compose<G, G2> composed_type;
	typedef matrix_row_transform<M, F, composed_type> type;
	
	static type create(matrix_row_transform<M, F, G> const& v, G2 const& g2){
		return type(v.matrix(), v.f(), composed_type(v.g(),g2) );
	}
};



////////////////////////////////////
//// Vector-Set Fold
////////////////////////////////////

template<class S, class F, class G>
struct fold_vector_set_optimizer;

template<class M, class F, class G>
struct fold_vector_set_optimizer<vector_set<M, row_major>, F, G>{
	typedef matrix_row_transform<M, F, G> type;
	static type create(vector_set<M, row_major> const& set, F const& f, G const& g){
		return type(set.expression(), f, g);
	}
};

template<class M, class F, class G>
struct fold_vector_set_optimizer<vector_set<M, column_major>, F, G>{
	typedef matrix_transpose_optimizer<M> opt;
	typedef matrix_row_transform<typename opt::type, F, G> type;
	static type create(vector_set<M, column_major> const& set, F const& f, G const& g){
		return type(opt::create(set.expression()), f, g);
	}
};

//~ template<class S, class M>
//~ struct vector_set_matrix_prod_optimizer;

//~ template<class M1, class M2>
//~ struct vector_set_matrix_prod_optimizer<vector_set<M1, row_major>, M2>{
	//~ typedef matrix_matrix_prod_optimizer<M1, M2> opt;
	//~ typedef vector_set<typename opt::type, row_major> type;
	//~ static type create(vector_set<M1, row_major> const& set, typename M2::const_closure_type const& m2){
		//~ return as_set(opt::create(set.expression(), m2), row_major());
	//~ }
//~ };

//~ template<class M1, class M2>
//~ struct vector_set_matrix_prod_optimizer<vector_set<M1, column_major>, M2>{
	//~ typedef matrix_transpose_optimizer<M2> trans_opt;
	//~ typedef matrix_matrix_prod_optimizer<typename trans_opt::type, M1> opt;
	//~ typedef vector_set<typename opt::type, column_major> type;
	//~ static type create(vector_set<M1, column_major> const& set, typename M2::const_closure_type const& m2){
		//~ return as_set(opt::create(trans_opt::create(m2),set.expression()), column_major());
	//~ }
//~ };

//~ template<class S, class V>
//~ struct vector_set_inner_prod_optimizer;

//~ template<class M, class V>
//~ struct vector_set_inner_prod_optimizer<vector_set<M, row_major>, V>{
	//~ typedef matrix_vector_prod_optimizer<M, V> opt;
	//~ typedef typename opt::type type;
	//~ static type create(vector_set<M, row_major> const& set, typename V::const_closure_type const& v){
		//~ return opt::create(set.expression(), v);
	//~ }
//~ };
//~ template<class M, class V>
//~ struct vector_set_inner_prod_optimizer<vector_set<M, column_major>, V>{
	//~ typedef matrix_transpose_optimizer<M> trans_opt;
	//~ typedef matrix_vector_prod_optimizer<typename trans_opt::type, V> opt;
	//~ typedef typename opt::type type;
	//~ static type create(vector_set<M, column_major> const& set, typename V::const_closure_type const& v){
		//~ return opt::create(trans_opt::create(set.expression()), v);
	//~ }
//~ };


}}
#endif
