/*!
 * \brief       Expression Optimizations for proxy expressions
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
 #ifndef SHARK_LINALG_BLAS_MATRIX_EXPRESSION_OPTIMIZERS_HPP
#define SHARK_LINALG_BLAS_MATRIX_EXPRESSION_OPTIMIZERS_HPP

#include "matrix_proxy_optimizers.hpp"

namespace shark {namespace blas {namespace detail{
	
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

//the helper guards against the case that applying the simplifications of matrix_transpose
//can be the identity -> guard against infinite loops
template<class M1, class M1Simplified, class V>
struct matrix_vector_prod_transpose_helper{
private:
	typedef matrix_vector_prod_optimizer<M1Simplified, V> inner_opt;
public:
	typedef typename inner_opt::type type;
	static type create(typename M1Simplified::const_closure_type const& m, typename V::const_closure_type const& v){
		return inner_opt::create(m,v);
	}
};
template<class M, class V>
struct matrix_vector_prod_transpose_helper<M,M,V>{
	typedef matrix_vector_prod<M,V> type;
	
	static type create(typename M::const_closure_type const& m, typename V::const_closure_type const& v){
		return type(m,v);
	}
};

//simplify expressions with transposed matrix arguments.(used for product of types xA=>A^Tx)
template<class M, class V>
struct matrix_vector_prod_optimizer<matrix_transpose<M>,V>{
private:
	typedef typename matrix_transpose<M>::const_closure_type closure;
	typedef matrix_transpose_optimizer<typename const_expression<M>::type > transpose_opt;//simplify the matrix transpose statement
	typedef matrix_vector_prod_transpose_helper<
		closure, 
		typename transpose_opt::type,V
	> inner_opt;//call recursively on the simplified type and guard against identity transformations
public:
	typedef typename inner_opt::type type;
	static type create(closure const& m, typename V::const_closure_type const& v){
		return inner_opt::create(transpose_opt::create(m.expression()),v);
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

}}}
#endif
