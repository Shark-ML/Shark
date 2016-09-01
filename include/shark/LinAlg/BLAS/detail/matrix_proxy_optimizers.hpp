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
 #ifndef SHARK_LINALG_BLAS_MATRIX_PROXY_OPTIMIZERS_HPP
#define SHARK_LINALG_BLAS_MATRIX_PROXY_OPTIMIZERS_HPP

#include "matrix_proxy_classes.hpp"
#include "matrix_expression_classes.hpp"

namespace shark {namespace blas {namespace detail{
	
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

//(M1 M2)^T = M2^T M1^T
template<class M1, class M2>
struct matrix_transpose_optimizer<matrix_matrix_prod<M1,M2> >{
	typedef matrix_transpose_optimizer<typename const_expression<M2>::type> left_opt;
	typedef matrix_transpose_optimizer<typename const_expression<M1>::type> right_opt;
	typedef matrix_matrix_prod<typename left_opt::type,typename right_opt::type > type;
	
	static type create(matrix_matrix_prod<M1,M2> const& m){
		return type(left_opt::create(m.rhs()),right_opt::create(m.lhs()));
	}
};

}}}
#endif
