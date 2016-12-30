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
#ifndef SHARK_LINALG_BLAS_DECOMPOSITIONS_HPP
#define SHARK_LINALG_BLAS_DECOMPOSITIONS_HPP



#include "kernels/trsm.hpp"
#include "kernels/trsv.hpp"
#include "kernels/potrf.hpp"
#include "kernels/pstrf.hpp"
//~ #include "kernels/getrf.hpp"
#include "assignment.hpp"
#include "permutation.hpp"
#include "matrix_expression.hpp"
#include "vector_proxy.hpp"
#include "vector_expression.hpp"

namespace shark{namespace blas{
template<class D, class Device>
struct solver{
	typedef D decomposition_type;
	typedef Device device_type;
	
	decomposition_type const& operator()() const {
		return *static_cast<decomposition_type const*>(this);
	}

	decomposition_type& operator()() {
		return *static_cast<decomposition_type*>(this);
	}
};


/// \brief Lower triangular Cholesky decomposition.
///
///  Given an \f$ m \times m \f$ symmetric positive definite matrix
///  \f$A\f$, represents the lower triangular matrix \f$L\f$ such that
///  \f$A = LL^T \f$.
///  
/// This decomposition is a corner block of many linear algebra routines
/// especially for solving symmetric positive definite systems of equations.
/// 
/// Unlike many other decompositions, this decomposition is fast,
/// numerically stable and can be updated when the original matrix is changed
/// (rank-k updates of the form A<-alpha A + VCV^T)
template<class MatrixStorage>
class cholesky_decomposition:
	public solver<
		cholesky_decomposition<MatrixStorage>, 
		typename MatrixStorage::device_type
>{
private:
	typedef typename MatrixStorage::value_type value_type;
public:
	typedef typename MatrixStorage::device_type device_type;
	cholesky_decomposition(){}
	template<class E>
	cholesky_decomposition(matrix_expression<E,device_type> const& e):m_cholesky(e){
		kernels::potrf<lower>(m_cholesky);
	}
	
	template<class E>
	void decompose(matrix_expression<E,device_type> const& e){
		m_cholesky.resize(e().size1(), e().size2());
		noalias(m_cholesky) = e;
		kernels::potrf<lower>(m_cholesky);
	}
	
	MatrixStorage const& lower_factor()const{
		return m_cholesky;
	}
	auto upper_factor()->decltype(trans(lower_factor())){
		return trans(m_cholesky);
	}
	
	template<class MatB>
	void solve(matrix_expression<MatB,device_type>& B, left){
		kernels::trsm<lower,left >(lower_factor(),B);
		kernels::trsm<upper,left >(upper_factor(),B);
	}
	template<class MatB>
	void solve(matrix_expression<MatB,device_type>& B, right){
		kernels::trsm<upper,right >(upper_factor(),B);
		kernels::trsm<lower,right >(lower_factor(),B);
	}
	template<class VecB, bool Left>
	void solve(vector_expression<VecB,device_type>&b, system_tag<Left>){
		kernels::trsv<lower, left>(lower_factor(),b);
		kernels::trsv<upper,left>(upper_factor(),b);
	}
	
	/// \brief Updates a covariance factor by a rank one update
	///
	/// Let \f$ A=LL^T \f$ be a matrix with its lower cholesky factor. Assume we want to update 
	/// A using a simple rank-one update \f$ A = \alpha A+ \beta vv^T \f$. This invalidates L and
	/// it needs to be recomputed which is O(n^3). instead we can update the factorisation
	/// directly by performing a similar, albeit more complex algorithm on L, which can be done
	/// in O(L^2). 
	/// 
	/// Alpha is not required to be positive, but if it is not negative, one has to be carefull
	/// that the update would keep A positive definite. Otherwise the decomposition does not
	/// exist anymore and an exception is thrown.
	///
	/// \param v the update vector
	/// \param alpha the scaling factor, must be positive.
	/// \param beta the update factor. it Can be positive or negative
	template<class VecV>
	void update(value_type alpha, value_type beta, vector_expression<VecV,device_type> const& v){
		//implementation blatantly stolen from Eigen
		std::size_t n = v().size();
		auto& L = m_cholesky;
		typename vector_temporary<VecV>::type temp = v;
		double beta_prime = 1;
		double a = std::sqrt(alpha);
		for(std::size_t j=0; j != n; ++j)
		{
			double Ljj = a * L(j,j);
			double dj = Ljj * Ljj;
			double wj = temp(j);
			double swj2 = beta * wj * wj;
			double gamma = dj * beta_prime + swj2;

			double x = dj + swj2/beta_prime;
			if (x <= 0.0)
				throw std::invalid_argument("[cholesky_decomposition::update] update makes matrix indefinite, no update available");
			double nLjj = std::sqrt(x);
			L(j,j) = nLjj;
			beta_prime += swj2/dj;
			
			// Update the terms of L
			if(j+1 <n)
			{
				subrange(column(L,j),j+1,n) *= a;
				noalias(subrange(temp,j+1,n)) -= (wj/Ljj) * subrange(column(L,j),j+1,n);
				if(gamma == 0)
					continue;
				subrange(column(L,j),j+1,n) *= nLjj/Ljj;
				noalias(subrange(column(L,j),j+1,n))+= (nLjj * beta*wj/gamma)*subrange(temp,j+1,n);
			}
		}
	}
	
	template<typename Archive>
	void serialize( Archive & ar, const std::size_t version ) {
		ar & m_cholesky;
	}
private:
	MatrixStorage m_cholesky;
};

//~ template<class MatrixStorage>
//~ class pivoting_lu_decomposition:
	//~ public solver<
		//~ pivoting_lu_decomposition<MatrixStorage>, 
		//~ typename MatrixStorage::device_type
//~ >{
//~ public:
	//~ template<class E>
	//~ pivoting_lu_decomposition(matrix_expression<E> const& e)
	//~ :m_factor(e), m_permutation(e().size1()){
		//~ kernels::getrf(m_factor,m_permutation);
	//~ }
	
	//~ MatrixStorage const& factor()const{
		//~ return m_factor;
	//~ }
	
	//~ permuttion_matrix const& permutation() const{
		//~ return m_permutation;
	//~ }
	
	//~ template<class MatB>
	//~ void solve(matrix_expression<MatB>& B, left){
		//~ swap_rows(m_permutation,B);
		//~ kernels::trsm<unit_lower,left>(m_factor,B);
		//~ kernels::trsm<upper,left>(m_factor,B);
		//~ swap_rows_inverted(m_permutation,B);
	//~ }
	//~ template<class MatB>
	//~ void solve(matrix_expression<MatB>& B, right){
		//~ swap_columns(m_permutation,B);
		//~ kernels::trsm<upper,right>(m_factor,B);
		//~ kernels::trsm<unit_lower,right>(m_factor,B);
		//~ swap_columns_inverted(m_permutation,B);
		//~ swap_rows_inverted(m_permutation,b);
	//~ }
	//~ template<class VecB>
	//~ void solve(vector_expression<VecB>& b, left){
		//~ swap_rows(m_permutation,b);
		//~ kernels::trsv<unit_lower,left>(m_factor,b);
		//~ kernels::trsv<upper,left>(m_factor,b);
		//~ swap_rows_inverted(m_permutation,b);
	//~ }
	//~ template<class VecB>
	//~ void solve(vector_expression<VecB>& b, right){
		//~ swap_rows(m_permutation,b);
		//~ kernels::trsv<upper,right>(m_factor,b);
		//~ kernels::trsv<unit_lower,right>(m_factor,b);
		//~ swap_rows_inverted(m_permutation,b);
	//~ }
//~ private:
	//~ MatrixStorage m_factor;
	//~ permutation_matrix m_permutation;
//~ };


// This is an implementation suggested by
// "Fast Computation of Moore-Penrose Inverse Matrices"
// applied to the special case of symmetric pos semi-def matrices
// trading numerical accuracy vs speed. We go for speed.
//
// The fact that A is not full rank means it is not invertable,
// so we solve it in a least squares sense.
//
// We use the formula for the pseudo-inverse:
// (P^T A P)^-1 = L(L^TL)^-1(L^TL)^-1 L^T
// where L is a matrix obtained by some rank revealing factorization
// P^T A P = L L^T 
// we chose a pivoting cholesky to make use of the fact that A is symmetric
// and all eigenvalues are >=0. If A has full rank, this reduces to
// the cholesky factor where the pivoting leads to slightly smaller numerical errors
// At a higher computational cost compared to the normal cholesky decomposition.
template<class MatrixStorage>
class symm_pos_semi_definite_solver:
	public solver<symm_pos_semi_definite_solver<MatrixStorage>, typename MatrixStorage::device_type>{
public:
	typedef typename MatrixStorage::device_type device_type;
	template<class E>
	symm_pos_semi_definite_solver(matrix_expression<E,device_type> const& e)
	:m_factor(e), m_permutation(e().size1()){
		m_rank = kernels::pstrf<lower>(m_factor,m_permutation);
		if(m_rank == e().size1()) return; //full rank, so m_factor is lower triangular and we are done
		
		auto L = columns(m_factor,0,m_rank);
		m_cholesky.decompose(prod(trans(L),L));
	}
	
	unsigned rank()const{
		return m_rank;
	}
	
	//compute C so that A^dagger = CC^T
	//where A^dagger is the moore-penrose inverse
	// m must be of size rank x n
	template<class Mat>
	void compute_inverse_factor(matrix_expression<Mat,device_type>& C){
		SIZE_CHECK(C().size1() == m_rank);
		SIZE_CHECK(C().size2() == m_factor.size1());
		if(m_rank == m_factor.size1()){//matrix has full rank
			//initialize as identity matrix and solve
			noalias(C) = identity_matrix<double>( m_factor.size1());
			swap_columns_inverted(m_permutation,C);
			kernels::trsm<lower,left>(m_factor,C);
		}else{
			auto L = columns(m_factor,0,m_rank);
			noalias(C) = trans(L);
			m_cholesky.solve(C,left());
			swap_columns_inverted(m_permutation,C);
		}
		
	}
	template<class MatB>
	void solve(matrix_expression<MatB,device_type>& B, left){
		swap_rows(m_permutation,B);
		if(m_rank == 0){//matrix is zero
			B().clear();
		}else if(m_rank == m_factor.size1()){//matrix has full rank
			kernels::trsm<lower,left>(m_factor,B);
			kernels::trsm<upper,left>(trans(m_factor),B);
		}else{//matrix is missing rank
			auto L = columns(m_factor,0,m_rank);
			auto Z =  eval_block(prod(trans(L),B));
			m_cholesky.solve(Z,left());	
			m_cholesky.solve(Z,left());	
			noalias(B) = prod(L,Z);
		}
		swap_rows_inverted(m_permutation,B);
	}
	template<class MatB>
	void solve(matrix_expression<MatB,device_type>& B, right){
		//compute using symmetry of the system of equations
		auto transB = trans(B);
		solve(transB, left());
	}
	template<class VecB, bool Left>
	void solve(vector_expression<VecB,device_type>& b, system_tag<Left>){
		swap_rows(m_permutation,b);
		if(m_rank == 0){//matrix is zero
			b().clear();
		}else if(m_rank == m_factor.size1()){//matrix has full rank
			kernels::trsv<lower,left >(m_factor,b);
			kernels::trsv<upper,left >(trans(m_factor),b);
		}else{//matrix is missing rank
			auto L = columns(m_factor,0,m_rank);
			auto z =  eval_block(prod(trans(L),b));
			m_cholesky.solve(z,left());	
			m_cholesky.solve(z,left());	
			noalias(b) = prod(L,z);
		}
		swap_rows_inverted(m_permutation,b);
	}
private:
	unsigned int m_rank;
	MatrixStorage m_factor;
	cholesky_decomposition<MatrixStorage> m_cholesky;
	permutation_matrix m_permutation;
};

namespace detail{
	template<bool Left>
	struct cg_system_tag{
		double epsilon = 1.e-10;
		unsigned max_iterations = 0; 
		cg_system_tag() = default;
		cg_system_tag(double epsilon, unsigned max_iterations)
		:epsilon(epsilon), max_iterations(max_iterations){}
	};
}

template<class MatA>
class cg_solver:
	public solver<cg_solver<MatA>, typename MatA::device_type>{
private:
	typedef typename vector_temporary<MatA>::type vector_type;
	typedef typename matrix_temporary<MatA>::type matrix_type;
public:
	typedef typename MatA::const_closure_type matrix_closure_type;
	typedef typename MatA::device_type device_type;
	cg_solver(matrix_closure_type const& e):m_expression(e){}
	
	template<class MatB>
	void solve(matrix_expression<MatB,device_type>& B, left, double epsilon = 1.e-10, unsigned max_iterations = 0){
		matrix_type X = B;
		cg(m_expression,X, B, epsilon, max_iterations);
		noalias(B) = X;
	}
	template<class MatB>
	void solve(matrix_expression<MatB,device_type>& B, right, double epsilon = 1.e-10, unsigned max_iterations = 0){
		auto transB = trans(B);
		matrix_type X = transB;
		cg(m_expression,X,transB, epsilon, max_iterations);
		noalias(transB) = X;
	}
	template<class VecB, bool Left>
	void solve(vector_expression<VecB,device_type>&b, system_tag<Left>, double epsilon = 1.e-10, unsigned max_iterations = 0){
		vector_type x = b;
		cg(m_expression,x,b,epsilon,max_iterations);
		noalias(b) = x;
	}
	
	template<class VecB, bool Left>
	void solve(vector_expression<VecB,device_type>&b, detail::cg_system_tag<Left> tag){
		solve(b, system_tag<Left>(), tag.epsilon, tag.max_iterations);
	}
	template<class MatB, bool Left>
	void solve(matrix_expression<MatB,device_type>&B, detail::cg_system_tag<Left> tag){
		solve(B, system_tag<Left>(), tag.epsilon, tag.max_iterations);
	}
private:
	template<class MatT, class VecB, class VecX>
	void cg(
		matrix_expression<MatT, device_type> const& A,
		vector_expression<VecB, device_type>& x,
		vector_expression<VecX, device_type> const& b,
		double epsilon,
		unsigned int max_iterations
	){
		SIZE_CHECK(A().size1() == A().size2());
		SIZE_CHECK(A().size1() == b().size());
		SIZE_CHECK(A().size1() == x().size());
				
		std::size_t dim = b().size();
		
		//initialize point. 
		vector_type residual = b - prod(A,x);
		//check if provided solution is better than starting at 0
		if(norm_inf(residual) > norm_inf(b)){
			x().clear();
			residual = b;
		}
		
		vector_type next_residual(dim); //the next residual
		vector_type p = residual; //the search direction- initially it is the gradient direction
		vector_type Ap(dim); //stores prod(A,p)
		
		for(std::size_t iter = 0;; ++iter){
			if(max_iterations != 0 && iter >= max_iterations) break;
			noalias(Ap) = prod(A,p);
			double rsqr = norm_sqr(residual);
			double alpha = rsqr/inner_prod(p,Ap);
			noalias(x) += alpha * p;
			noalias(next_residual) = residual - alpha * Ap; 
			if(norm_inf(next_residual) < epsilon)
				break;
			
			double beta = inner_prod(next_residual,next_residual)/rsqr;
			p *= beta;
			noalias(p) += next_residual;
			swap(residual,next_residual);
		}
	}
	
	template<class MatT, class MatB, class MatX>
	void cg(
		matrix_expression<MatT, device_type> const& A,
		matrix_expression<MatX, device_type>& X,
		matrix_expression<MatB, device_type> const& B,
		double epsilon,
		unsigned int max_iterations
	){
		SIZE_CHECK(A().size1() == A().size2());
		SIZE_CHECK(A().size1() == B().size1());
		SIZE_CHECK(A().size1() == X().size1());
		SIZE_CHECK(B().size2() == X().size2());
				
		std::size_t dim = B().size1();
		std::size_t num_rhs = B().size2();
		
		//initialize gradient given the starting point
		matrix_type residual = B - prod(A,X);
		//check for each rhs whether the starting point is better than starting from scratch
		for(std::size_t i = 0; i != num_rhs; ++i){
			if(norm_inf(column(residual,i)) <= norm_inf(column(residual,i))){
				column(X,i).clear();
				noalias(column(residual,i)) = column(B,i);
			}
		}
		
		vector_type next_residual(dim); //the next residual of a column
		matrix_type P = residual; //the search direction- initially it is the gradient direction
		matrix_type AP(dim, num_rhs); //stores prod(A,p)
		
		for(std::size_t iter = 0;; ++iter){
			if(max_iterations != 0 && iter >= max_iterations) break;
			//compute the product for all rhs at the same time
			noalias(AP) = prod(A,P);
			//for each rhs apply a step of cg
			for(std::size_t i = 0; i != num_rhs; ++i){
				auto r = column(residual,i);
				//skip this if we are done already
				//otherwise we might run into numerical troubles later on
				if(norm_inf(r) < epsilon) continue;
				
				auto x = column(X,i);
				auto p = column(P,i);
				auto Ap = column(AP,i);
				double rsqr = norm_sqr(r);
				double alpha = rsqr/inner_prod(p,Ap);
				noalias(x) += alpha * p;
				noalias(next_residual) = r - alpha * Ap; 
				double beta = inner_prod(next_residual,next_residual)/rsqr;
				p *= beta;
				noalias(p) += next_residual;
				noalias(r) = next_residual;
			}
			//if all solutions are within tolerance, we are done
			if(max(abs(residual)) < epsilon)
				break;
		}
	}
	
	matrix_closure_type m_expression;
};

template<class MatA, class Triangular>
class triangular_decomposition:
	public solver<
		triangular_decomposition<MatA, Triangular>, 
		typename MatA::device_type
>{
public:
	typedef typename MatA::const_closure_type matrix_closure_type;
	typedef typename MatA::device_type device_type;
	triangular_decomposition(matrix_closure_type const& e):m_matrix(e){}
	
	matrix_closure_type triangular_factor()const{
		return m_matrix;
	}
	
	template<class MatB, bool Left>
	void solve(matrix_expression<MatB, device_type>& B, system_tag<Left> ){
		kernels::trsm<Triangular,system_tag<Left> >(m_matrix,B);
	}
	template<class VecB, bool Left>
	void solve(vector_expression<VecB, device_type>& b, system_tag<Left> ){
		kernels::trsv<Triangular,system_tag<Left> >(m_matrix,b);
	}
private:
	matrix_closure_type m_matrix;
};

struct symm_pos_def{};
struct symm_semi_pos_def{};
struct conjugate_gradient{};
struct indefinite_full_rank{};


//solvers for triangular matrices
template<class MatA, class VecB, bool Left, class Device, bool Upper, bool Unit>
matrix_vector_solve<
	MatA,VecB, system_tag<Left>,
	triangular_decomposition<MatA, triangular_tag<Upper, Unit>>
>
solve(
	matrix_expression<MatA, Device> const& A,
	vector_expression<VecB, Device> const& b,
	triangular_tag<Upper,Unit>, system_tag<Left>
){
	SIZE_CHECK(A().size1() ==  A().size2());
	SIZE_CHECK(A().size1() ==  b().size());
	typedef triangular_decomposition<MatA, triangular_tag<Upper, Unit> > Decomp;
	return matrix_vector_solve<MatA,VecB,system_tag<Left>,Decomp>(A(),b());
}
template<class MatA, class MatB, bool Left, class Device, bool Upper, bool Unit>
matrix_matrix_solve<
	MatA,MatB, system_tag<Left>,
	triangular_decomposition<MatA, triangular_tag<Upper, Unit>>
>
solve(
	matrix_expression<MatA, Device> const& A,
	matrix_expression<MatB, Device> const& B,
	triangular_tag<Upper,Unit>, system_tag<Left>
){
	SIZE_CHECK(A().size1() ==  A().size2());
	typedef triangular_decomposition<MatA, triangular_tag<Upper, Unit> > Decomp;
	return matrix_matrix_solve<MatA,MatB,system_tag<Left>,Decomp>(A(),B());
}


//solvers for symmetric positive definite systems of equations
template<class MatA, class VecB, bool Left, class Device>
matrix_vector_solve<
	MatA,VecB, system_tag<Left>,
	cholesky_decomposition<typename matrix_temporary<MatA>::type>
>
solve(
	matrix_expression<MatA, Device> const& A,
	vector_expression<VecB, Device> const& b,
	symm_pos_def, system_tag<Left>
){
	SIZE_CHECK(A().size1() ==  A().size2());
	SIZE_CHECK(A().size1() ==  b().size());
	typedef cholesky_decomposition<typename matrix_temporary<MatA>::type> Decomp;
	return matrix_vector_solve<MatA,VecB, system_tag<Left>, Decomp>(A(),b());
}

template<class MatA, class MatB, bool Left, class Device>
matrix_matrix_solve<
	MatA,MatB, system_tag<Left>,
	cholesky_decomposition<typename matrix_temporary<MatA>::type>
>
solve(
	matrix_expression<MatA, Device> const& A,
	matrix_expression<MatB, Device> const& B,
	symm_pos_def, system_tag<Left>
){
	SIZE_CHECK(A().size1() ==  A().size2());
	typedef cholesky_decomposition<typename matrix_temporary<MatA>::type> Decomp;
	return matrix_matrix_solve<MatA,MatB,system_tag<Left>, Decomp>(A(),B());
}

//solvers for symmetric positive semi-definite systems of equations
template<class MatA, class VecB, bool Left, class Device>
matrix_vector_solve<
	MatA,VecB, system_tag<Left>,
	symm_pos_semi_definite_solver<typename matrix_temporary<MatA>::type>
>
solve(
	matrix_expression<MatA, Device> const& A,
	vector_expression<VecB, Device> const& b,
	symm_semi_pos_def, system_tag<Left>
){
	SIZE_CHECK(A().size1() ==  A().size2());
	SIZE_CHECK(A().size1() ==  b().size());
	typedef symm_pos_semi_definite_solver<typename matrix_temporary<MatA>::type> Decomp;
	return matrix_vector_solve<MatA,VecB, system_tag<Left>, Decomp>(A(),b());
}

template<class MatA, class MatB, bool Left, class Device>
matrix_matrix_solve<
	MatA,MatB, system_tag<Left>,
	symm_pos_semi_definite_solver<typename matrix_temporary<MatA>::type>
>
solve(
	matrix_expression<MatA, Device> const& A,
	matrix_expression<MatB, Device> const& B,
	symm_semi_pos_def, system_tag<Left>
){
	SIZE_CHECK(A().size1() ==  A().size2());
	typedef symm_pos_semi_definite_solver<typename matrix_temporary<MatA>::type> Decomp;
	return matrix_matrix_solve<MatA,MatB,system_tag<Left>, Decomp>(A(),B());
}

//~ //solvers for general full rank systems of equations
//~ template<class MatA, class VecB, bool Left, class Device>
//~ matrix_vector_solve<
	//~ MatA,VecB, system_tag<Left>,
	//~ pivoting_lu_decomposition<typename matrix_temporary<MatA>::type>
//~ >
//~ solve(
	//~ matrix_expression<MatA, Device> const& A,
	//~ vector_expression<VecB, Device> const& b,
	//~ indefinite_full_rank, system_tag<Left>
//~ ){
	//~ SIZE_CHECK(A().size1() ==  A().size2());
	//~ SIZE_CHECK(A().size1() ==  b().size());
	//~ typedef pivoting_lu_decomposition<typename matrix_temporary<MatA>::type Decomp;
	//~ return matrix_vector_solve<MatA,VecB,system_tag<Left>, Decomp>(A(),b());
//~ }

//~ template<class MatA, class MatB, bool Left, class Device>
//~ matrix_matrix_solve<
	//~ MatA,MatB, system_tag<Left>,
	//~ pivoting_lu_decomposition<typename matrix_temporary<MatA>::type>
//~ >
//~ solve(
	//~ matrix_expression<MatA, Device> const& A,
	//~ matrix_expression<MatB, Device> const& B,
	//~ indefinite_full_rank, system_tag<Left>
//~ ){
	//~ SIZE_CHECK(A().size1() ==  A().size2());
	//~ SIZE_CHECK(A().size1() ==  B().size1());
	//~ typedef pivoting_lu_decomposition<typename matrix_temporary<MatA>::type Decomp;
	//~ return matrix_matrix_solve<MatA,MatB,system_tag<Left>, Decomp>(A(),B());
//~ }

//conjugate gradient solvers
template<class MatA, class VecB, bool Left, class Device>
matrix_vector_solve<MatA,VecB, detail::cg_system_tag<Left>,cg_solver<MatA> >
solve(
	matrix_expression<MatA, Device> const& A,
	vector_expression<VecB, Device> const& b,
	conjugate_gradient, system_tag<Left>, 
	double epsilon = 1.e-10,
	unsigned int max_iterations = 0
){
	SIZE_CHECK(A().size1() ==  A().size2());
	SIZE_CHECK(A().size1() ==  b().size());
	return matrix_vector_solve<MatA,VecB,detail::cg_system_tag<Left>, cg_solver<MatA> >(A(),b(),detail::cg_system_tag<Left>(epsilon, max_iterations));
}

template<class MatA, class MatB, bool Left, class Device>
matrix_matrix_solve<MatA,MatB, detail::cg_system_tag<Left>, cg_solver<MatA> >
solve(
	matrix_expression<MatA, Device> const& A,
	matrix_expression<MatB, Device> const& B,
	conjugate_gradient, system_tag<Left>,
	double epsilon = 1.e-10,
	unsigned int max_iterations = 0
){
	SIZE_CHECK(A().size1() ==  A().size2());
	return matrix_matrix_solve<MatA,MatB,detail::cg_system_tag<Left>, cg_solver<MatA> >(A(),B(),detail::cg_system_tag<Left>(epsilon, max_iterations));
}

}}
#endif