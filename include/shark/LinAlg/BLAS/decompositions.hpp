/*!
 * \brief       Defines types for matrix decompositions and solvers
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
#ifndef REMORA_DECOMPOSITIONS_HPP
#define REMORA_DECOMPOSITIONS_HPP

#include "kernels/trsm.hpp"
#include "kernels/trsv.hpp"
#include "kernels/potrf.hpp"
#include "kernels/pstrf.hpp"
#include "kernels/getrf.hpp"
#include "kernels/syev.hpp"
#include "assignment.hpp"
#include "permutation.hpp"
#include "matrix_expression.hpp"
#include "proxy_expressions.hpp"
#include "vector_expression.hpp"

namespace remora{
template<class D, class Device>
struct solver_expression{
	typedef Device device_type;
	
	D const& operator()() const {
		return *static_cast<D const*>(this);
	}

	D& operator()() {
		return *static_cast<D*>(this);
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
	public solver_expression<
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

	auto upper_factor()const -> decltype(trans(std::declval<MatrixStorage const&>())){
		return trans(m_cholesky);
	}
	
	template<class MatB>
	void solve(matrix_expression<MatB,device_type>& B, left)const{
		kernels::trsm<lower,left >(lower_factor(),B);
		kernels::trsm<upper,left >(upper_factor(),B);
	}
	template<class MatB>
	void solve(matrix_expression<MatB,device_type>& B, right)const{
		kernels::trsm<upper,right >(upper_factor(),B);
		kernels::trsm<lower,right >(lower_factor(),B);
	}
	template<class VecB, bool Left>
	void solve(vector_expression<VecB,device_type>&b, system_tag<Left>)const{
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
		if(beta == 0){
			m_cholesky *= std::sqrt(alpha);
			return;
		}
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
			if(j+1 <n){
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


/// \brief Symmetric eigenvalue decomposition A=QDQ^T
///
/// every symmetric matrix can be decomposed into its eigenvalue decomposition
/// A=QDQ^T, where Q is an orthogonal matrix with Q^TQ=QQ^T=I
/// and D is the diagonal matrix of eigenvalues of A.
template<class MatrixStorage>
class symm_eigenvalue_decomposition:
	public solver_expression<
		symm_eigenvalue_decomposition<MatrixStorage>, 
		typename MatrixStorage::device_type
>{
private:
	typedef typename MatrixStorage::value_type value_type;
	typedef typename vector_temporary<MatrixStorage>::type VectorStorage;
public:
	typedef typename MatrixStorage::device_type device_type;
	symm_eigenvalue_decomposition(){}
	template<class E>
	symm_eigenvalue_decomposition(matrix_expression<E,device_type> const& e){
		decompose(e);
	}
	
	template<class E>
	void decompose(matrix_expression<E,device_type> const& e){
		REMORA_SIZE_CHECK(e().size1() ==  e().size2());
		m_eigenvectors.resize(e().size1(),e().size1());
		m_eigenvalues.resize(e().size1());
		noalias(m_eigenvectors) = e;

		kernels::syev(m_eigenvectors,m_eigenvalues);
	}
	
	MatrixStorage const& Q()const{
		return m_eigenvectors;
	}
	VectorStorage const& D()const{
		return m_eigenvalues;
	}
	
	
	template<class MatB>
	void solve(matrix_expression<MatB,device_type>& B, left)const{
		B() = Q() % to_diagonal(elem_inv(D()))% trans(Q()) % B;
	}
	template<class MatB>
	void solve(matrix_expression<MatB,device_type>& B, right)const{
		auto transB = trans(B);
		solve(transB,left());
	}
	template<class VecB>
	void solve(vector_expression<VecB,device_type>&b, left)const{
		b() = Q() % safe_div(trans(Q()) % b,D() ,0.0);
	}
	
	template<class VecB>
	void solve(vector_expression<VecB,device_type>&b, right)const{
		solve(b,left());
	}
	
	template<typename Archive>
	void serialize( Archive & ar, const std::size_t version ) {
		ar & m_eigenvectors;
		ar & m_eigenvalues;
	}
private:
	MatrixStorage m_eigenvectors;
	VectorStorage m_eigenvalues;
};




template<class MatrixStorage>
class pivoting_lu_decomposition:
public solver_expression<
	pivoting_lu_decomposition<MatrixStorage>, 
	typename MatrixStorage::device_type
>{
public:
	typedef typename MatrixStorage::device_type device_type;
	template<class E>
	pivoting_lu_decomposition(matrix_expression<E,device_type> const& e)
	:m_factor(e), m_permutation(e().size1()){
		kernels::getrf(m_factor,m_permutation);
	}
	
	MatrixStorage const& factor()const{
		return m_factor;
	}
	
	permutation_matrix const& permutation() const{
		return m_permutation;
	}
	
	template<class MatB>
	void solve(matrix_expression<MatB,device_type>& B, left)const{
		swap_rows(m_permutation,B);
		kernels::trsm<unit_lower,left>(m_factor,B);
		kernels::trsm<upper,left>(m_factor,B);
	}
	template<class MatB>
	void solve(matrix_expression<MatB,device_type>& B, right)const{
		kernels::trsm<upper,right>(m_factor,B);
		kernels::trsm<unit_lower,right>(m_factor,B);
		swap_columns_inverted(m_permutation,B);
	}
	template<class VecB>
	void solve(vector_expression<VecB,device_type>& b, left)const{
		swap_rows(m_permutation,b);
		kernels::trsv<unit_lower,left>(m_factor,b);
		kernels::trsv<upper,left>(m_factor,b);
	}
	template<class VecB>
	void solve(vector_expression<VecB,device_type>& b, right)const{
		kernels::trsv<upper,right>(m_factor,b);
		kernels::trsv<unit_lower,right>(m_factor,b);
		swap_rows_inverted(m_permutation,b);
	}
private:
	MatrixStorage m_factor;
	permutation_matrix m_permutation;
};


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
	public solver_expression<symm_pos_semi_definite_solver<MatrixStorage>, typename MatrixStorage::device_type>{
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
	
	std::size_t rank()const{
		return m_rank;
	}
	
	//compute C so that A^dagger = CC^T
	//where A^dagger is the moore-penrose inverse
	// m must be of size rank x n
	template<class Mat>
	void compute_inverse_factor(matrix_expression<Mat,device_type>& C)const{
		REMORA_SIZE_CHECK(C().size1() == m_rank);
		REMORA_SIZE_CHECK(C().size2() == m_factor.size1());
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
	void solve(matrix_expression<MatB,device_type>& B, left)const{
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
	void solve(matrix_expression<MatB,device_type>& B, right)const{
		//compute using symmetry of the system of equations
		auto transB = trans(B);
		solve(transB, left());
	}
	template<class VecB, bool Left>
	void solve(vector_expression<VecB,device_type>& b, system_tag<Left>)const{
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
	std::size_t m_rank;
	MatrixStorage m_factor;
	cholesky_decomposition<MatrixStorage> m_cholesky;
	permutation_matrix m_permutation;
};

template<class MatA>
class cg_solver:public solver_expression<cg_solver<MatA>, typename MatA::device_type>{
public:
	typedef typename MatA::const_closure_type matrix_closure_type;
	typedef typename MatA::device_type device_type;
	cg_solver(matrix_closure_type const& e, double epsilon = 1.e-10, unsigned int max_iterations = 0)
	:m_expression(e), m_epsilon(epsilon), m_max_iterations(max_iterations){}
	
	template<class MatB>
	void solve(
		matrix_expression<MatB,device_type>& B, left,
		double epsilon, unsigned max_iterations
	)const{
		typename matrix_temporary<MatB>::type X = B;
		cg(m_expression,X, B, epsilon, max_iterations);
		noalias(B) = X;
	}
	template<class MatB>
	void solve(
		matrix_expression<MatB,device_type>& B, right,
		double epsilon, unsigned max_iterations
	)const{
		auto transB = trans(B);
		typename transposed_matrix_temporary<MatB>::type X = transB;
		cg(m_expression,X,transB, epsilon, max_iterations);
		noalias(transB) = X;
	}
	template<class VecB, bool Left>
	void solve(
		vector_expression<VecB,device_type>&b, system_tag<Left>, 
		double epsilon, unsigned max_iterations
	)const{
		typename vector_temporary<VecB>::type x = b;
		cg(m_expression,x,b,epsilon,max_iterations);
		noalias(b) = x;
	}
	
	template<class VecB, bool Left>
	void solve(vector_expression<VecB,device_type>&b, system_tag<Left> tag)const{
		solve(b, tag, m_epsilon, m_max_iterations);
	}
	template<class MatB, bool Left>
	void solve(matrix_expression<MatB,device_type>&B, system_tag<Left> tag)const{
		solve(B, tag, m_epsilon, m_max_iterations);
	}
private:
	template<class MatT, class VecB, class VecX>
	void cg(
		matrix_expression<MatT, device_type> const& A,
		vector_expression<VecX, device_type>& x,
		vector_expression<VecB, device_type> const& b,
		double epsilon,
		unsigned int max_iterations
	)const{
		REMORA_SIZE_CHECK(A().size1() == A().size2());
		REMORA_SIZE_CHECK(A().size1() == b().size());
		REMORA_SIZE_CHECK(A().size1() == x().size());
		typedef typename vector_temporary<VecX>::type vector_type;
		
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
	)const{
		REMORA_SIZE_CHECK(A().size1() == A().size2());
		REMORA_SIZE_CHECK(A().size1() == B().size1());
		REMORA_SIZE_CHECK(A().size1() == X().size1());
		REMORA_SIZE_CHECK(B().size2() == X().size2());
		typedef typename vector_temporary<MatX>::type vector_type;
		typedef typename matrix_temporary<MatX>::type matrix_type;
		
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
	double m_epsilon;
	unsigned int m_max_iterations;
};


/////////////////////////////////////////////////////////////////
////Traits connecting decompositions/solvers with tags
/////////////////////////////////////////////////////////////////
struct symm_pos_def{ typedef symm_pos_def transposed_orientation;};
struct symm_semi_pos_def{ typedef symm_semi_pos_def transposed_orientation;};
struct indefinite_full_rank{ typedef indefinite_full_rank transposed_orientation;};
struct conjugate_gradient{
	typedef conjugate_gradient transposed_orientation;
	double epsilon;
	unsigned max_iterations;
	conjugate_gradient(double epsilon = 1.e-10, unsigned max_iterations = 0)
	:epsilon(epsilon), max_iterations(max_iterations){}
};


namespace detail{
template<class MatA, class SolverTag>
struct solver_traits;

template<class MatA, bool Upper, bool Unit>
struct solver_traits<MatA,triangular_tag<Upper,Unit> >{
	class type: public solver_expression<type,typename MatA::device_type>{
	public:
		typedef typename MatA::const_closure_type matrix_closure_type;
		typedef typename MatA::device_type device_type;
		type(matrix_closure_type const& e, triangular_tag<Upper,Unit>):m_matrix(e){}
		
		template<class MatB, bool Left>
		void solve(matrix_expression<MatB, device_type>& B, system_tag<Left> ){
			kernels::trsm<triangular_tag<Upper,Unit>,system_tag<Left> >(m_matrix,B);
		}
		template<class VecB, bool Left>
		void solve(vector_expression<VecB, device_type>& b, system_tag<Left> ){
			kernels::trsv<triangular_tag<Upper,Unit>,system_tag<Left> >(m_matrix,b);
		}
	private:
		matrix_closure_type m_matrix;
	};
};
	
template<class MatA>
struct solver_traits<MatA,symm_pos_def>{
	struct type : public cholesky_decomposition<typename matrix_temporary<MatA>::type>{
		template<class M>
		type(M const& m, symm_pos_def)
		:cholesky_decomposition<typename matrix_temporary<MatA>::type>(m){}
	};
};

template<class MatA>
struct solver_traits<MatA,indefinite_full_rank>{
	struct type : public pivoting_lu_decomposition<typename matrix_temporary<MatA>::type>{
		template<class M>
		type(M const& m, indefinite_full_rank)
		:pivoting_lu_decomposition<typename matrix_temporary<MatA>::type>(m){}
	};
};

template<class MatA>
struct solver_traits<MatA,symm_semi_pos_def>{
	struct type : public symm_pos_semi_definite_solver<typename matrix_temporary<MatA>::type>{
		template<class M>
		type(M const& m, symm_semi_pos_def)
		:symm_pos_semi_definite_solver<typename matrix_temporary<MatA>::type>(m){}
	};
};

template<class MatA>
struct solver_traits<MatA,conjugate_gradient>{
	struct type : public cg_solver<MatA>{
		template<class M>
		type(M const& m, conjugate_gradient t):cg_solver<MatA>(m,t.epsilon,t.max_iterations){}
	};
};

}

template<class MatA, class SolverType>
struct solver:public detail::solver_traits<MatA,SolverType>::type{
	template<class M>
	solver(M const& m, SolverType t = SolverType()): detail::solver_traits<MatA,SolverType>::type(m,t){}
};


}
#endif
