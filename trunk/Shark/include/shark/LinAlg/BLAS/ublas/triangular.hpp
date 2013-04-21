#ifndef SHARK_BLAS_TRIANGULAR_HPP
#define SHARK_BLAS_TRIANGULAR_HPP

#include <shark/LinAlg/BLAS/ublas/matrix.hpp>
#include <shark/LinAlg/BLAS/ublas/matrix_expression.hpp>
// Iterators based on ideas of Jeremy Siek

namespace shark {
namespace blas {

//Lower triangular(row-major) - vector
template<bool Unit, class E1, class E2>
void inplace_solve(
	matrix_expression<E1> const& e1, vector_expression<E2> &e2,
        lower_tag, column_major_tag
) {
	SIZE_CHECK(e1().size1() == e1().size2());
	SIZE_CHECK(e1().size2() == e2().size());
	
	typedef typename E1::value_type value_type;
	
	std::size_t size = e2().size();
	for (std::size_t n = 0; n < size; ++ n) {
		if(!Unit){
			RANGE_CHECK(e1()(n, n) != value_type());//matrix is singular
			e2()(n) /= e1()(n, n);
		}
		if (e2()(n) != value_type/*zero*/()){
			matrix_column<E1 const> col = column(e1(),n);
			noalias(subrange(e2(),n+1,size)) -= e2()(n) * subrange(col,n+1,size);
		}
	}
}
//Lower triangular(column-major) - vector
template<bool Unit, class E1, class E2>
void inplace_solve(
	matrix_expression<E1> const& e1, vector_expression<E2> &e2,
        lower_tag, row_major_tag
) {
	SIZE_CHECK(e1().size1() == e1().size2());
	SIZE_CHECK(e1().size2() == e2().size());
	
	typedef typename E1::value_type value_type;
	
	std::size_t size = e2().size();
	for (std::size_t n = 0; n < size; ++ n) {
		matrix_row<E1 const> matRow = row(e1(),n);
		e2()(n) -= inner_prod(subrange(matRow,0,n),subrange(e2(),0,n));

		if(!Unit){
			RANGE_CHECK(e1()(n, n) != value_type());//matrix is singular
			e2()(n) /= e1()(n, n);
		}
	}
}

//upper triangular(column-major)-vector
template<bool Unit, class E1, class E2>
void inplace_solve(
	matrix_expression<E1> const& e1, vector_expression<E2> &e2,
        upper_tag, column_major_tag
) {
	SIZE_CHECK(e1().size1() == e1().size2());
	SIZE_CHECK(e1().size2() == e2().size());
	
	typedef typename E1::value_type value_type;
	
	std::size_t size = e2().size();
	for (std::size_t i = 0; i < size; ++ i) {
		std::size_t n = size-i-1;
		if(!Unit){
			RANGE_CHECK(e1()(n, n) != value_type());//matrix is singular
			e2()(n) /= e1()(n, n);
		}
		if (e2()(n) != value_type/*zero*/()) {
			matrix_column<E1 const> col = column(e1(),n);
			noalias(subrange(e2(),0,n)) -= e2()(n) * subrange(col,0,n);
		}
	}
}
//upper triangular(row-major)-vector
template<bool Unit, class E1, class E2>
void inplace_solve(
	matrix_expression<E1> const& e1, vector_expression<E2> &e2,
        upper_tag, row_major_tag
) {
	SIZE_CHECK(e1().size1() == e1().size2());
	SIZE_CHECK(e1().size2() == e2().size());
	
	typedef typename E1::value_type value_type;
	
	std::size_t size = e1().size1();
	for (std::size_t i = 0; i < size; ++ i) {
		std::size_t n = size-i-1;
		matrix_row<E1 const> matRow = row(e1(),n);
		e2()(n) -= inner_prod(subrange(matRow,n+1,size),subrange(e2(),n+1,size));
		if(!Unit){
			RANGE_CHECK(e1()(n, n) != value_type());//matrix is singular
			e2()(n) /= e1()(n, n);
		}
	}
}

//public interface

// Dispatcher for Systems of the form Ax=b
template<class E1, class E2>
void inplace_solve(matrix_expression<E1> const& e1, vector_expression<E2> &e2, lower_tag tag){
	typedef typename E1::orientation_category orientation_category;
	inplace_solve<false>(e1, e2, lower_tag(), orientation_category());
}
template<class E1, class E2>
void inplace_solve(matrix_expression<E1> const& e1, vector_expression<E2> &e2, unit_lower_tag){
	typedef typename E1::orientation_category orientation_category;
	inplace_solve<true>(e1, e2, lower_tag(), orientation_category());
}

template<class E1, class E2>
void inplace_solve(matrix_expression<E1> const& e1, vector_expression<E2> &e2, upper_tag){
	typedef typename E1::orientation_category orientation_category;
	inplace_solve<false>(e1, e2,upper_tag(), orientation_category());
}
template<class E1, class E2>
void inplace_solve(matrix_expression<E1> const& e1, vector_expression<E2> &e2, unit_upper_tag){
	typedef typename E1::orientation_category orientation_category;
	inplace_solve<true>(e1, e2,upper_tag(), orientation_category());
}

// Dispatcher for Systems of the form x^TA=b
template<class E1, class E2>
void inplace_solve(vector_expression<E1> &e1, const matrix_expression<E2> &e2, lower_tag) {
	inplace_solve(trans(e2), e1, upper_tag());
}
template<class E1, class E2>
void inplace_solve(vector_expression<E1> &e1, const matrix_expression<E2> &e2, upper_tag) {
	inplace_solve(trans(e2), e1, lower_tag());
}
template<class E1, class E2>
void inplace_solve(vector_expression<E1> &e1, const matrix_expression<E2> &e2, unit_lower_tag) {
	inplace_solve(trans(e2), e1, unit_upper_tag());
}
template<class E1, class E2>
void inplace_solve(vector_expression<E1> &e1, const matrix_expression<E2> &e2, unit_upper_tag) {
	inplace_solve(trans(e2), e1, unit_lower_tag());
}


// Operations:
//  k * n * (n - 1) / 2 + k * n = k * n * (n + 1) / 2 multiplications,
//  k * n * (n - 1) / 2 additions

// Lower triangular(column major) - matrix
template<bool Unit, class E1, class E2>
void inplace_solve(
	matrix_expression<E1> const& e1, matrix_expression<E2> &e2, 
	lower_tag, column_major_tag
) {
	SIZE_CHECK(e1().size1() == e1().size2());
	SIZE_CHECK(e1().size2() == e2().size1());
	
	typedef typename E1::value_type value_type;
	
	std::size_t size1 = e2().size1();
	std::size_t size2 = e2().size2();
	for (std::size_t n = 0; n < size1; ++ n) {
		matrix_column<E1 const> columnTriangular = column(e1(),n);
		for (std::size_t l = 0; l < size2; ++ l) {
			if(!Unit){
				RANGE_CHECK(e1()(n, n) != value_type());//matrix is singular
				e2()(n, l) /= e1()(n, n);
			}
			if (e2()(n, l) != value_type/*zero*/()) {
				matrix_column<E2> columnMatrix = column(e2(),l);
				noalias(subrange(columnMatrix,n+1,size1)) -= e2()(n,l) * subrange(columnTriangular,n+1,size1);
			}
		}
	}
}
// Lower triangular(row major) - matrix
template<bool Unit, class E1, class E2>
void inplace_solve(
	matrix_expression<E1> const& e1, matrix_expression<E2> &e2, 
	lower_tag, row_major_tag
) {
	SIZE_CHECK(e1().size1() == e1().size2());
	SIZE_CHECK(e1().size2() == e2().size1());
	
	typedef typename E1::value_type value_type;
	
	std::size_t size1 = e2().size1();
	for (std::size_t n = 0; n < size1; ++ n) {
		for (std::size_t m = 0; m < n; ++m) {
			noalias(row(e2(),n)) -= e1()(n,m)*row(e2(),m);
		}
		if(!Unit){
			RANGE_CHECK(e1()(n, n) != value_type());//matrix is singular
			row(e2(),n)/=e1()(n, n);
		}
	}
}

//Upper triangular(column major) - matrix
template<bool Unit, class E1, class E2>
void inplace_solve(
	matrix_expression<E1> const& e1, matrix_expression<E2> &e2,
        upper_tag, column_major_tag
) {
	SIZE_CHECK(e1().size1() == e1().size2());
	SIZE_CHECK(e1().size2() == e2().size1());
	
	typedef typename E1::value_type value_type;
	
	std::size_t size1 = e2().size1();
	std::size_t size2 = e2().size2();
	for (std::size_t i = 0; i < size1; ++ i) {
		std::size_t n = size1-i-1;
		matrix_column<E1 const> columnTriangular = column(e1(),n);
		if(!Unit){
			RANGE_CHECK(e1()(n, n) != value_type());//matrix is singular
			row(e2(),n) /= e1()(n, n);
		}
		for (std::size_t l = 0; l < size2; ++ l) {
			if (e2()(n, l) != value_type/*zero*/()) {
				matrix_column<E2> columnMatrix = column(e2(),l);
				noalias(subrange(columnMatrix,0,n)) -= e2()(n,l) * subrange(columnTriangular,0,n);
			}
		}
	}
}

//Upper triangular(row major) - matrix
template<bool Unit, class E1, class E2>
void inplace_solve(
	matrix_expression<E1> const& e1, matrix_expression<E2> &e2,
        upper_tag, row_major_tag
) {
	SIZE_CHECK(e1().size1() == e1().size2());
	SIZE_CHECK(e1().size2() == e2().size1());
	
	typedef typename E1::value_type value_type;
	
	std::size_t size1 = e2().size1();
	for (std::size_t i = 0; i < size1; ++ i) {
		std::size_t n = size1-i-1;
		for (std::size_t m = n+1; m < size1; ++m) {
			noalias(row(e2(),n)) -= e1()(n,m)*row(e2(),m);
		}
		if(!Unit){
			RANGE_CHECK(e1()(n, n) != value_type());//matrix is singular
			row(e2(),n)/=e1()(n, n);
		}
	}
}


// Dispatcher
template<class E1, class E2>
void inplace_solve(matrix_expression<E1> const& e1, matrix_expression<E2> &e2, lower_tag) {
	typedef typename E1::orientation_category orientation_category;
	inplace_solve<false>(e1, e2, lower_tag(), orientation_category());
}
template<class E1, class E2>
void inplace_solve(matrix_expression<E1> const& e1, matrix_expression<E2> &e2, upper_tag) {
	typedef typename E1::orientation_category orientation_category;
	inplace_solve<false>(e1, e2, upper_tag(), orientation_category());
}
template<class E1, class E2>
void inplace_solve(matrix_expression<E1> const& e1, matrix_expression<E2> &e2, unit_lower_tag) {
	typedef typename E1::orientation_category orientation_category;
	inplace_solve<true>(e1, e2, lower_tag(), orientation_category());
}
template<class E1, class E2>
void inplace_solve(matrix_expression<E1> const& e1, matrix_expression<E2> &e2, unit_upper_tag) {
	typedef typename E1::orientation_category orientation_category;
	inplace_solve<true>(e1, e2, upper_tag(), orientation_category());
}


}
}

#endif
