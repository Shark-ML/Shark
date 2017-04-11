/*!
 *
 *
 * \brief       -
 *
 * \author      O. Krause
 * \date        2011
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
#ifndef REMORA_KERNELS_DEFAULT_TRSV_HPP
#define REMORA_KERNELS_DEFAULT_TRSV_HPP

#include "../../expression_types.hpp" //vector/matrix_expression
#include "../../assignment.hpp" //plus_assign
#include "../dot.hpp" //dot kernel
#include "../../detail/vector_proxy_classes.hpp" //vector_range
#include "../../detail/matrix_proxy_classes.hpp" //matrix_row, matrix_transpose
#include "../../detail/structure.hpp" //structure tags
#include <stdexcept> //exception when matrix is singular
#include <type_traits> //std::false_type marker for unoptimized

namespace remora {namespace bindings {

//Lower triangular(row-major) - vector
template<bool Unit, class MatA, class V>
void trsv_impl(
	matrix_expression<MatA, cpu_tag> const& A,
	vector_expression<V, cpu_tag> &b,
    lower, column_major, left
) {
	SIZE_CHECK(A().size1() == A().size2());
	SIZE_CHECK(A().size2() == b().size());

	typedef typename MatA::value_type value_type;
	typedef matrix_transpose<typename const_expression<MatA>::type> TransA;
	TransA transA(A());
	std::size_t size = b().size();
	for (std::size_t n = 0; n != size; ++ n) {
		if(!Unit){
			if(A()(n, n) == value_type()){//matrix is singular
				throw std::invalid_argument("[TRSV] Matrix is singular!");
			}
			b()(n) /= A()(n, n);
		}
		if (b()(n) != value_type/*zero*/()){
			matrix_row<TransA> colA = row(transA,n);
			vector_range<V> blower(b(),n+1,size);
			vector_range<matrix_row<TransA> > colAlower(colA,n+1,size);
			plus_assign(blower,colAlower,-b()(n));
		}
	}
}
//Lower triangular(column-major) - vector
template<bool Unit, class MatA, class V>
void trsv_impl(
	matrix_expression<MatA, cpu_tag> const& A,
	vector_expression<V, cpu_tag> &b,
    lower, row_major, left
) {
	SIZE_CHECK(A().size1() == A().size2());
	SIZE_CHECK(A().size2() == b().size());

	typedef typename MatA::value_type value_type;

	std::size_t size = b().size();
	for (std::size_t n = 0; n < size; ++ n) {
		typedef matrix_row<typename const_expression<MatA>::type> RowA;
		RowA matRow(A(),n);
		vector_range<V> blower(b(),0,n);
		vector_range<RowA> matRowLower(matRow,0,n);
		value_type value;
		kernels::dot(blower,matRowLower,value);
		b()(n) -= value;
		if(!Unit){
			if(A()(n, n) == value_type()){//matrix is singular
				throw std::invalid_argument("[TRSV] Matrix is singular!");
			}
			b()(n) /= A()(n, n);
		}
	}
}

//upper triangular(column-major)-vector
template<bool Unit, class MatA, class V>
void trsv_impl(
	matrix_expression<MatA, cpu_tag> const& A,
	vector_expression<V, cpu_tag> &b,
    upper, column_major, left
) {
	SIZE_CHECK(A().size1() == A().size2());
	SIZE_CHECK(A().size2() == b().size());

	typedef typename MatA::value_type value_type;
	typedef matrix_transpose<typename const_expression<MatA>::type> TransA;
	TransA transA(A());
	std::size_t size = b().size();
	for (std::size_t i = 0; i < size; ++ i) {
		std::size_t n = size-i-1;
		if(!Unit){
			if(A()(n, n) == value_type()){//matrix is singular
				throw std::invalid_argument("[TRSV] Matrix is singular!");
			}
			b()(n) /= A()(n, n);
		}
		if (b()(n) != value_type/*zero*/()) {
			matrix_row<TransA> colA = row(transA,n);
			vector_range<V> blower(b(),0,n);
			vector_range<matrix_row<TransA> > colAlower(colA,0,n);
			plus_assign(blower,colAlower, -b()(n));
		}
	}
}
//upper triangular(row-major)-vector
template<bool Unit, class MatA, class V>
void trsv_impl(
	matrix_expression<MatA, cpu_tag> const& A,
	vector_expression<V, cpu_tag> &b,
    upper, row_major, left
) {
	SIZE_CHECK(A().size1() == A().size2());
	SIZE_CHECK(A().size2() == b().size());

	typedef typename MatA::value_type value_type;

	std::size_t size = A().size1();
	for (std::size_t i = 0; i < size; ++ i) {
		std::size_t n = size-i-1;
		typedef matrix_row<typename const_expression<MatA>::type> RowA;
		RowA matRow(A(),n);
		vector_range<V> blower(b(),n+1,size);
		vector_range<RowA> matRowLower(matRow,n+1,size);
		value_type value;
		kernels::dot(blower,matRowLower,value);
		b()(n) -= value;
		if(!Unit){
			if(A()(n, n) == value_type()){//matrix is singular
				throw std::invalid_argument("[TRSV] Matrix is singular!");
			}
			b()(n) /= A()(n, n);
		}
	}
}


//right is mapped onto left via transposing A
template<bool Unit, class Triangular, class Orientation, class MatA, class V>
void trsv_impl(
	matrix_expression<MatA, cpu_tag> const& A,
	vector_expression<V, cpu_tag> &b,
    Triangular, Orientation, right
) {
	matrix_transpose<typename const_expression<MatA>::type> transA(A());
	trsv_impl<Unit>(
		transA, b,
		typename Triangular::transposed_orientation(),
		typename Orientation::transposed_orientation(),
		left()
	);
}

//dispatcher
template <class Triangular, class Side,typename MatA, typename V>
void trsv(
	matrix_expression<MatA, cpu_tag> const& A,
	vector_expression<V, cpu_tag> & b,
	std::false_type//unoptimized
){
	trsv_impl<Triangular::is_unit>(
		A, b,
		triangular_tag<Triangular::is_upper,false>(),
		typename MatA::orientation(), Side());
}

}}
#endif
