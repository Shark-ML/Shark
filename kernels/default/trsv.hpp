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

#include "../../detail/traits.hpp" //structure tags, expression types etc
#include "../../assignment.hpp" //plus_assign
#include "../dot.hpp" //dot kernel
#include "../../proxy_expressions.hpp" //range, row, transpose

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
	REMORA_SIZE_CHECK(A().size1() == A().size2());
	REMORA_SIZE_CHECK(A().size2() == b().size());

	typedef typename MatA::value_type value_type;
	typedef device_traits<cpu_tag>::multiply_and_add<value_type> MultAdd;
	std::size_t size = b().size();
	for (std::size_t n = 0; n != size; ++ n) {
		if(!Unit){
			if(A()(n, n) == value_type()){//matrix is singular
				throw std::invalid_argument("[TRSV] Matrix is singular!");
			}
			b()(n) /= A()(n, n);
		}
		if (b()(n) != value_type/*zero*/()){
			auto blower = subrange(b,n+1,size);
			auto colAlower = subrange(column(A,n),n+1,size);
			kernels::assign(blower, colAlower, MultAdd(-b()(n)));
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
	REMORA_SIZE_CHECK(A().size1() == A().size2());
	REMORA_SIZE_CHECK(A().size2() == b().size());

	typedef typename V::value_type value_type;

	std::size_t size = b().size();
	for (std::size_t n = 0; n < size; ++ n) {
		value_type value;
		kernels::dot(subrange(b,0,n),subrange(row(A,n),0,n),value);
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
	REMORA_SIZE_CHECK(A().size1() == A().size2());
	REMORA_SIZE_CHECK(A().size2() == b().size());

	typedef typename MatA::value_type value_type;
	typedef device_traits<cpu_tag>::multiply_and_add<value_type> MultAdd;
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
			auto blower = subrange(b(),0,n);
			auto colAlower = subrange(column(A,n),0,n);
			kernels::assign(blower, colAlower, MultAdd(-b()(n)));
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
	REMORA_SIZE_CHECK(A().size1() == A().size2());
	REMORA_SIZE_CHECK(A().size2() == b().size());

	typedef typename MatA::value_type value_type;

	std::size_t size = A().size1();
	for (std::size_t i = 0; i < size; ++ i) {
		std::size_t n = size-i-1;
		value_type value;
		kernels::dot(subrange(b(),n+1,size),subrange(row(A,n),n+1,size),value);
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
	trsv_impl<Unit>(
		trans(A), b,
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
		typename MatA::orientation(), Side()
	);
}

}}
#endif
