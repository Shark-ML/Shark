/*!
 * \brief       Kernels for matrix-expression assignments
 * 
 * \author      O. Krause
 * \date        2013
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
#ifndef REMORA_KERNELS_MATRIX_ASSIGN_HPP
#define REMORA_KERNELS_MATRIX_ASSIGN_HPP

#include "default/matrix_assign.hpp"
#ifdef REMORA_USE_GPU
#include "gpu/matrix_assign.hpp"
#endif
#include <type_traits>

namespace remora {namespace kernels{
	
template<class F, class M, class Device>
void apply(
	matrix_expression<M, Device>& m, 
	F const&f
){
	if(m().size1() == 0|| m().size2() == 0) return;
	typedef typename M::orientation orientation;
	bindings::matrix_apply(m, f, orientation());
}	

//////////////////////////////////////////////////////
////Scalar Assignment to Matrix
/////////////////////////////////////////////////////

// Dispatcher
template<class F, class M, class Device>
void assign(
	matrix_expression<M, Device>& m, 
	typename M::value_type t
){
	if(m().size1() == 0|| m().size2() == 0) return;
	typedef typename M::orientation orientation;
	bindings::matrix_assign<F> (m, t, orientation());
}

/////////////////////////////////////////////////////////////////
//////Matrix Assignment implementing op=
////////////////////////////////////////////////////////////////

namespace detail{

//general dispatcher: if the second argument has an unknown orientation
// it is chosen the same as the first one
template<class M, class E, class EOrientation, class TagE, class TagM, class Device>
void matrix_assign(
	matrix_expression<M, Device>& m, 
	matrix_expression<E, Device> const& e,
	row_major, EOrientation ,TagE tagE, TagM tagM
) {
	typedef typename std::conditional<
		std::is_same<EOrientation, unknown_orientation>::value,
		row_major,
		typename E::orientation
	>::type Orientation;
	bindings::matrix_assign(m, e, typename M::orientation(), Orientation(), tagE, tagM);
}

//general dispatcher: if the first argument is column major, we transpose the whole expression
//so that it  is row-major, this saves us to implment everything twice.
template<class M, class E,class EOrientation, class TagE, class TagM, class Device>
void matrix_assign(
	matrix_expression<M, Device>& m, 
	matrix_expression<E, Device> const& e,
	column_major, EOrientation,TagE tagE, TagM tagM
) {
	typedef typename M::orientation::transposed_orientation::orientation TMOrientation;
	typedef typename E::orientation::transposed_orientation::orientation TEOrientation;
	auto transM = trans(m);
	auto transE = trans(e);
	//dispatch to first version
	matrix_assign(transM, transE, TMOrientation(), TEOrientation(), tagE, tagM);
}
}

// Dispatcher
template<class M, class E, class Device>
void assign(
	matrix_expression<M, Device>& m,
	matrix_expression<E, Device> const& e
){
	REMORA_SIZE_CHECK(m().size1() == e().size1());
	REMORA_SIZE_CHECK(m().size2() == e().size2());
	if(m().size1() == 0|| m().size2() == 0) return;
	typedef typename M::orientation::orientation MOrientation;
	typedef typename E::orientation::orientation EOrientation;
	typedef typename M::evaluation_category::tag MCategory;
	typedef typename E::evaluation_category::tag ECategory;
	detail::matrix_assign(m, e, MOrientation(), EOrientation(), MCategory(), ECategory());
}


///////////////////////////////////////////////////////////////////////////////////////////
//////Matrix Assignment With Functor implementing +=,-=...
///////////////////////////////////////////////////////////////////////////////////////////

namespace detail{
// general dispatcher: if the second argument has an unknown orientation
// it is chosen the same as the first one
template<class F, class M, class E, class EOrientation, class TagE, class TagM, class Device>
void matrix_assign_functor(
	matrix_expression<M, Device>& m, 
	matrix_expression<E, Device> const& e,
	F f,
	row_major, EOrientation ,TagE tagE, TagM tagM
) {
	typedef typename std::conditional<
		std::is_same<EOrientation, unknown_orientation>::value,
		row_major,
		typename E::orientation
	>::type Orientation;
	bindings::matrix_assign_functor(m, e, f, typename M::orientation(), Orientation(), tagE, tagM);
}

//general dispatcher: if the first argument is column major, we transpose the whole expression
//so that it  is row-major, this saves us to implment everything twice.
template<class F, class M, class E,class EOrientation, class TagE, class TagM, class Device>
void matrix_assign_functor(
	matrix_expression<M, Device>& m, 
	matrix_expression<E, Device> const& e,
	F f,
	column_major, EOrientation,TagE tagE, TagM tagM
) {
	typedef typename M::orientation::transposed_orientation::orientation TMOrientation;
	typedef typename E::orientation::transposed_orientation::orientation TEOrientation;
	
	auto transM = trans(m);
	auto transE = trans(e);
	matrix_assign_functor(transM, transE, f, TMOrientation(), TEOrientation(), tagE, tagM);
}

}


//First Level Dispatcher, dispatches by orientation
template<class F, class M, class E, class Device>
void assign(
	matrix_expression<M, Device>& m, 
	matrix_expression<E, Device> const&e,
	F const& f
){
	REMORA_SIZE_CHECK(m().size1()  == e().size1());
	REMORA_SIZE_CHECK(m().size2()  == e().size2());
	if(m().size1() == 0|| m().size2() == 0) return;
	typedef typename M::orientation::orientation MOrientation;
	typedef typename E::orientation::orientation EOrientation;
	typedef typename M::evaluation_category::tag MCategory;
	typedef typename E::evaluation_category::tag ECategory;
	detail::matrix_assign_functor(m, e, f, MOrientation(), EOrientation(), MCategory(), ECategory());
}

}}

#endif
