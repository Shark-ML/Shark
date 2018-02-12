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
#ifndef REMORA_KERNELS_DEFAULT_MATRIX_ASSIGN_HPP
#define REMORA_KERNELS_DEFAULT_MATRIX_ASSIGN_HPP

#include "../vector_assign.hpp"
#include "../../proxy_expressions.hpp"
#include "../../detail/traits.hpp"
#include <algorithm>
#include <vector>
#include <iostream>
namespace remora{namespace bindings{

// Explicitly iterating row major
template<class F, class M>
void matrix_apply(
	matrix_expression<M, cpu_tag>& m,
	F const& f,
	row_major
){
	for(std::size_t i = 0; i != m().size1(); ++i){
		auto rowM = row(m,i);
		kernels::apply(rowM,f);
	}
}
// Explicitly iterating column major
template<class F, class M>
void matrix_apply(
	matrix_expression<M, cpu_tag>& m,
	F const& f,
	column_major
){
	for(std::size_t j = 0; j != m().size2(); ++j){
		auto columnM = column(m,j);
		kernels::apply(columnM,f);
	}
}
// Spcial case triangular packed - just calls the first two implementations.
template<class F, class M, class Orientation, class Triangular>
void matrix_apply(
	matrix_expression<M, cpu_tag>& m,
	F const& f,
	triangular<Orientation,Triangular>
){
	matrix_apply(m,f,Orientation());
}

//////////////////////////////////////////////////////
////Scalar Assignment to Matrix
/////////////////////////////////////////////////////


// Explicitly iterating row major
template<class F, class M, class Orientation>
void matrix_assign(
	matrix_expression<M, cpu_tag>& m,
	typename M::value_type t,
	Orientation
){
	F f;
	matrix_apply(m, [=](typename M::value_type x){return f(x,t);},Orientation());

}



/////////////////////////////////////////////////////////////////
//////Matrix Assignment implementing op=
////////////////////////////////////////////////////////////////

// direct assignment without functor
// the cases were both arguments have the same orientation and the left hand side
// is dense can be implemented using assign.
template<class M, class E,class TagE>
void matrix_assign(
	matrix_expression<M, cpu_tag>& m,
	matrix_expression<E, cpu_tag> const& e,
	row_major, row_major,dense_tag, TagE
) {
	for(std::size_t i = 0; i != m().size1(); ++i){
		auto rowM = row(m,i);
		kernels::assign(rowM,row(e,i));
	}
}

// direct assignment for sparse matrices with the same orientation
template<class M, class E>
void matrix_assign(
	matrix_expression<M, cpu_tag>& m,
	matrix_expression<E, cpu_tag> const& e,
	row_major, row_major,sparse_tag, sparse_tag
) {
	m().clear();
	for(std::size_t i = 0; i != m().size1(); ++i){
		auto m_pos = m().major_begin(i);
		auto end = e().major_end(i);
		for(auto it = e().major_begin(i); it != end; ++it){
			m_pos = m().set_element(m_pos,it.index(),*it);
		}
	}
}

//remain the versions where both arguments do not have the same orientation

//dense-dense case
template<class M, class E>
void matrix_assign(
	matrix_expression<M, cpu_tag>& m,
	matrix_expression<E, cpu_tag> const& e,
	row_major, column_major,dense_tag, dense_tag
) {
	//compute blockwise and wrelem the transposed block.
	std::size_t const blockSize = 8;
	typename M::value_type blockStorage[blockSize][blockSize];

	typedef typename M::size_type size_type;
	size_type size1 = m().size1();
	size_type size2 = m().size2();
	for (size_type iblock = 0; iblock < size1; iblock += blockSize){
		for (size_type jblock = 0; jblock < size2; jblock += blockSize){
			std::size_t blockSizei = std::min(blockSize,size1-iblock);
			std::size_t blockSizej = std::min(blockSize,size2-jblock);

			//compute block values
			for (size_type j = 0; j < blockSizej; ++j){
				for (size_type i = 0; i < blockSizei; ++i){
					blockStorage[i][j] = e()(iblock+i,jblock+j);
				}
			}

			//copy block in a different order to m
			for (size_type i = 0; i < blockSizei; ++i){
				for (size_type j = 0; j < blockSizej; ++j){
					m()(iblock+i,jblock+j) = blockStorage[i][j];
				}
			}
		}
	}
}

// dense-sparse
template<class M, class E>
void matrix_assign(
	matrix_expression<M, cpu_tag>& m,
	matrix_expression<E, cpu_tag> const& e,
	row_major, column_major,dense_tag, sparse_tag
) {
	for(std::size_t j = 0; j != m().size2(); ++j){
		auto columnM = column(m,j);
		kernels::assign(columnM,column(e,j));
	}
}

//sparse-sparse
template<class M, class E>
void matrix_assign(
	matrix_expression<M, cpu_tag>& m,
	matrix_expression<E, cpu_tag> const& e,
	row_major, column_major,sparse_tag,sparse_tag
) {
	//apply the transposition of e()
	//first evaluate e and fill the values into a vector which is then sorted by row_major order
	//this gives this algorithm a run time of  O(eval(e)+k*log(k))
	//where eval(e) is the time to evaluate and k*log(k) the number of non-zero elements
	typedef typename M::value_type value_type;
	typedef typename M::size_type size_type;
	typedef row_major::sparse_element<value_type> Element;
	std::vector<Element> elements;

	size_type size2 = m().size2();
	for(size_type j = 0; j != size2; ++j){
		auto pos_e = e().major_begin(j);
		auto end_e = e().major_end(j);
		for(; pos_e != end_e; ++pos_e){
			Element element;
			element.i = pos_e.index();
			element.j = j;
			element.value = *pos_e;
			elements.push_back(element);
		}
	}
	std::sort(elements.begin(),elements.end());
	//fill m with the contents
	m().clear();
	size_type num_elems = size_type(elements.size());
	for(size_type current = 0; current != num_elems;){
		//count elements in row and reserve enough space for it
		size_type row = elements[current].i;
		size_type row_end = current;
		while(row_end != num_elems && elements[row_end].i == row)
			++ row_end;
		m().major_reserve(row,row_end - current);

		//copy elements
		auto row_pos = m().major_begin(row);
		for(; current != row_end; ++current){
			row_pos = m().set_element(row_pos,elements[current].j,elements[current].value);
		}
	}
}

//triangular row_major,row_major
template<class M, class E, bool Upper, bool Unit>
void matrix_assign(
	matrix_expression<M, cpu_tag>& m,
	matrix_expression<E, cpu_tag> const& e,
	triangular<row_major,triangular_tag<Upper, false> >, triangular<row_major,triangular_tag<Upper, Unit> >,
	packed_tag, packed_tag
) {
	for(std::size_t i = 0; i != m().size1(); ++i){
		auto mpos = m().major_begin(i);
		auto epos = e().major_begin(i);
		auto eend = e().major_end(i);
		if(Unit && Upper){
			*mpos = 1;
			++mpos;
		}
		REMORA_SIZE_CHECK(mpos.index() == epos.index());
		for(; epos != eend; ++epos,++mpos){
			*mpos = *epos;
		}
		if(Unit && Upper){
			*mpos = 1;
		}
	}
}

////triangular row_major,column_major
//todo: this is suboptimal as we do strided access!!!!
template<class M, class E,class Triangular>
void matrix_assign(
	matrix_expression<M, cpu_tag>& m,
	matrix_expression<E, cpu_tag> const& e,
	triangular<row_major,Triangular>, triangular<column_major,Triangular>,
	packed_tag, packed_tag
) {
	for(std::size_t i = 0; i != m().size1(); ++i){
		auto mpos = m().major_begin(i);
		auto mend = m().major_end(i);
		for(; mpos!=mend; ++mpos){
			*mpos = e()(i,mpos.index());
		}
	}
}

///////////////////////////////////////////////////////////////////////////////////////////
//////Matrix Assignment With Functor implementing +=,-=...
///////////////////////////////////////////////////////////////////////////////////////////

//when both are row-major and target is dense we can map to vector case
template<class F, class M, class E, class TagE>
void matrix_assign_functor(
	matrix_expression<M, cpu_tag>& m,
	matrix_expression<E, cpu_tag> const& e,
	F f,
	row_major, row_major, dense_tag, TagE
) {
	for(std::size_t i = 0; i != m().size1(); ++i){
		auto rowM = row(m,i);
		kernels::assign(rowM,row(e,i),f);
	}
}
template<class F, class M, class E>
void matrix_assign_functor(
	matrix_expression<M, cpu_tag>& m,
	matrix_expression<E, cpu_tag> const& e,
	F f,
	row_major, row_major, sparse_tag, sparse_tag
) {
	typedef typename M::value_type value_type;
	value_type zero = value_type();
	typedef row_major::sparse_element<value_type> Element;
	std::vector<Element> elements;
	
	for(std::size_t i = 0; i != major_size(m); ++i){
		//first merge the two rows in elements using the functor
		
		elements.clear();
		auto m_pos = m().major_begin(i);
		auto m_end = m().major_end(i);
		auto e_pos = e().major_begin(i);
		auto e_end = e().major_end(i);
		
		while(m_pos != m_end && e_pos != e_end){
			if(m_pos.index() < e_pos.index()){
				elements.push_back({i,m_pos.index(), f(*m_pos, zero)});
				++m_pos;
			}else if( m_pos.index() == e_pos.index()){
				elements.push_back({i,m_pos.index(), f(*m_pos ,*e_pos)});
				++m_pos;
				++e_pos;
			}
			else{ //m_pos.index() > e_pos.index()
				elements.push_back({i,e_pos.index(), f(zero,*e_pos)});
				++e_pos;
			}
		}
		for(;m_pos != m_end;++m_pos){
			elements.push_back({i,m_pos.index(), f(*m_pos, zero)});
		}
		for(;e_pos != e_end; ++e_pos){
			elements.push_back({i,e_pos.index(), f(zero, *e_pos)});
		}
		
		//clear contents of m and fill with elements
		m().clear_range(m().major_begin(i),m().major_end(i));
		m().major_reserve(i,elements.size());
		m_pos = m().major_begin(i);
		for(auto elem: elements){
			m_pos = m().set_element(m_pos, elem.j, elem.value);
		}
	}
}
	

//we only need to implement the remaining versions for column major second argument

//dense-dense case
template<class F,class M, class E>
void matrix_assign_functor(
	matrix_expression<M, cpu_tag>& m,
	matrix_expression<E, cpu_tag> const& e,
	F f,
	row_major, column_major,dense_tag, dense_tag
) {
	//compute blockwise and wrelem the transposed block.
	std::size_t const blockSize = 16;
	typename M::value_type blockStorage[blockSize][blockSize];

	typedef typename M::size_type size_type;
	size_type size1 = m().size1();
	size_type size2 = m().size2();
	for (size_type iblock = 0; iblock < size1; iblock += blockSize){
		for (size_type jblock = 0; jblock < size2; jblock += blockSize){
			std::size_t blockSizei = std::min(blockSize,size1-iblock);
			std::size_t blockSizej = std::min(blockSize,size2-jblock);

			//fill the block with the values of e
			for (size_type j = 0; j < blockSizej; ++j){
				for (size_type i = 0; i < blockSizei; ++i){
					blockStorage[i][j] = e()(iblock+i,jblock+j);
				}
			}

			//compute block values and store in m
			for (size_type i = 0; i < blockSizei; ++i){
				for (size_type j = 0; j < blockSizej; ++j){
					m()(iblock+i,jblock+j) = f(m()(iblock+i,jblock+j), blockStorage[i][j]);
				}
			}
		}
	}
}

//dense-sparse case
template<class F,class M, class E>
void matrix_assign_functor(
	matrix_expression<M, cpu_tag>& m,
	matrix_expression<E, cpu_tag> const& e,
	F f,
	row_major, column_major,dense_tag, sparse_tag
) {
	for(std::size_t j = 0; j != m().size2(); ++j){
		auto columnM = column(m,j);
		kernels::assign(columnM,column(e,j),f);
	}
}

//sparse-sparse
template<class F,class M, class E>
void matrix_assign_functor(
	matrix_expression<M, cpu_tag>& m,
	matrix_expression<E, cpu_tag> const& e,
	F f,
	row_major, column_major,sparse_tag t,sparse_tag
) {
	typename matrix_temporary<M>::type eTrans = e;//explicit calculation of the transpose for now
	matrix_assign_functor(m,eTrans,f,row_major(),row_major(),t,t);
}


//kernels for triangular
template<class F, class M, class E, class Triangular, class Tag1, class Tag2>
void matrix_assign_functor(
	matrix_expression<M, cpu_tag>& m,
	matrix_expression<E, cpu_tag> const& e,
	F f,
	triangular<row_major,Triangular>, triangular<row_major,Triangular>,
	Tag1, Tag2
) {
	//there is nothing we can do if F does not leave the non-stored elements 0
	//this is the case for all current assignment functors, but you never know :)

	for(std::size_t i = 0; i != m().size1(); ++i){
		auto mpos = m().major_begin(i);
		auto epos = e().major_begin(i);
		auto mend = m().major_end(i);
		REMORA_SIZE_CHECK(mpos.index() == epos.index());
		for(; mpos!=mend; ++mpos,++epos){
			*mpos = f(*mpos,*epos);
		}
	}
}

//todo: this is suboptimal as we do strided access!!!!
template<class F, class M, class E, class Triangular, class Tag1, class Tag2>
void matrix_assign_functor(
	matrix_expression<M, cpu_tag>& m,
	matrix_expression<E, cpu_tag> const& e,
	F f,
	triangular<row_major,Triangular>, triangular<column_major,Triangular>,
	Tag1, Tag2
) {
	//there is nothing we can do, if F does not leave the non-stored elements 0

	for(std::size_t i = 0; i != m().size1(); ++i){
		auto mpos = m().major_begin(i);
		auto mend = m().major_end(i);
		for(; mpos!=mend; ++mpos){
			*mpos = f(*mpos,e()(i,mpos.index()));
		}
	}
}


}}

#endif
