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
#ifndef SHARK_LINALG_BLAS_KERNELS_DEFAULT_MATRIX_ASSIGN_HPP
#define SHARK_LINALG_BLAS_KERNELS_DEFAULT_MATRIX_ASSIGN_HPP

#include "../vector_assign.hpp"
#include "../../detail/traits.hpp"
#include <algorithm>
#include <vector>
namespace shark {namespace blas {namespace bindings{
	
//////////////////////////////////////////////////////
////Scalar Assignment to Matrix
/////////////////////////////////////////////////////


// Explicitly iterating row major
template<class F, class M>
void matrix_assign(
	matrix_expression<M, cpu_tag> &m, 
	typename M::value_type t, 
	row_major
){
	for(std::size_t i = 0; i != m().size1(); ++i){
		auto rowM = row(m,i);
		kernels::assign<F>(rowM,t);
	}
}
// Explicitly iterating column major
template<class F, class M>
void matrix_assign(
	matrix_expression<M, cpu_tag> &m, 
	typename M::value_type t, 
	column_major
){
	for(std::size_t j = 0; j != m().size2(); ++j){
		auto columnM = column(m,j);
		kernels::assign<F>(columnM,t);
	}
}
// Spcial case triangular packed - just calls the first two implementations.
template<class F, class M, class Orientation, class Triangular>
void matrix_assign(
	matrix_expression<M, cpu_tag> &m, 
	typename M::value_type t, 
	triangular<Orientation,Triangular>
){
	matrix_assign<F>(m,t,Orientation());
}


/////////////////////////////////////////////////////////////////
//////Matrix Assignment implementing op=
////////////////////////////////////////////////////////////////

//direct assignment without functor
//the cases were both arguments have the same orientation can be implemented using assign.
template<class M, class E,class TagE, class TagM>
void matrix_assign(
	matrix_expression<M, cpu_tag> &m, 
	matrix_expression<E, cpu_tag> const& e,
	row_major, row_major,TagE, TagM
) {
	for(std::size_t i = 0; i != m().size1(); ++i){
		auto rowM = row(m,i);
		kernels::assign(rowM,row(e,i));
	}
}

//remain the versions where both argumnts to not have the same orientation

//dense-dense case
template<class M, class E>
void matrix_assign(
	matrix_expression<M, cpu_tag> &m, 
	matrix_expression<E, cpu_tag> const& e,
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
	matrix_expression<M, cpu_tag> &m, 
	matrix_expression<E, cpu_tag> const& e,
	row_major, column_major,dense_tag, sparse_tag
) {
	for(std::size_t j = 0; j != m().size2(); ++j){
		auto columnM = column(m,j);
		kernels::assign(columnM,column(e,j));
	}
}


//sparse-dense
template<class M, class E>
void matrix_assign(
	matrix_expression<M, cpu_tag> &m, 
	matrix_expression<E, cpu_tag> const& e,
	row_major, column_major, sparse_tag, dense_tag
) {
	for(std::size_t i = 0; i != m().size1(); ++i){
		auto rowM = row(m,i);
		kernels::assign(rowM,row(e,i));
	}
}

//sparse-sparse
template<class M, class E>
void matrix_assign(
	matrix_expression<M, cpu_tag> &m, 
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
	size_type size1 = m().size1();
	for(size_type j = 0; j != size2; ++j){
		typename E::const_column_iterator pos_e = e().column_begin(j);
		typename E::const_column_iterator end_e = e().column_end(j);
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
	m().reserve(elements.size());//reserve a bit of space
	for(size_type current = 0; current != elements.size();){
		//count elements in row and reserve enough space for it
		size_type row = elements[current].i;
		size_type row_end = current;
		while(row_end != elements.size() && elements[row_end].i == row) 
			++ row_end;
		m().reserve_row(row,row_end - current);
		
		//copy elements
		typename M::row_iterator row_pos = m().row_begin(row); 
		for(; current != row_end; ++current){
			row_pos = m().set_element(row_pos,elements[current].j,elements[current].value);
			++row_pos;
		}
	}
}

//packed row_major,row_major
template<class M, class E,class Triangular>
void matrix_assign(
	matrix_expression<M, cpu_tag> &m, 
	matrix_expression<E, cpu_tag> const& e,
	triangular<row_major,Triangular>, triangular<row_major,Triangular>,
	packed_tag, packed_tag
) {
	typedef typename M::row_iterator MIter;
	typedef typename E::const_row_iterator EIter;
	
	for(std::size_t i = 0; i != m().size1(); ++i){
		MIter mpos = m().row_begin(i);
		EIter epos = e().row_begin(i);
		MIter mend = m().row_end(i);
		SIZE_CHECK(mpos.index() == epos.index());
		for(; mpos!=mend; ++mpos,++epos){
			*mpos = *epos;
		}
	}
}

////packed row_major,column_major
//todo: this is suboptimal as we do strided access!!!!
template<class M, class E,class Triangular>
void matrix_assign(
	matrix_expression<M, cpu_tag> &m, 
	matrix_expression<E, cpu_tag> const& e,
	triangular<row_major,Triangular>, triangular<column_major,Triangular>,
	packed_tag, packed_tag
) {
	typedef typename M::row_iterator MIter;

	for(std::size_t i = 0; i != m().size1(); ++i){
		MIter mpos = m().row_begin(i);
		MIter mend = m().row_end(i);
		for(; mpos!=mend; ++mpos){
			*mpos = e()(i,mpos.index());
		}
	}
}

///////////////////////////////////////////////////////////////////////////////////////////
//////Matrix Assignment With Functor implementing +=,-=...
///////////////////////////////////////////////////////////////////////////////////////////

//when both are row-major we can map to vector case
//this is not necessarily efficient if m is sparse.
template<class F, class M, class E, class TagE, class TagM>
void matrix_assign_functor(
	matrix_expression<M, cpu_tag> &m, 
	matrix_expression<E, cpu_tag> const& e,
	F f,
	row_major, row_major,TagM, TagE
) {
	for(std::size_t i = 0; i != m().size1(); ++i){
		auto rowM = row(m,i);
		kernels::assign(rowM,row(e,i),f);
	}
}

//we only need to implement the remaining versions for column major second argument

//dense-dense case
template<class F,class M, class E>
void matrix_assign_functor(
	matrix_expression<M, cpu_tag> &m, 
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
	matrix_expression<M, cpu_tag> &m, 
	matrix_expression<E, cpu_tag> const& e,
	F f,
	row_major, column_major,dense_tag, sparse_tag
) {
	for(std::size_t j = 0; j != m().size2(); ++j){
		auto columnM = column(m,j);
		kernels::assign(columnM,column(e,j),f);
	}
}

//sparse-dense case
template<class F,class M, class E>
void matrix_assign_functor(
	matrix_expression<M, cpu_tag> &m, 
	matrix_expression<E, cpu_tag> const& e,
	F f,
	row_major, column_major, sparse_tag, dense_tag
) {
	for(std::size_t i = 0; i != m().size1(); ++i){
		auto rowM = row(m,i);
		kernels::assign(rowM,row(e,i),f);
	}
}

//sparse-sparse
template<class F,class M, class E>
void matrix_assign_functor(
	matrix_expression<M, cpu_tag> &m, 
	matrix_expression<E, cpu_tag> const& e,
	F f,
	row_major, column_major,sparse_tag t,sparse_tag
) {
	typename matrix_temporary<M>::type eTrans = e;//explicit calculation of the transpose for now
	matrix_assign_functor(m,eTrans,f,row_major(),row_major(),t,t);
	//~ F<typename M::iterator::reference, typename E::value_type> f;
	//~ //first evaluate e and fill the values  togethe into a vector which 
	//~ //is then sorted by row_major order
	//~ //this gives this algorithm a run time of  O(eval(e)+k*log(k))
	//~ //where eval(e) is the time to evaluate and k*log(k) the number of non-zero elements
	//~ typedef typename M::value_type value_type;
	//~ typedef typename M::size_type size_type;
	//~ typedef row_major::sparse_element<value_type> Element;
	//~ std::vector<Element> elements;
	
	//~ size_type size2 = m().size2();
	//~ size_type size1 = m().size1();
	//~ for(size_type j = 0; j != size2; ++j){
		//~ typename E::const_column_iterator pos_e = e().column_begin(j);
		//~ typename E::const_column_iterator end_e = e().column_end(j);
		//~ for(; pos_e != end_e; ++pos_e){
			//~ Element element;
			//~ element.i = pos_e.index();
			//~ element.j = j;
			//~ element.value = *pos_e;
			//~ elements.push_back(element);
		//~ }
	//~ }
	//~ std::sort(elements.begin(),elements.end());

	
	//~ //assign the contents to m, applying the functor every time
	//~ //that is we assign it for every element for e and m
	//~ m().reserve(elements.size());//reserve a bit of space, we might need more, though.
	//~ std::vector<Element>::const_iterator elem = elements.begin();
	//~ std::vector<Element>::const_iterator elem_end = elements.end();
	//~ value_type zero = value_type();
	//~ for(size_type row = 0; row != m().size2(); ++row){
		//~ //todo pre-reserve enough space in the row of m()
		//~ //merge both rows with f as functor
		//~ typename M::row_iterator it = m().row_begin(row);
		//~ while(it != m().row_end(row) && elem != elem_end && elem->i == row) {
			//~ size_type it_index = it.index();
			//~ size_type elem_index = elem->j;
			//~ if (it_index == elem_index) {
				//~ f(*it, *elem);
				//~ ++ elem;
			//~ } else if (it_index < elem_index) {
				//~ f(*it, zero);
			//~ } else{//it_index > elem_index so insert new element in v()
				//~ it = m().set_element(it,elem_index,zero); 
				//~ f(*it, *elem);
				//~ ++elem;
			//~ }
			//~ ++it;
		//~ }
		//~ //apply f to remaining elemms in the row
		//~ for(;it != v().end();++it) {
			//~ f(*it, zero);
		//~ }
		//~ //add missing elements
		//~ for(;elem != elem_end;++it,++elem) {
			//~ it = m().set_element(it,elem.index(),zero); 
			//~ f(*it, zero);
		//~ }
	//~ }
}


//kernels for packed
template<class F, class M, class E, class Triangular>
void matrix_assign_functor(
	matrix_expression<M, cpu_tag> &m, 
	matrix_expression<E, cpu_tag> const& e,
	F f,
	triangular<row_major,Triangular>, triangular<row_major,Triangular>
) {
	typedef typename M::row_iterator MIter;
	typedef typename E::const_row_iterator EIter;
	//there is nothing we can do if F does not leave the non-stored elements 0
	//this is the case for all current assignment functors, but you never know :)
	static_assert(F::left_zero_identity || F::right_zero_identity, "cannot handle the given packed matrix assignment function");

	for(std::size_t i = 0; i != m().size1(); ++i){
		MIter mpos = m().row_begin(i);
		EIter epos = e().row_begin(i);
		MIter mend = m().row_end(i);
		SIZE_CHECK(mpos.index() == epos.index());
		for(; mpos!=mend; ++mpos,++epos){
			*mpos = f(*mpos,*epos);
		}
	}
}

//todo: this is suboptimal as we do strided access!!!!
template<class F, class M, class E, class Triangular>
void matrix_assign_functor(
	matrix_expression<M, cpu_tag> &m, 
	matrix_expression<E, cpu_tag> const& e,
	F f,
	triangular<row_major,Triangular>, triangular<column_major,Triangular>
) {
	typedef typename M::row_iterator MIter;
	//there is nothing we can do, if F does not leave the non-stored elements 0
	static_assert(F::left_zero_identity, "cannot handle the given packed matrix assignment function");
	
	for(std::size_t i = 0; i != m().size1(); ++i){
		MIter mpos = m().row_begin(i);
		MIter mend = m().row_end(i);
		for(; mpos!=mend; ++mpos){
			*mpos = f(*mpos,e()(i,mpos.index()));
		}
	}
}


}}}

#endif
