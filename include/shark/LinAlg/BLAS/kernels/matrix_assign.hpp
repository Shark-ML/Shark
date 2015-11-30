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
#ifndef SHARK_LINALG_BLAS_KERNELS_MATRIX_ASSIGN_HPP
#define SHARK_LINALG_BLAS_KERNELS_MATRIX_ASSIGN_HPP

#include "vector_assign.hpp"
#include <algorithm>
namespace shark {
namespace blas {

namespace detail{

//base version of the matrix_transpose proxy which does not rely on the existence of the assignment functions
//we define it here so that we don't need to duplicate all versions of assign for the case of row/column major
//first argument
template<class M>
class internal_transpose_proxy: public matrix_expression<internal_transpose_proxy<M> > {
public:
	typedef typename M::size_type size_type;
	typedef typename M::difference_type difference_type;
	typedef typename M::value_type value_type;
	typedef typename M::scalar_type scalar_type;
	typedef typename M::const_reference const_reference;
	typedef typename reference<M>::type reference;
	typedef typename M::const_pointer const_pointer;
	typedef typename M::pointer pointer;

	typedef typename M::index_type index_type;
	typedef typename M::const_index_pointer const_index_pointer;
	typedef typename index_pointer<M>::type index_pointer;

	typedef typename closure<M>::type matrix_closure_type;
	typedef const internal_transpose_proxy<M> const_closure_type;
	typedef internal_transpose_proxy<M> closure_type;
	typedef typename M::orientation::transposed_orientation orientation;
	typedef typename M::storage_category storage_category;
	typedef typename M::evaluation_category evaluation_category;

	// Construction and destruction
	explicit internal_transpose_proxy(matrix_closure_type m):
		m_expression(m) {}

	// Accessors
	size_type size1() const {
		return m_expression.size2();
	}
	size_type size2() const {
		return m_expression.size1();
	}

	// Expression accessors
	matrix_closure_type const &expression() const{
		return m_expression;
	}
	matrix_closure_type &expression(){
		return m_expression;
	}

	// Element access
	reference operator()(size_type i, size_type j)const{
		return m_expression(j, i);
	}

	typedef typename matrix_closure_type::const_column_iterator const_row_iterator;
	typedef typename matrix_closure_type::column_iterator row_iterator;
	typedef typename matrix_closure_type::const_row_iterator const_column_iterator;
	typedef typename matrix_closure_type::row_iterator column_iterator;

	//iterators
	const_row_iterator row_begin(std::size_t i) const {
		return m_expression.column_begin(i);
	}
	const_row_iterator row_end(std::size_t i) const {
		return m_expression.column_end(i);
	}
	const_column_iterator column_begin(std::size_t j) const {
		return m_expression.row_begin(j);
	}
	const_column_iterator column_end(std::size_t j) const {
		return m_expression.row_end(j);
	}

	row_iterator row_begin(std::size_t i) {
		return m_expression.column_begin(i);
	}
	row_iterator row_end(std::size_t i) {
		return m_expression.column_end(i);
	}
	column_iterator column_begin(std::size_t j) {
		return m_expression.row_begin(j);
	}
	column_iterator column_end(std::size_t j) {
		return m_expression.row_end(j);
	}
	
	typedef typename major_iterator<internal_transpose_proxy<M> >::type major_iterator;
	
	major_iterator set_element(major_iterator pos, size_type index, value_type value){
		return m_expression.set_element(pos,index,value);
	}
	
	major_iterator clear_range(major_iterator start, major_iterator end){
		return m_expression.clear_range(start,end);
	}
	
	major_iterator clear_element(major_iterator elem){
		return m_expression.clear_element(elem);
	}
	
	void clear(){
		expression().clear();
	}
	
	void reserve(size_type non_zeros) {
		m_expression.reserve(non_zeros);
	}
	
	void reserve_row(size_type row, size_type non_zeros) {
		m_expression.reserve_row(row,non_zeros);
	}
	void reserve_column(size_type column, size_type non_zeros) {
		m_expression.reserve_column(column,non_zeros);
	}
private:
	matrix_closure_type m_expression;
};
	
}
	
//////////////////////////////////////////////////////
////Scalar Assignment to Matrix
/////////////////////////////////////////////////////

namespace kernels{
// Explicitly iterating row major
template<template <class T1, class T2> class F, class M>
void assign(
	matrix_expression<M> &m, 
	typename M::value_type t, 
	row_major
){
	for(std::size_t i = 0; i != m().size1(); ++i){
		matrix_row<M> rowM(m(),i);
		assign<F>(rowM,t);
	}
}
// Explicitly iterating column major
template<template <class T1, class T2> class F, class M>
void assign(
	matrix_expression<M> &m, 
	typename M::value_type t, 
	column_major
){
	for(std::size_t i = 0; i != m().size2(); ++i){
		matrix_column<M> columnM(m(),i);
		assign<F>(columnM,t);
	}
}


// Spcial case packed - just calls the first two implementations.
template<template <class T1, class T2> class F, class M, class Orientation, class Triangular>
void assign(
	matrix_expression<M> &m, 
	typename M::value_type t, 
	packed<Orientation,Triangular>
){
	assign<F>(m,t,Orientation());
}

// Dispatcher
template<template <class T1, class T2> class F, class M>
void assign(
	matrix_expression<M> &m, 
	typename M::value_type t
){
	typedef typename M::orientation orientation;
	assign<F> (m, t, orientation());
}

/////////////////////////////////////////////////////////////////
//////Matrix Assignment implmenting op=
////////////////////////////////////////////////////////////////

//direct assignment without functor
//the cases were both arguments have the same orientation can be implemented using assign.
template<class M, class E,class TagE, class TagM>
void assign(
	matrix_expression<M> &m, 
	matrix_expression<E> const& e,
	row_major, row_major,TagE, TagM
) {
	for(std::size_t i = 0; i != m().size1(); ++i){
		matrix_row<M> rowM(m(),i);
		kernels::assign(rowM,row(e,i));
	}
}

//remain the versions where both argumnts to not have the same orientation

//dense-dense case
template<class M, class E>
void assign(
	matrix_expression<M> &m, 
	matrix_expression<E> const& e,
	row_major, column_major,dense_random_access_iterator_tag, dense_random_access_iterator_tag
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
void assign(
	matrix_expression<M> &m, 
	matrix_expression<E> const& e,
	row_major, column_major,dense_random_access_iterator_tag, sparse_bidirectional_iterator_tag
) {
	for(std::size_t i = 0; i != m().size2(); ++i){
		matrix_column<M> columnM(m(),i);
		kernels::assign(columnM,column(e,i));
	}
}


//sparse-dense
template<class M, class E>
void assign(
	matrix_expression<M> &m, 
	matrix_expression<E> const& e,
	row_major, column_major, sparse_bidirectional_iterator_tag, dense_random_access_iterator_tag
) {
	for(std::size_t i = 0; i != m().size1(); ++i){
		matrix_column<M> rowM(m(),i);
		kernels::assign(rowM,row(e,i));
	}
}

//sparse-sparse
template<class M, class E>
void assign(
	matrix_expression<M> &m, 
	matrix_expression<E> const& e,
	row_major, column_major,sparse_bidirectional_iterator_tag,sparse_bidirectional_iterator_tag
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
void assign(
	matrix_expression<M> &m, 
	matrix_expression<E> const& e,
	packed<row_major,Triangular>, packed<row_major,Triangular>,
	packed_random_access_iterator_tag, packed_random_access_iterator_tag
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
void assign(
	matrix_expression<M> &m, 
	matrix_expression<E> const& e,
	packed<row_major,Triangular>, packed<column_major,Triangular>,
	packed_random_access_iterator_tag, packed_random_access_iterator_tag
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

//general dispatcher: if the second argument has an unknown orientation
// it is chosen the same as the first one
template<class M, class E, class TagE, class TagM>
void assign(
	matrix_expression<M> &m, 
	matrix_expression<E> const& e,
	row_major, unknown_orientation ,TagE tagE, TagM tagM
) {
	assign(m,e,row_major(),row_major(),tagE,tagM);
}

//general dispatcher: if the first argumeent is column major, we transpose the whole expression
//so that it  is row-major, this saves us to implment everything twice.
template<class M, class E,class EOrientation, class TagE, class TagM>
void assign(
	matrix_expression<M> &m, 
	matrix_expression<E> const& e,
	column_major, EOrientation,TagE tagE, TagM tagM
) {
	typedef typename EOrientation::transposed_orientation TEOrientation;
	detail::internal_transpose_proxy<M> transM(m());
	detail::internal_transpose_proxy<E const> transE(e());
	assign(transM,transE,row_major(),TEOrientation(),tagE,tagM);
}

//first argument is column_major->transpose to row_major
template<class M, class E,class EOrientation, class Triangular, class TagM, class TagE>
void assign(
	matrix_expression<M> &m, 
	matrix_expression<E> const& e,
	packed<column_major,Triangular>, packed<EOrientation,Triangular>,
	TagM tagM, TagE tagE
) {
	typedef typename M::orientation::transposed_orientation TMPacked;
	typedef typename E::orientation::transposed_orientation TEPacked;
	detail::internal_transpose_proxy<M> transM(m());
	detail::internal_transpose_proxy<E const> transE(e());
	assign(transM,transE,TMPacked(),TEPacked(),tagM,tagE);
}

// Dispatcher
template<class M, class E>
void assign(matrix_expression<M> &m, const matrix_expression<E> &e) {
	SIZE_CHECK(m().size1() == e().size1());
	SIZE_CHECK(m().size2() == e().size2());
	typedef typename M::orientation MOrientation;
	typedef typename E::orientation EOrientation;
	typedef typename major_iterator<M>::type::iterator_category MCategory;
	typedef typename major_iterator<E>::type::iterator_category ECategory;
	assign(m, e, MOrientation(),EOrientation(),MCategory(), ECategory());
}


///////////////////////////////////////////////////////////////////////////////////////////
//////Matrix Assignment With Functor implementing +=,-=...
///////////////////////////////////////////////////////////////////////////////////////////

//we only implement the case where the target is row major and than map the rest into these
	
//when both are row-major we can map to vector case
//this is not necessarily efficient if m is sparse.
template<template <class, class> class F, class M, class E, class TagE, class TagM>
void assign(
	matrix_expression<M> &m, 
	matrix_expression<E> const& e,
	row_major, row_major,TagM, TagE
) {
	for(std::size_t i = 0; i != m().size1(); ++i){
		matrix_row<M> rowM(m(),i);
		kernels::assign<F>(rowM,row(e,i));
	}
}

//we only need to implement the remaining versions for column major second argument

//dense-dense case
template<template <class, class> class F,class M, class E>
void assign(
	matrix_expression<M> &m, 
	matrix_expression<E> const& e,
	row_major, column_major,dense_random_access_iterator_tag, dense_random_access_iterator_tag
) {
	F<typename major_iterator<M>::type::reference, typename E::value_type> f;
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
					f(m()(iblock+i,jblock+j), blockStorage[i][j]);
				}
			}
		}
	}
}

//dense-sparse case
template<template <class, class> class F,class M, class E>
void assign(
	matrix_expression<M> &m, 
	matrix_expression<E> const& e,
	row_major, column_major,dense_random_access_iterator_tag, sparse_bidirectional_iterator_tag
) {
	for(std::size_t i = 0; i != m().size2(); ++i){
		matrix_column<M> columnM(m(),i);
		kernels::assign<F>(columnM,column(e,i));
	}
}

//sparse-dense case
template<template <class, class> class F,class M, class E>
void assign(
	matrix_expression<M> &m, 
	matrix_expression<E> const& e,
	row_major, column_major, sparse_bidirectional_iterator_tag, dense_random_access_iterator_tag
) {
	for(std::size_t i = 0; i != m().size1(); ++i){
		matrix_column<M> rowM(m(),i);
		kernels::assign<F>(rowM,row(e,i));
	}
}

//sparse-sparse
template<template <class, class> class F,class M, class E>
void assign(
	matrix_expression<M> &m, 
	matrix_expression<E> const& e,
	row_major, column_major,sparse_bidirectional_iterator_tag t,sparse_bidirectional_iterator_tag
) {
	typename matrix_temporary<M>::type eTrans = e;//explicit calculation of the transpose for now
	assign<F>(m,eTrans,row_major(),row_major(),t,t);
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
template<template <class, class> class F, class M, class E, class Triangular>
void assign(
	matrix_expression<M> &m, 
	matrix_expression<E> const& e,
	packed<row_major,Triangular>, packed<row_major,Triangular>
) {
	typedef typename M::row_iterator MIter;
	typedef typename E::const_row_iterator EIter;
	typedef F<typename MIter::reference,typename  EIter::value_type> Function;
	//there is nothing we can do, if F does not leave the non-stored elements 0
	//this is the case for all current aissgnment functors, but you never know :)
	BOOST_STATIC_ASSERT( Function::left_zero_identity || Function::right_zero_identity);
	
	Function f;
	for(std::size_t i = 0; i != m().size1(); ++i){
		MIter mpos = m().row_begin(i);
		EIter epos = e().row_begin(i);
		MIter mend = m().row_end(i);
		SIZE_CHECK(mpos.index() == epos.index());
		for(; mpos!=mend; ++mpos,++epos){
			f(*mpos,*epos);
		}
	}
}

//todo: this is suboptimal as we do strided access!!!!
template<template <class, class> class F, class M, class E, class Triangular>
void assign(
	matrix_expression<M> &m, 
	matrix_expression<E> const& e,
	packed<row_major,Triangular>, packed<column_major,Triangular>
) {
	typedef typename M::row_iterator MIter;
	typedef typename E::const_row_iterator EIter;
	typedef F<typename MIter::reference,typename EIter::value_type> Function;
	//there is nothing we can do, if F does not leave the non-stored elements 0
	BOOST_STATIC_ASSERT( Function::left_zero_identity);
	
	Function f;
	for(std::size_t i = 0; i != m().size1(); ++i){
		MIter mpos = m().row_begin(i);
		MIter mend = m().row_end(i);
		for(; mpos!=mend; ++mpos){
			f(*mpos,e()(i,mpos.index()));
		}
	}
}

//second level of dispatcher dispatches to actual computation kernels

//first standard structured matrices which only differ in row_major/column_major 
//for these we standardize input so that the first argument is row_major. Further
//we choose in the case if unknown_orientation the second argument to be the same 
//as the first. more than one dispatcher can be called every time!


//everything fulfilled -> dispatch sparse/dense computing versions
template<template <class, class> class F, class M, class E>
void assign(
	matrix_expression<M> &m, 
	matrix_expression<E> const& e,
	row_major, row_major
) {
	typedef typename M::const_row_iterator::iterator_category MCategory;
	typedef typename E::const_row_iterator::iterator_category ECategory;
	assign<F>(m,e,row_major(),row_major(),MCategory(),ECategory());
}
//everything fulfilled -> dispatch sparse/dense computing versions
template<template <class, class> class F, class M, class E>
void assign(
	matrix_expression<M> &m, 
	matrix_expression<E> const& e,
	row_major, column_major
) {
	typedef typename M::const_row_iterator::iterator_category MCategory;
	typedef typename E::const_column_iterator::iterator_category ECategory;
	assign<F>(m,e,row_major(),column_major(),MCategory(),ECategory());
}

//first argument is row_major, second is unknown->choose unknown to be row_major
template<template <class, class> class F,class M, class E>
void assign(
	matrix_expression<M> &m, 
	matrix_expression<E> const& e,
	row_major, unknown_orientation
) {
	assign<F>(m,e,row_major(),row_major());
}


//first argument is column_major->transpose to row_major
template<template <class, class> class F, class M, class E,class EOrientation>
void assign(
	matrix_expression<M> &m, 
	matrix_expression<E> const& e,
	column_major, EOrientation
) {
	typedef typename EOrientation::transposed_orientation TEOrientation;
	detail::internal_transpose_proxy<M> transM(m());
	detail::internal_transpose_proxy<E const> transE(e());
	assign<F>(transM,transE,row_major(),TEOrientation());
}

//now dispatch packed matrices. Again we dispatch so that the default is row_major
//we also ensure here that the triangular structure is compatible

//first argument is column_major->transpose to row_major
template<template <class, class> class F, class M, class E,class EOrientation, class Triangular>
void assign(
	matrix_expression<M> &m, 
	matrix_expression<E> const& e,
	packed<column_major,Triangular>, packed<EOrientation,Triangular>
) {
	typedef typename M::orientation::transposed_orientation TMPacked;
	typedef typename E::orientation::transposed_orientation TEPacked;
	detail::internal_transpose_proxy<M> transM(m());
	detail::internal_transpose_proxy<E const> transE(e());
	assign<F>(transM,transE,TMPacked(),TEPacked());
}


//First Level Dispatcher, dispatches by orientation
template<template <class,class> class F, class M, class E>
void assign(matrix_expression<M> &m, const matrix_expression<E> &e) {
	SIZE_CHECK(m().size1()  == e().size1());
	SIZE_CHECK(m().size2()  == e().size2());
	typedef typename M::orientation MOrientation;
	typedef typename E::orientation EOrientation;
	
	assign<F>(m, e, MOrientation(),EOrientation());
}

}}}

#endif
