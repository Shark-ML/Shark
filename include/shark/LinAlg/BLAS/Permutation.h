/**
*
*  \brief Helper functions for linear algebra component.
*
*  \author O.Krause, T.Glasmachers, T. Voss
*  \date 2010-2011
*
*  \par Copyright (c) 1998-2007:
*      Institut f&uuml;r Neuroinformatik<BR>
*      Ruhr-Universit&auml;t Bochum<BR>
*      D-44780 Bochum, Germany<BR>
*      Phone: +49-234-32-25558<BR>
*      Fax:   +49-234-32-14209<BR>
*      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
*      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
*      <BR>
*
*
*  <BR><HR>
*  This file is part of Shark. This library is free software;
*  you can redistribute it and/or modify it under the terms of the
*  GNU General Public License as published by the Free Software
*  Foundation; either version 3, or (at your option) any later version.
*
*  This library is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with this library; if not, see <http://www.gnu.org/licenses/>.
*
*/
#ifndef SHARK_LINALG_BLAS_PERMUTATION_H
#define SHARK_LINALG_BLAS_PERMUTATION_H

#include <shark/LinAlg/BLAS/ublas.h>

namespace shark { namespace blas{

	///\brief implements row pivoting at matrix A using permutation P
	///
	///by convention it is not allowed that P(i) < i. 
	template<class M, class Permutation>
	void swapRows(Permutation const& P, blas::matrix_expression<M>& A){
		blas::swap_rows(P,A());
	}
	
	///\brief implements column pivoting of vector v using permutation P
	///
	///by convention it is not allowed that P(i) < i. 
	template<class V, class Permutation>
	void swapRows(Permutation const& P, blas::vector_expression<V>& v){
		blas::swap_rows(P,v());
	}
	
	///\brief implements the inverse row pivoting of vector v using permutation P
	///
	///This is the inverse operation to swapRows. 
	template<class V, class Permutation>
	void swapRowsInverted(Permutation const& P, blas::vector_expression<V>& v){
		for(std::size_t i = P.size(); i != 0; --i){
			std::size_t k = i-1;
			if(k != P(k)){
				using std::swap;
				swap(v()(k),v()(P(k)));
			}
		}
	}
	
	///\brief implements column pivoting at matrix A using permutation P
	///
	///by convention it is not allowed that P(i) < i. 
	template<class M, class Permutation>
	void swapColumns(Permutation const& P, blas::matrix_expression<M>& A){
		for(std::size_t i = 0; i != P.size(); ++i){
			if(i != P(i)){
				column(A(),i).swap(column(A(),P(i)));
			}
		}
	}
	
	///\brief implements the inverse row pivoting at matrix A using permutation P
	///
	///This is the inverse operation to swapRows. 
	template<class M, class Permutation>
	void swapRowsInverted(Permutation const& P, blas::matrix_expression<M>& A){
		for(std::size_t i = P.size(); i != 0; --i){
			std::size_t k = i-1;
			if(k != P(k)){
				row(A(),k).swap(row(A(),P(k)));
			}
		}
	}
	
	///\brief implements the inverse column pivoting at matrix A using permutation P
	///
	///This is the inverse operation to swapColumns. 
	template<class M, class Permutation>
	void swapColumnsInverted(Permutation const& P, blas::matrix_expression<M>& A){
		for(std::size_t i = P.size(); i != 0; --i){
			std::size_t k = i-1;
			if(k != P(k)){
				column(A(),k).swap(column(A(),P(k)));
			}
		}
	}
	
	///\brief Implements full pivoting at matrix A using permutation P
	///
	///full pivoting does swap rows and columns such that the diagonal element
	///A_ii is then at position A_P(i)P(i)
	///by convention it is not allowed that P(i) < i. 
	template<class M, class Permutation>
	void swapFull(Permutation const& P, blas::matrix_expression<M>& A){
		for(std::size_t i = 0; i != P.size(); ++i){
			if(i != P(i)){
				row(A(),i).swap(row(A(),P(i)));
				column(A(),i).swap(column(A(),P(i)));
			}
		}
	}
	///\brief implements the inverse full pivoting at matrix A using permutation P
	///
	///This is the inverse operation to swapColumns. 
	template<class M, class Permutation>
	void swapFullInverted(Permutation const& P, blas::matrix_expression<M>& A){
		for(std::size_t i = P.size(); i != 0; --i){
			std::size_t k = i-1;
			if(k != P(k)){
				row(A(),k).swap(row(A(),P(k)));
				column(A(),k).swap(column(A(),P(k)));
			}
		}
	}
	
/** @}*/
}}
#endif
