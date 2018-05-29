/*!
 * \brief       Classes used for matrix expressions.
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
 #ifndef REMORA_VECTOR_PROXY_SET_CLASSES_HPP
#define REMORA_VECTOR_PROXY_SET_CLASSES_HPP

#include "traits.hpp"
#include "../expression_types.hpp"

namespace remora{

template<class E, class O>
class vector_set:public vector_set_expression<vector_set<E, O>, typename E::device_type >{
public:
	typedef typename closure<E>::type expression_closure_type;
	typedef typename E::size_type size_type;
	typedef typename E::value_type value_type;
	typedef typename E::const_reference const_reference;
	typedef typename reference<E>::type reference;

	typedef vector_set<typename E::const_closure_type, O> const_closure_type;
	typedef vector_set closure_type;
	typedef O point_orientation;
	//~ typedef typename E::orientation storage_orientation;
	typedef typename E::evaluation_category evaluation_category;
	typedef typename E::device_type device_type;

	// Construction and destruction
	explicit vector_set(expression_closure_type const& e):m_expression(e){}

	size_type size() const{
		return point_orientation::index_M(m_expression.size1(), m_expression.size2());
	}
	size_type point_size() const{
		return point_orientation::index_m(m_expression.size1(), m_expression.size2());
	}
	
	expression_closure_type const& expression() const{
		return m_expression;
	}
	typename device_traits<device_type>::queue_type& queue()const{
		return m_expression.queue();
	}
	
	//computation kernels
	template<class MatX>
	void assign_to(matrix_expression<MatX, device_type>& X, typename MatX::value_type alpha)const{
		assign(X, m_expression, alpha);
	}
	template<class MatX>
	void plus_assign_to(matrix_expression<MatX, device_type>& X, typename MatX::value_type alpha)const{
		plus_assign(X, m_expression, alpha);
	}
private:
	expression_closure_type m_expression;
};	

}
#endif
