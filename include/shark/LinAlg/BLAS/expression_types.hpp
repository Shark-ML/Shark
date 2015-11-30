/*!
 * \brief       Defines the basic types of CRTP base-classes
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
#ifndef SHARK_LINALG_BLAS_EXPRESSION_TYPE_HPP
#define SHARK_LINALG_BLAS_EXPRESSION_TYPE_HPP

namespace shark {
namespace blas {

/** \brief Base class for Vector Expression models
 *
 * it does not model the Vector Expression concept but all derived types should.
 * The class defines a common base type and some common interface for all
 * statically derived Vector Expression classes.
 * We implement the casts to the statically derived type.
 */
template<class E>
struct vector_expression {
	typedef E expression_type;

	const expression_type &operator()() const {
		return *static_cast<const expression_type *>(this);
	}

	expression_type &operator()() {
		return *static_cast<expression_type *>(this);
	}
};

/** \brief Base class for Vector container models
 *
 * it does not model the Vector concept but all derived types should.
 * The class defines a common base type and some common interface for all
 * statically derived Vector classes
 * We implement the casts to the statically derived type.
 */
template<class C>
struct vector_container:public vector_expression<C> {
	typedef C container_type;

	const container_type &operator()() const {
		return *static_cast<const container_type *>(this);
	}

	container_type &operator()() {
		return *static_cast<container_type *>(this);
	}
};


/** \brief Base class for Matrix Expression models
 *
 * it does not model the Matrix Expression concept but all derived types should.
 * The class defines a common base type and some common interface for all
 * statically derived Matrix Expression classes
 * We iboost::mplement the casts to the statically derived type.
 */
template<class E>
struct matrix_expression {
	typedef E expression_type;

	const expression_type &operator()() const {
		return *static_cast<const expression_type *>(this);
	}

	expression_type &operator()() {
		return *static_cast<expression_type *>(this);
	}
};

/** \brief Base class for expressions of matrix sets
 *
 * The matrix set expression type is similar to a tensor type. However it behaves
 * like a vector of matrices with elements of the vector being matrices. Moreover
 * all usual operations can be used. There is no distinction to the sizes of the matrices
 * and all matrices may have different dimensionalities.
 *
 * it does not model the Matrix Expression concept but all derived types should.
 * The class defines a common base type and some common interface for all
 * statically derived Matrix Expression classes
 * We iboost::mplement the casts to the statically derived type.
 */
template<class E>
struct matrix_set_expression {
	typedef E expression_type;

	const expression_type &operator()() const {
		return *static_cast<const expression_type *>(this);
	}

	expression_type &operator()() {
		return *static_cast<expression_type *>(this);
	}
};

/** \brief Base class for expressions of vector sets
 *
 * The vector set expression type is similar to a matrix type. However it behaves
 * like a vector of vectors with elements of the vector being vectors. Moreover
 * all usual vector-space operations can be used . There is no distinction to the sizes of the elements
 * and all vectors may have different dimensionalities.
 *
 * it does not model the Matrix Expression concept but all derived types should.
 * The class defines a common base type and some common interface for all
 * statically derived Matrix Expression classes
 * We iboost::mplement the casts to the statically derived type.
 */
template<class E>
struct vector_set_expression {
	typedef E expression_type;

	const expression_type &operator()() const {
		return *static_cast<const expression_type *>(this);
	}

	expression_type &operator()() {
		return *static_cast<expression_type *>(this);
	}
};

/** \brief Base class for Matrix container models
 *
 * it does not model the Matrix concept but all derived types should.
 * The class defines a common base type and some common interface for all
 * statically derived Matrix classes
 * We implement the casts to the statically derived type.
 */
template<class C>
struct matrix_container: public matrix_expression<C> {
	typedef C container_type;

	const container_type &operator()() const {
		return *static_cast<const container_type *>(this);
	}

	container_type &operator()() {
		return *static_cast<container_type *>(this);
	}
};

template<class P>
struct temporary_proxy:public P{
	temporary_proxy(P const& p):P(p){}
	
	template<class E>
	P& operator=(E const& e){
		return static_cast<P&>(*this) = e;
	}
	
	P& operator=(temporary_proxy<P> const& e){
		return static_cast<P&>(*this) = e;
	}
};

}
}

#endif
