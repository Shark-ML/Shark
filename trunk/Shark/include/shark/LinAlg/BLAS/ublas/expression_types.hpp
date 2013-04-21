//
//  Copyright (c) 2000-2010
//  Joerg Walter, Mathias Koch. David Bellot
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  GeNeSys mbH & Co. KG in producing this work.
//
#ifndef _BOOST_UBLAS_EXPRESSION_TYPE_
#define _BOOST_UBLAS_EXPRESSION_TYPE_

#include <shark/LinAlg/BLAS/ublas/exception.hpp>
#include <shark/LinAlg/BLAS/ublas/traits.hpp>
#include <shark/LinAlg/BLAS/ublas/functional.hpp>

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

}
}

#endif
