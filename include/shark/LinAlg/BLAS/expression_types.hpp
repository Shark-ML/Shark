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

// Assignment proxy.
// Provides temporary free assigment when LHS has no alias on RHS
template<class C>
class noalias_proxy{
public:
	typedef typename C::closure_type closure_type;
	typedef typename C::scalar_type scalar_type;

	noalias_proxy(C &lval): m_lval(lval) {}

	noalias_proxy(const noalias_proxy &p):m_lval(p.m_lval) {}

	template <class E>
	closure_type &operator= (const E &e) {
		m_lval.assign(e);
		return m_lval;
	}

	template <class E>
	closure_type &operator+= (const E &e) {
		m_lval.plus_assign(e);
		return m_lval;
	}

	template <class E>
	closure_type &operator-= (const E &e) {
		m_lval.minus_assign(e);
		return m_lval;
	}
	
	template <class E>
	closure_type &operator*= (const E &e) {
		m_lval.multiply_assign(e);
		return m_lval;
	}

	template <class E>
	closure_type &operator/= (const E &e) {
		m_lval.divide_assign(e);
		return m_lval;
	}
	
	//this is not needed, but prevents errors when fr example doing noalias(x)*=2;
	closure_type &operator*= (scalar_type t) {
		m_lval *= t;
		return m_lval;
	}

	//this is not needed, but prevents errors when for example doing noalias(x)/=2;
	closure_type &operator/= (scalar_type t) {
		m_lval *=t;
		return m_lval;
	}

private:
	closure_type m_lval;
};

// Improve syntax of efficient assignment where no aliases of LHS appear on the RHS
//  noalias(lhs) = rhs_expression
template <class C>
noalias_proxy<C> noalias(matrix_expression<C>& lvalue) {
	return noalias_proxy<C> (lvalue());
}
template <class C>
noalias_proxy<C> noalias(vector_expression<C>& lvalue) {
	return noalias_proxy<C> (lvalue());
}

template <class C>
noalias_proxy<C> noalias(matrix_set_expression<C>& lvalue) {
	return noalias_proxy<C> (lvalue());
}
template <class C>
noalias_proxy<C> noalias(vector_set_expression<C>& lvalue) {
	return noalias_proxy<C> (lvalue());
}
template <class C>
noalias_proxy<C> noalias(temporary_proxy<C> lvalue) {
	return noalias_proxy<C> (static_cast<C&>(lvalue));
}

}
}

#endif
