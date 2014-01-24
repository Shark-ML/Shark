/*!
 * 
 * \file        Initialize.h
 *
 * \brief       Easy initialization of vectors. 
 * 
 * Often, you want to initialize a vector using already available data or you know a fixed initialization. In this case, this header helps, since
 * it is possible to initialize a vector using
 * init(vec) << a,b,c,...;
 * also if you want to split a vector into several smaller parts, you can write
 * init(vec) >> a,b,c,...;
 * a,b,c are allowed to be vectors, vector expressions or single values. However, all vectors needs to be initialized to the correct size. It is checked
 * in debug mode, that size(vec) equals size(a,b,c,...). The usage is not restricted to initialization, but can be used for any assignment where a 
 * vector is constructed from several parts. For example the construction of parameter vectors in AbstractModel::parameterVector.
 * 
 * 
 *
 * \author      O.Krause, T.Glasmachers
 * \date        2010-2011
 *
 *
 * \par Copyright 1995-2014 Shark Development Team
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
//===========================================================================
#ifndef SHARK_LINALG_INITIALIZE_H
#define SHARK_LINALG_INITIALIZE_H

#include "Impl/Initialize.h"
namespace shark{ namespace blas{

/**
 * \ingroup shark_globals
 *
 * @{
 */	

///\brief Starting-point for the initialization sequence.
///
///Usage: init(vector)<<a,b,c where vector is a ublas vector or sub-vector and a,b,c are either scalars or vectors.
///In debug mode, it is checked that size(vector) == size(a,b,c)
template<class Source>
detail::ADLVector<Source&> init(vector_container<Source>& source){
	return detail::ADLVector<Source&>(source());
}
///\brief Starting-point for the initialization sequence.
///
///Usage: init(vector)<<a,b,c where vector is a ublas vector or sub-vector and a,b,c are either scalars or vectors.
///In debug mode, it is checked that size(vector) == size(a,b,c)
template<class Source>
detail::ADLVector<const Source&> init(const vector_container<Source>& source){
	return detail::ADLVector<const Source&>(source());
}
///\brief Starting-point for the initialization sequence when used for splitting the vector.
///
///Usage: init(vector)>>a,b,c where vector is a ublas vector or sub-vector and a,b,c are mutable scalars or vectors.
///In debug mode, it is checked that size(vector) == size(a,b,c)
//~ template<class Source>
//~ detail::ADLVector<const Source&> init(const vector_expression<Source>& source){
	//~ return detail::ADLVector<const Source&>(source());
//~ }


///\brief Specialization for ublas vector_range. 
template<class Source>
detail::ADLVector<vector_range<Source> > 
init(const vector_range<Source>& source){
	return detail::ADLVector<vector_range<Source> >(source);
}

///\brief Specialization for matrix rows.
template<class Source>
detail::ADLVector<matrix_row<Source> > 
init(const matrix_row<Source>& source){
	return detail::ADLVector<matrix_row<Source> >(source);
}
///\brief Specialization for matrix columns.
template<class Source>
detail::ADLVector<matrix_column<Source> > 
init(const matrix_row<Source>& source){
	return detail::ADLVector<matrix_column<Source> >(source);
}

//matrices as arguments

///\brief Linearizes a matrix as a set of row vectors and treats them as a set of vectors for initialization.
template<class Matrix>
detail::MatrixExpression<const Matrix> toVector(const matrix_expression<Matrix>& matrix){
	return detail::MatrixExpression<const Matrix>(matrix());
}
///\brief Linearizes a matrix as a set of row vectors and treats them as a set of vectors for initialization.
template<class Matrix>
detail::MatrixExpression<Matrix> toVector(matrix_expression<Matrix>& matrix){
	return detail::MatrixExpression<Matrix>(matrix());
}

//parameterizable objects as arguments
///\brief Uses the parameters of a parameterizable object for initialization.
///
///The object doesn't have to be derived from IParameterizable, but needs to offer the methods
template<class T>
detail::ParameterizableExpression<const T> parameters(const T& object){
	return detail::ParameterizableExpression<const T>(object);
}
///\brief Uses the parameters of a parameterizable object for initialization.
///
///The object doesn't have to be derived from IParameterizable, but needs to offer the methods
template<class T>
detail::ParameterizableExpression<T> parameters(T& object){
	return detail::ParameterizableExpression<T>(object);
}


//ranges as arguments

///\brief Uses a range of vectors for initialization.
///
///Sometimes not a single vector but a set of vectors is needed as argument like
///std::deque<RealVector> set;
///in this case, vectorSet is needed
///init(vec)<<vec1,vectorSet(set),vec2;
template<class T>
detail::InitializerRange<typename T::const_iterator,detail::VectorExpression<const typename T::value_type&> > 
vectorSet(const T& range){
	return detail::InitializerRange<
		typename T::const_iterator,
		detail::VectorExpression<const typename T::value_type&> 
	>(range.begin(),range.end());
}
///\brief Uses a range of vectors for splitting and initialization.
///
///Sometimes not a single vector but a set of vectors is needed as argument like
///std::deque<RealVector> set;
///in this case, vectorSet is needed
///init(vec)>>vec1,vectorSet(set),vec2;
template<class T>
detail::InitializerRange<typename T::iterator,detail::VectorExpression<typename T::value_type&> > 
vectorSet(T& range){
	return detail::InitializerRange<
		typename T::iterator,
		detail::VectorExpression<typename T::value_type&> 
	>(range.begin(),range.end());
}

///\brief Uses a range of vectors for initialization.
///
///Sometimes not a single matrix but a set of matrices is needed as argument like
///std::deque<RealMatrix> set;
///in this case, matrixSet is needed
///init(vec)<<vec1,matrixSet(set),vec2;
template<class T>
detail::InitializerRange<typename T::const_iterator,detail::MatrixExpression<const typename T::value_type> > 
matrixSet(const T& range){
	return detail::InitializerRange<
		typename T::const_iterator,
		detail::MatrixExpression<const typename T::value_type> 
	>(range.begin(),range.end());
}
///\brief Uses a range of vectors for splitting and initialization.
///
///Sometimes not a single matrix but a set of matrices is needed as argument like
///std::deque<RealMatrix> set;
///in this case, vectorSet is needed
///init(vec)>>vec1,matrixSet(set),vec2;
template<class T>
detail::InitializerRange<typename T::iterator,detail::MatrixExpression<typename T::value_type> > 
matrixSet(T& range){
	return detail::InitializerRange<
		typename T::iterator,
		detail::MatrixExpression<typename T::value_type> 
	>(range.begin(),range.end());
}

///\brief Uses a range of parametrizable objects for initialization.
///
///The objects in the set must offer the methods described by IParameterizable. Also pointer to objects are allowed
template<class T>
detail::InitializerRange<typename T::const_iterator, detail::ParameterizableExpression<const typename T::value_type> > 
parameterSet(const T& range){
	return detail::InitializerRange<
		typename T::const_iterator,
		detail::ParameterizableExpression<const typename T::value_type>
	>(range.begin(),range.end());
}
///\brief Uses a range of parametrizable objects for splitting and initialization.
///
///The objects in the set must offer the methods described by IParameterizable. Also pointer to objects are allowed
template<class T>
detail::InitializerRange<typename T::iterator, detail::ParameterizableExpression<typename T::value_type> > 
parameterSet(T& range){
	return detail::InitializerRange<
		typename T::iterator,
		detail::ParameterizableExpression<typename T::value_type>
	>(range.begin(),range.end());
}

/** @} */
}}

#endif