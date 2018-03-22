//===========================================================================
/*!
 * 
 *
 * \brief       Class Describing the Shape of an Input
 * 
 * 
 *
 * \author      O. Krause
 * \date        2017
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://shark-ml.org/>
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

#ifndef SHARK_CORE_SHAPE_H
#define SHARK_CORE_SHAPE_H

#include <vector>
#include <initializer_list>
#include <ostream>
#include <boost/serialization/vector.hpp>
namespace shark{


/// \brief Represents the Shape of an input or output
///
/// Mostly used for vector data, the Shape describes
/// The expected structure of a model. 
/// A N-D shape with shape variables (n1,n2,..nN)
/// expects an input of size n1*n2*...*nN which is then interpreted as tensor
/// with the dimensionalities n1 x n2 x ... x nN. 
/// A batch of inputs is then treated as each element having this shape, so
/// the batch size is not a part of the shape.
///
/// The standard shape
/// is the 1-D shape just describing that a model interprets every
/// input as 1-D input.
/// A 0-D shape describes the inputs of a model where the input can not be
/// described by a shape, for example a class label or other scalar values are 0d shapes.
/// A 3-D shape could describe an image patch with rows x columns x channels.
///
/// Shapes can be flattened, this way a 3-D image patch can also be treated as a simple
/// vector input.
///
/// Shark currently does not enforce Shapes, it only checks that input data is compatible
/// to a shape, i.e. a vector has the right number of dimensions.
class Shape{
public:
	Shape(): m_numElements(1){}
	Shape(std::size_t size): m_dims(1,size), m_numElements(size){}
	Shape(std::initializer_list<std::size_t> dims): m_dims(dims){
		m_numElements = 1;
		for(auto dim: m_dims){
			m_numElements *= dim;
		}
	}
	std::size_t size()const{
		return m_dims.size();
	}
	std::size_t operator[](std::size_t i) const{
		return m_dims[i];
	}
	std::size_t numElements()const{
		return m_numElements;
	}
	
	///\brief Returns a 1-D shape with the same number of elements
	Shape flatten() const{
		return Shape({m_numElements});
	}
	
	//stride of elements in memory when increasing dimension dim by 1
	//assuming the underlying memory is contiguous
	std::size_t stride(std::size_t dim) const{
		std::size_t val = 1;
		if(size() == 0) return val;
		for(std::size_t i = size() - 1; i != dim; --i){
			val *= m_dims[i];
		}
		return val;
	}
	
	template<class Archive>
	void serialize(Archive & archive,unsigned int version){
		archive & m_dims;
		archive & m_numElements;
	}
private:
	std::vector<std::size_t> m_dims;
	std::size_t m_numElements;
};

inline bool operator == (Shape const& shape1, Shape const& shape2){
	if(shape1.size() != shape2.size())
		return false;
	for(std::size_t i = 0; i != shape1.size(); ++i){
		if(shape1[i] != shape2[i]){
			return false;
		}
	}
	return true;
}

inline bool operator != (Shape const& shape1, Shape const& shape2){
	return ! (shape1 == shape2);
}

template<class E, class T>
std::basic_ostream<E, T> &operator << (std::basic_ostream<E, T> &os, Shape const& shape) {
	os<<'(';
	for(std::size_t i = 0; i != shape.size(); ++i){
		os<<shape[i];
		if(i != shape.size() -1)
			os<<", ";
	}
	os<<')';
	return os;
}

}
#endif
