/*!
 * 
 *
 * \brief       Model for scaling and translation of data vectors.
 * 
 * 
 *
 * \author      T. Glasmachers
 * \date        2013
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
#ifndef SHARK_MODELS_NORMALIZER_H
#define SHARK_MODELS_NORMALIZER_H

#include <shark/Models/AbstractModel.h>
#include <shark/LinAlg/Base.h>


namespace shark {


///
/// \brief "Diagonal" linear model for data normalization.
///
/// \par
/// The Normalizer is a restricted and often more efficient variant of
/// the LinearModel class. It restricts the linear model in two respects:
/// (1) input and output dimension must agree,
/// (2) computations are independent for each component.
/// This is useful mostly for data normalization (therefore the name).
/// The model's operation is of the form \f$ x \mapsto A x + b \f$ where
/// A is a diagonal matrix. This reduces memory requirements to linear,
/// which is why there is no sparse version of this model (as opposed to
/// the more general linear model). Also, the addition of b is optional.
///
///
/// \ingroup models
template <class VectorType = RealVector>
class Normalizer : public AbstractModel<VectorType, VectorType>
{
public:
	typedef AbstractModel<VectorType, VectorType, VectorType> base_type;
	typedef Normalizer<VectorType> self_type;

	typedef typename base_type::BatchInputType BatchInputType;
	typedef typename base_type::BatchOutputType BatchOutputType;

	/// \brief Construction from dimension
	Normalizer(std::size_t dimension = 0, bool hasOffset = false)
	{ setStructure(dimension,hasOffset);}

	/// \brief Construction from matrix and and optional offset vector
	Normalizer(VectorType const& diagonal, VectorType const& offset = VectorType())
	{ setStructure(diagonal, offset);}


	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "Normalizer"; }

	/// \brief derivative storage object (empty for this model)
	boost::shared_ptr<State> createState() const{
		return boost::shared_ptr<State>(new EmptyState());
	}
	
	/// \brief check for the presence of an offset term
	bool hasOffset() const{
		return !m_b.empty();
	}
	
	/// \brief obtain the input dimension
	Shape inputShape() const{
		return m_A.size();
	}

	/// \brief obtain the output dimension
	Shape outputShape() const{
		return m_A.size();
	}

	/// \brief return the diagonal of the matrix
	VectorType const& diagonal() const{
		return m_A;
	}

	/// \brief return the offset vector
	VectorType const& offset() const{
		return m_b;
	}
	
	/// \brief obtain the parameter vector
	VectorType parameterVector() const{
		return m_A | m_b;
	}

	/// \brief overwrite the parameter vector
	void setParameterVector(VectorType const& newParameters){
		SIZE_CHECK(newParameters.size() == numberOfParameters());
		std::size_t dim = m_A.size();
		noalias(m_A) = subrange(newParameters,0,dim);
		noalias(m_b) = subrange(newParameters, dim, newParameters.size());
	}

	/// \brief return the number of parameter
	std::size_t numberOfParameters() const{
		return m_A.size() + m_b.size();
	}

	/// \brief overwrite structure and parameters
	void setStructure(VectorType const& diagonal, VectorType const& offset = VectorType()){
		m_A = diagonal;
		m_b = offset;
	}

	/// \brief overwrite structure and parameters
	void setStructure(std::size_t dimension, bool hasOffset = false){
		m_A.resize(dimension);
		m_b.resize(hasOffset? dimension : 0);
	}

	using base_type::eval;

	/// \brief Evaluate the model: output = matrix * input + offset.
	void eval(BatchInputType const& input, BatchOutputType& output) const{
		SIZE_CHECK(input.size2() == m_A.size());
		output.resize(input.size1(), input.size2());
		noalias(output) = input * repeat(m_A,input.size1());
		if (hasOffset()){
			noalias(output) += repeat(m_b,input.size1());
		}
	}

	/// \brief Evaluate the model: output = matrix * input + offset.
	void eval(BatchInputType const& input, BatchOutputType& output, State& state) const{
		eval(input, output);
	}

	/// from ISerializable
	void read(InArchive& archive){
		archive & m_A;
		archive & m_b;
	}

	/// from ISerializable
	void write(OutArchive& archive) const{
		archive & m_A;
		archive & m_b;
	}

protected:
	VectorType m_A;                ///< matrix A (see class documentation)
	VectorType m_b;                        ///< vector b (see class documentation)
};


}
#endif
