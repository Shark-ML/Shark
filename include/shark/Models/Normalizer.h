/*!
*  \brief Model for scaling and translation of data vectors.
*
*  \author  T. Glasmachers
*  \date    2013
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
#ifndef SHARK_MODELS_NORMALIZER_H
#define SHARK_MODELS_NORMALIZER_H

#include <shark/Models/AbstractModel.h>
#include <shark/LinAlg/Base.h>
#include <shark/LinAlg/BLAS/Initialize.h>


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
template <class DataType = RealVector>
class Normalizer : public AbstractModel<DataType, DataType>
{
public:
	typedef AbstractModel<DataType, DataType> base_type;
	typedef Normalizer<DataType> self_type;

	typedef typename base_type::BatchInputType BatchInputType;
	typedef typename base_type::BatchOutputType BatchOutputType;

	/// Constructor of an invalid model; use setStructure later
	Normalizer()
	{
		base_type::m_name = "Normalizer";
	}

	/// copy constructor
	Normalizer(const self_type& model)
	: m_A(model.m_A)
	, m_b(model.m_b)
	, m_hasOffset(model.m_hasOffset)
	{
		base_type::m_name = "Normalizer";
	}

	/// Construction from dimension
	Normalizer(std::size_t dimension, bool hasOffset = false)
	: m_A(dimension, dimension)
	, m_b(dimension)
	, m_hasOffset(hasOffset)
	{
		base_type::m_name = "Normalizer";
	}

	/// Construction from matrix
	Normalizer(RealDiagonalMatrix matrix)
	: m_A(matrix)
	, m_hasOffset(false)
	{
		base_type::m_name = "Normalizer";
	}

	/// Construction from matrix and vector
	Normalizer(RealDiagonalMatrix matrix, RealVector vector)
	: m_A(matrix)
	, m_b(vector)
	, m_hasOffset(true)
	{
		base_type::m_name = "Normalizer";
	}


	/// swap
	friend void swap(const Normalizer& model1, const Normalizer& model2)
	{
		std::swap(model1.m_A, model2.m_A);
		std::swap(model1.m_b, model2.m_b);
		std::swap(model1.m_hasOffset, model2.m_hasOffset);
	}

	/// assignment operator
	const self_type operator = (const self_type& model)
	{
		m_A = model.m_A;
		m_b = model.m_b;
		m_hasOffset = model.m_hasOffset;
	}

	/// derivative storage object (empty for this model)
	boost::shared_ptr<State> createState() const
	{
		return boost::shared_ptr<State>(new EmptyState());
	}


	/// check if the model is properly initialized
	bool isValid() const
	{
		return (m_A.size1() != 0);
	}

	/// check for the presence of an offset term
	bool hasOffset() const
	{
		return m_hasOffset;
	}

	/// return a copy of the matrix
	RealDiagonalMatrix const& matrix() const
	{
		SHARK_CHECK(isValid(), "[Normalizer::matrix] model is not initialized");
		return m_A;
	}

	/// return the offset vector
	RealMatrix const& vector() const
	{
		SHARK_CHECK(isValid(), "[Normalizer::vector] model is not initialized");
		return m_b;
	}

	/// obtain the input dimension
	std::size_t inputSize() const
	{
		SHARK_CHECK(isValid(), "[Normalizer::inputSize] model is not initialized");
		return m_A.size1();
	}

	/// obtain the output dimension
	std::size_t outputSize() const
	{
		SHARK_CHECK(isValid(), "[Normalizer::outputSize] model is not initialized");
		return m_A.size1();
	}

	/// obtain the parameter vector
	RealVector parameterVector() const
	{
		SHARK_CHECK(isValid(), "[Normalizer::parameterVector] model is not initialized");
		std::size_t dim = m_A.size1();
		if (hasOffset())
		{
			RealVector param(2 * dim);
			for (std::size_t i=0; i<dim; i++)
			{
				param(i) = m_A(i, i);
				param(dim + i) = m_b(i);
			}
			return param;
		}
		else
		{
			RealVector param(dim);
			for (std::size_t i=0; i<dim; i++)
			{
				param(i) = m_A(i, i);
			}
			return param;
		}
	}

	/// overwrite the parameter vector
	void setParameterVector(RealVector const& newParameters)
	{
		SHARK_CHECK(isValid(), "[Normalizer::setParameterVector] model is not initialized");
		std::size_t dim = m_A.size1();
		if (hasOffset())
		{
			SIZE_CHECK(newParameters.size() == 2 * dim);
			for (std::size_t i=0; i<dim; i++)
			{
				m_A(i, i) = newParameters(i);
				m_b(i) = newParameters(dim + i);
			}
		}
		else
		{
			SIZE_CHECK(newParameters.size() == dim);
			for (std::size_t i=0; i<dim; i++)
			{
				m_A(i, i) = newParameters(i);
			}
		}
	}

	/// return the number of parameter
	std::size_t numberOfParameters() const
	{
		SHARK_CHECK(isValid(), "[Normalizer::numberOfParameters] model is not initialized");
		return (m_hasOffset) ? m_A.size1() + m_b.size() : m_A.size1();
	}

	/// overwrite structure and parameters
	void setStructure(RealDiagonalMatrix matrix)
	{
		m_A = matrix;
		m_hasOffset = false;
	}

	/// overwrite structure and parameters
	void setStructure(std::size_t dimension, bool hasOffset = false)
	{
		m_A.resize(dimension, dimension);
		m_hasOffset = hasOffset;
		if (hasOffset) m_b.resize(dimension);
	}

	/// overwrite structure and parameters
	void setStructure(RealDiagonalMatrix matrix, RealVector offset)
	{
		SHARK_CHECK(matrix.size1() == matrix.size2() && matrix.size1() == offset.size(), "[Normalizer::setStructure] dimension conflict");
		m_A = matrix;
		m_b = offset;
		m_hasOffset = true;
	}

	using base_type::eval;

	/// \brief Evaluate the model: output = matrix * input + offset.
	void eval(BatchInputType const& input, BatchOutputType& output) const
	{
		SHARK_CHECK(isValid(), "[Normalizer::eval] model is not initialized");
		output.resize(input.size1(), input.size2(), false);
		noalias(output) = prod(input, m_A);
		// fast_prod(input, m_A, output);    // not defined for dense+sparse mixtures
		if (hasOffset())
		{
			for (std::size_t i=0; i<input.size1(); i++) row(output, i) += m_b;
		}
	}

	/// \brief Evaluate the model: output = matrix * input + offset.
	void eval(BatchInputType const& input, BatchOutputType& output, State& state) const
	{
		eval(input, output);
	}

	/// from ISerializable
	void read(InArchive& archive)
	{
		// believe it or not: ublas diagonal matrix cannot be serialized... REALLY???
		// archive & m_A;

		RealVector a;
		archive & a;
		m_A.resize(a.size(), a.size(), false);
		for (std::size_t i=0; i<a.size(); i++) m_A(i, i) = a(i);

		archive & m_b;
		archive & m_hasOffset;
	}

	/// from ISerializable
	void write(OutArchive& archive) const
	{
		// believe it or not: ublas diagonal matrix cannot be serialized... REALLY???
		// archive & m_A;

		RealVector a(m_A.size1());
		for (std::size_t i=0; i<a.size(); i++) a(i) = m_A(i, i);
		archive & a;

		archive & m_b;
		archive & m_hasOffset;
	}

protected:
	RealDiagonalMatrix m_A;                ///< matrix A (see class documentation)
	RealVector m_b;                        ///< vector b (see class documentation)
	bool m_hasOffset;                      ///< if true: add offset therm b; if false: don't.
};


}
#endif
