//===========================================================================
/*!
 * 
 *
 * \brief       Kernel on a finite, discrete space.
 * 
 * 
 *
 * \author      T. Glasmachers
 * \date        2012
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
//===========================================================================

#ifndef SHARK_MODELS_KERNELS_DISCRETEKERNEL_H
#define SHARK_MODELS_KERNELS_DISCRETEKERNEL_H


#include <shark/Models/Kernels/AbstractKernelFunction.h>
#include <shark/LinAlg/Base.h>
#include <vector>

namespace shark {


///
/// \brief Kernel on a finite, discrete space.
///
/// \par
/// This class represents a kernel function on a finite
/// space with N elements. Wlog, this space is represented
/// by the integers 0, ..., N-1. The kernel function is
/// defined by a symmetric, positive semi-definite N x N
/// matrix.
///
class DiscreteKernel : public AbstractKernelFunction<std::size_t>
{
public:
	typedef AbstractKernelFunction<std::size_t> base_type;

	/// \brief Construction of the kernel from a positive definite, symmetric matrix.
	DiscreteKernel(RealMatrix const& matrix)
	: m_matrix(matrix)
	{
		SHARK_CHECK(matrix.size1() == matrix.size2(), "[DiscreteKernel::DiscreteKernel] kernel matrix must be square");
#ifdef DEBUG
		for (std::size_t i=0; i<matrix.size1(); i++)
		{
			for (std::size_t j=0; j<i; j++)
			{
				SHARK_CHECK(matrix(i, j) == matrix(j, i), "[DiscreteKernel::DiscreteKernel] kernel matrix must be symmetric");
			}
		}
#endif
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "DiscreteKernel"; }

	RealVector parameterVector() const
	{
		return RealVector();
	}

	void setParameterVector(RealVector const& newParameters)
	{
		SIZE_CHECK(newParameters.size() == 0);
	}

	std::size_t numberOfParameters() const
	{
		return 0;
	}

	/// \brief Cardinality of the discrete space.
	std::size_t size() const
	{ return m_matrix.size1(); }
	
	///\brief DiscreteKernels don't have a state so they return an EmptyState object
	boost::shared_ptr<State> createState()const{
		return boost::shared_ptr<State>(new EmptyState());
	}

	/// \brief Evaluates the kernel function.
	///
	/// The function returns the stored similarity value.
	double eval(ConstInputReference x1, ConstInputReference x2)const
	{
		return m_matrix(x1, x2);
	}
	
	/// \brief Evaluates the kernel function for every point combination of the two batches
	///
	/// The function returns the stored similarity value.
	void eval(ConstBatchInputReference batchX1, ConstBatchInputReference batchX2, RealMatrix& result, State& state) const{
		eval(batchX1,batchX2,result);
	}
	/// \brief Evaluates the kernel function for every point combination of the two batches
	///
	/// The function returns the stored similarity value.
	void eval(ConstBatchInputReference batchX1, ConstBatchInputReference batchX2, RealMatrix& result) const{
		std::size_t sizeX1=shark::size(batchX1);
		std::size_t sizeX2=shark::size(batchX2);
		ensure_size(result,sizeX1,sizeX2);
		for(std::size_t i = 0; i != sizeX1; ++i)
			for(std::size_t j = 0; j != sizeX2; ++j)
				result(i,j)=m_matrix(i,j); 
	}


	/// From ISerializable.
	void read(InArchive& ar)
	{ ar >> m_matrix; }

	/// From ISerializable.
	void write(OutArchive& ar) const
	{ ar << m_matrix; }

protected:
	/// kernel matrix
	RealMatrix m_matrix;
};


}
#endif
