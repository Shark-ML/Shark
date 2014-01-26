//===========================================================================
/*!
 * 
 *
 * \brief       Product of kernel functions.
 * 
 * 
 *
 * \author      T. Glasmachers, O.Krause
 * \date        2012
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

#ifndef SHARK_MODELS_KERNELS_PRODUCTKERNEL_H
#define SHARK_MODELS_KERNELS_PRODUCTKERNEL_H


#include <shark/Models/Kernels/AbstractKernelFunction.h>

namespace shark{


///
/// \brief Product of kernel functions.
///
/// \par
/// The product of any number of kernels is again a valid kernel.
/// This class supports a kernel af the form
/// \f$ k(x, x') = k_1(x, x') \cdot k_2(x, x') \cdot \dots \cdot k_n(x, x') \f$
/// for any number of base kernels. All kernels need to be defined
/// on the same input space.
///
/// \par
/// Derivatives are currently not implemented. Only the plain
/// kernel value can be computed. Everyone is free to add this
/// functionality :)
///
template<class InputType>
class ProductKernel : public AbstractKernelFunction<InputType>
{
private:
	typedef AbstractKernelFunction<InputType> base_type;
public:
	typedef AbstractKernelFunction<InputType> SubKernel;
	typedef typename base_type::BatchInputType BatchInputType;
	typedef typename base_type::ConstInputReference ConstInputReference;
	typedef typename base_type::ConstBatchInputReference ConstBatchInputReference;
	/// \brief Default constructor.
	ProductKernel(){
		// this->m_features |= base_type::HAS_FIRST_PARAMETER_DERIVATIVE;
		// this->m_features |= base_type::HAS_SECOND_PARAMETER_DERIVATIVE;
		// this->m_features |= base_type::HAS_FIRST_INPUT_DERIVATIVE;
		// this->m_features |= base_type::HAS_SECOND_INPUT_DERIVATIVE;
		this->m_features |= base_type::IS_NORMALIZED;    // an "empty" product is a normalized kernel (k(x, x) = 1).
	}

	/// \brief Constructor for a product of two kernels.
	ProductKernel(SubKernel* k1, SubKernel* k2){
		// this->m_features |= base_type::HAS_FIRST_PARAMETER_DERIVATIVE;
		// this->m_features |= base_type::HAS_SECOND_PARAMETER_DERIVATIVE;
		// this->m_features |= base_type::HAS_FIRST_INPUT_DERIVATIVE;
		// this->m_features |= base_type::HAS_SECOND_INPUT_DERIVATIVE;
		this->m_features |= base_type::IS_NORMALIZED;    // an "empty" product is a normalized kernel (k(x, x) = 1).
		addKernel(k1);
		addKernel(k2);
	}
	ProductKernel(std::vector<SubKernel*> kernels){
		// this->m_features |= base_type::HAS_FIRST_PARAMETER_DERIVATIVE;
		// this->m_features |= base_type::HAS_SECOND_PARAMETER_DERIVATIVE;
		// this->m_features |= base_type::HAS_FIRST_INPUT_DERIVATIVE;
		// this->m_features |= base_type::HAS_SECOND_INPUT_DERIVATIVE;
		this->m_features |= base_type::IS_NORMALIZED;    // an "empty" product is a normalized kernel (k(x, x) = 1).
		for(std::size_t i = 0; i != kernels.size(); ++i)
			addKernel(kernels[i]);
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "ProductKernel"; }

	/// \brief Add one more kernel to the expansion.
	///
	/// \param  k  The pointer is expected to remain valid during the lifetime of the ProductKernel object.
	///
	void addKernel(SubKernel* k){
		SHARK_ASSERT(k != NULL);

		m_kernels.push_back(k);
		m_numberOfParameters += k->numberOfParameters();
		if (! k->isNormalized()) this->m_features.reset(base_type::IS_NORMALIZED);   // products of normalized kernels are normalized.
	}

	RealVector parameterVector() const{
		RealVector ret(m_numberOfParameters);
		init(ret) << parameterSet(m_kernels);
		return ret;
	}

	void setParameterVector(RealVector const& newParameters){
		SIZE_CHECK(newParameters.size() == m_numberOfParameters);
		init(newParameters) >> parameterSet(m_kernels);
	}

	std::size_t numberOfParameters() const{
		return m_numberOfParameters;
	}

	/// \brief evaluates the kernel function
	///
	/// This function returns the product of all sub-kernels.
	double eval(ConstInputReference x1, ConstInputReference x2) const{
		double prod = 1.0;
		for (std::size_t i=0; i<m_kernels.size(); i++) 
			prod *= m_kernels[i]->eval(x1, x2);
		return prod;
	}
	
	void eval(ConstBatchInputReference batchX1, ConstBatchInputReference batchX2, RealMatrix& result) const{
		std::size_t sizeX1=shark::size(batchX1);
		std::size_t sizeX2=shark::size(batchX2);
		
		//evaluate first kernel to initialize the result
		m_kernels[0]->eval(batchX1,batchX2,result);
		
		RealMatrix kernelResult(sizeX1,sizeX2);
		for(std::size_t i = 1; i != m_kernels.size(); ++i){
			m_kernels[i]->eval(batchX1,batchX2,kernelResult);
			noalias(result) *= kernelResult;
		}
	}
	
	void eval(ConstBatchInputReference batchX1, ConstBatchInputReference batchX2, RealMatrix& result, State& state) const{
		eval(batchX1,batchX2,result);
	}
		
	/// From ISerializable.
	void read(InArchive& ar){
		for(std::size_t i = 0;i != m_kernels.size(); ++i ){
			ar >> *m_kernels[i];
		}
		ar >> m_numberOfParameters;
	}

	/// From ISerializable.
	void write(OutArchive& ar) const{
		for(std::size_t i = 0;i != m_kernels.size(); ++i ){
			ar << const_cast<AbstractKernelFunction<InputType> const&>(*m_kernels[i]);//prevent serialization warning
		}
		ar << m_numberOfParameters;
	}

protected:
	std::vector<SubKernel*> m_kernels;           ///< vector of sub-kernels
	std::size_t m_numberOfParameters;        ///< total number of parameters in the product (this is redundant information)
};


}
#endif
