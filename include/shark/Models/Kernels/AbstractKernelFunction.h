//===========================================================================
/*!
 * 
 *
 * \brief       abstract super class of all kernel functions
 * \file
 * 
 *
 * \author      T.Glasmachers, O. Krause, M. Tuma
 * \date        2010-2012
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

#ifndef SHARK_MODELS_KERNELS_ABSTRACTKERNELFUNCTION_H
#define SHARK_MODELS_KERNELS_ABSTRACTKERNELFUNCTION_H

#include <shark/Models/Kernels/AbstractMetric.h>
#include <shark/LinAlg/Base.h>
#include <shark/Core/Flags.h>
#include <shark/Core/State.h>
namespace shark {

#ifdef SHARK_COUNT_KERNEL_LOOKUPS
	#define INCREMENT_KERNEL_COUNTER( counter ) { counter++; }
#else
	#define INCREMENT_KERNEL_COUNTER( counter ) {  }
#endif
	
///\defgroup kernels Kernels
///\ingroup models
///
/// A kernel is a positive definite function k(x,y), which can be understood as a generalized scalar product. Kernel methods.
/// like support vector machines or gaussian processes rely on the kernels.

/// \brief Base class of all Kernel functions.
///
/// \par
/// A (Mercer) kernel is a symmetric positive definite
/// function of two parameters. It is (currently) used
/// in two contexts in Shark, namely for kernel methods
/// such as support vector machines (SVMs), and for
/// radial basis function networks.
///
/// \par
/// In Shark a kernel function class represents a parametric
/// family of such kernel functions: The AbstractKernelFunction
/// interface inherits the IParameterizable interface.
/// \ingroup kernels
template<class InputTypeT>
class AbstractKernelFunction : public AbstractMetric<InputTypeT>
{
private:
	typedef AbstractMetric<InputTypeT> base_type;
	typedef Batch<InputTypeT> Traits;
public:
	/// \brief  Input type of the Kernel.
	typedef typename base_type::InputType InputType;
	/// \brief batch input type of the kernel
	typedef  typename base_type::BatchInputType BatchInputType;
	/// \brief Const references to InputType
	typedef typename base_type::ConstInputReference ConstInputReference;
	/// \brief Const references to BatchInputType
	typedef typename base_type::ConstBatchInputReference ConstBatchInputReference;

	AbstractKernelFunction() { }
	
	/// enumerations of kerneland metric features (flags)
	enum Feature {
		HAS_FIRST_PARAMETER_DERIVATIVE = 1,    ///< is the kernel differentiable w.r.t. its parameters?
		HAS_FIRST_INPUT_DERIVATIVE 	   = 2,    ///< is the kernel differentiable w.r.t. its inputs?
		IS_NORMALIZED                  = 4 ,   ///< does k(x, x) = 1 hold for all inputs x?
		SUPPORTS_VARIABLE_INPUT_SIZE = 8 ///< Input arguments must have same size, but not the same size in different calls to eval
	};
	
	/// This statement declares the member m_features. See Core/Flags.h for details.
	SHARK_FEATURE_INTERFACE;
	
	bool hasFirstParameterDerivative()const{
		return m_features & HAS_FIRST_PARAMETER_DERIVATIVE;
	}
	bool hasFirstInputDerivative()const{
		return m_features & HAS_FIRST_INPUT_DERIVATIVE;
	}
	bool isNormalized() const{
		return m_features & IS_NORMALIZED;
	}
	bool supportsVariableInputSize() const{
		return m_features & SUPPORTS_VARIABLE_INPUT_SIZE;
	}

	///\brief Creates an internal state of the kernel.
	///
	///The state is needed when the derivatives are to be
	///calculated. Eval can store a state which is then reused to speed up
	///the calculations of the derivatives. This also allows eval to be
	///evaluated in parallel!
	virtual boost::shared_ptr<State> createState()const
	{
		SHARK_RUNTIME_CHECK(!hasFirstParameterDerivative() && !hasFirstInputDerivative(), "createState must be overridden by kernels with derivatives");
		return boost::shared_ptr<State>(new EmptyState());
	}

	///////////////////////////////////////////SINGLE ELEMENT INTERFACE///////////////////////////////////////////

	/// \brief Evaluates the kernel function.
	virtual double eval(ConstInputReference x1, ConstInputReference x2) const = 0;

	/// \brief Convenience operator which evaluates the kernel function.
	inline double operator () (ConstInputReference x1, ConstInputReference x2) const {
		return eval(x1, x2);
	}

	//////////////////////////////////////BATCH INTERFACE///////////////////////////////////////////
	
	/// \brief Evaluates the subset of the KernelGram matrix which is defined by X1(rows) and X2 (columns).
	///
	/// The result matrix is filled in with the values result(i,j) = kernel(x1[i], x2[j]);
	/// The State object is filled in with data used in subsequent derivative computations.
	virtual void eval(ConstBatchInputReference batchX1, ConstBatchInputReference batchX2, RealMatrix& result, State& state) const = 0;

	/// \brief Evaluates the subset of the KernelGram matrix which is defined by X1(rows) and X2 (columns).
	///
	/// The result matrix is filled in with the values result(i,j) = kernel(x1[i], x2[j]);
	virtual void eval(ConstBatchInputReference batchX1, ConstBatchInputReference batchX2, RealMatrix& result) const {
		boost::shared_ptr<State> state = createState();
		eval(batchX1, batchX2, result, *state);
	}

	/// \brief Evaluates the subset of the KernelGram matrix which is defined by X1(rows) and X2 (columns).
	///
	/// Convenience operator.
	/// The result matrix is filled in with the values result(i,j) = kernel(x1[i], x2[j]);
	inline RealMatrix operator () (ConstBatchInputReference batchX1, ConstBatchInputReference batchX2) const{ 
		RealMatrix result;
		eval(batchX1, batchX2, result);
		return result;
	}

	/// \brief Computes the gradient of the parameters as a weighted sum over the gradient of all elements of the batch.
	///
	/// The default implementation throws a "not implemented" exception.
	virtual void weightedParameterDerivative(
		ConstBatchInputReference batchX1, 
		ConstBatchInputReference batchX2, 
		RealMatrix const& coefficients,
		State const& state, 
		RealVector& gradient
	) const {
		SHARK_FEATURE_EXCEPTION(HAS_FIRST_PARAMETER_DERIVATIVE);
	}

	/// \brief Calculates the derivative of the inputs X1 (only x1!).
	///
	/// The i-th row of the resulting matrix is a weighted sum of the form:
	/// c[i,0] * k'(x1[i], x2[0]) + c[i,1] * k'(x1[i], x2[1]) + ... + c[i,n] * k'(x1[i], x2[n]).
	///
	/// The default implementation throws a "not implemented" exception.
	virtual void weightedInputDerivative( 
		ConstBatchInputReference batchX1, 
		ConstBatchInputReference batchX2, 
		RealMatrix const& coefficientsX2,
		State const& state, 
		BatchInputType& gradient
	) const {
		SHARK_FEATURE_EXCEPTION(HAS_FIRST_INPUT_DERIVATIVE);
	}


	//////////////////////////////////NORMS AND DISTANCES/////////////////////////////////

	/// Computes the squared distance in the kernel induced feature space.
	virtual double featureDistanceSqr(ConstInputReference x1, ConstInputReference x2) const{
		if (isNormalized()){
			double k12 = eval(x1, x2);
			return (2.0 - 2.0 * k12);
		} else {
			double k11 = eval(x1, x1);
			double k12 = eval(x1, x2);
			double k22 = eval(x2, x2);
			return (k11 - 2.0 * k12 + k22);
		}
	}
	
	virtual RealMatrix featureDistanceSqr(ConstBatchInputReference batchX1,ConstBatchInputReference batchX2) const{
		std::size_t sizeX1 = batchSize(batchX1);
		std::size_t sizeX2 = batchSize(batchX2);
		RealMatrix result=(*this)(batchX1,batchX2);
		result *= -2.0;
		if (isNormalized()){
			noalias(result) += 2.0;
		} else {
			//compute self-product
			RealVector kx2(sizeX2);
			for(std::size_t i = 0; i != sizeX2;++i){
				kx2(i)=eval(getBatchElement(batchX2,i),getBatchElement(batchX2,i));
			}
			for(std::size_t j = 0; j != sizeX1;++j){
				double kx1=eval(getBatchElement(batchX1,j),getBatchElement(batchX1,j));
				noalias(row(result,j)) += kx1 + kx2;
			}
		}
		return result;
	}
};


}
#endif
