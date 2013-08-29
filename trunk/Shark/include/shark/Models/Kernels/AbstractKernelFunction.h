//===========================================================================
/*!
*
*  \brief abstract super class of all kernel functions
*
*  \author  T.Glasmachers, O. Krause, M. Tuma
*  \date    2010-2012
*
*  \par Copyright (c) 1999-2012:
*      Institut f&uuml;r Neuroinformatik<BR>
*      Ruhr-Universit&auml;t Bochum<BR>
*      D-44780 Bochum, Germany<BR>
*      Phone: +49-234-32-27974<BR>
*      Fax:   +49-234-32-14209<BR>
*      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
*      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
*
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
//===========================================================================

#ifndef SHARK_MODELS_KERNELS_ABSTRACTKERNELFUNCTION_H
#define SHARK_MODELS_KERNELS_ABSTRACTKERNELFUNCTION_H


#include <cmath>
#include <shark/LinAlg/Base.h>
#include <shark/Data/BatchInterface.h>
#include <shark/Core/IParameterizable.h>
#include <shark/Core/ISerializable.h>
#include <shark/Core/IConfigurable.h>
#include <shark/Core/INameable.h>
#include <shark/Core/Flags.h>
#include <shark/Core/State.h>
#include <shark/Core/Traits/ProxyReferenceTraits.h>
namespace shark {

#ifdef SHARK_COUNT_KERNEL_LOOKUPS
	#define INCREMENT_KERNEL_COUNTER( counter ) { counter++; }
#else
	#define INCREMENT_KERNEL_COUNTER( counter ) {  }
#endif

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
///
template<class InputTypeT>
class AbstractKernelFunction : public INameable, public IParameterizable, public ISerializable, public IConfigurable
{
private:
	/// \brief Meta type describing properties of batches.
	typedef Batch<InputTypeT> Traits;

public:
	/// \brief  Input type of the Kernel.
	typedef InputTypeT InputType;
	/// \brief batch input type of the kernel
	typedef typename Traits::type BatchInputType;
	/// \brief Const references to InputType
	typedef typename ConstProxyReference<InputType const>::type ConstInputReference;
	/// \brief Const references to BatchInputType
	typedef typename ConstProxyReference<BatchInputType const>::type ConstBatchInputReference;

	AbstractKernelFunction() { }
	virtual ~AbstractKernelFunction() { }

	/// configure the kernel
	void configure( PropertyTree const& node ){}

	/// enumerations of kernel features (flags)
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

	/// \brief From ISerializable, reads a kernel from an archive.
	virtual void read( InArchive & archive ){
		m_features.read(archive);
		RealVector p;
		archive & p;
		setParameterVector(p);
	}

	/// \brief From ISerializable, writes a kernel to an archive.
	///
	/// The default implementation just saves the parameters.
	virtual void write( OutArchive & archive ) const{
		m_features.write(archive);
		RealVector p = parameterVector();
		archive & p;
	}

	///\brief Creates an internal state of the kernel.
	///
	///The state is needed when the derivatives are to be
	///calculated. Eval can store a state which is then reused to speed up
	///the calculations of the derivatives. This also allows eval to be
	///evaluated in parallel!
	virtual boost::shared_ptr<State> createState()const
	{
		if (hasFirstParameterDerivative() || hasFirstInputDerivative())
		{
			throw SHARKEXCEPTION("[AbstractKernelFunction::createState] createState must be overridden by kernels with derivatives");
		}
		return boost::shared_ptr<State>(new EmptyState());
	}

	///////////////////////////////////////////SINGLE ELEMENT INTERFACE///////////////////////////////////////////
	// By default, this is mapped to the batch case.

	/// \brief Evaluates the kernel function.
	virtual double eval(ConstInputReference x1, ConstInputReference x2) const{
		RealMatrix res;
		BatchInputType b1 = Traits::createBatch(x1,1);
		BatchInputType b2 = Traits::createBatch(x2,1);
		get(b1,0) = x1;
		get(b2,0) = x2;
		eval(b1, b2, res);
		return res(0, 0);
	}

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
		throw SHARKEXCEPTION("[AbstractKernelFunction::weightedParameterDerivative] weightedParameterDerivative(...) not implemented.");
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
		throw SHARKEXCEPTION("[AbstractKernelFunction::weightedInputDerivative] weightedInputDerivative(...) not implemented");
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
		std::size_t sizeX1=shark::size(batchX1);
		std::size_t sizeX2=shark::size(batchX2);
		RealMatrix result=(*this)(batchX1,batchX2);
		result*=-2;
		if (isNormalized()){
			noalias(result)+=RealScalarMatrix(sizeX1,sizeX2,2.0);
		} else {
			//compute self-product
			RealVector kx2(sizeX2);
			for(std::size_t i = 0; i != sizeX2;++i){
				kx2(i)=eval(get(batchX2,i),get(batchX2,i));
			}
			for(std::size_t j = 0; j != sizeX1;++j){
				double kx1=eval(get(batchX1,j),get(batchX1,j));
				noalias(row(result,j))+= blas::repeat(kx1,sizeX2)+kx2;
			}
		}
		return result;
	}
	

	/// \brief Computes the distance in the kernel induced feature space.
	double featureDistance(ConstInputReference x1, ConstInputReference x2) const {
		return std::sqrt(featureDistanceSqr(x1, x2));
	}
};


}
#endif
