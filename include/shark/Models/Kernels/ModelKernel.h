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

#ifndef SHARK_MODELS_KERNELS_MODEL_KERNEL_H
#define SHARK_MODELS_KERNELS_MODEL_KERNEL_H


#include <shark/Models/Kernels/AbstractKernelFunction.h>
#include <shark/LinAlg/Base.h>
#include <shark/Models/AbstractModel.h>
#include <vector>
#include <boost/scoped_ptr.hpp>

namespace shark {

namespace detail{
template<class InputType, class IntermediateType>
class ModelKernelImpl : public AbstractKernelFunction<InputType>
{
private:
	typedef AbstractKernelFunction<InputType> base_type;
public:
	typedef typename base_type::BatchInputType BatchInputType;
	typedef typename base_type::ConstInputReference ConstInputReference;
	typedef typename base_type::ConstBatchInputReference ConstBatchInputReference;
	typedef AbstractKernelFunction<IntermediateType> Kernel;
	typedef AbstractModel<InputType,IntermediateType> Model;
private:
	struct InternalState: public State{
		boost::shared_ptr<State> kernelStateX1X2;
		boost::shared_ptr<State> kernelStateX2X1;
		boost::shared_ptr<State> modelStateX1;
		boost::shared_ptr<State> modelStateX2;
		typename Model::BatchOutputType intermediateX1;
		typename Model::BatchOutputType intermediateX2;
	};
public:
	
	ModelKernelImpl(Kernel* kernel, Model* model):mpe_kernel(kernel),mpe_model(model){
		if(kernel->hasFirstParameterDerivative() 
		&& kernel->hasFirstInputDerivative() 
		&& model->hasFirstParameterDerivative())
			this->m_features|=base_type::HAS_FIRST_PARAMETER_DERIVATIVE;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "ModelKernel"; }

	std::size_t numberOfParameters()const{
		return mpe_kernel->numberOfParameters()+mpe_model->numberOfParameters();
	}
	RealVector parameterVector() const{ 
		RealVector params(numberOfParameters());
		init(params) << parameters(*mpe_kernel),parameters(*mpe_model);
		return params;
	}
	void setParameterVector(RealVector const& newParameters){ 
		SIZE_CHECK(newParameters.size() == numberOfParameters()); 
		init(newParameters) >> parameters(*mpe_kernel),parameters(*mpe_model);
	}
	
	boost::shared_ptr<State> createState()const{
		InternalState* s = new InternalState();
		boost::shared_ptr<State> sharedState(s);//create now to allow for destructor to be called in case of exception
		s->kernelStateX1X2 = mpe_kernel->createState();
		s->kernelStateX2X1 = mpe_kernel->createState();
		s->modelStateX1 = mpe_model->createState();
		s->modelStateX2 = mpe_model->createState();
		return sharedState;
	}

	double eval(ConstInputReference x1, ConstInputReference x2) const{
		return mpe_kernel->eval((*mpe_model)(x1),(*mpe_model)(x2));
	}
	
	void eval(ConstBatchInputReference x1, ConstBatchInputReference x2, RealMatrix& result, State& state) const{
		InternalState& s=state.toState<InternalState>();
		mpe_model->eval(x1,s.intermediateX1,*s.modelStateX1);
		mpe_model->eval(x2,s.intermediateX2,*s.modelStateX2);
		mpe_kernel->eval(s.intermediateX2,s.intermediateX1,result,*s.kernelStateX2X1);
		mpe_kernel->eval(s.intermediateX1,s.intermediateX2,result,*s.kernelStateX1X2);
		
	}
	
	void eval(ConstBatchInputReference batchX1, ConstBatchInputReference batchX2, RealMatrix& result) const{
		return mpe_kernel->eval((*mpe_model)(batchX1),(*mpe_model)(batchX2),result);
	}
	
	void weightedParameterDerivative(
		ConstBatchInputReference batchX1, 
		ConstBatchInputReference batchX2, 
		RealMatrix const& coefficients,
		State const& state, 
		RealVector& gradient
	) const{
		gradient.resize(numberOfParameters());
		InternalState const& s=state.toState<InternalState>();
		
		//compute derivative of the kernel wrt. parameters
		RealVector kernelGrad;
		mpe_kernel->weightedParameterDerivative(
			s.intermediateX1,s.intermediateX2,
			coefficients,*s.kernelStateX1X2,kernelGrad
		);
		//compute derivative of the kernel wrt left and right parameter
		typename Model::BatchOutputType inputDerivativeX1, inputDerivativeX2;
		mpe_kernel->weightedInputDerivative(
			s.intermediateX1,s.intermediateX2,
			coefficients,*s.kernelStateX1X2,inputDerivativeX1
		);
		mpe_kernel->weightedInputDerivative(
			s.intermediateX2,s.intermediateX1,
			trans(coefficients),*s.kernelStateX2X1,inputDerivativeX2
		);
		
		//compute derivative of model wrt parameters
		RealVector modelGradX1,modelGradX2;
		mpe_model->weightedParameterDerivative(batchX1,inputDerivativeX1,*s.modelStateX1,modelGradX1);
		mpe_model->weightedParameterDerivative(batchX2,inputDerivativeX2,*s.modelStateX2,modelGradX2);
		init(gradient) << kernelGrad, (modelGradX1+modelGradX2);
	}
	
	void read(InArchive& ar){
		if(mpe_kernel == NULL)
			throw SHARKEXCEPTION("[ModelKernel::read] the kernel function is NULL, kernel needs to be constructed prior to read in");
		if(mpe_model == NULL)
			throw SHARKEXCEPTION("[ModelKernel::read] the model is NULL, model needs to be constructed prior to read in");
		ar >> *mpe_kernel;
		ar >> *mpe_model;
	}

	void write(OutArchive& ar) const{
		ar << *mpe_kernel;
		ar << *mpe_model;
	}
	
private:
	Kernel* mpe_kernel;
	Model* mpe_model;
};
}


///  \brief Kernel function that uses a Model as transformation function for another kernel
///
/// Using an Abstractmodel \f$ f: X \rightarrow X' \f$ and an inner kernel 
/// \f$k: X' \times X' \rightarrow \mathbb{R} \f$, this class defines another kernel 
/// \f$K: X \times X \rightarrow \mathbb{R}\f$ using
/// \f[
/// K(x,y) = k(f(x),f(y))
///\f]
/// If the inner kernel \f$k\f$ suports both input, as well as parameter derivative and
/// the model also supports the parameter derivative, the kernel \f$K\f$ also
/// supports the first parameter derivative using
/// \f[
/// \frac{\partial}{\partial \theta} K(x,y) = 
///	\frac{\partial}{\partial f(x)} k(f(x),f(y))\frac{\partial}{\partial \theta} f(x)
///	+\frac{\partial}{\partial f(y)} k(f(x),f(y))\frac{\partial}{\partial \theta} f(y)
///\f]
/// This requires the derivative of the inputs of the kernel wrt both parameters which,
/// by limitation of the current kernel interface, requires to compute \f$k(f(x),f(y))\f$ and \f$k(f(y),f(x))\f$. 
template<class InputType=RealVector>
class ModelKernel: public AbstractKernelFunction<InputType>{
private:
	typedef AbstractKernelFunction<InputType> base_type;
public:	
	typedef typename base_type::BatchInputType BatchInputType;
	typedef typename base_type::ConstInputReference ConstInputReference;
	typedef typename base_type::ConstBatchInputReference ConstBatchInputReference;
	
	template<class IntermediateType>
	ModelKernel(
		AbstractKernelFunction<IntermediateType>* kernel, 
		AbstractModel<InputType,IntermediateType>* model
	):m_wrapper(new detail::ModelKernelImpl<InputType,IntermediateType>(kernel,model)){
		if(m_wrapper->hasFirstParameterDerivative())
			this->m_features|=base_type::HAS_FIRST_PARAMETER_DERIVATIVE;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "ModelKernel"; }

	/// \brief Returns the number of parameters.
	std::size_t numberOfParameters()const{
		return m_wrapper->numberOfParameters();
	}
	///\brief Returns the concatenated parameters of kernel and model.
	RealVector parameterVector() const{ 
		return m_wrapper->parameterVector();
	}
	///\brief Sets the concatenated parameters of kernel and model.
	void setParameterVector(RealVector const& newParameters){ 
		m_wrapper->setParameterVector(newParameters);
	}
	
	///\brief Returns the internal state object used for eval and the derivatives.
	boost::shared_ptr<State> createState()const{
		return m_wrapper->createState();
	}

	///\brief Computes K(x,y) for a single input pair.
	double eval(ConstInputReference x1, ConstInputReference x2) const{
		return m_wrapper->eval(x1,x2);
	}
	
	/// \brief For two batches X1 and X2 computes the matrix k_ij=K(X1_i,X2_j) and stores the state for the derivatives.
	void eval(ConstBatchInputReference batchX1, ConstBatchInputReference batchX2, RealMatrix& result, State& state) const{
		return m_wrapper->eval(batchX1,batchX2,result,state);
	}
	/// \brief For two batches X1 and X2 computes the matrix k_ij=K(X1_i,X2_j).
	void eval(ConstBatchInputReference batchX1, ConstBatchInputReference batchX2, RealMatrix& result) const{
		m_wrapper->eval(batchX1,batchX2,result);
	}
	
	///\brief After a call to eval with state, computes the derivative wrt all parameters of the kernel and the model.
	///
	/// This is computed over the whole kernel matrix k_ij created by eval and summed up using the coefficients c
	/// thus this call returns \f$ \sum_{i,j} c_{ij} \frac{\partial}{\partial \theta} k(x^1_i,x^2_j)\f$.
	void weightedParameterDerivative(
		ConstBatchInputReference batchX1, 
		ConstBatchInputReference batchX2, 
		RealMatrix const& coefficients,
		State const& state, 
		RealVector& gradient
	) const{
		m_wrapper->weightedParameterDerivative(batchX1,batchX2,coefficients,state,gradient);
	}

	///\brief Stores the kernel to an Archive.
	void write(OutArchive& ar) const{
		ar<< *m_wrapper;
	}
	///\brief Reads the kernel from an Archive.
	void read(OutArchive& ar) const{
		ar >> *m_wrapper;
	}
private:
	boost::scoped_ptr<AbstractKernelFunction<InputType> > m_wrapper;
};

typedef ModelKernel<RealVector> DenseModelKernel;
typedef ModelKernel<CompressedRealVector> CompressedModelKernel;


}
#endif
