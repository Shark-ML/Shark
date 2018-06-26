//===========================================================================
/*!
 * 
 *
 * \brief       Affine linear kernel function expansion
 * 
 * \par
 * Affine linear kernel expansions resulting from Support
 * vector machine (SVM) training and other kernel methods.
 * 
 * 
 * 
 *
 * \author      T. Glasmachers
 * \date        2007-2011
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


#ifndef SHARK_MODELS_KERNELEXPANSION_H
#define SHARK_MODELS_KERNELEXPANSION_H

#include <shark/Models/Classifier.h>
#include <shark/Models/Kernels/AbstractKernelFunction.h>
#include <shark/Data/Dataset.h>
#include <shark/Data/DataView.h>

namespace shark {


///
/// \brief Linear model in a kernel feature space.
///
/// An affine linear kernel expansion is a model of the type
/// \f[ x : \mathbb{R}^d \to \mathbb{R}^d \enspace , \enspace x \mapsto \sum_{n=1}^{\ell} \alpha_n k(x_n, x) + b \enspace ,\f]
/// with parameters \f$ \alpha_n \in \mathbb{R}^{d} \f$ for all
/// \f$ n \in \{1, \dots, \ell\} \f$ and \f$ b \in \mathbb{R}^d \f$.
///
/// One way in which the possibility for vector-valued input and output of dimension \f$ d \f$ may be interpreted
/// is as allowing for a KernelExpansion model for \f$ d \f$ different classes/outputs in multi-class problems. Then,
/// the i-th column of the matrix #m_alpha is the KernelExpansion for class/output i.
///
/// For a choice of kernel, see \ref kernels.
///
/// \tparam InputType Type of basis elements supplied to the kernel
/// \ingroup models
template<class InputType>
class KernelExpansion : public AbstractModel<InputType, RealVector>
{
public:
	typedef AbstractKernelFunction<InputType> KernelType;
	typedef AbstractModel<InputType, RealVector> base_type;
	typedef typename base_type::BatchInputType BatchInputType;
	typedef typename base_type::BatchOutputType BatchOutputType;

	// //////////////////////////////////////////////////////////
	// ////////////      CONSTRUCTORS       /////////////////////
	// //////////////////////////////////////////////////////////

	KernelExpansion():mep_kernel(NULL){}
		
	KernelExpansion(KernelType* kernel):mep_kernel(kernel){
		SHARK_ASSERT(kernel != NULL);
	}
		
	KernelExpansion(KernelType* kernel, Data<InputType> const& basis,bool offset, std::size_t outputs = 1){
		SHARK_ASSERT(kernel != NULL);
		setStructure(kernel, basis,offset,outputs);
	}
	
	void setStructure(KernelType* kernel, Data<InputType> const& basis,bool offset, std::size_t outputs = 1){
		SHARK_ASSERT(kernel != NULL);
		mep_kernel = kernel;
		if(offset)
			m_b.resize(outputs);
		m_basis = basis;
		m_alpha.resize(basis.numberOfElements(), outputs);
		m_alpha.clear();
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "KernelExpansion"; }

	/// dimensionality of the output RealVector
	Shape outputShape() const{
		return m_alpha.size2();
	}
	
	Shape inputShape() const{
		return Shape();
	}

	// //////////////////////////////////////////////////////////
	// ///////////   ALL THINGS KERNEL     //////////////////////
	// //////////////////////////////////////////////////////////

	KernelType const* kernel() const{
		return mep_kernel;
	}
	KernelType* kernel(){
		return mep_kernel;
	}
	void setKernel(KernelType* kernel){
		mep_kernel = kernel;
	}

	// //////////////////////////////////////////////////////////
	// ///////    ALL THINGS ALPHA AND OFFSET    ////////////////
	// //////////////////////////////////////////////////////////

	bool hasOffset() const{
		return m_b.size() != 0;
	}
	RealMatrix& alpha(){
		return m_alpha;
	}
	RealMatrix const& alpha() const{
		return m_alpha;
	}
	double& alpha(std::size_t example, std::size_t cls){
		return m_alpha(example, cls);
	}
	double const& alpha(std::size_t example, std::size_t cls) const{
		return m_alpha(example, cls);
	}
	RealVector& offset(){
		SHARK_RUNTIME_CHECK(hasOffset(), "[KernelExpansion::offset] invalid call for object without offset term");
		return m_b;
	}
	RealVector const& offset() const{
		SHARK_RUNTIME_CHECK(hasOffset(), "[KernelExpansion::offset] invalid call for object without offset term");
		return m_b;
	}
	double& offset(std::size_t cls){
		SHARK_RUNTIME_CHECK(hasOffset(), "[KernelExpansion::offset] invalid call for object without offset term");
		return m_b(cls);
	}
	double const& offset(std::size_t cls) const{
		SHARK_RUNTIME_CHECK(hasOffset(), "[KernelExpansion::offset] invalid call for object without offset term");
		return m_b(cls);
	}

	// //////////////////////////////////////////////////////////
	// ////////    ALL THINGS UNDERLYING DATA    ////////////////
	// //////////////////////////////////////////////////////////


	Data<InputType> const& basis() const {
		return m_basis;
	}

	Data<InputType>& basis() {
        return m_basis;
    }
    
    /// The sparsify method removes non-support-vectors from
	/// its set of basis vectors and the coefficient matrix.
	void sparsify(){
		std::size_t ic = m_basis.numberOfElements();
		std::vector<std::size_t> svIndices;
		for (std::size_t i=0; i != ic; ++i){
			if (blas::norm_1(row(m_alpha, i)) > 0.0){
				svIndices.push_back(i);
			}
		}
		//project basis on the support vectors
		m_basis = toDataset(subset(elements(m_basis),svIndices));
		
		//reduce alpha to it's support vector variables
		RealMatrix a(svIndices.size(), m_alpha.size2());
		for (std::size_t i=0; i!= svIndices.size(); ++i){
			noalias(row(a,i)) = row(m_alpha,svIndices[i]); 
		}
		swap(m_alpha,a);
	}

	// //////////////////////////////////////////////////////////
	// ////////    ALL THINGS KERNEL PARAMETERS    //////////////
	// //////////////////////////////////////////////////////////

	RealVector parameterVector() const{
		if (hasOffset()){
			return  to_vector(m_alpha) | m_b;
		}
		else{
			return to_vector(m_alpha);
		}
	}

	void setParameterVector(RealVector const& newParameters){
		SHARK_RUNTIME_CHECK(newParameters.size() == numberOfParameters(), "Invalid size of the parameter vector");
		std::size_t numParams = m_alpha.size1() * m_alpha.size2();
		noalias(to_vector(m_alpha)) = subrange(newParameters, 0, numParams);
		if (hasOffset())
			noalias(m_b) = subrange(newParameters, numParams, numParams + m_b.size());
	}

	std::size_t numberOfParameters() const{
		if (hasOffset()) 
			return m_alpha.size1() * m_alpha.size2() + m_b.size();
		else 
			return m_alpha.size1() * m_alpha.size2();
	}

	// //////////////////////////////////////////////////////////
	// ////////       ALL THINGS EVALUATION        //////////////
	// //////////////////////////////////////////////////////////
	
	boost::shared_ptr<State> createState()const{
		return boost::shared_ptr<State>(new EmptyState());
	}

	using AbstractModel<InputType, RealVector>::eval;
	void eval(BatchInputType const& patterns, BatchOutputType& output)const{
		std::size_t numPatterns = batchSize(patterns);
		SHARK_ASSERT(mep_kernel != NULL);

		output.resize(numPatterns,m_alpha.size2());
		if (hasOffset())
			output = repeat(m_b,numPatterns);
		else
			output.clear();

		std::size_t batchStart = 0;
		for (std::size_t i=0; i != m_basis.numberOfBatches(); i++){
			std::size_t batchEnd = batchStart+batchSize(m_basis.batch(i));
			//evaluate kernels
			//results in a matrix of the form where a column consists of the kernel evaluation of 
			//pattern i with respect to the batch of the basis,this gives a good memory alignment
			//in the following matrix matrix product
			RealMatrix kernelEvaluations = (*mep_kernel)(m_basis.batch(i),patterns);
			
			//get the part of the alpha matrix which is suitable for this batch
			auto batchAlpha = subrange(m_alpha,batchStart,batchEnd,0,m_alpha.size2());
			noalias(output) += prod(trans(kernelEvaluations),batchAlpha);
			batchStart = batchEnd;
		}
	}
	void eval(BatchInputType const& patterns, BatchOutputType& outputs, State & state)const{
		eval(patterns, outputs);
	}

	// //////////////////////////////////////////////////////////
	// ////////      ALL THINGS SERIALIZATION      //////////////
	// //////////////////////////////////////////////////////////

	/// From ISerializable, reads a model from an archive
	void read( InArchive & archive ){
		SHARK_ASSERT(mep_kernel != NULL);

		archive >> m_alpha;
		archive >> m_b;
		archive >> m_basis;
		archive >> (*mep_kernel);
	}

	/// From ISerializable, writes a model to an archive
	void write( OutArchive & archive ) const{
		SHARK_ASSERT(mep_kernel != NULL);

		archive << m_alpha;
		archive << m_b;
		archive << m_basis;
		archive << const_cast<KernelType const&>(*mep_kernel);//prevent compilation warning
	}

// //////////////////////////////////////////////////////////
// ////////              MEMBERS               //////////////
// //////////////////////////////////////////////////////////

protected:
	/// kernel function used in the expansion
	KernelType* mep_kernel;

	/// "support" basis vectors
	Data<InputType> m_basis;
	
	/// kernel coefficients
	RealMatrix m_alpha;

	/// offset or bias term
	RealVector m_b;
};

///
/// \brief Linear classifier in a kernel feature space.
///
/// This model is a simple wrapper for the KernelExpansion calculating the arg max
/// of the outputs of the model. This is the model used by kernel classifier models like SVMs.
///
/// \ingroup models
template<class InputType>
struct KernelClassifier: public Classifier<KernelExpansion<InputType> >{
	typedef AbstractKernelFunction<InputType> KernelType;
	typedef KernelExpansion<InputType> KernelExpansionType;

	KernelClassifier()
	{ }
	KernelClassifier(KernelType* kernel)
	: Classifier<KernelExpansion<InputType> >(KernelExpansionType(kernel))
	{ }
	KernelClassifier(KernelExpansionType const& decisionFunction)
	: Classifier<KernelExpansion<InputType> >(decisionFunction)
	{ }

	std::string name() const
	{ return "KernelClassifier"; }
};


}
#endif
