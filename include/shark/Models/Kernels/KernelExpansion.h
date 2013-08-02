//===========================================================================
/*!
 *  \brief Affine linear kernel function expansion
 *
 *  \par
 *  Affine linear kernel expansions resulting from Support
 *  vector machine (SVM) training and other kernel methods.
 *
 *
 *  \author  T. Glasmachers
 *  \date    2007-2011
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


#ifndef SHARK_MODELS_KERNELEXPANSION_H
#define SHARK_MODELS_KERNELEXPANSION_H

#include <shark/Models/AbstractModel.h>
#include <shark/Models/Kernels/AbstractKernelFunction.h>
#include <shark/LinAlg/BLAS/Initialize.h>
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
/// \tparam InputType Type of basis elements supplied to the kernel
///
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

	KernelExpansion(bool offset, unsigned int outputs = 1)
	: mep_kernel(NULL), m_offset(offset), m_outputs(outputs){
		m_alpha.resize(0, outputs);
		if (m_offset) m_b = RealZeroVector(outputs);
	}

	KernelExpansion(Data<InputType> const& basis, bool offset, unsigned int outputs = 1)
	: mep_kernel(NULL), m_basis(basis), m_offset(offset), m_outputs(outputs){
		m_alpha = RealZeroMatrix(basis.numberOfElements(), outputs);
		if (m_offset) m_b = RealZeroVector(outputs);
	}

	KernelExpansion(KernelType* kernel, bool offset, unsigned int outputs = 1)
	: mep_kernel(kernel), m_offset(offset), m_outputs(outputs),m_alpha(0,outputs), m_basisSize(0){
		SHARK_ASSERT(mep_kernel != NULL);
		if (m_offset) m_b = RealZeroVector(outputs);
	}

	KernelExpansion(KernelType* kernel, Data<InputType> const& basis, bool offset, unsigned int outputs = 1)
	: mep_kernel(kernel), m_offset(offset), m_outputs(outputs){
		m_alpha = RealZeroMatrix(basis.numberOfElements(), outputs);
		SHARK_ASSERT(mep_kernel != NULL);
		setBasis(basis);
		if (m_offset) m_b = RealZeroVector(outputs);
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "KernelExpansion"; }

	/// dimensionality of the output RealVector
	size_t outputSize() const{
		return m_outputs;
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
		return m_offset;
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
	void setAlpha(const RealMatrix& alpha) const{
		SHARK_CHECK(alpha.size1() == m_alpha.size1() && alpha.size2() == m_alpha.size2(),
				"[KernelExpansion::setAlpha] invalid matrix size");
		m_alpha = alpha;
	}
	RealVector& offset(){
		SHARK_CHECK(m_offset, "[KernelExpansion::offset] invalid call for object without offset term");
		return m_b;
	}
	RealVector const& offset() const{
		SHARK_CHECK(m_offset, "[KernelExpansion::offset] invalid call for object without offset term");
		return m_b;
	}
	double& offset(unsigned int cls){
		SHARK_CHECK(m_offset, "[KernelExpansion::offset] invalid call for object without offset term");
		return m_b(cls);
	}
	double const& offset(unsigned int cls) const{
		SHARK_CHECK(m_offset, "[KernelExpansion::offset] invalid call for object without offset term");
		return m_b(cls);
	}
	void setOffset(const RealVector& b) const{
		SHARK_CHECK(m_offset, "[KernelExpansion::setOffset] invalid call for object without offset term");
		SHARK_CHECK(b.size() == m_b.size(), "[KernelExpansion::setOffset] invalid vector size");
		m_b = b;
	}

	// //////////////////////////////////////////////////////////
	// ////////    ALL THINGS UNDERLYING DATA    ////////////////
	// //////////////////////////////////////////////////////////


	Data<InputType> const& basis() const {
		return m_basis;
	}

	void setBasis(Data<InputType> const& basis){
		m_basisSize = basis.numberOfElements();
		m_basis = basis;
		m_alpha = RealZeroMatrix(m_basisSize, m_outputs);
		if (m_offset) 
			m_b = RealZeroVector(m_outputs);
	}

	/// The sparsify method removes non-support-vectors from
	/// its set of basis vectors and the coefficient matrix.
	void sparsify(){
		std::size_t ic = m_basis.numberOfElements();
		std::vector<std::size_t> svIndices;
		for (std::size_t i=0; i != ic; ++i){
			if (blas::norm_1(RealMatrixRow(m_alpha, i)) > 0.0){
				svIndices.push_back(i);
			}
		}
		//project basis on the support vectors
		m_basis = toDataset(subset(toView(m_basis),svIndices));
		
		//reduce alpha to it's support vector variables
		RealMatrix a(svIndices.size(), m_outputs);
		for (std::size_t i=0; i!= svIndices.size(); ++i){
			noalias(row(a,i)) = row(m_alpha,svIndices[i]); 
		}
		swap(m_alpha,a);
		
		// old version
		//~ std::size_t ic = m_basis.numberOfElements();
		//~ std::size_t sv = 0;
		//~ for (std::size_t i=0; i != ic; i++) 
			//~ if (blas::norm_1(RealMatrixRow(m_alpha, i)) > 0.0) 
				//~ sv++;

		//~ RealMatrix a(sv, m_outputs);
		//~ Data<InputType> b(sv,m_basis(0));

		//~ for (std::size_t s=0, i=0; i!= ic; i++){
			//~ if (blas::norm_1(RealMatrixRow(m_alpha, i)) > 0.0){
				//~ RealMatrixRow(a, s) = RealMatrixRow(m_alpha, i);
				//~ noalias(b(s)) = m_basis(i);
				//~ s++;
			//~ }
		//~ }
		//~ m_alpha = a;
		//~ m_basis = b;
	}

	// //////////////////////////////////////////////////////////
	// ////////    ALL THINGS KERNEL PARAMETERS    //////////////
	// //////////////////////////////////////////////////////////

	RealVector parameterVector() const{
		RealVector ret(numberOfParameters());
		if (m_offset){
			init(ret) << toVector(m_alpha), m_b;
		}
		else{
			init(ret) << toVector(m_alpha);
		}
		return ret;
	}

	void setParameterVector(RealVector const& newParameters){
		SHARK_CHECK(newParameters.size() == numberOfParameters(), "[KernelExpansion::setParameterVector] invalid size of the parameter vector");

		if (m_offset)
			init(newParameters) >> toVector(m_alpha), m_b;
		else
			init(newParameters) >> toVector(m_alpha);
	}

	std::size_t numberOfParameters() const{
		if (m_offset) 
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
		std::size_t numPatterns = boost::size(patterns);
		SHARK_ASSERT(mep_kernel != NULL);

		output.resize(numPatterns,outputSize());
		if (m_offset)
			output = repeat(m_b,numPatterns);
		else
			output.clear();

		std::size_t batchStart = 0;
		for (std::size_t i=0; i != m_basis.numberOfBatches(); i++){
			std::size_t batchEnd = batchStart+boost::size(m_basis.batch(i));
			//evaluate kernels
			//results in a matrix of the form where a column consists of the kernel evaluation of 
			//pattern i with respect to the batch of the basis,this gives a good memory alignment
			//in the following matrix matrix product
			RealMatrix kernelEvaluations = (*mep_kernel)(m_basis.batch(i),patterns);
			
			//get the part of the alpha matrix which is suitable for this batch
			ConstRealSubMatrix batchAlpha = subrange(m_alpha,batchStart,batchEnd,0,outputSize());
			fast_prod(trans(kernelEvaluations),batchAlpha,output,1.0);
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
		archive >> m_offset;
		archive >> m_outputs;
		archive >> m_basis;
		archive >> (*mep_kernel);
	}

	/// From ISerializable, writes a model to an archive
	void write( OutArchive & archive ) const{
		SHARK_ASSERT(mep_kernel != NULL);

		archive << m_alpha;
		archive << m_b;
		archive << m_offset;
		archive << m_outputs;
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
	
	/// is the bias/offset adaptive?
	bool m_offset;
	
	/// dimension of output
	unsigned int m_outputs;

	/// kernel coefficients
	RealMatrix m_alpha;
	
	///number of elements in the basis
	std::size_t m_basisSize;

	/// offset or bias term
	RealVector m_b;
};


}
#endif
