//===========================================================================
/*!
 * 
 *
 * \brief       Trainer for a Regularization Network or a Gaussian Process
 * 
 * 
 * 
 *
 * \author      T. Glasmachers
 * \date        2007-2012
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


#ifndef SHARK_ALGORITHMS_REGULARIZATIONNETWORKTRAINER_H
#define SHARK_ALGORITHMS_REGULARIZATIONNETWORKTRAINER_H


#include <shark/Algorithms/Trainers/AbstractSvmTrainer.h>
#include <shark/Models/Kernels/KernelHelpers.h>


namespace shark {


///
/// \brief Training of a regularization network.
///
/// A regularization network is a kernel-based model for
/// regression problems. Given are data tuples
/// \f$ (x_i, y_i) \f$ with x-component denoting input and
/// y-component denoting a real-valued label (see the tutorial on
/// label conventions; the implementation uses RealVector),
/// a kernel function k(x, x') and a regularization
/// constant \f$ C  > 0\f$. Let H denote the kernel induced
/// reproducing kernel Hilbert space of k, and let \f$ \phi \f$
/// denote the corresponding feature map.
/// Then the SVM regression function is of the form
/// \f[
///     f(x) = \langle w, \phi(x) \rangle + b
/// \f]
/// with coefficients w and b given by the (primal)
/// optimization problem
/// \f[
///     \min \frac{1}{2} \|w\|^2 + C \sum_i L(y_i, f(x_i)),
/// \f]
/// where the simple quadratic loss is employed:
/// \f[
///     L(y, f(x)) = (y - f(x))^2
/// \f]
/// Regularization networks can be interpreted as a special
/// type of support vector machine (for regression, with
/// squared loss, and thus with non-sparse weights).
///
/// Training a regularization network is identical to training a
/// Gaussian process for regression. The parameter \f$ C \f$ then
/// corresponds precision of the noise (denoted by \f$ \beta \f$ in
/// Bishop's textbook). The precision is the inverse of the variance
/// of the noise. The variance of the noise is denoted by \f$
/// \sigma_n^2 \f$ in the textbook by Rasmussen and
/// Williams. Accordingly, \f$ C = 1/\sigma_n^2 \f$.
/// \ingroup supervised_trainer
template <class InputType>
class RegularizationNetworkTrainer : public AbstractSvmTrainer<InputType, RealVector,KernelExpansion<InputType> >
{
public:
	typedef AbstractModel<InputType, RealVector> ModelType;
	typedef AbstractKernelFunction<InputType> KernelType;
	typedef AbstractSvmTrainer<InputType, RealVector, KernelExpansion<InputType> > base_type;

	/// \param kernel Kernel
	/// \param betaInv Inverse precision, equal to assumed noise variance, equal to inverse regularization parameter C 
	/// \param unconstrained Indicates exponential encoding of the regularization parameter 
	RegularizationNetworkTrainer(KernelType* kernel, double betaInv, bool unconstrained = false)
	: base_type(kernel, 1.0 / betaInv, false, unconstrained)
	{ }

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "RegularizationNetworkTrainer"; }

	/// \brief Returns the assumed noise variance (i.e., 1/C) 
	double noiseVariance() const
	{ return 1.0 / this->C(); }
	/// \brief Sets the assumed noise variance (i.e., 1/C) 
	void setNoiseVariance(double betaInv)
	{ this->C() = 1.0 / betaInv; }

	/// \brief Returns the precision (i.e., C), the inverse of the assumed noise variance 
	double precision() const
	{ return this->C(); }
	/// \brief Sets the precision (i.e., C), the inverse of the assumed noise variance 
	void setPrecision(double beta)
	{ this->C() = beta; }

	void train(KernelExpansion<InputType>& svm, const LabeledData<InputType, RealVector>& dataset){
		svm.setStructure(base_type::m_kernel,dataset.inputs(),true, labelDimension(dataset));
		
		// Setup the kernel matrix
		RealMatrix M = calculateRegularizedKernelMatrix(*(this->m_kernel),dataset.inputs(), noiseVariance());
		RealMatrix V = createBatch<RealVector>(elements(dataset.labels()));
		RealVector mean = sum(as_columns(V))/V.size1();
		noalias(V) -= blas::repeat(mean,V.size1());

		//check whether lambda is large enough to make the eigenvalues numerically stable
		if(noiseVariance()/max(diag(M)) < 1.e-5)
			noalias(svm.alpha()) = inv(M,blas::symm_semi_pos_def()) % V;
		else//we think now it is stable so we can use the fast pure cholesky decomposition
			noalias(svm.alpha()) = inv(M,blas::symm_pos_def()) % V;
		noalias(svm.offset()) = mean;
	}
};


// A regularization network can be interpreted as a Gaussian
// process, with the same trainer:
#define GaussianProcessTrainer RegularizationNetworkTrainer


}
#endif
