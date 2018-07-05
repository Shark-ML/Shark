//===========================================================================
/*!
 * 
 *
 * \brief       Evidence for model selection of a regularization network/Gaussian process.


 * 
 *
 * \author      C. Igel, T. Glasmachers, O. Krause
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

#ifndef SHARK_OBJECTIVEFUNCTIONS_NEGATIVEGAUSSIANPROCESSEVIDENCE_H
#define SHARK_OBJECTIVEFUNCTIONS_NEGATIVEGAUSSIANPROCESSEVIDENCE_H

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/Models/Kernels/KernelHelpers.h>

#include <shark/LinAlg/Base.h>
namespace shark {


///
/// \brief Evidence for model selection of a regularization network/Gaussian process.
///
/// Let \f$M\f$ denote the (kernel Gram) covariance matrix and
/// \f$t\f$ the corresponding label vector.  For the evidence we have: 
/// \f[ E = 1/2 \cdot [ -\log(\det(M)) - t^T M^{-1} t - N \log(2 \pi)] \f]
///
/// The evidence is also known as marginal (log)likelihood. For
/// details, please see:
///
/// C.E. Rasmussen & C.K.I. Williams, Gaussian
/// Processes for Machine Learning, section 5.4, MIT Press, 2006
///
/// C.M. Bishop, Pattern Recognition and Machine Learning, section
/// 6.4.3, Springer, 2006
///
/// The regularization parameter can be encoded in different ways.
/// The exponential encoding is the proper choice for unconstraint optimization.
/// Be careful not to mix up different encodings between trainer and evidence.
/// \ingroup kerneloptimization
template<class InputType = RealVector, class OutputType = RealVector, class LabelType = RealVector>
class NegativeGaussianProcessEvidence : public AbstractObjectiveFunction< RealVector, double >
{
public:
	typedef LabeledData<InputType,LabelType> DatasetType;
	typedef AbstractKernelFunction<InputType> KernelType;

	/// \param dataset: training data for the Gaussian process
	/// \param kernel: pointer to external kernel function
	/// \param unconstrained: exponential encoding of regularization parameter for unconstraint optimization
	NegativeGaussianProcessEvidence(
		DatasetType const& dataset,
		KernelType* kernel,
		bool unconstrained = false
	): m_dataset(dataset)
	, mep_kernel(kernel)
	, m_unconstrained(unconstrained)
	{
		if (kernel->hasFirstParameterDerivative()) m_features |= HAS_FIRST_DERIVATIVE;
		setThreshold(0.);
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "NegativeGaussianProcessEvidence"; }
	
	std::size_t numberOfVariables()const{
		return 1+ mep_kernel->numberOfParameters();
	}

	/// Let \f$M\f$ denote the (kernel Gram) covariance matrix and
	/// \f$t\f$ the label vector.  For the evidence we have: \f[ E= 1/2 \cdot [ -\log(\det(M)) - t^T M^{-1} t - N \log(2 \pi) ] \f]
	double eval(const RealVector& parameters) const {
		std::size_t N  = m_dataset.numberOfElements(); 
		std::size_t kp = mep_kernel->numberOfParameters();
		// check whether argument has right dimensionality
		SHARK_ASSERT(1+kp == parameters.size());

		// keep track of how often the objective function is called
		m_evaluationCounter++;
		
		//set parameters
		double betaInv = parameters.back();
		if(m_unconstrained)
			betaInv = std::exp(betaInv); // for unconstraint optimization
		mep_kernel->setParameterVector(subrange(parameters,0,kp));
		
		
		//generate kernel matrix and label vector
		RealMatrix M = calculateRegularizedKernelMatrix(*mep_kernel,m_dataset.inputs(),betaInv);
		RealMatrix t = createBatch<RealVector>(elements(m_dataset.labels()));

		blas::cholesky_decomposition<RealMatrix> cholesky(M);
		
		//compute the determinant of M using the cholesky factorization M=AA^T:
		//ln det(M) = 2 trace(ln A)
		double logDet = 2* trace(log(cholesky.lower_factor()));
		
		//we need to compute t^T M^-1 t 
		//= t^T (AA^T)^-1 t= t^T (A^-T A^-1)=||A^-1 t||^2
		//so we will first solve the triangular System Az=t
		//and then compute ||z||^2
		RealMatrix z = solve(cholesky.lower_factor(),t,blas::lower(), blas::left());

		// equation (6.69) on page 311 in the book C.M. Bishop, Pattern Recognition and Machine Learning, Springer, 2006
		// e = 1/2 \cdot [ -log(det(M)) - t^T M^{-1} t - N log(2 \pi) ]
		double e = 0.5 * (-logDet - norm_sqr(to_vector(z)) - N * std::log(2.0 * M_PI));

		// return the *negative* evidence
		return -e;
	}

	/// Let \f$M\f$ denote the regularized (kernel Gram) covariance matrix.
	/// For the evidence we have:
	/// \f[ E = 1/2 \cdot [ -\log(\det(M)) - t^T M^{-1} t - N \log(2 \pi) ] \f]
	/// For a kernel parameter \f$p\f$ and \f$C = \beta^{-1}\f$ we get the derivatives:
	/// \f[  dE/dC = 1/2 \cdot [ -tr(M^{-1}) + (M^{-1} t)^2 ] \f]
	/// \f[  dE/dp = 1/2 \cdot [ -tr(M^{-1} dM/dp) + t^T (M^{-1} dM/dp M^{-1}) t ] \f]
	double evalDerivative(const RealVector& parameters, FirstOrderDerivative& derivative) const {
		std::size_t N  = m_dataset.numberOfElements(); 
		std::size_t kp = mep_kernel->numberOfParameters();

		// check whether argument has right dimensionality
		SHARK_ASSERT(1 + kp == parameters.size());
		derivative.resize(1 + kp);
		
		// keep track of how often the objective function is called
		m_evaluationCounter++;

		//set parameters
		double betaInv = parameters.back();
		if(m_unconstrained)
			betaInv = std::exp(betaInv); // for unconstraint optimization
		mep_kernel->setParameterVector(subrange(parameters,0,kp));
		
		//generate kernel matrix and label vector
		RealMatrix M = calculateRegularizedKernelMatrix(*mep_kernel,m_dataset.inputs(),betaInv);
		RealMatrix t = createBatch<RealVector>(elements(m_dataset.labels()));

		//compute cholesky decomposition of M
		blas::cholesky_decomposition<RealMatrix> cholesky(M);
		//we dont need M anymore, so save a lot of memory by freeing the memory of M
		M=RealMatrix();
		
		// compute derivative w.r.t. kernel parameters
		//the derivative is defined as:
		//dE/da = -tr(IM dM/da) +t^T IM dM/da IM t
		// where IM is the inverse matrix of M, tr is the trace and a are the parameters of the kernel
		//by substituting z = IM t we can expand the operations to:
		//dE/da = -(sum_i sum_j IM_ij * dM_ji/da)+(sum_i sum_j dM_ij/da *z_i * z_j)
		//           =  sum_i sum_j (-IM_ij+z_i * z_j) * dM_ij/da
		// with W = -IM + zz^T we get
		// dE/da = sum_i sum_j W dM_ij/da
		//this can be calculated as blockwise derivative.
		
		//compute inverse matrix from the cholesky decomposition 
		RealMatrix W= blas::identity_matrix<double>(N);
		cholesky.solve(W,blas::left());

		//calculate z = Wt=M^-1 t
		RealMatrix z = prod(W,t);
		
		// W is already initialized as the inverse of M, so we only need 
		// to change the sign and add z. to calculate W fully
		W*=-1;
		noalias(W) += prod(z,trans(z));
		
		
		//now calculate the derivative
		RealVector kernelGradient = 0.5*calculateKernelMatrixParameterDerivative(*mep_kernel,m_dataset.inputs(),W);
		
		// compute derivative w.r.t. regularization parameter
		//we have: dE/dC = 1/2 * [ -tr(M^{-1}) + (M^{-1} t)^2
		// which can also be written as 1/2 tr(W)
		double betaInvDerivative = 0.5 * trace(W) ;
		if(m_unconstrained) 
			betaInvDerivative *= betaInv;
		
		//merge both derivatives and since we return the negative evidence, multiply with -1
		noalias(derivative) = - (kernelGradient | betaInvDerivative);

		// truncate gradient vector 
		for(std::size_t i=0; i<derivative.size(); i++) 
			if(std::abs(derivative(i)) < m_derivativeThresholds(i)) derivative(i) = 0;

		// compute the evidence
		//compute determinant of M (see eval for why this works)
		double logDetM = 2* trace(log(cholesky.lower_factor()));
		double e = 0.5 * (-logDetM - inner_prod(to_vector(t), to_vector(z)) - N * std::log(2.0 * M_PI));
		return -e;
	}
	
	/// set threshold value for truncating partial derivatives
	void setThreshold(double d) {
		m_derivativeThresholds = RealVector(mep_kernel->numberOfParameters() + 1, d); // plus one parameter for the prior 
	}

	/// set threshold values for truncating partial derivatives
	void setThresholds(RealVector &c) {
		SHARK_ASSERT(m_derivativeThresholds.size() == c.size());
		m_derivativeThresholds = c;
	}
		

private:
	/// pointer to external data set
	DatasetType m_dataset;

	/// thresholds for setting derivatives to zero
	RealVector  m_derivativeThresholds;

	/// pointer to external kernel function
	KernelType* mep_kernel;

	/// Indicates whether log() of the regularization parameter is
	/// considered. This is useful for unconstraint
	/// optimization. The default value is false.
	bool m_unconstrained; 
};


}
#endif
