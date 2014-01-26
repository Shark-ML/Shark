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

#ifndef SHARK_OBJECTIVEFUNCTIONS_NEGATIVEGAUSSIANPROCESSEVIDENCE_H
#define SHARK_OBJECTIVEFUNCTIONS_NEGATIVEGAUSSIANPROCESSEVIDENCE_H

#include <shark/ObjectiveFunctions/DataObjectiveFunction.h>
#include <shark/Models/Kernels/KernelHelpers.h>

#include <shark/LinAlg/Base.h>
#include <shark/LinAlg/solveTriangular.h>
#include <shark/LinAlg/Cholesky.h>
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
template<class InputType = RealVector, class OutputType = RealVector, class LabelType = RealVector>
class NegativeGaussianProcessEvidence : public SupervisedObjectiveFunction<InputType,LabelType>
{
private:
	typedef SupervisedObjectiveFunction<InputType, LabelType> base_type;
public:
	typedef typename base_type::DatasetType DatasetType;
	typedef AbstractKernelFunction<InputType> KernelType;

	/// \param kernel: pointer to external kernel function
	/// \param unconstrained: exponential encoding of regularization parameter for unconstraint optimization
	NegativeGaussianProcessEvidence(KernelType* kernel, bool unconstrained = false)
	:mep_kernel(kernel), m_unconstrained(unconstrained)
	{
		if (kernel->hasFirstParameterDerivative()) this->m_features |= base_type::HAS_FIRST_DERIVATIVE;
		setThreshold(0.);
	}

	/// \param dataset: training data for the Gaussian process
	/// \param kernel: pointer to external kernel function
	/// \param unconstrained: exponential encoding of regularization parameter for unconstraint optimization
	NegativeGaussianProcessEvidence(DatasetType const& dataset,
					KernelType* kernel,
					bool unconstrained = false)
		: m_dataset(dataset)
		, mep_kernel(kernel)
		, m_unconstrained(unconstrained)
	{
		if (kernel->hasFirstParameterDerivative()) this->m_features |= base_type::HAS_FIRST_DERIVATIVE;
		setThreshold(0.);
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "NegativeGaussianProcessEvidence"; }

	/// inherited from SupervisedObjectiveFunction
	void setDataset(DatasetType const& dataset) {
		m_dataset = dataset;
	}
	
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
		this->m_evaluationCounter++;
		
		//set parameters
		RealVector kernelParams(kp);
		double betaInv = 0;
		blas::init(parameters) >> kernelParams, betaInv;
		if(m_unconstrained)
			betaInv = std::exp(betaInv); // for unconstraint optimization
		mep_kernel->setParameterVector(kernelParams);
		
		
		//generate kernel matrix and label vector
		RealMatrix M = calculateRegularizedKernelMatrix(*mep_kernel,m_dataset.inputs(),betaInv);
		//~ RealVector t = generateLabelVector();
		RealVector t = column(createBatch<RealVector>(m_dataset.labels().elements()),0);

		RealMatrix choleskyFactor(N,N);
		choleskyDecomposition(M, choleskyFactor);
		
		//compute the determinant of M using the cholesky factorization M=AA^T:
		//ln det(M) = 2 trace(ln A)
		double logDet = 2* trace(log(choleskyFactor));
		
		//we need to compute t^T M^-1 t 
		//= t^T (AA^T)^-1 t= t^T (A^-T A^-1)=||A^-1 t||^2
		//so we will first solve the triangular System Az=t
		//and then compute ||z||^2
		//since we don't need t anymore after that, we solve in-place and omit z
		blas::solveTriangularSystemInPlace<blas::SolveAXB,blas::Lower>(choleskyFactor,t);

		// equation (6.69) on page 311 in the book C.M. Bishop, Pattern Recognition and Machine Learning, Springer, 2006
		// e = 1/2 \cdot [ -log(det(M)) - t^T M^{-1} t - N log(2 \pi) ]
		double e = 0.5 * (-logDet - norm_sqr(t) - N * std::log(2.0 * M_PI));

		// return the *negative* evidence
		return -e;
	}

	/// Let \f$M\f$ denote the regularized (kernel Gram) covariance matrix.
	/// For the evidence we have:
	/// \f[ E = 1/2 \cdot [ -\log(\det(M)) - t^T M^{-1} t - N \log(2 \pi) ] \f]
	/// For a kernel parameter \f$p\f$ and \f$C = \beta^{-1}\f$ we get the derivatives:
	/// \f[  dE/dC = 1/2 \cdot [ -tr(M^{-1}) + (M^{-1} t)^2 ] \f]
	/// \f[  dE/dp = 1/2 \cdot [ -tr(M^{-1} dM/dp) + t^T (M^{-1} dM/dp M^{-1}) t ] \f]
	double evalDerivative(const RealVector& parameters, typename base_type::FirstOrderDerivative& derivative) const {
		std::size_t N  = m_dataset.numberOfElements(); 
		std::size_t kp = mep_kernel->numberOfParameters();

		// check whether argument has right dimensionality
		SHARK_ASSERT(1 + kp == parameters.size());
		derivative.resize(1 + kp);
		
		// keep track of how often the objective function is called
		this->m_evaluationCounter++;

		//set parameters
		RealVector kernelParams(kp);
		double betaInv = 0;
		blas::init(parameters) >> kernelParams, betaInv;
		if(m_unconstrained)
			betaInv = std::exp(betaInv); // for unconstraint optimization
		mep_kernel->setParameterVector(kernelParams);
		
		
		//generate kernel matrix and label vector
		RealMatrix M = calculateRegularizedKernelMatrix(*mep_kernel,m_dataset.inputs(),betaInv);
		//~ RealVector t = generateLabelVector();
		RealVector t = column(createBatch<RealVector>(m_dataset.labels().elements()),0);
		
		//new way to compute inverse and logDetM
		RealMatrix choleskyFactor(N,N);
		choleskyDecomposition(M, choleskyFactor);
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
		
		//compute inverse matrix from the cholesky dcomposition 
		//using forward-backward substitution,
		RealMatrix W=RealIdentityMatrix(N);
		blas::solveTriangularCholeskyInPlace<blas::SolveAXB>(choleskyFactor,W);
		
		//calculate z = Wt=M^-1 t
		RealVector z(N);
		axpy_prod(W,t,z);
		
		// W is already initialized as the inverse of M, so we only need 
		// to change the sign and add z. to calculate W fully
		W*=-1;
		W+=outer_prod(z,z);
		
		
		//now calculate the derivative
		RealVector kernelGradient = 0.5*calculateKernelMatrixParameterDerivative(*mep_kernel,m_dataset.inputs(),W);
		
		// compute derivative w.r.t. regularization parameter
		//we have: dE/dC = 1/2 * [ -tr(M^{-1}) + (M^{-1} t)^2
		// which can also be written as 1/2 tr(W)
		double betaInvDerivative = 0.5 * trace(W) ;
		if(m_unconstrained) 
			betaInvDerivative *= betaInv;
		
		//merge both derivatives and since we return the negative evidence, multiply with -1
		blas::init(derivative)<<kernelGradient,betaInvDerivative;
		derivative *= -1.0;

		// truncate gradient vector 
		for(std::size_t i=0; i<derivative.size(); i++) 
			if(std::abs(derivative(i)) < m_derivativeThresholds(i)) derivative(i) = 0;

		// compute the evidence
		//compute determinant of M (see eval for why this works)
		double logDetM = 2* trace(log(choleskyFactor));
		double e = 0.5 * (-logDetM - inner_prod(t, z) - N * std::log(2.0 * M_PI));
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
	//~ RealVector generateLabelVector()const{
		//~ std::size_t N  = m_dataset.numberOfElements(); 
		//~ RealVector t(N);
		//~ std::size_t startX = 0;//start of the current batch
		//~ for (std::size_t i=0; i<m_dataset.numberOfBatches(); i++){
			//~ std::size_t sizeX=size(m_dataset.batch(i));
			//~ for(std::size_t k =0; k != sizeX; ++k){
				//~ t(startX+k)=m_dataset.batch(i).label(k,0);
			//~ }
		//~ }
		//~ return t;
	//~ }
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
