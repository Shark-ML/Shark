//===========================================================================
/*!
 * 
 *
 * \brief       Derivative of a C-SVM hypothesis w.r.t. its hyperparameters.
 * 
 * \par
 * This class provides two main member functions for computing the
 * derivative of a C-SVM hypothesis w.r.t. its hyperparameters. First,
 * the derivative is prepared in general. Then, the derivative can be
 * computed comparatively cheaply for any input sample. Needs to be
 * supplied with pointers to a KernelExpansion and CSvmTrainer.
 * 
 * 
 *
 * \author      M. Tuma, T. Glasmachers
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


#ifndef SHARK_MODELS_CSVMDERIVATIVE_H
#define SHARK_MODELS_CSVMDERIVATIVE_H

#include <shark/Core/INameable.h>
#include <shark/Core/ISerializable.h>
#include <shark/Algorithms/Trainers/CSvmTrainer.h>
#include <shark/Models/Kernels/KernelExpansion.h>
namespace shark {


/// \brief
///
/// This class provides two main member functions for computing the
/// derivative of a C-SVM hypothesis w.r.t. its hyperparameters.
/// The constructor takes a pointer to a KernelClassifier and an SvmTrainer,
/// in the assumption that the former was trained by the latter. It heavily
/// accesses their members to calculate the derivative of the alpha and offset
/// values w.r.t. the SVM hyperparameters, that is, the regularization
/// parameter C and the kernel parameters. This is done in the member function
/// prepareCSvmParameterDerivative called by the constructor. After this initial,
/// heavier computation step, modelCSvmParameterDerivative can be called on an
/// input sample to the SVM model, and the method will yield the derivative of
/// the hypothesis w.r.t. the SVM hyperparameters.
///
/// \tparam InputType Same basis type as the kernel expansion's
/// \tparam CacheType While the basic cache type defaults to float in the QP algorithms, it here defaults to double, because the SVM derivative benefits a lot from higher precision.
///
template< class InputType, class CacheType = double >
class CSvmDerivative : public ISerializable, public INameable
{
public:
	typedef CacheType QpFloatType;
	typedef KernelClassifier<InputType> KeType;
	typedef AbstractKernelFunction<InputType> KernelType;
	typedef CSvmTrainer<InputType, QpFloatType> TrainerType;

protected:

	// key external members through which main information is obtained
	KernelExpansion<InputType>* mep_ke;  ///< pointer to the KernelExpansion which has to have been been trained by the below SvmTrainer
	TrainerType* mep_tr; ///< pointer to the SvmTrainer with which the above KernelExpansion has to have been trained
	KernelType* mep_k; ///< convenience pointer to the underlying kernel function
	RealMatrix& m_alpha; ///< convenience reference to the alpha values of the KernelExpansion
	Data<InputType> m_basis; ///< convenience reference to the underlying data of the KernelExpansion
	RealVector m_db_dParams_from_solver; ///< convenience access to the correction term from the solver, for the rare case that there are no free SVs

	// convenience copies from the CSvmTrainer and the underlying kernel function
	double m_C; ///< the regularization parameter value with which the SvmTrainer trained the KernelExpansion
	bool m_unconstrained; ///< is the unconstrained flag of the SvmTrainer set? Influences the derivatives!
	std::size_t m_nkp; ///< convenience member holding the Number of Kernel Parameters.
	std::size_t m_nhp; ///< convenience member holding the Number of Hyper Parameters.

	// information calculated from the KernelExpansion state in the prepareDerivative-step
	std::size_t m_noofFreeSVs; ///< number of free SVs
	std::size_t m_noofBoundedSVs; ///< number of bounded SVs
	std::vector< std::size_t > m_freeAlphaIndices; ///< indices of free SVs
	std::vector< std::size_t > m_boundedAlphaIndices; ///< indices of bounded SVs
	RealVector m_freeAlphas; 	///< free non-SV alpha values
	RealVector m_boundedAlphas; ///< bounded non-SV alpha values
	RealVector m_boundedLabels; ///< labels of bounded non-SVs

	/// Main member and result, computed in the prepareDerivative-step:
	/// Stores the derivative of the **free** alphas w.r.t. SVM hyperparameters as obtained
	/// through the CSvmTrainer (for C) and through the kernel (for the kernel parameters).
	/// Each row corresponds to one **free** alpha, each column to one hyperparameter.
	/// The **last** column is the derivative of (free_alphas, b) w.r.t C. All **previous**
	/// columns are w.r.t. the kernel parameters.
	RealMatrix m_d_alphab_d_theta;

public:

	/// Constructor. Only sets up the main pointers and references to the external instances and data, and
	/// performs basic sanity checks.
	/// \param ke pointer to the KernelExpansion which has to have been been trained by the below SvmTrainer
	/// \param trainer pointer to the SvmTrainer with which the above KernelExpansion has to have been trained
	CSvmDerivative( KeType* ke, TrainerType* trainer )
	: mep_ke( &ke->decisionFunction() ),
	  mep_tr( trainer ),
	  mep_k( trainer->kernel() ),
	  m_alpha( mep_ke->alpha() ),
	  m_basis( mep_ke->basis() ),
	  m_db_dParams_from_solver( trainer->get_db_dParams() ),
	  m_C ( trainer->C() ),
	  m_unconstrained( trainer->isUnconstrained() ),
	  m_nkp( trainer->kernel()->numberOfParameters() ),
	  m_nhp( trainer->kernel()->numberOfParameters()+1 )
	{
		SHARK_RUNTIME_CHECK( mep_ke->kernel() == trainer->kernel(), "[CSvmDerivative::CSvmDerivative] KernelExpansion and SvmTrainer must use the same KernelFunction.");
		SHARK_RUNTIME_CHECK( mep_ke != NULL, "[CSvmDerivative::CSvmDerivative] KernelExpansion cannot be NULL.");
		SHARK_RUNTIME_CHECK( mep_ke->outputShape().numElements() == 1, "[CSvmDerivative::CSvmDerivative] only defined for binary SVMs.");
		SHARK_RUNTIME_CHECK( mep_ke->hasOffset() == 1, "[CSvmDerivative::CSvmDerivative] only defined for SVMs with offset.");
		SHARK_RUNTIME_CHECK( m_alpha.size2() == 1, "[CSvmDerivative::CSvmDerivative] this class is only defined for binary SVMs.");
		prepareCSvmParameterDerivative(); //main
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "CSvmDerivative"; }

	inline const KeType* ke() { return mep_ke; }
	inline const TrainerType* trainer() { return mep_tr; }

	//! Computes the derivative of the model w.r.t. regularization and kernel parameters.
	//! Be sure to call prepareCSvmParameterDerivative after SVM training and before calling this function!
	//! \param input an example to be scored by the SVM
	//! \param derivative a vector of derivatives of the score. The last entry is w.r.t. C.
	void modelCSvmParameterDerivative(const InputType& input, RealVector& derivative )
	{
		// create temporary batch helpers
		RealMatrix unit_weights(1,1,1.0);
		RealMatrix bof_results(1,1);
		typename Batch<InputType>::type bof_xi = Batch<InputType>::createBatch(input,1);
		typename Batch<InputType>::type bof_input = Batch<InputType>::createBatch(input,1);
		getBatchElement(bof_input, 0) = input; //fixed over entire function scope
		auto elements = m_basis.elements();

		// init helpers
		RealVector der( m_nhp );
		boost::shared_ptr<State> state = mep_k->createState(); //state from eval and for derivatives
		derivative.resize( m_nhp );

		// start calculating derivative
		noalias(derivative) = row(m_d_alphab_d_theta,m_noofFreeSVs); //without much thinking, we add db/d(\theta) to all derivatives
		// first: go through free SVs and add their contributions (the actual ones, which use the matrix d_alphab_d_theta)
		for ( std::size_t i=0; i<m_noofFreeSVs; i++ ) {
			getBatchElement(bof_xi, 0) = elements[m_freeAlphaIndices[i]];
			mep_k->eval( bof_input, bof_xi, bof_results, *state );
			double ker = bof_results(0,0);
			double cur_alpha = m_freeAlphas(i);
			mep_k->weightedParameterDerivative( bof_input, bof_xi, unit_weights, *state, der );
			noalias(derivative) += ker * row(m_d_alphab_d_theta,i); //for C, simply add up the individual contributions
			noalias(subrange(derivative,0,m_nkp))+= cur_alpha*der;
		}
		// second: go through all bounded SVs and add their "trivial" derivative contributions
		for ( std::size_t i=0; i<m_noofBoundedSVs; i++ ) {
			getBatchElement(bof_xi, 0) = elements[m_boundedAlphaIndices[i]];
			mep_k->eval( bof_input, bof_xi, bof_results, *state );
			double ker = bof_results(0,0);
			double cur_label = m_boundedLabels(i);
			mep_k->weightedParameterDerivative( bof_input, bof_xi, unit_weights, *state, der );
			derivative( m_nkp ) += ker * cur_label; //deriv of bounded alpha w.r.t. C is simply the label
			noalias(subrange(derivative,0,m_nkp))+= cur_label * m_C * der;
		}
		if ( m_unconstrained )
			derivative( m_nkp ) *= m_C; //compensate for log encoding via chain rule
			//(the kernel parameter derivatives are correctly differentiated according to their
			// respective encoding via the kernel's derivative, so we don't need to correct for those.)

		// in some rare cases, there are no free SVs and we have to manually correct the derivatives using a correcting term from the SvmTrainer.
		if ( m_noofFreeSVs == 0 ) {
			noalias(derivative) += m_db_dParams_from_solver;
		}
	}

	//! Whether there are free SVs in the solution. Useful to monitor for degenerate solutions
	//! with only bounded and no free SVs. Be sure to call prepareCSvmParameterDerivative after
	//! SVM training and before calling this function.
	bool hasFreeSVs() { return ( m_noofFreeSVs != 0 ); }
	//! Whether there are bounded SVs in the solution. Useful to monitor for degenerate solutions
	//! with only bounded and no free SVs. Be sure to call prepareCSvmParameterDerivative after
	//! SVM training and before calling this function.
	bool hasBoundedSVs() { return ( m_noofBoundedSVs != 0 ); }

	/// Access to the matrix of SVM coefficients' derivatives. Derivative w.r.t. C is last.
	const RealMatrix& get_dalphab_dtheta() {
		return m_d_alphab_d_theta;
	}

	/// From ISerializable, reads a network from an archive
	virtual void read( InArchive & archive ) {
	}
	/// From ISerializable, writes a network to an archive
	virtual void write( OutArchive & archive ) const {
	}

private:

	///////////  DERIVATIVE OF BINARY (C-)SVM  ////////////////////

	//! Fill m_d_alphab_d_theta, the matrix of derivatives of free SVs w.r.t. C-SVM hyperparameters
	//! as obtained through the CSvmTrainer (for C) and through the kernel (for the kernel parameters).
	//! \par
	//!  Note: we follow the alpha-encoding-conventions of Glasmacher's dissertation, where the alpha values
	//!  are re-encoded by multiplying each with the corresponding label
	//!
	void prepareCSvmParameterDerivative() {
		// init convenience size indicators
		std::size_t numberOfAlphas = m_alpha.size1();

		// first round through alphas: count free and bounded SVs
		for ( std::size_t i=0; i<numberOfAlphas; i++ ) {
			double cur_alpha = m_alpha(i,0); //we assume (and checked) that there is only one class
			if ( cur_alpha != 0.0 ) {
				if ( cur_alpha == m_C || cur_alpha == -m_C ) { //the svm formulation using reparametrized alphas is assumed
					m_boundedAlphaIndices.push_back(i);
				} else {
					m_freeAlphaIndices.push_back(i);
				}
			}
		}
		m_noofFreeSVs = m_freeAlphaIndices.size(); //don't forget to add b to the count where appropriate
		m_noofBoundedSVs = m_boundedAlphaIndices.size();
		// in contrast to the Shark2 implementation, we here don't store useless constants (i.e., 0, 1, -1), but only the derivs w.r.t. (\alpha_free, b)
		m_d_alphab_d_theta.resize(m_noofFreeSVs+1, m_nhp);
		m_d_alphab_d_theta.clear();
		m_freeAlphaIndices.push_back( numberOfAlphas ); //b is always free (but don't forget to add to count manually)

		// 2nd round through alphas: build up the RealVector of free and bounded alphas (needed for matrix-vector-products later)
		m_freeAlphas.resize( m_noofFreeSVs+1);
		m_boundedAlphas.resize( m_noofBoundedSVs );
		m_boundedLabels.resize( m_noofBoundedSVs );
		for ( std::size_t i=0; i<m_noofFreeSVs; i++ )
			m_freeAlphas(i) = m_alpha( m_freeAlphaIndices[i], 0 );
		m_freeAlphas( m_noofFreeSVs ) = mep_ke->offset(0);
		for ( std::size_t i=0; i<m_noofBoundedSVs; i++ ) {
			double cur_alpha = m_alpha( m_boundedAlphaIndices[i], 0 );
			m_boundedAlphas(i) = cur_alpha;
			m_boundedLabels(i) = ( (cur_alpha > 0.0) ? 1.0 : -1.0 );
		}
		
		//if there are no free support vectors, we are done.
		if ( m_noofFreeSVs == 0 ) {
			return;
		}

		// set up helper variables.
		// 		See Tobias Glasmacher's dissertation, chapter 9.3, for a calculation of the derivatives as well as
		// 		for a definition of these variables. -> It's very easy to follow this code with that chapter open.
		//		The Keerthi-paper "Efficient method for gradient-based..." is also highly recommended for cross-reference.
		RealVector der( m_nkp ); //derivative storage helper
		boost::shared_ptr<State> state = mep_k->createState(); //state object for derivatives

		// create temporary batch helpers
		RealMatrix unit_weights(1,1,1.0);
		RealMatrix bof_results(1,1);
		typename Batch<InputType>::type bof_xi;
		typename Batch<InputType>::type bof_xj;
		if ( m_noofFreeSVs != 0 ) {
			bof_xi = Batch<InputType>::createBatchFromShape( m_basis.shape(), 1 ); //any input works
			bof_xj = Batch<InputType>::createBatchFromShape( m_basis.shape(), 1 ); //any input works
		} else if ( m_noofBoundedSVs != 0 ) {
			bof_xi = Batch<InputType>::createBatchFromShape( m_basis.shape(), 1 ); //any input works
			bof_xj = Batch<InputType>::createBatchFromShape( m_basis.shape(), 1 ); //any input works
		} else {
			throw SHARKEXCEPTION("[CSvmDerivative::prepareCSvmParameterDerivative] Something went very wrong.");
		}
		auto elements = m_basis.elements();

		
		// initialize H and dH
		//H is the the matrix 
		//H = (K 1; 1 0)
		// where the ones are row or column vectors and 0 is a scalar.
		// K is the kernel matrix spanned by the free support vectors
		RealMatrix H( m_noofFreeSVs + 1, m_noofFreeSVs + 1,0.0);
		std::vector< RealMatrix > dH( m_nkp , RealMatrix(m_noofFreeSVs+1, m_noofFreeSVs+1));
		for ( std::size_t i=0; i<m_noofFreeSVs; i++ ) {
			getBatchElement(bof_xi, 0) = elements[m_freeAlphaIndices[i]]; //fixed over outer loop
			// fill the off-diagonal entries..
			for ( std::size_t j=0; j<i; j++ ) {
				getBatchElement(bof_xj, 0) = elements[m_freeAlphaIndices[j]]; //get second sample into a batch
				mep_k->eval( bof_xi, bof_xj, bof_results, *state );
				H( i,j ) = H( j,i ) = bof_results(0,0);
				mep_k->weightedParameterDerivative( bof_xi, bof_xj, unit_weights, *state, der );
				for ( std::size_t k=0; k<m_nkp; k++ ) {
					dH[k]( i,j ) = dH[k]( j,i ) = der(k);
				}
			}
			// ..then fill the diagonal entries..
			mep_k->eval( bof_xi, bof_xi, bof_results, *state );
			H( i,i ) = bof_results(0,0);
			H( i, m_noofFreeSVs) = H( m_noofFreeSVs, i) = 1.0;
			mep_k->weightedParameterDerivative( bof_xi, bof_xi, unit_weights, *state, der );
			for ( std::size_t k=0; k<m_nkp; k++ ) {
				dH[k]( i,i ) = der(k);
			}
			// ..and finally the last row/column (pertaining to the offset parameter b)..
			for (std::size_t k=0; k<m_nkp; k++)
				dH[k]( m_noofFreeSVs, i ) = dH[k]( i, m_noofFreeSVs ) = 0.0;
		}

		// ..the lower-right-most entry gets set separately:
		for ( std::size_t k=0; k<m_nkp; k++ ) {
			dH[k]( m_noofFreeSVs, m_noofFreeSVs ) = 0.0;
		}
		
		// initialize R and dR
		RealMatrix R( m_noofFreeSVs+1, m_noofBoundedSVs );
		std::vector< RealMatrix > dR( m_nkp, RealMatrix(m_noofFreeSVs+1, m_noofBoundedSVs));
		for ( std::size_t i=0; i<m_noofBoundedSVs; i++ ) {
			getBatchElement(bof_xi, 0) = elements[m_boundedAlphaIndices[i]]; //fixed over outer loop
			for ( std::size_t j=0; j<m_noofFreeSVs; j++ ) { //this time, we (have to) do it row by row
				getBatchElement(bof_xj, 0) = elements[m_freeAlphaIndices[j]]; //get second sample into a batch
				mep_k->eval( bof_xi, bof_xj, bof_results, *state );
				R( j,i ) = bof_results(0,0);
				mep_k->weightedParameterDerivative( bof_xi, bof_xj, unit_weights, *state, der );
				for ( std::size_t k=0; k<m_nkp; k++ )
					dR[k]( j,i ) = der(k);
			}
			R( m_noofFreeSVs, i ) = 1.0; //last row is for b
			for ( std::size_t k=0; k<m_nkp; k++ )
				dR[k]( m_noofFreeSVs, i ) = 0.0;
		}
		
		
		//O.K.: A big step of the computation of the derivative m_d_alphab_d_theta is
		// the multiplication with H^{-1} B. (where B are the other terms).
		// However  instead of storing m_d_alphab_d_theta_i = -H^{-1}*b_i
		//we store _i and compute the multiplication with the inverse
		//afterwards by solving the system Hx_i = b_i 
		//for i = 1....m_nkp+1
		//this is a lot faster and numerically more stable.

		// compute the derivative of (\alpha, b) w.r.t. C
		if ( m_noofBoundedSVs > 0 ) {
			noalias(column(m_d_alphab_d_theta,m_nkp)) = prod( R, m_boundedLabels);
		}
		// compute the derivative of (\alpha, b) w.r.t. the kernel parameters
		for ( std::size_t k=0; k<m_nkp; k++ ) {
			RealVector sum = prod( dH[k], m_freeAlphas ); //sum = dH * \alpha_f
			if(m_noofBoundedSVs > 0)
				noalias(sum) += prod( dR[k], m_boundedAlphas ); // sum += dR * \alpha_r , i.e., the C*y_g is expressed as alpha_g
			//fill the remaining columns of the derivative matrix (except the last, which is for C)
			noalias(column(m_d_alphab_d_theta,k)) = sum;
		}
		
		//lastly solve the system Hx_i = b_i 
		// MAJOR STEP: this is the achilles heel of the current implementation, cf. keerthi 2007
		// TODO: mtq: explore ways for speed-up..
		//compute via moore penrose pseudo-inverse
		noalias(m_d_alphab_d_theta) = - solveH(H,m_d_alphab_d_theta);
		
		// that's all, folks; we're done.
	}
	
	RealMatrix solveH(RealMatrix const& H, RealMatrix const& rhs){
		//implement using moore penrose pseudo inverse
		RealMatrix HTH=prod(trans(H),H);
		RealMatrix result = solve(HTH,prod(H,rhs),blas::symm_semi_pos_def(),blas::left());
		return result;
	}
};//class


}//namespace
#endif
