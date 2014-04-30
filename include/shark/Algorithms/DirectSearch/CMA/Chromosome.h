/*!
 * 
 *
 * \brief       CMAChromosomeof the CMA-ES.
 * 
 * 
 *
 * \author      T.Voss, T. Glasmachers, O.Krause
 * \date        2010-2011
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
#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_CMA_CHROMOSOME_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_CMA_CHROMOSOME_H

#include <shark/Statistics/Distributions/MultiVariateNormalDistribution.h>

namespace shark {

/**
* \brief Models a CMAChromosomeof the elitist (MO-)CMA-ES that encodes strategy parameters.
*/
struct CMAChromosome{
	enum IndividualSuccess{
		Successful = 1,
		Unsuccessful = 2,
		Failure = 3
	};
	//~ MultiVariateNormalDistribution m_mutationDistribution; ///< Models the search distribution
	MultiVariateNormalDistributionCholesky m_mutationDistribution; ///< Models the search distribution using a cholsky matrix
	//~ RealMatrix m_inverseCholesky;///< inverse cholesky matrix

	RealVector m_evolutionPath; ///< Low-pass filtered accumulation of successful mutative steps.
	RealVector m_lastStep; ///< The most recent mutative step.

	double m_stepSize; ///< The step-size used to scale the normally-distributed mutative steps. Dynamically adapted during the run.
	double m_stepSizeDampingFactor; ///< Damping factor \f$d\f$ used in the step-size update procedure.
	double m_stepSizeLearningRate; ///< The learning rate for the step-size.
	double m_successProbability; ///< Current success probability of this parameter set.
	double m_targetSuccessProbability; ///< Target success probability, close \f$ \frac{1}{5}\f$.
	double m_evolutionPathLearningRate; ///< Learning rate (constant) for updating the evolution path.
	double m_covarianceMatrixLearningRate; ///< Learning rate (constant) for updating the covariance matrix.
	double m_covarianceMatrixUnlearningRate; ///< Learning rate (constant) for unlearning unsuccessful directions from the covariance matrix.

	double m_successThreshold; ///< Success threshold \f$p_{\text{thresh}}\f$ for cutting off evolution path updates.

	CMAChromosome(){}
	CMAChromosome(
		std::size_t searchSpaceDimension,
		double successThreshold,
		double initialStepSize
	)
	: m_stepSize( initialStepSize )
	, m_covarianceMatrixLearningRate( 0 )
	, m_successThreshold(successThreshold)
	{
		m_mutationDistribution.resize( searchSpaceDimension );
		//~ m_inverseCholesky = blas::identity_matrix<double>( searchSpaceDimension );
		m_evolutionPath.resize( searchSpaceDimension );
		m_lastStep.resize( searchSpaceDimension );

		m_targetSuccessProbability = 1.0 / ( 5.0 + 1/2.0 );		
		m_successProbability = m_targetSuccessProbability;
		m_stepSizeDampingFactor = 1.0 + searchSpaceDimension / 2.;
		m_stepSizeLearningRate = m_targetSuccessProbability/ (2. + m_targetSuccessProbability );
		m_evolutionPathLearningRate = 2.0 / (2.0 + searchSpaceDimension);
		m_covarianceMatrixLearningRate = 2.0 / (sqr(searchSpaceDimension) + 6.);
		m_covarianceMatrixUnlearningRate = 0.4/( std::pow(searchSpaceDimension, 1.6 )+1. );
	}
	
	/**
	* \brief Updates a \f$(\mu+1)\f$-MO-CMA-ES chromosome of an successful offspring individual. It is assumed that unsuccessful individuals are not selected for future mutation.
	*
	* Updates strategy parameters according to:
	* \f{align*}
	*	\bar{p}_{\text{succ}} & \leftarrow & (1-c_p)\bar{p}_{\text{succ}} + c_p \\
	*	\sigma & \leftarrow & \sigma \cdot e^{\frac{1}{d}\frac{\bar{p}_{\text{succ}} - p^{\text{target}}_{\text{succ}}}{1-p^{\text{target}}_{\text{succ}}}}\\
	*	\vec{p}_c & \leftarrow & (1-c_c) \vec{p}_c + \mathbb{1}_{\bar{p}_{\text{succ}} < p_{\text{thresh}}} \sqrt{c_c (2 - c_c)} \vec{x}_{\text{step}} \\
	*	\vec{C} & \leftarrow & (1-c_{\text{cov}}) \vec{C} + c_{\text{cov}} \left(  \vec{p}_c \vec{p}_c^T + \mathbb{1}_{\bar{p}_{\text{succ}} \geq p_{\text{thresh}}} c_c (2 - c_c) \vec{C} \right)
	* \f}
	*/	
	void updateAsOffspring() {
		m_successProbability = (1 - m_stepSizeLearningRate) * m_successProbability + m_stepSizeLearningRate;
		m_stepSize *= ::exp( 1./m_stepSizeDampingFactor * (m_successProbability - m_targetSuccessProbability) / (1-m_targetSuccessProbability) );
		
		double evolutionpathUpdateWeight=m_evolutionPathLearningRate * ( 2.-m_evolutionPathLearningRate );
		if( m_successProbability < m_successThreshold ) {
			m_evolutionPath *= 1 - m_evolutionPathLearningRate;
			noalias(m_evolutionPath) += std::sqrt( evolutionpathUpdateWeight ) * m_lastStep;
			rankOneUpdate(1 - m_covarianceMatrixLearningRate,m_covarianceMatrixLearningRate,m_evolutionPath);
		} else {
			roundUpdate();
		}
	}
	
	/**
	* \brief Updates a \f$(\mu+1)\f$-MO-CMA-ES chromosome of a parent individual.
	*
	* This is called when the parent individual survived the last selection process. The update process depends now on how the offspring fares:
	* It can be successful, unsuccesful or a complete failure.
	* 
	* Based on whether it is succesful or not, the global stepsize is adapted as for the child. In the case of a failure the direction of that individual is actively 
	* purged from the Covariance matrix to make this offspring less likely.
	*/	
	void updateAsParent(IndividualSuccess offspringSuccess) {
		m_successProbability = (1 - m_stepSizeLearningRate) * m_successProbability + m_stepSizeLearningRate * (offspringSuccess == Successful);
		m_stepSize *= ::exp( 1./m_stepSizeDampingFactor * (m_successProbability - m_targetSuccessProbability) / (1-m_targetSuccessProbability) );
		
		if(offspringSuccess != Failure) return;
		
		if( m_successProbability < m_successThreshold ) {
			//check whether the step is admissible with the proposed update weight
			double rate = m_covarianceMatrixUnlearningRate;
			double stepNormSqr = norm_sqr( m_lastStep );
			if( 1 <  m_covarianceMatrixUnlearningRate*(2*stepNormSqr-1) ){
				rate = 0.5/(2*stepNormSqr-1);//make the update shorter
				return; //better be safe for now
			}
			rankOneUpdate(1-rate,rate,m_lastStep);
		} else {
			roundUpdate();
		}
	}

	/**
	* \brief Serializes the CMAChromosometo the supplied archive.
	* \tparam Archive The type of the archive the CMAChromosomeshall be serialized to.
	* \param [in,out] archive The archive to serialize to.
	* \param [in] version Version information (optional and not used here).
	*/
	template<typename Archive>
	void serialize( Archive & archive, const unsigned int version ) {

		archive & BOOST_SERIALIZATION_NVP( m_mutationDistribution );
		//~ archive & BOOST_SERIALIZATION_NVP( m_inverseCholesky );

		archive & BOOST_SERIALIZATION_NVP( m_evolutionPath );
		archive & BOOST_SERIALIZATION_NVP( m_lastStep );

		archive & BOOST_SERIALIZATION_NVP( m_stepSize );
		archive & BOOST_SERIALIZATION_NVP( m_stepSizeDampingFactor );
		archive & BOOST_SERIALIZATION_NVP( m_stepSizeLearningRate );
		archive & BOOST_SERIALIZATION_NVP( m_successProbability );
		archive & BOOST_SERIALIZATION_NVP( m_targetSuccessProbability );
		archive & BOOST_SERIALIZATION_NVP( m_evolutionPathLearningRate );
		archive & BOOST_SERIALIZATION_NVP( m_covarianceMatrixLearningRate );
		archive & BOOST_SERIALIZATION_NVP( m_covarianceMatrixUnlearningRate );
	}
private:
	/// \brief Performs a rank one update to the cholesky factor. 
	///
	/// This also requries an update of the inverse cholesky factor, that is the only reason, it exists.
	void rankOneUpdate(double alpha, double beta, RealVector const& v){
		//~ RealMatrix & C = m_mutationDistribution.covarianceMatrix();
		//~ noalias(C) = alpha*C - beta * outer_prod( m_lastStep, m_lastStep );
		//~ m_mutationDistribution.update();
		
		RealMatrix& A =m_mutationDistribution.lowerCholeskyFactor();
		A *= alpha; 
		choleskyUpdate(A,v,beta);
		//~ RealVector w = prod(m_inverseCholesky,v);
		//~ if(norm_inf(w) < 1.e-20) return; //precision under which we assum that the update is mostly noise.
		//~ RealVector wInv = prod(w,m_inverseCholesky);
		
		//~ double normWSqr =norm_sqr(w);
		//~ double a = std::sqrt(alpha);
		//~ double root = std::sqrt(1+beta/alpha*normWSqr);
		//~ double b = a/normWSqr * (root-1);
		//~ RealMatrix& A =m_mutationDistribution.lowerCholeskyFactor();
		//~ noalias(A) =a*A+b*outer_prod(v,w);
		//~ noalias(m_inverseCholesky) = 1.0/a * m_inverseCholesky - b/ (a*a+a*b*normWSqr)*outer_prod(w,wInv);
	}
	
	/// \brief Performs an update step which makes the distribution more round
	///
	/// This is called, when the distribution is too successful as this indicates that the step size  
	/// in some direction is too small to be useful
	void roundUpdate(){
		double evolutionpathUpdateWeight = m_evolutionPathLearningRate * ( 2.-m_evolutionPathLearningRate );
		m_evolutionPath *= 1 - m_evolutionPathLearningRate;
		rankOneUpdate(
			1 - m_covarianceMatrixLearningRate+evolutionpathUpdateWeight,
			m_covarianceMatrixLearningRate,
			m_evolutionPath
		);
	}
};
}

#endif 