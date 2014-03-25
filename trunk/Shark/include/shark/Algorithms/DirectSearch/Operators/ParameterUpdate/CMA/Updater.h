#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_PARAMETER_UPDATE_CMA_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_PARAMETER_UPDATE_CMA_H

namespace shark {
namespace elitist_cma {
/**
* \brief Strategy parameter update that implements the 
* strategy parameter update procedure of the \f$(\mu+1)\f$-MO-CMA-ES.
*/
template<typename Individual, typename Chromosome, unsigned int ChromosomeIndex>
struct Updater {

	/**
	* \brief Chromosome type.
	*/
	typedef Chromosome chromosome_type;

	/**
	* \brief Individual type.
	*/
	typedef Individual individual_type;

	/**
	* \brief Updates a \f$(\mu+1)\f$-MO-CMA-ES individual.
	*
	* Integrates with the STL algorithms and allows for updating
	* whole population:
	* \code
	* Population pop( 101 );
	* // update the first 100 individuals
	* std::transform( pop.begin(), pop.begin() + 100, Updater() );
	* \endcode
	*
	* \param [in,out] individual The individual to be initialized.
	*/	
	void operator()( individual_type & individual ) {
		(*this)( individual.template get<ChromosomeIndex>() );
	}

	/**
	* \brief Updates a \f$(\mu+1)\f$-MO-CMA-ES chromosome.
	*
	* Updates strategy parameters according to:
	* \f{align*}
	*	\vec{p}_c & \leftarrow & (1-c_c) \vec{p}_c + \mathbb{1}_{\bar{p}_{\text{succ}} < p_{\text{thresh}}} \sqrt{c_c (2 - c_c)} \vec{x}_{\text{step}} \\
	*	\vec{C} & \leftarrow & (1-c_{\text{cov}}) \vec{C} + c_{\text{cov}} \left(  \vec{p}_c \vec{p}_c^T + \mathbb{1}_{\bar{p}_{\text{succ}} \geq p_{\text{thresh}}} c_c (2 - c_c) \vec{C} \right) \\
	*	\bar{p}_{\text{succ}} & \leftarrow & (1-c_p)\bar{p}_{\text{succ}} + c_p p_{\text{succ}} \\
	*	\sigma & \leftarrow & \sigma \cdot e^{\frac{1}{d}\frac{\bar{p}_{\text{succ}} - p^{\text{target}}_{\text{succ}}}{1-p^{\text{target}}_{\text{succ}}}}
	* \f}
	*
	* \param [in,out] c The chromosome to be updated.
	*/	
	void operator()( chromosome_type & c ) {

		if( c.m_needsCovarianceUpdate ) {
			if( c.m_successProbability < m_successThreshold ) {
				c.m_evolutionPath = (1-c.m_evolutionPathLearningRate)*c.m_evolutionPath + ::sqrt( c.m_evolutionPathLearningRate * ( 2.-c.m_evolutionPathLearningRate ) ) * c.m_lastStep;
				c.m_mutationDistribution.covarianceMatrix() = (1.-c.m_covarianceMatrixLearningRate)*c.m_mutationDistribution.covarianceMatrix() + c.m_covarianceMatrixLearningRate * blas::outer_prod( c.m_evolutionPath , c.m_evolutionPath );
			} else {
				c.m_evolutionPath = (1-c.m_evolutionPathLearningRate)*c.m_evolutionPath;
				c.m_mutationDistribution.covarianceMatrix() = (1.-c.m_covarianceMatrixLearningRate)*c.m_mutationDistribution.covarianceMatrix() + 
					c.m_covarianceMatrixLearningRate * ( 
					blas::outer_prod( c.m_evolutionPath , c.m_evolutionPath ) + 
					c.m_evolutionPathLearningRate*(2-c.m_evolutionPathLearningRate)*c.m_mutationDistribution.covarianceMatrix() 
					);

			}

			c.m_mutationDistribution.update();
			c.m_needsCovarianceUpdate = false;
		}
		c.m_successProbability = (1 - c.m_stepSizeLearningRate) * c.m_successProbability + c.m_stepSizeLearningRate * ( c.m_noSuccessfulOffspring / c.m_lambda );
		c.m_stepSize *= ::exp( 1./c.m_stepSizeDampingFactor * (c.m_successProbability - c.m_targetSuccessProbability) / (1-c.m_targetSuccessProbability) );

		c.m_noSuccessfulOffspring = 0;
	}

	/**
	* \brief Stores/restores the updater's state.
	* \tparam Archive Type of the archive.
	* \param [in,out] archive The archive to serialize to.
	* \param [in] version Version number, currently unused.
	*/		    
	template<typename Archive>
	void serialize( Archive & archive, const unsigned int version ) {
		archive & m_successThreshold;
	}

	double m_successThreshold; ///< Success threshold \f$p_{\text{thresh}}\f$ for cutting off evolution path updates.
};
}
}

#endif
