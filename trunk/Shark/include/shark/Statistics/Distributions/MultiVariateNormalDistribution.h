/*!
 *
 *
 * \brief       Implements a multi-variate normal distribution with zero mean.
 * 
 * 
 *
 * \author      T.Voss
 * \date        2010
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
#ifndef MULTIVARIATENORMALDISTRIBUTION_H
#define MULTIVARIATENORMALDISTRIBUTION_H

#include <shark/LinAlg/eigenvalues.h>
#include <shark/LinAlg/Cholesky.h>
#include <shark/Rng/GlobalRng.h>

namespace shark {

/// \brief Implements a multi-variate normal distribution with zero mean.
class MultiVariateNormalDistribution {
public:
	///\brief Result type of a sampling operation.
	/// 
	/// The first element is the result of sampling this distribution, the
	/// second element is the original standard-normally distributed vector drawn
	/// for sampling purposes.
	typedef std::pair<RealVector,RealVector> result_type;

	/// \brief Constructor
	/// \param [in] Sigma covariance matrix
	MultiVariateNormalDistribution( RealMatrix const& Sigma ) {
		m_covarianceMatrix = Sigma;
		update();
	}
	
	MultiVariateNormalDistribution(){} 
	
	/// \brief Stores/Restores the distribution from the supplied archive.
	/// \param [in,out] ar The archive to read from/write to.
	/// \param [in] version Currently unused.
	template<typename Archive>
	void serialize( Archive & ar, const unsigned int version ) {
		ar & BOOST_SERIALIZATION_NVP( m_covarianceMatrix );
		ar & BOOST_SERIALIZATION_NVP( m_eigenVectors );
		ar & BOOST_SERIALIZATION_NVP( m_eigenValues );
	}

	/// \brief Resizes the distribution. Updates both eigenvectors and eigenvalues.
	/// \param [in] size The new size of the distribution
	void resize( unsigned int size ) {
		m_covarianceMatrix = blas::identity_matrix<double>( size );
		m_eigenValues = blas::repeat(1.0,size);
		m_eigenVectors = blas::identity_matrix<double>( size );
	}

	/// \brief Accesses the covariance matrix defining the distribution.
	RealMatrix const& covarianceMatrix() const {
		return m_covarianceMatrix;
	}
	
	/// \brief Accesses a mutable reference to the covariance matrix 
	/// defining the distribution. Allows for l-value semantics.
	/// 
	/// ATTENTION: If the reference is altered, update needs to be called manually.
	RealMatrix& covarianceMatrix() {
		return m_covarianceMatrix;
	}
	
	/// \brief Sets the covariance matrix and updates the internal variables. This is expensive
	void setCovarianceMatrix(RealMatrix const& matrix){
		covarianceMatrix() = matrix;
		update();
	}

	/// \brief Accesses an immutable reference to the eigenvectors of the covariance matrix.
	const RealMatrix & eigenVectors() const {
		return m_eigenVectors;
	}

	/// \brief Accesses an immutable reference to the eigenvalues of the covariance matrix.
	RealVector const& eigenValues() const {
		return m_eigenValues;
	}

	/// \brief Samples the distribution.
	result_type operator()() const {
		RealVector result( m_eigenValues.size(), 0. );
		RealVector z( m_eigenValues.size() );
		
		for( unsigned int i = 0; i < result.size(); i++ ) {
			z( i ) = Rng::gauss( 0., 1. );
		}

		for( unsigned int i = 0; i < result.size(); i++ )
			for( unsigned int j = 0; j < result.size(); j++ )
				result( i ) += m_eigenVectors( i, j ) * std::sqrt( std::abs( m_eigenValues(j) ) ) * z( j );

		return( std::make_pair( result, z ) );
	}	    

	/// \brief Calculates the evd of the current covariance matrix.
	void update() {
		eigensymm( m_covarianceMatrix, m_eigenVectors, m_eigenValues );
	}			

private:
	RealMatrix m_covarianceMatrix; ///< Covariance matrix of the mutation distribution.
	RealMatrix m_eigenVectors; ///< Eigenvectors of the covariance matrix.
	RealVector m_eigenValues; ///< Eigenvalues of the covariance matrix.
};

/// \brief Multivariate normal distribution with zero mean using a cholesky decomposition
class MultiVariateNormalDistributionCholesky {
public:
	/// \brief Result type of a sampling operation.
	/// 
	/// The first element is the result of sampling this distribution, the
	/// second element is the original standard-normally distributed vector drawn
	/// for sampling purposes.
	typedef std::pair<RealVector,RealVector> result_type;

	/// \brief Constructor
	/// \param [in] covariance covariance matrix
	MultiVariateNormalDistributionCholesky( RealMatrix const& covariance ) {
		setCovarianceMatrix(covariance);
	}
	
	MultiVariateNormalDistributionCholesky(){} 
	
	/// \brief Stores/Restores the distribution from the supplied archive.
	///\param [in,out] ar The archive to read from/write to.
	///\param [in] version Currently unused.
	template<typename Archive>
	void serialize( Archive & ar, const unsigned int version ) {
		ar & BOOST_SERIALIZATION_NVP( m_lowerCholesky);
	}

	/// \brief Resizes the distribution. Updates both eigenvectors and eigenvalues.
	/// \param [in] size The new size of the distribution
	void resize( unsigned int size ) {
		m_lowerCholesky = blas::identity_matrix<double>( size );
	}
	
	/// \brief Returns the size of the created vectors
	std::size_t size()const{
		return m_lowerCholesky.size1();
	}
	
	/// \brief Sets the new covariance matrix by computing the new cholesky dcomposition
	void setCovarianceMatrix(RealMatrix const& matrix){
		choleskyDecomposition(matrix,m_lowerCholesky);
	}

	/// \brief Returns the lower cholsky factor.
	RealMatrix const& lowerCholeskyFactor() const {
		return m_lowerCholesky;
	}

	/// \brief Returns the lower cholesky factor.
	RealMatrix& lowerCholeskyFactor(){
		return m_lowerCholesky;
	}

	/// \brief Samples the distribution.
	///
	/// Returns a vector pair (y,z) where  y=Lz and, L is the lower cholesky factor and z is a vector
	/// of normally distributed numbers. Thus y is the real sampled point.
	result_type operator()() const {
		result_type result;
		RealVector& z = result.second;
		RealVector& y = result.first;
		z.resize(size());
		y.resize(size());
		
		for( unsigned int i = 0; i != size(); i++ ) {
			z( i ) = Rng::gauss( 0, 1 );
		}
		
		axpy_prod(m_lowerCholesky,z,y,false);

		return result;
	}

private:
	RealMatrix m_lowerCholesky; ///< The lower cholesky factor (actually any is oaky as long as it is the left)
};


}

#endif
