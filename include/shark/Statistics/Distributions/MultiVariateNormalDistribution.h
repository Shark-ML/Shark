/*!
 *
 *
 * \brief       Implements a multi-variate normal distribution with zero mean.
 * 
 * 
 *
 * \author      T.Voss, O.Krause
 * \date        2016
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
#ifndef SHARK_STATISTICS_MULTIVARIATENORMALDISTRIBUTION_H
#define SHARK_STATISTICS_MULTIVARIATENORMALDISTRIBUTION_H

#include <shark/LinAlg/Base.h>
#include <shark/Core/Random.h>
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
	MultiVariateNormalDistribution(RealMatrix const& Sigma ) {
		m_covarianceMatrix = Sigma;
		update();
	}
	
	/// \brief Constructor
	MultiVariateNormalDistribution(){} 
	
	/// \brief Stores/Restores the distribution from the supplied archive.
	/// \param [in,out] ar The archive to read from/write to.
	/// \param [in] version Currently unused.
	template<typename Archive>
	void serialize( Archive & ar, const std::size_t version ) {
		ar & BOOST_SERIALIZATION_NVP( m_covarianceMatrix );
		ar & BOOST_SERIALIZATION_NVP( m_decomposition );
	}

	/// \brief Resizes the distribution. Updates both eigenvectors and eigenvalues.
	/// \param [in] size The new size of the distribution
	void resize( std::size_t size ) {
		m_covarianceMatrix = blas::identity_matrix<double>( size );
		update();
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
	RealMatrix const& eigenVectors() const {
		return m_decomposition.Q();
	}

	/// \brief Accesses an immutable reference to the eigenvalues of the covariance matrix.
	RealVector const& eigenValues() const {
		return m_decomposition.D();
	}

	/// \brief Samples the distribution.
	template<class randomType>
	result_type operator()(randomType& rng) const {
		RealVector z( m_covarianceMatrix.size1() );
		
		for( std::size_t i = 0; i < z.size(); i++ ) {
			z( i ) = random::gauss(rng, 0., 1. );
		}
		
		RealVector result = m_decomposition.Q() % to_diagonal(sqrt(max(eigenValues(),0))) % z;
		return std::make_pair( result, z );
	}	    

	/// \brief Calculates the evd of the current covariance matrix.
	void update() {
		m_decomposition.decompose(m_covarianceMatrix);
	}

private:
	RealMatrix m_covarianceMatrix; ///< Covariance matrix of the mutation distribution.
	blas::symm_eigenvalue_decomposition<RealMatrix> m_decomposition; /// < Eigenvalue decomposition of the covarianceMatrix
};

/// \brief Multivariate normal distribution with zero mean using a cholesky decomposition
class MultiVariateNormalDistributionCholesky{
public:
	/// \brief Result type of a sampling operation.
	/// 
	/// The first element is the result of sampling this distribution, the
	/// second element is the original standard-normally distributed vector drawn
	/// for sampling purposes.
	typedef std::pair<RealVector,RealVector> result_type;

	/// \brief Constructor
	/// \param [in] rng the random number generator
	/// \param [in] covariance covariance matrix
	MultiVariateNormalDistributionCholesky( RealMatrix const& covariance){
		setCovarianceMatrix(covariance);
	}
	
	MultiVariateNormalDistributionCholesky(){} 
	
	/// \brief Stores/Restores the distribution from the supplied archive.
	///\param [in,out] ar The archive to read from/write to.
	///\param [in] version Currently unused.
	template<typename Archive>
	void serialize( Archive & ar, const std::size_t version ) {
		ar & BOOST_SERIALIZATION_NVP( m_cholesky);
	}

	/// \brief Resizes the distribution. Updates both eigenvectors and eigenvalues.
	/// \param [in] size The new size of the distribution
	void resize( std::size_t size ) {
		m_cholesky = blas::identity_matrix<double>( size );
	}
	
	/// \brief Returns the size of the created vectors
	std::size_t size()const{
		return m_cholesky.lower_factor().size1();
	}

	/// \brief Returns the matrix holding the lower cholesky factor A
	blas::matrix<double,blas::column_major> const& lowerCholeskyFactor()const{
		return m_cholesky.lower_factor();
	}

	
	/// \brief Sets the new covariance matrix by computing the new cholesky dcomposition
	void setCovarianceMatrix(RealMatrix const& matrix){
		m_cholesky.decompose(matrix);
	}

	/// \brief Updates the covariance matrix of the distribution to C<- alpha*C+beta * vv^T
	void rankOneUpdate(double alpha, double beta, RealVector const& v){
		m_cholesky.update(alpha,beta,v);
	}
	
	template<class randomType, class Vector1, class Vector2>
	void generate(randomType& rng, Vector1& y, Vector2& z)const{
		z.resize(size());
		y.resize(size());
		for( std::size_t i = 0; i != size(); i++ ) {
			z( i ) = random::gauss(rng, 0, 1 );
		}
		noalias(y) = blas::triangular_prod<blas::lower>(m_cholesky.lower_factor(),z);
	}

	/// \brief Samples the distribution.
	///
	/// Returns a vector pair (y,z) where  y=Lz and, L is the lower cholesky factor and z is a vector
	/// of normally distributed numbers. Thus y is the real sampled point.
	template<class randomType>
	result_type operator()(randomType& rng) const {
		result_type result;
		RealVector& z = result.second;
		RealVector& y = result.first;
		generate(rng,y,z);
		return result;
	}

private:
	blas::cholesky_decomposition<blas::matrix<double,blas::column_major> > m_cholesky; ///< The lower cholesky factor (actually any is okay as long as it is the left)
};


}

#endif
