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
#include <shark/Rng/GlobalRng.h>

namespace shark {

	namespace detail {

		/**
		* \brief Implements a multi-variate normal distribution with zero mean.
		* \tparam Rng  Random number generator type for sampling the distribution, needs to be a model of shark::GlobalRng.
		* \tparam MatrixType The underlying matrix type.
		* \tparam VectorType The underlying vector type.
		*/
		template<typename Rng, typename MatrixType, typename VectorType>
		class TypedMultiVariateNormalDistribution {
		public:

			typedef TypedMultiVariateNormalDistribution< Rng, MatrixType, VectorType > this_type;

			/**
			* \brief Result type of a sampling operation.
			* 
			* The first element is the result of sampling this distribution, the
			* second element is the original standard-normally distributed vector drawn
			* for sampling purposes.
			*/
			typedef std::pair<VectorType,VectorType> ResultType;

			/**
			* \brief Default c'tor.
			* \param [in] dimension Size of the distribution.
			*/
			TypedMultiVariateNormalDistribution() {
				//resize(dimension);
			}

			/**
			* \brief Constructor
			* \param [in] Sigma covariance matrix
			*/
			TypedMultiVariateNormalDistribution( MatrixType &Sigma ) {
				m_covarianceMatrix = Sigma;
				update();
			}


			/**
			* \brief Stores the distribution in the supplied archive.
			* \param [in,out] archive The archive to write to.
			*/
			template<typename Archive>
			void write( Archive & archive ) const {
				archive & BOOST_SERIALIZATION_NVP( m_covarianceMatrix );
				archive & BOOST_SERIALIZATION_NVP( m_eigenVectors );
				archive & BOOST_SERIALIZATION_NVP( m_eigenValues );
			}

			/**
			* \brief Restores the distribution from the supplied archive.
			* \param [in,out] archive The archive to read from.
			*/
			template<typename Archive>
			void read( Archive & archive ) {
				archive & BOOST_SERIALIZATION_NVP( m_covarianceMatrix );
				archive & BOOST_SERIALIZATION_NVP( m_eigenVectors );
				archive & BOOST_SERIALIZATION_NVP( m_eigenValues );
			}

			/**
			* \brief Stores/Restores the distribution from the supplied archive.
			* \param [in,out] ar The archive to read from/write to.
			* \param [in] version Currently unused.
			*/
			template<typename Archive>
			void serialize( Archive & ar, const unsigned int version ) {
				ar & BOOST_SERIALIZATION_NVP( m_covarianceMatrix );
				ar & BOOST_SERIALIZATION_NVP( m_eigenVectors );
				ar & BOOST_SERIALIZATION_NVP( m_eigenValues );
			}

			/**
			* \brief Resizes the distribution. Updates both eigenvectors and eigenvalues.
			* \param [in] size The new size of the distribution
			*/
			void resize( unsigned int size ) {
				m_covarianceMatrix = blas::identity_matrix<double>( size );
				update();
			}

			/**
			* \brief Accesses the covariance matrix defining the distribution.
			*/
			const MatrixType & covarianceMatrix() const {
				return( m_covarianceMatrix );
			}

			/**
			* \brief Set covariance matrix, update intenally eigenvectors and eigenvalues.
			* \param [in] size Covariance matrix
			*/
			void setCovarianceMatrix( MatrixType &Sigma ) {
				m_covarianceMatrix = Sigma;
				update();
			}

			/**
			* \brief Accesses a mutable reference to the covariance matrix 
			* defining the distribution. Allows for l-value semantics.
			* 
			* ATTENTION: If the reference is altered, update needs to be called manually.
			*/
			MatrixType & covarianceMatrix() {
				return m_covarianceMatrix;
			}

			/**
			* \brief Accesses a mutable reference to the eigenvectors of the covariance matrix.
			*/
			MatrixType & eigenVectors() {
				return m_eigenVectors;
			}

			/**
			* \brief Accesses an immutable reference to the eigenvectors of the covariance matrix.
			*/
			const MatrixType & eigenVectors() const {
				return m_eigenVectors;
			}

			/**
			* \brief Accesses a mutable reference to the eigenvalues of the covariance matrix.
			*/
			VectorType & eigenValues() {
				return m_eigenValues;
			}

			/**
			* \brief Accesses an immutable reference to the eigenvalues of the covariance matrix.
			*/
			const VectorType & eigenValues() const {
				return m_eigenValues;
			}

			/**
			* \brief Adjusts the covariance matrix and updates both eigenvectors and eigenvalues.
			* \param [in] m The new covariance matrix.
			*/
			void setCovarianceMatrix( const MatrixType & m ) {
				m_covarianceMatrix = m;
				update();
			}

			/**
			* \brief Samples the distribution.
			*/
			ResultType operator()() const {
				VectorType result( m_eigenValues.size(), 0. );
				VectorType z( m_eigenValues.size() );
				
				for( unsigned int i = 0; i < result.size(); i++ ) {
					z( i ) = Rng::gauss( 0., 1. );
				}

				for( unsigned int i = 0; i < result.size(); i++ )
					for( unsigned int j = 0; j < result.size(); j++ )
						result( i ) += m_eigenVectors( i, j ) * ::sqrt( ::fabs( m_eigenValues(j) ) ) * z( j );

				return( std::make_pair( result, z ) );
			}	    

			/**
			* \brief Calculates the evd of the current covariance matrix.
			*/
			void update() {
				eigensymm( m_covarianceMatrix, m_eigenVectors, m_eigenValues );
			}			

			/**
			* \brief Prints the distribution in human-readable form to the supplied stream.
			*/
			template<typename Stream>
			void print( Stream & s ) const {
				s << "C: " << m_covarianceMatrix << std::endl;
				s << "B: " << m_eigenVectors << std::endl;
				s << "D: " << m_eigenValues << std::endl;
			}

			/**
			* \brief Checks two distributions for equality.
			* 
			*/
			bool operator==( const this_type & rhs ) const {
				return( 
					shark::blas::norm_1( m_covarianceMatrix - rhs.m_covarianceMatrix ) == 0 &&
					shark::blas::norm_1( m_eigenVectors - rhs.m_eigenVectors ) == 0 &&
					shark::blas::norm_2( m_eigenValues - rhs.m_eigenValues ) == 0
				);
			}
			// protected:
			MatrixType m_covarianceMatrix; ///< Covariance matrix of the mutation distribution.
			MatrixType m_eigenVectors; ///< Eigenvectors of the covariance matrix.
			VectorType m_eigenValues; ///< Eigenvalues of the covariance matrix.
		};
	}

	/**
	* \brief Injects a multi-variate normal distribution in the shark namespace.
	*/
	typedef detail::TypedMultiVariateNormalDistribution<Rng,RealMatrix,RealVector> MultiVariateNormalDistribution;

}

#endif
