//===========================================================================
/*!
 *
 *
 * \brief       PCA
 *
 *
 *
 * \author      T.Glasmachers, Christian Igel
 * \date        2010-2011
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
#define SHARK_COMPILE_DLL
#include <shark/Data/Statistics.h>
#include <shark/Algorithms/Trainers/PCA.h>

using namespace shark;

/// Set the input data, which is stored in the PCA object.
void PCA::setData(UnlabeledData<RealVector> const& inputs) {
	SHARK_RUNTIME_CHECK(inputs.numberOfElements() >= 2, "Input needs to contain at least two points");
	m_l = inputs.numberOfElements(); ///< number of data points
	PCAAlgorithm algorithm = m_algorithm;
	m_n = dataDimension(inputs);

	if(algorithm == AUTO)  {
		if(m_n > m_l) algorithm = SMALL_SAMPLE; // more attributes than data points
		else algorithm = STANDARD;
	}

	// decompose covariance matrix
	if(algorithm == STANDARD) { // standard case
		RealMatrix S(m_n,m_n);//covariance matrix
		meanvar(inputs,m_mean,S);
		blas::symm_eigenvalue_decomposition<RealMatrix> eigen(S);
		m_eigenvectors = eigen.Q();
		m_eigenvalues = eigen.D();
	} else {
		//let X0 be the design matrix having all inputs as rows
		//we want to avoid to form it directly but us it's batch represntation in the dataset
		m_mean = shark::mean(inputs);
		RealMatrix S(m_l,m_l,0.0);//S=X0 X0^T
		//as every batch is only  a fraction of all samples, we have to loop through all
		//combinations of the batches of first and second argument and calculate a block of S
		//so if X0^T = (B0^T,B1^T)
		//than S = B0 B0^T B0 B1^T
		//               B1 B0^T  B1 B1^T
		std::size_t start1 = 0;
		for(std::size_t b1 = 0; b1 != inputs.numberOfBatches(); ++b1){
			std::size_t batchSize1 = inputs.batch(b1).size1();
			RealMatrix X1 = inputs.batch(b1)-repeat(m_mean,batchSize1);
			std::size_t start2 = 0;
			//calculate off-diagonal blocks
			//and the block X2 X1^T is the transpose of X1X2^T and thus can bee calculated for free.
			for(std::size_t b2 = 0; b2 != b1; ++b2){
				std::size_t batchSize2 = inputs.batch(b2).size1();
				RealMatrix X2 = inputs.batch(b2)-repeat(m_mean,batchSize2);
				auto X1X2T= subrange(S,start1,start1+batchSize1,start2,start2+batchSize2);
				auto X2X1T= subrange(S,start2,start2+batchSize2,start1,start1+batchSize1);
				noalias(X1X2T) = prod(X1,trans(X2));// X1 X2^T
				noalias(X2X1T) = trans(X1X2T);// X2 X1^T
				start2+=batchSize2;
			}
			//diagonal block
			auto X1X1T= subrange(S,start1,start1+batchSize1,start1,start1+batchSize1);
			noalias(X1X1T) = prod(X1,trans(X1));
			start1+=batchSize1;
		}
		S /= m_l;
		
		blas::symm_eigenvalue_decomposition<RealMatrix> eigen(S);
		m_eigenvalues = eigen.D();
		m_eigenvectors.resize(m_n,m_l);
		m_eigenvectors.clear();
		// compute true eigenvectors
		//eigenv=X0^T U
		std::size_t batchStart  = 0;
		for(std::size_t b = 0; b != inputs.numberOfBatches(); ++b){
			std::size_t batchSize = inputs.batch(b).size1();
			std::size_t batchEnd = batchStart+batchSize;
			RealMatrix X = inputs.batch(b)-repeat(m_mean,batchSize);
			noalias(m_eigenvectors) += prod(trans(X),rows(eigen.Q(),batchStart,batchEnd));
			batchStart = batchEnd;
		}
		//normalize
		for(std::size_t i=0; i != m_l; i++)
			column(m_eigenvectors, i) /= norm_2(column(m_eigenvectors, i));
	}
}

//! Returns a model mapping the original data to the
//! m-dimensional PCA coordinate system.
void PCA::encoder(LinearModel<>& model, std::size_t m) {
	if(!m) m = std::min(m_n,m_l);

	RealMatrix A = trans(columns(m_eigenvectors, 0, m) );
	RealVector offset = -prod(A, m_mean);
	if(m_whitening){
		for(std::size_t i=0; i<A.size1(); i++) {
			//take care of numerical difficulties for very small eigenvalues.
			if(m_eigenvalues(i)/m_eigenvalues(0) < 1.e-15){
				row(A,i).clear();
				offset(i) = 0;
			}
			else{
				row(A, i) /= std::sqrt(m_eigenvalues(i));
				offset(i) /= std::sqrt(m_eigenvalues(i));
			}
		}
	}
	model.setStructure(A, offset);
}

//! Returns a model mapping encoded data from the
//! m-dimensional PCA coordinate system back to the
//! n-dimensional original coordinate system.
void PCA::decoder(LinearModel<>& model, std::size_t m) {
	if(!m) m = std::min(m_n,m_l);
	if( m == m_n && !m_whitening){
		model.setStructure(m_eigenvectors, m_mean);
	}
	RealMatrix A = columns(m_eigenvectors, 0, m);
	if(m_whitening){
		for(std::size_t i=0; i<A.size2(); i++) {
			//take care of numerical difficulties for very small eigenvalues.
			if(m_eigenvalues(i)/m_eigenvalues(0) < 1.e-15){
				column(A,i).clear();
			}
			else{
				column(A, i) = column(A, i) * std::sqrt(m_eigenvalues(i));
			}
		}
	}

	model.setStructure(A, m_mean);
}
