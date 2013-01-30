//===========================================================================
/*!
 *  \brief PCA
 *
 *  \author T.Glasmachers, Christian Igel
 *  \date 2010-2011
 *
 *  <BR><HR>
 *  This file is part of Shark. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 3, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, see <http://www.gnu.org/licenses/>.
 *
 */
//===========================================================================
#include <shark/LinAlg/eigenvalues.h>
#include <shark/Algorithms/Trainers/PCA.h>

using namespace shark;

	/// Set the input data, which is stored in the PCA object.
void PCA::setData(UnlabeledData<RealVector> const& inputs) {
	SHARK_CHECK(inputs.numberOfElements() >= 2, "[PCA::train] input needs to contain at least two points");
	m_l = inputs.numberOfElements(); ///< number of data points
	PCAAlgorithm algorithm = m_algorithm;
	m_n = dataDimension(inputs); 
	
	if(algorithm == AUTO)  {
		if(m_n > m_l) algorithm = SMALL_SAMPLE; // more attributes than data points
		else algorithm = STANDARD;
	}
	
	// compute mean
	m_mean = inputs(0);
	for(std::size_t i=1; i<m_l; i++) 
		noalias(m_mean) += inputs(i);
	m_mean /= m_l;
		
	// build design matrix
	RealMatrix X0(m_l,m_n);           ///< design matrix, mean-free
	for(std::size_t i=0; i< m_l; i++) 
		noalias(row(X0, i)) = inputs(i) - m_mean;
	
	
	// decompose covariance matrix
	
	if(algorithm == STANDARD) { // standard case
		RealMatrix S(m_n,m_n);
		symmRankKUpdate(trans(X0),S);
		S /= m_l;
		m_eigenvalues.resize(m_n);
		m_eigenvectors.resize(m_n, m_n);
		eigensymm(S, m_eigenvectors, m_eigenvalues);
	} else {
		RealMatrix S(m_l,m_l);
		symmRankKUpdate(X0,S);
		S /= m_l;
		m_eigenvalues.resize(m_l);
		m_eigenvectors.resize(m_n, m_l);
		RealMatrix U(m_l, m_l);
		eigensymm(S, U, m_eigenvalues);
		// compute true eigenvectors
		fast_prod(trans(X0),U,m_eigenvectors);
		for(std::size_t i=0; i<m_l; i++)
			column(m_eigenvectors, i) /= norm_2(column(m_eigenvectors, i));
	}
}

//! Returns a model mapping the original data to the
//! m-dimensional PCA coordinate system.
void PCA::encoder(LinearModel<>& model, std::size_t m) {
	if(!m) m = m_n;
	RealVector offset;
	// in the hope to save some time, we split the simple
	// computation in four very similar cases
	if(m != m_n) { // dimensionality reduction
		offset = -prod(trans( blas::subrange(m_eigenvectors,0, m_eigenvectors.size1(), 0, m) ), m_mean);
		if(!m_whitening) // no whitening
			model.setStructure(RealMatrix(trans(blas::subrange(m_eigenvectors, 0, m_eigenvectors.size1(), 0, m) )), offset);
		else { // whitening
			RealMatrix A = trans(blas::subrange(m_eigenvectors, 0, m_eigenvectors.size1(), 0, m) );
			for(std::size_t i=0; i<A.size1(); i++) {
				row(A, i) = row(A, i) / std::sqrt(m_eigenvalues(i));
				offset(i) /= std::sqrt(m_eigenvalues(i));
			}
			model.setStructure(A, offset);
		}
	} else { // no dimensionality redunction 
		offset = -prod(trans(m_eigenvectors), m_mean);
		if(!m_whitening) // no whitening
			model.setStructure(RealMatrix(trans(m_eigenvectors)), offset);
		else { // whitening
			RealMatrix A = 	trans(m_eigenvectors);
			for(std::size_t i=0; i<A.size1(); i++) {
				row(A, i) = row(A, i) / std::sqrt(m_eigenvalues(i));
				offset(i) /= std::sqrt(m_eigenvalues(i));
			}
			model.setStructure(A, offset);
		}
	}
	SHARK_CHECK(model.hasOffset(), "[PCA] model must have an offset");
}

//! Returns a model mapping encoded data from the
//! m-dimensional PCA coordinate system back to the
//! n-dimensional original coordinate system.
void PCA::decoder(LinearModel<>& model, std::size_t m) {
	if(!m) m = m_n;
	RealVector offset;
	// in the hope to save some time, we split the simple
	// computation in four very similar cases
	if(m != m_n) { // reduced dimension
		if(!m_whitening) // no whitening
			model.setStructure(RealMatrix(blas::subrange(m_eigenvectors, 0, m_eigenvectors.size1(), 0, m) ), m_mean);
		else { // whitening
			RealVector offset = m_mean;
			RealMatrix A = blas::subrange(m_eigenvectors, 0, m_eigenvectors.size1(), 0, m);
			for(std::size_t i=0; i<A.size2(); i++) {
				column(A, i) = column(A, i) * std::sqrt(m_eigenvalues(i));
			}
			model.setStructure(A, offset);
		}
	} else { // full dimension
		if(!m_whitening) // no whitening
			model.setStructure(m_eigenvectors, m_mean);
		else { // whitening
			RealVector offset = m_mean;
			RealMatrix A =  m_eigenvectors;
			for(std::size_t i=0; i<A.size2(); i++) {
				column(A, i) = column(A, i) * std::sqrt(m_eigenvalues(i));
			}
			model.setStructure(A, offset);
		}
	}
	SHARK_CHECK(model.hasOffset(), "[PCA] model must have an offset");
}
