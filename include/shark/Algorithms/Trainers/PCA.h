//===========================================================================
/*!
 * 
 *
 * \brief       Principal Component Analysis
 * 
 * 
 * 
 *
 * \author      T. Glasmachers, C. Igel
 * \date        2010, 2011
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


#ifndef SHARK_ALGORITHMS_TRAINER_PCA_H
#define SHARK_ALGORITHMS_TRAINER_PCA_H

#include <shark/Models/LinearModel.h>
#include <shark/Algorithms/Trainers/AbstractTrainer.h>

namespace shark{

/*!
 *  \brief Principal Component Analysis
 *
 *  The Principal Component Analysis, also known as
 *  Karhunen-Loeve transformation, takes a symmetric
 *  \f$ n \times n \f$ matrix \f$ A \f$ and uses its decomposition
 *
 *  \f$
 *      A = \Gamma \Lambda \Gamma^T,
 *  \f$
 *
 *  where \f$ \Lambda \f$ is the diagonal matrix of eigenvalues
 *  of \f$ A \f$ and \f$ \Gamma \f$ is the orthogonal matrix
 *  with the corresponding eigenvectors as columns.
 *  \f$ \Lambda \f$ then defines a successive orthogonal rotation
 *  that maximizes the variances of the coordinates, i.e. the
 *  coordinate system is rotated in such a way that the correlation
 *  between the new axes becomes zero. If there are \f$ p \f$ axes,
 *  the first axis is rotated in a way that the points on the new axis
 *  have maximum variance. Then the remaining \f$ p - 1 \f$ axes are
 *  rotated such that a another axis covers a maximum part of the rest
 *  variance, that is not covered by the first axis. After the
 *  rotation of \f$ p - 1 \f$ axes, the rotation destination of axis
 *  no. \f$ p \f$ is fixed.  An application for PCA is the reduction
 *  of dimensions by skipping the components with the least
 *  corresponding eigenvalues/variances. Furthermore, the eigenvalues
 *  may be rescaled to one, resulting in a whitening of the data.
 */
class PCA : public AbstractUnsupervisedTrainer<LinearModel<> >
{
private:
	typedef AbstractUnsupervisedTrainer<LinearModel<> > base_type;
public:
	enum PCAAlgorithm { STANDARD, SMALL_SAMPLE, AUTO };

	/// Constructor.
	/// The parameter defines whether the model should also
	/// whiten the data.
	PCA(bool whitening = false) 
	: m_whitening(whitening){
		m_algorithm = AUTO;
	};
	/// Constructor.
	/// The parameter defines whether the model should also
	/// whiten the data.
	/// The eigendecomposition of the data is stored inthe PCA object.
	PCA(UnlabeledData<RealVector> const& inputs, bool whitening = false) 
	: m_whitening(whitening){
		m_algorithm = AUTO;
		setData(inputs);
	};

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "PCA"; }

	/// If set to true, the encoded data has unit variance along
	/// the new coordinates.
	void setWhitening(bool whitening) {
		m_whitening = whitening;
	}

	/// Train the model to perform PCA. The model must be a
	/// LinearModel object with offset, and its output dimension
	/// defines the number of principal components
	/// represented. The model returned is the one given by the
	/// econder() function (i.e., mapping from the original input
	/// space to the PCA coordinate system).
	void train(LinearModel<>& model, UnlabeledData<RealVector> const& inputs) {
		std::size_t m = model.outputSize(); ///< reduced dimensionality
		setData(inputs);   // compute PCs
		encoder(model, m); // define the model 
	}


	//! Sets the input data and performs the PCA. This is a
	//! computationally costly operation. The eigendecomposition
	//! of the data is stored inthe PCA object.
	void setData(UnlabeledData<RealVector> const& inputs);

	//! Returns a model mapping the original data to the
	//! m-dimensional PCA coordinate system.
	void encoder(LinearModel<>& model, std::size_t m = 0);

	//! Returns a model mapping encoded data from the
	//! m-dimensional PCA coordinate system back to the
	//! n-dimensional original coordinate system.
	void decoder(LinearModel<>& model, std::size_t m = 0);

	/// Eigenvalues of last training. The number of eigenvalues
	//! is equal to the minimum of the input dimensions (i.e.,
	//! number of attributes) and the number of data points used
	//! for training the PCA.
	RealVector const& eigenvalues() const {
		return m_eigenvalues;
	}
	/// Returns ith eigenvalue.
	double eigenvalue(std::size_t i) const {
		SIZE_CHECK( i < m_l );
		if( i < m_eigenvalues.size()) 
			return m_eigenvalues(i);
		return 0.;
	}

	//! Eigenvectors of last training. The number of eigenvectors
	//! is equal to the minimum of the input dimensions (i.e.,
	//! number of attributes) and the number of data points used
	//! for training the PCA.
	RealMatrix const& eigenvectors() const{
		return m_eigenvectors;
	}

	/// mean of last training
	RealVector const& mean() const{
		return m_mean;
	}

protected:
	bool m_whitening;          ///< normalize variance yes/no
	RealMatrix m_eigenvectors; ///< eigenvectors
	RealVector m_eigenvalues;  ///< eigenvalues
	RealVector m_mean;	   ///< mean value

	std::size_t m_n;           ///< number of attributes
	std::size_t m_l;           ///< number of training data points

	PCAAlgorithm m_algorithm;  ///< whether to use design matrix or its transpose for building covariance matrix
};


}
#endif // SHARK_ML_TRAINER_PCA_H
