//===========================================================================
/*!
 * 
 *
 * \brief       Data normalization to zero mean, unit variance and zero covariance while keping the original coordinate system
 * 
 * 
 * 
 *
 * \author      T. Glasmachers
 * \date        2010
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


#ifndef SHARK_ALGORITHMS_TRAINERS_NORMALIZECOMPONENTSZCA_H
#define SHARK_ALGORITHMS_TRAINERS_NORMALIZECOMPONENTSZCA_H


#include <shark/Models/LinearModel.h>
#include <shark/Algorithms/Trainers/AbstractTrainer.h>
#include <shark/Data/Statistics.h>

namespace shark {


/// \brief Train a linear model to whiten the data
///
/// ZCA does whitening in the sense that it sets the mean to zero and the covariance to the Identity.
/// However in contrast to NormalizeComponentsWhitening it makes sure that the initial and end coordinate
/// system are the same and just rescales the data.  The effect is, that image data still resembles images
/// after applying ZCA in contrast to other methods which rotate the data randomly.
/// \ingroup unsupervised_trainer
class NormalizeComponentsZCA : public AbstractUnsupervisedTrainer<LinearModel<RealVector> >
{
	typedef AbstractUnsupervisedTrainer<LinearModel<RealVector> > base_type;
public:

	double m_targetVariance;
	NormalizeComponentsZCA(double targetVariance = 1.0){ 
		SHARK_RUNTIME_CHECK(targetVariance > 0.0, "Target variance must be positive");
		m_targetVariance = targetVariance;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "NormalizeComponentsZCA"; }

	void train(ModelType& model, UnlabeledData<RealVector> const& input){
		std::size_t dc = dataDimension(input);
		SHARK_RUNTIME_CHECK(input.numberOfElements() >= dc + 1, "Input needs to contain more points than there are input dimensions");

		// dense model with bias having input and output dimension equal to data dimension
		model.setStructure(dc, dc, true); 

		RealVector mean;
		RealMatrix covariance;
		meanvar(input, mean, covariance);
		
		blas::symm_eigenvalue_decomposition<RealMatrix> eigen(covariance);
		covariance=RealMatrix(); //covariance not needed anymore
		
		
		RealMatrix ZCAMatrix = eigen.Q() % to_diagonal(elem_inv(sqrt(eigen.D()))) % trans(eigen.Q());
		ZCAMatrix *= std::sqrt(m_targetVariance);

		RealVector offset = -prod(ZCAMatrix,mean);

		model.setStructure(ZCAMatrix, offset);
	}
};


}
#endif
