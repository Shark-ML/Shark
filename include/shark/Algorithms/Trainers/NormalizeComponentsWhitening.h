//===========================================================================
/*!
 * 
 *
 * \brief       Data normalization to zero mean, unit variance and zero covariance 
 * 
 * 
 * 
 *
 * \author      T. Glasmachers,O.Krause
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
//===========================================================================


#ifndef SHARK_ALGORITHMS_TRAINERS_NORMALIZECOMPONENTSWHITENING_H
#define SHARK_ALGORITHMS_TRAINERS_NORMALIZECOMPONENTSWHITENING_H

#include <shark/Core/DLLSupport.h>
#include <shark/Models/LinearModel.h>
#include <shark/Algorithms/Trainers/AbstractTrainer.h>

namespace shark {


/// \brief Train a linear model to whiten the data.
///
/// computes a linear model that normlizes the data to be 0 mean, a given target variance and covariance 0.
/// By default the trainer makes the data unit variance, but the target variance can be changed as well.
/// \ingroup unsupervised_trainer
class NormalizeComponentsWhitening : public AbstractUnsupervisedTrainer<LinearModel<RealVector> >
{
public:
	SHARK_EXPORT_SYMBOL NormalizeComponentsWhitening(double targetVariance = 1.0);

	/// \brief From INameable: return the class name.
	SHARK_EXPORT_SYMBOL std::string name() const;

	SHARK_EXPORT_SYMBOL void train(ModelType& model, UnlabeledData<RealVector> const& input);

private:
	double m_targetVariance;
};


}
#endif
