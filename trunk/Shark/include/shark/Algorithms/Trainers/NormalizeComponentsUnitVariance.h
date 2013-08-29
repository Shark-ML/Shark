//===========================================================================
/*!
 *  \brief Data normalization to zero mean and unit variance
 *
 *
 *  \author T. Glasmachers
 *  \date 2010, 2013
 *
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


#ifndef SHARK_ALGORITHMS_TRAINERS_NORMALIZECOMPONENTSUNITVARIANCE_H
#define SHARK_ALGORITHMS_TRAINERS_NORMALIZECOMPONENTSUNITVARIANCE_H


#include <shark/Models/Normalizer.h>
#include <shark/Algorithms/Trainers/AbstractTrainer.h>
#include <shark/Data/Statistics.h>

namespace shark {


///
/// \brief Train a linear model to normalize the components of a dataset to unit variance, and optionally to zero mean.
///
/// \par
/// Normalizing the components of a dataset works via
/// training a Normalizer model. This model is then
/// applied to the dataset in order to perform the
/// normalization. The same model can be applied to
/// different datasets.
///
/// \par
/// The typical use case is that the Normalizer
/// model is trained on the training data. Later, as
/// "test" data comes in, the same model is used, of
/// course without being recalibrated. Thus, the model
/// used for normalization must be independent of the
/// dataset it was trained on.
///
/// \par
/// Note that subtracting the mean destroys sparsity.
/// Therefore this feature is turned off by default.
/// If you have non-sparse data and you need to
/// move data to zero mean, not only to unit variance,
/// then enable the flag zeroMean in the constructor.
///
template <class DataType = RealVector>
class NormalizeComponentsUnitVariance : public AbstractUnsupervisedTrainer< Normalizer<DataType> >
{
public:
	typedef AbstractUnsupervisedTrainer< Normalizer<DataType> > base_type;

	NormalizeComponentsUnitVariance(bool zeroMean = false)
	: m_zeroMean(zeroMean){ }

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "NormalizeComponentsUnitVariance"; }

	void train(Normalizer<DataType>& model, UnlabeledData<DataType> const& input)
	{
		SHARK_CHECK(input.numberOfElements() >= 2, "[NormalizeComponentsUnitVariance::train] input needs to consist of at least two points");
		std::size_t dc = dataDimension(input);

		RealVector mean;
		RealVector variance;
		meanvar(input, mean, variance);

		RealVector diagonal(dc);
		RealVector vector(dc);

		for (std::size_t d=0; d != dc; d++){
			double stddev = std::sqrt(variance(d));
			if (stddev == 0.0)
			{
				diagonal(d) = 0.0;
				vector(d) = 0.0;
			}
			else
			{
				diagonal(d) = 1.0 / stddev;
				vector(d) = -mean(d) / stddev;
			}
		}

		if (m_zeroMean) 
			model.setStructure(diagonal, vector);
		else 
			model.setStructure(diagonal);
	}

protected:
	bool m_zeroMean;
};


}
#endif
