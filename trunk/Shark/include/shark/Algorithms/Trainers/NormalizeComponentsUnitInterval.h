//===========================================================================
/*!
 * 
 *
 * \brief       Data normalization to the unit interval
 * 
 * 
 * 
 *
 * \author      T. Glasmachers
 * \date        2010, 2013
 *
 *
 * \par Copyright 1995-2015 Shark Development Team
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


#ifndef SHARK_ALGORITHMS_TRAINERS_NORMALIZECOMPONENTSUNITINTERVAL_H
#define SHARK_ALGORITHMS_TRAINERS_NORMALIZECOMPONENTSUNITINTERVAL_H


#include <shark/Models/Normalizer.h>
#include <shark/Algorithms/Trainers/AbstractTrainer.h>

namespace shark{


///
/// \brief Train a model to normalize the components of a dataset to fit into the unit inverval
///
/// \par
/// Normalizing the components of a dataset works via
/// training a LinearMap model. This model is then
/// applied to the dataset in order to perform the
/// normalization. The same model can be applied to
/// different datasets.
///
/// \par
/// The typical use case is that the AffineLinearMap
/// model is trained on the training data. Later, as
/// "test" data comes in, the same model is used, of
/// course without being recalibrated. Thus, the model
/// used for normalization must be independent of the
/// dataset it was trained on.
///
/// \par
/// Note that the transformation represented by this
/// trainer destroys sparsity of the data. Therefore
/// one may prefer NormalizeComponentsUnitVariance
/// particularly on sparse data.
///
template <class DataType = RealVector>
class NormalizeComponentsUnitInterval : public AbstractUnsupervisedTrainer< Normalizer<DataType> >
{
public:
	typedef AbstractUnsupervisedTrainer< Normalizer<DataType> > base_type;

	NormalizeComponentsUnitInterval()
	{ }

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "NormalizeComponentsUnitInterval"; }

	void train(Normalizer<DataType>& model, UnlabeledData<DataType> const& input)
	{
		//SHARK_CHECK(model.hasOffset(), "[NormalizeComponentsUnitInterval::train] model must have an offset term");
		std:: size_t ic = input.numberOfElements();
		SHARK_CHECK(ic >= 2, "[NormalizeComponentsUnitInterval::train] input needs to consist of at least two points");
		std::size_t dc = dataDimension(input);

		RealVector min = input.element(0);
		RealVector max = input.element(0);
		for(std::size_t i=1; i != ic; i++){
			for(std::size_t d = 0; d != dc; d++){
				double x = input.element(i)(d);
				min(d) = std::min(min(d), x);
				max(d) = std::max(max(d), x);
			}
		}

		RealVector diagonal(dc);
		RealVector offset(dc);

		for (std::size_t d=0; d != dc; d++)
		{
			if (min(d) == max(d))
			{
				diagonal(d) = 0.0;
				offset(d) = -min(d) + 0.5;
			}
			else
			{
				double n = 1.0 / (max(d) - min(d));
				diagonal(d) = n;
				offset(d) = -min(d) * n;
			}
		}

		model.setStructure(diagonal, offset);
	}
};


}
#endif
