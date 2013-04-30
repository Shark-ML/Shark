/*
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
*/
#ifndef SHARK_UNSUPERVISED_RBM_DATAEVALUATOR_H
#define SHARK_UNSUPERVISED_RBM_DATAEVALUATOR_H

#include <shark/Unsupervised/RBM/Sampling/GibbsOperator.h>
namespace shark{
namespace detail{
///\brief function used by gradient approximators to calculate the gradient of the data
template<class AverageEnergyGradient,class VectorType,class RBM>
void evaluateData( AverageEnergyGradient& averageGradient, Data<VectorType> const& data, RBM& rbm ){
	typedef GibbsOperator<RBM> Operator;
	typedef Batch<typename Operator::HiddenSample> HiddenTraits;
	typedef Batch<typename Operator::VisibleSample> VisibleTraits;
	
	Operator dataEvaluator(&rbm);
	
	//calculate the expectation of the energy gradient with respect to the data
	BOOST_FOREACH(RealMatrix const& batch,data.batches()) {
		typename HiddenTraits::type hiddenSamples(batch.size1(),rbm.numberOfHN());
		typename VisibleTraits::type visibleSamples(batch.size1(),rbm.numberOfVN());
		
		dataEvaluator.createSample(hiddenSamples,visibleSamples,batch);
		averageGradient.addVH(hiddenSamples, visibleSamples);
	}
}

}
}

#endif
