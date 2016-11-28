//===========================================================================
/*!
 *
 *
 * \brief       General functions for Tree modeling.
 *
 *
 *
 * \author      K. N. Hansen, J. Kremer, J. Wrigley
 * \date        2011-2016
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
#include <shark/Algorithms/Trainers/CARTcommon.h>
#include <shark/Core/Math.h>
#include <shark/Algorithms/Trainers/RFTrainer.h>

#include <limits>

namespace shark {
namespace detail{
namespace cart {


/**
 * Calculates the Gini impurity of a node. The impurity is defined as
 * gini = 1-sum_j p(j|t)^2
 * i.e the 1 minus the sum of the squared probability of observing class j in node t
 */
double gini(ClassVector const& countVector, std::size_t n)
{
	if(!n) return 1.;
	return 1.-sum(sqr(countVector/static_cast<double>(n)));
}

/**
 * Calculates the misclassification error of a node.
 * ME = 1-max(count)/n
 */
double misclassificationError(ClassVector const& countVector, std::size_t n)
{
	if(!n) return 1.;
	double m = max(countVector);
	return 1.-m/n;
}

/**
 * Calculates the cross-entropy of a node.
 * CE = - sum_j(count[j]/n * log(count[j]/n))
 */
double crossEntropy(ClassVector const& countVector, std::size_t n)
{
	if(!n) return std::numeric_limits<double>::infinity();

	// eliminate zero counts
	ClassVector c(countVector);
	auto pivot = std::partition(c.begin(),c.end(),[&](unsigned value){return value!=0;});
	c.resize(pivot.index());

	RealVector p (c/double(n));
	return -sum(p*log(p));
}

/**
 * Create a count vector as used in the classification case.
 */
ClassVector createCountVector(
		DataView<ClassificationDataset const> const& elements,
		std::size_t labelCardinality)
{
	ClassVector countVector(labelCardinality);
	for(auto const& element: elements){
		++countVector[element.label];
	}
	return countVector;
}
ClassVector createCountVector(
		ClassificationDataset const& dataset,
		std::size_t labelCardinality)
{
	ClassVector countVector(labelCardinality);
	for(auto const& element: dataset.elements()){
		++countVector[element.label];
	}
	return countVector;
}

ImpurityMeasureFn setImpurityFn(ImpurityMeasure im){
	switch(im) {
		case ImpurityMeasure::gini: return gini;
		case ImpurityMeasure::misclassification: return misclassificationError;
		case ImpurityMeasure::crossEntropy: return crossEntropy;
	}
    return nullptr;
}
}}} // namespace shark::detail::cart
