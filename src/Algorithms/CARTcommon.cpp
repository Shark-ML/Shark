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

namespace shark {
namespace detail{
namespace cart {

RealVector hist(ClassVector const& countVector)
{

	RealVector histogram(countVector.size(),0.0);

	std::size_t totalElements = 0;

	for (std::size_t i = 0, s = countVector.size(); i<s; ++i){
		histogram(i) = countVector[i];
		totalElements += countVector[i];
	}
	histogram /= totalElements;

	return histogram;
}

///Calculates the Gini impurity of a node. The impurity is defined as
///1-sum_j p(j|t)^2
///i.e the 1 minus the sum of the squared probability of observing class j in node t
double gini(ClassVector const& countVector, std::size_t n)
{
	if(!n) return 1;

	double res = 0.;
	n *= n;
	for(auto const& i: countVector){
		res += sqr(i)/ static_cast<double>(n);
	}
	return 1-res;
}

// ME = 1-max(count)/n
double misclassificationError(ClassVector const& countVector, std::size_t n)
{
	if(!n) return 1.;
	auto m = countVector[0];
	for(auto c: countVector){
		m = std::max(m,c);
	}
	return 1-m/n;
}

// CE = - sum_j(count[j] log(count[j]/n))/n
double crossEntropy(ClassVector const& countVector, std::size_t n)
{
	return - sum<double>(0,n,[&](std::size_t i){
		if(!countVector[i]) return 0.;
		return countVector[i]*std::log(countVector[i]/n);
	})/n;
}


/// Create a count matrix as used in the classification case.
ClassVector createCountVector(
		DataView<ClassificationDataset const> const& elements,
		std::size_t labelCardinality)
{
	ClassVector countVector = ClassVector(labelCardinality);
	for(std::size_t i = 0, s = elements.size(); i<s; ++i){
		++countVector[elements[i].label];
	}
	return countVector;
}



}}} // namespace shark::detail::cart
