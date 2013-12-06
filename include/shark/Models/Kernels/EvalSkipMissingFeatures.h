//===========================================================================
/*!
 *  \brief Do special kernel evaluation by skipping missing features
 *
 *  \author  B. Li
 *  \date    2012
 *
 *  \par Copyright (c) 2012:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
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
#ifndef SHARK_MODELS_KERNELS_EVAL_SKIP_MISSING_FEATURES_H
#define SHARK_MODELS_KERNELS_EVAL_SKIP_MISSING_FEATURES_H

#include "shark/Core/Exception.h"
#include "shark/LinAlg/Base.h"
#include "shark/Models/Kernels/AbstractKernelFunction.h"
#include "shark/Models/Kernels/LinearKernel.h"
#include "shark/Models/Kernels/MonomialKernel.h"
#include "shark/Models/Kernels/PolynomialKernel.h"

#include <boost/optional.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <vector>

namespace shark {

/// Does a kernel function evaluation with Missing features in the inputs
/// @param kernelFunction The kernel function used to do evaluation
/// @param inputA a input
/// @param inputB another input
///
/// The kernel k(x,y) is evaluated taking missing features into account. For this it is checked whether a feature
/// of x or y is nan and in this case the corresponding features in @a inputA and @a inputB won't be considered.
template <typename InputType,typename InputTypeT1,typename InputTypeT2>
double evalSkipMissingFeatures(
	const AbstractKernelFunction<InputType>& kernelFunction,
	const InputTypeT1& inputA,
	const InputTypeT2& inputB)
{
	SIZE_CHECK(inputA.size() == inputB.size());
	// Do kernel type check
	if (!kernelFunction.supportsVariableInputSize())
		throw SHARKEXCEPTION("[evalSkipMissingFeatures] Kernel must support variable input size.");
	
	// Work out features that are valid for both dataset i and j, and also should not be filtered out by missingness
	// Because we won't exact length of valid features beforehand, so we choose to construct two vectors and then
	// construct another two InputTypes with them.
	typedef typename InputType::value_type InputValueType;
	std::vector<InputValueType> tempInputA;
	std::vector<InputValueType> tempInputB;
	tempInputA.reserve(inputA.size());
	tempInputB.reserve(inputB.size());
	for (std::size_t index = 0; index < inputA.size(); ++index)
	{
		//using namespace boost::math;
		if (!boost::math::isnan(inputA(index)) && !boost::math::isnan(inputB(index)))
		{
			tempInputA.push_back(inputA(index));
			tempInputB.push_back(inputB(index));
		}
	}

	SIZE_CHECK(tempInputA.size() == tempInputB.size());
	SIZE_CHECK(tempInputA.size() > 0);
	InputType validInputA(tempInputA.size());
	InputType validInputB(tempInputA.size());
	std::copy(tempInputA.begin(),tempInputA.end(),validInputA.begin());
	std::copy(tempInputB.begin(),tempInputB.end(),validInputB.begin());

	// And then pass them to the kernel for calculation
	return kernelFunction.eval(validInputA, validInputB);
}

/// Do kernel function evaluation while Missing features in the inputs
/// @param kernelFunction The kernel function used to do evaluation
/// @param inputA a input
/// @param inputB another input
/// @param missingness
///     used to decide which features in the inputs to take into consideration for the purpose of evaluation.
///     If a feature is NaN, then the corresponding features in @a inputA and @a inputB won't be considered.
template <typename InputType,typename InputTypeT1,typename InputTypeT2,typename InputTypeT3>
double evalSkipMissingFeatures(
	const AbstractKernelFunction<InputType>& kernelFunction,
	const InputTypeT1& inputA,
	const InputTypeT2& inputB,
	InputTypeT3 const& missingness)
{
	SIZE_CHECK(inputA.size() == inputB.size());
	//SIZE_CHECK(inputA.size() == missingness.size());
	// Do kernel type check
	if (!kernelFunction.supportsVariableInputSize())
		throw SHARKEXCEPTION("[evalSkipMissingFeatures] Kernel must support variable input size.");

	

	// Work out features that are valid for both dataset i and j, and also should not be filtered out by missingness
	// Because we won't exact length of valid features beforehand, so we choose to construct two vectors and then
	// construct another two InputTypes with them.
	typedef typename InputType::value_type InputValueType;
	std::vector<InputValueType> tempInputA;
	std::vector<InputValueType> tempInputB;
	tempInputA.resize(inputA.size());
	tempInputB.resize(inputB.size());
	for (std::size_t index = 0; index < inputA.size(); ++index)
	{
		using namespace boost::math;
		if (!std::isnan(inputA(index)) && !std::isnan(inputB(index)) && !std::isnan(missingness(index)))
		{
			tempInputA.push_back(inputA(index));
			tempInputB.push_back(inputB(index));
		}
	}

	SIZE_CHECK(tempInputA.size() == tempInputB.size());
	SIZE_CHECK(tempInputA.size() > 0);
	InputType validInputA(tempInputA.size());
	InputType validInputB(tempInputA.size());
	for (std::size_t i = 0; i < tempInputA.size(); ++i)
	{
		validInputA(i) = tempInputA[i];
		validInputB(i) = tempInputB[i];
	}

	// And then pass them to the kernel for calculation
	return kernelFunction.eval(validInputA, validInputB);
}

} // namespace shark {

#endif // SHARK_MODELS_KERNELS_EVAL_SKIP_MISSING_FEATURES_H
