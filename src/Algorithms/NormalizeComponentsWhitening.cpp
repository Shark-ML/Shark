//===========================================================================
/*!
 * 
 *
 * \brief       Normalizes Components by Whitening
 * 
 * \author      T.Glasmachers, Christian Igel
 * \date        2010-2011
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
#define SHARK_COMPILE_DLL
#include <shark/Algorithms/Trainers/NormalizeComponentsWhitening.h>
#include <shark/Data/Statistics.h>


using namespace shark;
using namespace blas;

NormalizeComponentsWhitening::NormalizeComponentsWhitening(double targetVariance){ 
	SHARK_RUNTIME_CHECK(targetVariance > 0.0, "Target variance must be positive");
	m_targetVariance = targetVariance;
}

/// \brief From INameable: return the class name.
std::string NormalizeComponentsWhitening::name() const
{ return "NormalizeComponentsWhitening"; }

void NormalizeComponentsWhitening::train(ModelType& model, UnlabeledData<RealVector> const& input){
	std::size_t dc = dataDimension(input);
	SHARK_RUNTIME_CHECK(input.numberOfElements() >= dc + 1, "Input needs to contain more points than there are input dimensions");
	SHARK_RUNTIME_CHECK(m_targetVariance > 0.0, "Target variance must be positive");

	RealVector mean;
	RealMatrix covariance;
	meanvar(input, mean, covariance);
	
	//compute the whitening factor taking into account that
	//it might not be full rank.
	symm_pos_semi_definite_solver<RealMatrix> solver(covariance);
	RealMatrix whiteningMatrix(solver.rank(),dc);
	solver.compute_inverse_factor(whiteningMatrix);
	whiteningMatrix *= std::sqrt(m_targetVariance);

	RealVector offset = -prod(whiteningMatrix,mean);
	model.setStructure(whiteningMatrix, offset);
}
