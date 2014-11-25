//===========================================================================
/*!
 * 
 *
 * \brief       Data normalization to zero mean, unit variance and zero covariance 
 * 
 * 
 * 
 *
 * \author      T. Glasmachers
 * \date        2010
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


#ifndef SHARK_ALGORITHMS_TRAINERS_NORMALIZECOMPONENTSWHITENING_H
#define SHARK_ALGORITHMS_TRAINERS_NORMALIZECOMPONENTSWHITENING_H


#include <shark/Models/LinearModel.h>
#include <shark/Algorithms/Trainers/AbstractTrainer.h>
#include <shark/Data/Statistics.h>
#include <shark/LinAlg/solveSystem.h>

namespace shark {


/// \brief Train a linear model to whiten the data
class NormalizeComponentsWhitening : public AbstractUnsupervisedTrainer<LinearModel<RealVector> >
{
	typedef AbstractUnsupervisedTrainer<LinearModel<RealVector> > base_type;
public:

	double m_targetVariance;
	NormalizeComponentsWhitening(double targetVariance = 1.0){ 
		SHARK_CHECK(targetVariance > 0.0, "[NormalizeComponentsWhitening::NormalizeComponentsWhitening] target variance must be positive");
		m_targetVariance = targetVariance;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "NormalizeComponentsWhitening"; }

	void train(ModelType& model, UnlabeledData<RealVector> const& input){
		std::size_t dc = dataDimension(input);
		SHARK_CHECK(input.numberOfElements() >= dc + 1, "[NormalizeComponentsWhitening::train] input needs to contain more points than there are input dimensions");
		SHARK_CHECK(m_targetVariance > 0.0, "[NormalizeComponentsWhitening::train] target variance must be positive");

		// dense model with bias having input and output dimension equal to data dimension
		model.setStructure(dc, dc, true); 

		RealVector mean;
		RealMatrix covariance;
		meanvar(input, mean, covariance);

		RealMatrix whiteningMatrix = createWhiteningMatrix(covariance);
		whiteningMatrix *= std::sqrt(m_targetVariance);

		RealVector offset(dc);
		axpy_prod(trans(whiteningMatrix),mean,offset);
		offset *= -1.0;

		model.setStructure(RealMatrix(trans(whiteningMatrix)), offset);
	}

private:
	RealMatrix createWhiteningMatrix(
		RealMatrix& covariance
	){
		using namespace blas;

		SIZE_CHECK(covariance.size1() == covariance.size2());
		std::size_t m = covariance.size1();
		//we use the inversed cholesky decomposition for whitening
		//since we have to assume that covariance does not have full rank, we use
		//the generalized decomposition
		RealMatrix whiteningMatrix(m,m,0.0);

		//do a pivoting cholesky decomposition
		//this destroys the covariance matrix as it is not neeeded anymore afterwards.
		PermutationMatrix permutation(m);
		std::size_t rank = pivotingCholeskyDecompositionInPlace(covariance,permutation);
		//only take the nonzero columns as C
		matrix_range<RealMatrix> C = columns(covariance,0,rank);

		//full rank, means that we can use the typical cholesky inverse with pivoting
		//so U is P C^-1 P^T
		if(rank == m){
			noalias(whiteningMatrix) = blas::identity_matrix<double>( m );
			solveTriangularSystemInPlace<SolveXAB,upper>(trans(C),whiteningMatrix);
			swap_full_inverted(permutation,whiteningMatrix);
			return whiteningMatrix;
		}
		//complex case. 
		//A' = P C(C^TC)^-1(C^TC)^-1 C^T P^T
		//=> P^T U P = C(C^TC)^-1
		//<=> P^T U P (C^TC) = C
		RealMatrix CTC(rank,rank);
		symm_prod(trans(C),CTC);

		matrix_range<RealMatrix> submat = columns(whiteningMatrix,0,rank);
		solveSymmSystem<SolveXAB>(CTC,submat,C);
		swap_full_inverted(permutation,whiteningMatrix);

		return whiteningMatrix;
	}
};


}
#endif
