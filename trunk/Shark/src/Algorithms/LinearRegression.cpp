//===========================================================================
/*!
 * 
 *
 * \brief       LinearRegression
 * 
 * 
 *
 * \author      O.Krause, T. Glasmachers
 * \date        2010-2011
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
#include <shark/LinAlg/solveSystem.h>
#include <shark/Algorithms/Trainers/LinearRegression.h>

using namespace shark;


LinearRegression::LinearRegression(double regularization){
	setRegularization(regularization);
}

void LinearRegression::train(LinearModel<>& model, LabeledData<RealVector, RealVector> const& dataset){
	std::size_t inputDim = inputDimension(dataset);
	std::size_t outputDim = labelDimension(dataset);
	std::size_t numInputs = dataset.numberOfElements();
	std::size_t numBatches = dataset.numberOfBatches();

	//Let P be the matrix of points with n rows and X=(P|1). the 1 rpresents the bias weight
	//Let A = X^T X + lambda * I
	//than whe have (for lambda = 0)
	//A = ( P^T P  P^T 1)
	//       ( 1^T P  1^T1)
	RealMatrix matA(inputDim+1,inputDim+1,0.0);
	blas::Blocking<RealMatrix> Ablocks(matA,inputDim,inputDim);
	//compute A and the label matrix batchwise
	typedef LabeledData<RealVector, RealVector>::const_batch_reference BatchRef;
	for (std::size_t b=0; b != numBatches; b++){
		BatchRef batch = dataset.batch(b);
		symm_prod(trans(batch.input),Ablocks.upperLeft(),false);
		noalias(column(Ablocks.upperRight(),0))+=sum_rows(batch.input);
	}
	row(Ablocks.lowerLeft(),0) = column(Ablocks.upperRight(),0);
	matA(inputDim,inputDim) = numInputs;
	//X^TX+=lambda* I
	diag(Ablocks.upperLeft())+= blas::repeat(m_regularization,inputDim);
	
	
	//we also need to compute X^T L= (P^TL, 1^T L) where L is the matrix of labels 
	RealMatrix XTL(inputDim + 1,outputDim,0.0);
	for (std::size_t b=0; b != numBatches; b++){
		BatchRef batch = dataset.batch(b);
		RealSubMatrix PTL = subrange(XTL,0,inputDim,0,outputDim);
		axpy_prod(trans(batch.input),batch.label,PTL,false);
		noalias(row(XTL,inputDim))+=sum_rows(batch.label);
	}	
	
	//we solve the system A Beta = X^T L
	//usually this is solved via the moore penrose inverse:
	//Beta = A^-1 T
	//but it is faster und numerically more stable, if we solve it as a symmetric system
	//w can use in-place solve
	RealMatrix&  beta = XTL;
	blas::solveSymmSemiDefiniteSystemInPlace<blas::SolveAXB>(matA,beta);
	
	RealMatrix matrix = subrange(trans(beta), 0, outputDim, 0, inputDim);
	RealVector offset = row(beta,inputDim);
	
	// write parameters into the model
	model.setStructure(matrix, offset);
}
