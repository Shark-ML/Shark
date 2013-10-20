//===========================================================================
/*!
 *  \brief Collection of functions dealing with typical tasks of kernels
 *
 *
 *  \author  O. Krause
 *  \date    2007-2012
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

#ifndef SHARK_MODELS_KERNELS_KERNELHELPERS_H
#define SHARK_MODELS_KERNELS_KERNELHELPERS_H

#include <shark/Models/Kernels/AbstractKernelFunction.h>
#include <shark/Data/Dataset.h>
#include <shark/Core/OpenMP.h>
namespace shark{
	
///  \brief Calculates the regularized kernel gram matrix of the points stored inside a dataset.
///
///  Regularization is applied by adding the regularizer on the diagonal
///  \param kernel the kernel for which to calculate the kernel gram matrix
///  \param dataset the set of points used in the gram matrix
///  \param matrix the target kernel matrix
///  \param regularizer the regularizer of the matrix which is always >= 0. default is 0.
template<class InputType, class M>
void calculateRegularizedKernelMatrix(
	AbstractKernelFunction<InputType>const& kernel,
	Data<InputType> const& dataset,
	blas::matrix_expression<M>& matrix,
	double regularizer = 0
){
	SHARK_CHECK(regularizer >= 0, "regularizer must be >=0");
	std::size_t B = dataset.numberOfBatches();
	//get start of all batches in the matrix
	//also include  the past the end position at the end
	std::vector<std::size_t> batchStart(B+1,0);
	for(std::size_t i = 1; i != B+1; ++i){
		batchStart[i] = batchStart[i-1]+ boost::size(dataset.batch(i-1));
	}
	SIZE_CHECK(batchStart[B] == dataset.numberOfElements());
	std::size_t N  = batchStart[B];//number of elements
	ensureSize(matrix,N,N);
	
	
	for (std::size_t i=0; i<B; i++){
		std::size_t startX = batchStart[i];
		std::size_t endX = batchStart[i+1];
		SHARK_PARALLEL_FOR(int j=0; j < B; j++){
			std::size_t startY = batchStart[j];
			std::size_t endY = batchStart[j+1];
			RealMatrix submatrix = kernel(dataset.batch(i), dataset.batch(j));
			noalias(subrange(matrix(),startX,endX,startY,endY))=submatrix;
			//~ if(i != j)
				//~ noalias(subrange(matrix(),startY,endY,startX,endX))=trans(submatrix);
		}
		for(std::size_t k = startX; k != endX; ++k){
			matrix()(k,k) += static_cast<typename M::value_type>(regularizer);
		}
	}
	
}

///  \brief Calculates the regularized kernel gram matrix of the points stored inside a dataset.
///
///  Regularization is applied by adding the regularizer on the diagonal
///  \param kernel the kernel for which to calculate the kernel gram matrix
///  \param dataset the set of points used in the gram matrix
///  \param regularizer the regularizer of the matrix which is always >= 0. default is 0.
/// \return the kernel gram matrix
template<class InputType>
RealMatrix calculateRegularizedKernelMatrix(
	AbstractKernelFunction<InputType>const& kernel,
	Data<InputType> const& dataset, 
	double regularizer = 0
){
	SHARK_CHECK(regularizer >= 0, "regularizer must be >=0");
	RealMatrix M;
	calculateRegularizedKernelMatrix(kernel,dataset,M,regularizer);
	return M;
}


/// \brief Efficiently calculates the weighted derivative of a Kernel Gram Matrix w.r.t the Kernel Parameters
///
/// The formula is \f$  \sum_i \sum_j w_{ij} k(x_i,x_j)\f$ where w_ij are the weights of the gradient and x_i x_j are
/// the datapoints defining the gram matrix and k is the kernel. For efficiency it is assumd that w_ij = w_ji.
///This method is only useful when the whole Kernel Gram Matrix neds to be computed to get the weights w_ij and
///only computing smaller blocks is not sufficient. 
///  \param kernel the kernel for which to calculate the kernel gram matrix
///  \param dataset the set of points used in the gram matrix
///  \param weights the weights of the derivative, they must be symmetric!
///  \return the weighted derivative w.r.t the parameters.
template<class InputType,class WeightMatrix>
RealVector calculateKernelMatrixParameterDerivative(
		AbstractKernelFunction<InputType> const& kernel,
		Data<InputType> const& dataset, 
		WeightMatrix const& weights
){
	std::size_t kp = kernel.numberOfParameters();
	RealMatrix block;//stores the kernel results of the block which we need to compute to get the State :(
	RealVector kernelGradient(kp);//weighted gradient summed over the whole kernel matrix
	zero(kernelGradient);
	
	//calculate the gradint blockwise taking symmetry into account.
	RealVector blockGradient(kp);//weighted gradient summed over the whole block
	boost::shared_ptr<State> state = kernel.createState();
	std::size_t startX = 0;
	for (std::size_t i=0; i<dataset.numberOfBatches(); i++){
		std::size_t sizeX=shark::size(dataset.batch(i));
		std::size_t startY = 0;//start of the current batch in y-direction
		for (std::size_t j=0; j <= i; j++){
			std::size_t sizeY=shark::size(dataset.batch(j));
			kernel.eval(dataset.batch(i), dataset.batch(j),block,*state);
			kernel.weightedParameterDerivative(
				dataset.batch(i), dataset.batch(j),//points
				subrange(weights,startX,startX+sizeX,startY,startY+sizeY),//weights
				*state,
				blockGradient
			);
			if(i != j)
				kernelGradient+=2*blockGradient;//Symmetry!
			else
				kernelGradient+=blockGradient;//middle blocks are symmetric
			startY+= sizeY;
		}
		startX+=sizeX;
	}
	return kernelGradient;
}

}
#endif