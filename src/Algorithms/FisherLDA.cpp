//===========================================================================
/*!
 *
 *
 * \brief       FisherLDA
 *
 *
 *
 * \author      O.Krause
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
#include <shark/Algorithms/Trainers/FisherLDA.h>
using namespace shark;


FisherLDA::FisherLDA(bool whitening, std::size_t dimensions){
	m_whitening = whitening;
	m_subspaceDimensions = dimensions;
}

void FisherLDA::train(LinearModel<>& model, LabeledData<RealVector, unsigned int> const& dataset){
	SHARK_RUNTIME_CHECK(!dataset.empty(), "Dataset can not be empty.");

	std::size_t inputDim = inputDimension(dataset);
	std::size_t nComp = m_subspaceDimensions? m_subspaceDimensions : numberOfClasses(dataset);

	RealVector mean(inputDim);
	RealMatrix scatter(inputDim, inputDim);
	meanAndScatter(dataset, mean, scatter);
	
	blas::symm_eigenvalue_decomposition<RealMatrix> eigen(scatter);
	
	//reduce the size of the covariance matrix the the needed
	//subspace
	RealMatrix subspaceDirections = trans(columns(eigen.Q(),0,nComp));
	if (m_whitening){
		for(std::size_t i = 0; i != nComp; ++i){
			if(eigen.D()(i) <= 0) continue;
			row(subspaceDirections,i) /= std::sqrt(eigen.D()(i));
		}
	}
	RealVector offset = -prod(subspaceDirections, mean);

	// write the parameters into the model
	model.setStructure(subspaceDirections, offset);
}

void FisherLDA::meanAndScatter(
	LabeledData<RealVector, unsigned int> const& dataset,
	RealVector& mean,
	RealMatrix& scatter)
{

	std::size_t classes = numberOfClasses(dataset);
	std::size_t inputs = dataset.numberOfElements();
	std::size_t inputDim = inputDimension(dataset);


	// intermediate results
	std::vector<RealVector> means(classes, RealVector(inputDim,0.0));
	std::vector<RealMatrix> covariances(classes, RealMatrix(inputDim, inputDim,0.0));
	std::vector<std::size_t> counter(classes, 0);   // counter for examples per class

	// calculate mean and scatter for every class.

	// for every example in set ...
	for(auto point: dataset.elements()){
		//find class
		std::size_t c= point.label;

		// count example
		counter[c] += 1;

		// add example to mean vector
		noalias(means[c])+=point.input;
		// add example to scatter matrix
		noalias(covariances[c]) += outer_prod( point.input, point.input );
	}

	// for every class ...
	for( std::size_t c = 0; c != classes ; c++ ) {
		// normalize mean vector
		means[c] /= counter[c];

		// make scatter mean free
		noalias(covariances[c]) -= outer_prod(counter[c]*means[c],means[c]);
	}

	// calculate global mean and final scatter

	RealMatrix Sb( inputDim, inputDim,0.0 ); // between-class scatter
	RealMatrix Sw( inputDim, inputDim,0.0 ); // within-class scatter

	// calculate global mean
	mean.clear();
	for (std::size_t c = 0; c != classes; c++)
		noalias(mean) += counter[c] * means[c]/inputs;
	mean /= inputs;

	// calculate between- and within-class scatters
	for (std::size_t c = 0; c != classes; c++) {
		RealVector diff = means[c] - mean;
		noalias(Sb) += outer_prod(counter[c] * diff,diff);
		noalias(Sw) += covariances[c];
	}

	// invert Sw
	noalias(scatter) = solve(Sw,Sb, blas::symm_pos_def(), blas::left());
}
