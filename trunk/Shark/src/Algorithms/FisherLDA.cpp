//===========================================================================
/*!
 *  \file FisherLDA.cpp
 *
 *  \brief FisherLDA
 *
 *  \author O.Krause
 *  \date 2010-2011
 *
 *  \par Copyright (c) 1998-2007:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
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
#include <shark/Algorithms/Trainers/FisherLDA.h>
#include <shark/LinAlg/eigenvalues.h>
#include <shark/LinAlg/Inverse.h>
using namespace shark;


FisherLDA::FisherLDA(bool whitening){ 
	m_whitening = whitening; 
}

void FisherLDA::train(LinearModel<>& model, LabeledData<RealVector, unsigned int> const& dataset){
	SHARK_CHECK(! dataset.empty(), "[FisherLDA::train] dataset is empty");

	std::size_t inputDim = model.inputSize();
	std::size_t nComp = model.outputSize();		// for dimensionality reduction

	RealVector mean(inputDim);
	RealMatrix scatter(inputDim, inputDim);
	meanAndScatter(dataset, mean, scatter);

	RealMatrix eigenvectors(inputDim, inputDim);
	RealVector eigenvalues(inputDim);
	if (m_whitening){
		RealMatrix h(inputDim, inputDim);
		rankDecomp(scatter, eigenvectors, h, eigenvalues);
	}
	else{
		eigensymm(scatter, eigenvectors, eigenvalues);
	}
	//reduce the size of the covariance matrix the the needed
	//subspace
	eigenvectors = blas::subrange(eigenvectors,0,inputDim,0,nComp);
	RealVector offset = -prod(trans(eigenvectors), mean);

	// write the parameters into the model
	model.setStructure(RealMatrix(trans(eigenvectors)), offset);
}

void FisherLDA::meanAndScatter(
	LabeledData<RealVector, unsigned int> const& dataset,
	RealVector& mean,
	RealMatrix& scatter)
{
	typedef LabeledData<RealVector,unsigned int>::const_element_iterator Iterator;
	std::size_t classes = numberOfClasses(dataset);
	std::size_t inputDim = inputDimension(dataset);

	// intermediate results
	std::vector<RealVector> means(classes, RealVector(inputDim));
	std::vector<RealMatrix> covariances(classes, RealMatrix(inputDim, inputDim));
	std::vector<std::size_t> counter(classes, 0);   // counter for examples per class

	// initialize vectors
	for (std::size_t i=0; i != classes; ++i) {
		means[i].clear();
		covariances[i].clear();
	}

	// calculate mean and scatter for every class.

	// for every example in set ...
	std::size_t inputs = dataset.numberOfElements();

	Iterator end = dataset.elemEnd();
	for(Iterator point = dataset.elemBegin(); point != end; ++point ) {
		//find class
		std::size_t c= point->label;

		// count example
		counter[c] += 1;

		// add example to mean vector
		noalias(means[c])+=point->input;
		// add example to scatter matrix
		noalias(covariances[c]) += outer_prod( point->input, point->input );
	}
	std::cout<<"end"<<std::endl;
	// for every class ...
	for( unsigned int c = 0; c != classes ; c++ ) {
		// normalize mean vector
		means[c] /= counter[c];

		// make scatter mean free
		noalias(covariances[c]) -= outer_prod(counter[c]*means[c],means[c]);
	}

	// calculate global mean and final scatter

	RealMatrix Sb( inputDim, inputDim ); // between-class scatter
	RealMatrix Sw( inputDim, inputDim ); // within-class scatter

	Sb.clear();
	Sw.clear();

	// calculate global mean
	mean.clear();
	for (unsigned int c = 0; c < classes; c++) 
		mean += counter[c] * means[c]/inputs;
	mean /= inputs;

	// calculate between- and within-class scatters
	for (unsigned int c = 0; c < classes; c++) {
		RealVector diff = means[c] - mean;
		Sb += outer_prod(counter[c] * diff,diff);
		Sw += covariances[c];
	}

	// invert Sw
	RealMatrix SwInv = invert( Sw );
	fast_prod(SwInv,Sb,scatter);
}
