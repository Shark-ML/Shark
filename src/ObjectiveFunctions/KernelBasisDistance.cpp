/*!
 * 
 *
 * \brief      -
 * \author    O.Krause
 * \date        2014
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

#include <shark/ObjectiveFunctions/KernelBasisDistance.h>
#include <shark/LinAlg/solveSystem.h>
#include <shark/Rng/GlobalRng.h>

using namespace shark;
using namespace blas;

KernelBasisDistance::KernelBasisDistance(
	KernelExpansion<RealVector>* kernelExpansion,
	std::size_t numApproximatingVectors
):mep_expansion(kernelExpansion),m_numApproximatingVectors(numApproximatingVectors){
	SHARK_CHECK(kernelExpansion != NULL, "[KernelBasisDistance] kernelExpansion must not be NULL");
	SHARK_CHECK(kernelExpansion->kernel() != NULL, "[KernelBasisDistance] kernelExpansion must have a kernel");
	mep_expansion -> sparsify(); //purge all non-support vectors

	m_features|=HAS_VALUE;
	m_features|=CAN_PROPOSE_STARTING_POINT;
	
	if(mep_expansion->kernel() -> hasFirstInputDerivative())
		m_features|=HAS_FIRST_DERIVATIVE;
}

void KernelBasisDistance::proposeStartingPoint(SearchPointType& startingPoint) const {
	Data<RealVector> const& expansionBasis = mep_expansion->basis();
	std::size_t dim = dataDimension(expansionBasis);
	std::size_t elems = mep_expansion->alpha().size1();
	startingPoint.resize(m_numApproximatingVectors * dim);
	for(std::size_t i = 0; i != m_numApproximatingVectors; ++i){
		std::size_t k = Rng::discrete(0,elems-1);
		noalias(subrange(startingPoint,i*dim,(i+1)*dim)) = expansionBasis.element(k);
	}
}

std::size_t KernelBasisDistance::numberOfVariables()const{
	return m_numApproximatingVectors  * dataDimension(mep_expansion->basis());
}

double KernelBasisDistance::eval(RealVector const& input) const{
	SIZE_CHECK(input.size() == numberOfVariables());
	
	//get access to the internal variables of the expansion
	Data<RealVector> const& expansionBasis = mep_expansion->basis();
	AbstractKernelFunction<RealVector> const& kernel = *mep_expansion->kernel();
	RealMatrix const& alpha= mep_expansion -> alpha();
	std::size_t dim = dataDimension(expansionBasis);
	std::size_t outputs = mep_expansion->outputSize();
	
	
	//set up system of equations
	RealMatrix basis = adapt_matrix(m_numApproximatingVectors,dim,&input(0));
	RealMatrix kBasis = kernel(basis,basis);
	//construct the linear part = K_xz \alpha!
	//we do this batch wise for every batch in the basis of the kernel expansion
	RealMatrix linear(m_numApproximatingVectors,outputs,0);
	std::size_t start = 0;
	for(std::size_t i = 0; i != expansionBasis.numberOfBatches(); ++i){
		RealMatrix const& batch = expansionBasis.batch(i);
		RealMatrix kernelBlock = kernel(basis,batch);
		noalias(linear) +=prod(kernelBlock,rows(alpha,start,start+batch.size1()));
		start +=batch.size1();
	}
	
	//solve for the optimal combination of kernel vectors beta
	RealMatrix beta = linear;
	solveSymmSemiDefiniteSystemInPlace<SolveAXB>(kBasis,beta);
	
	//return error of optimal beta
	RealMatrix kBeta = prod(kBasis,beta);
	double error = 0;
	for(std::size_t i = 0; i != outputs; ++i){
		error += 0.5*inner_prod(column(beta,i),column(kBeta,i));
		error -= inner_prod(column(linear,i),column(beta,i));
	}
	return error;
}

KernelBasisDistance::ResultType KernelBasisDistance::evalDerivative( const SearchPointType & input, FirstOrderDerivative & derivative ) const {
	SIZE_CHECK(input.size() == numberOfVariables());
	
	//get access to the internal variables of the expansion
	Data<RealVector> const& expansionBasis = mep_expansion->basis();
	AbstractKernelFunction<RealVector> const& kernel = *mep_expansion->kernel();
	RealMatrix const& alpha= mep_expansion -> alpha();
	std::size_t dim = dataDimension(expansionBasis);
	std::size_t outputs = mep_expansion->outputSize();
	RealMatrix basis = adapt_matrix(m_numApproximatingVectors,dim,&input(0));
	
	//set up system of equations and store the kernel states at the same time
	// (we assume here thyt everything fits into memory, which is the case as long as the number of
	// vectors to approximate is quite small)
	boost::shared_ptr<State> kBasisState = kernel.createState();
	RealMatrix kBasis;
	kernel.eval(basis,basis,kBasis,*kBasisState);
	//construct the linear part!
	std::vector<boost::shared_ptr<State> > linearState(expansionBasis.numberOfBatches());
	RealMatrix linear(m_numApproximatingVectors,outputs,0);
	std::size_t start = 0;
	for(std::size_t i = 0; i != expansionBasis.numberOfBatches(); ++i){
		RealMatrix const& batch = expansionBasis.batch(i);
		RealMatrix kernelBlock;
		linearState[i] = kernel.createState();
		kernel.eval(basis,batch,kernelBlock,*linearState[i]);
		noalias(linear) +=prod(kernelBlock,rows(alpha,start,start+batch.size1()));
		start +=batch.size1();
	}
	
	//solve for the optimal combination of kernel vectors beta
	RealMatrix beta= linear;
	solveSymmSemiDefiniteSystemInPlace<SolveAXB>(kBasis,beta);
	
	//compute derivative
	derivative.resize(input.size());
	RealMatrix baseDerivative(m_numApproximatingVectors,dim);
	kernel.weightedInputDerivative(basis,basis,prod(beta,trans(beta)),*kBasisState,baseDerivative);
	noalias(derivative) = adapt_vector(input.size(), &baseDerivative(0,0));
	start = 0;
	for(std::size_t i = 0; i != expansionBasis.numberOfBatches(); ++i){
		RealMatrix const& batch = expansionBasis.batch(i);
		kernel.weightedInputDerivative(
			basis,batch,
			prod(beta,trans(rows(alpha,start,start+batch.size1()))),
			*linearState[i],
			baseDerivative
		);
		noalias(derivative) -= adapt_vector(input.size(), &baseDerivative(0,0));
		start +=batch.size1();
	}
	
	//return error of optimal beta
	RealMatrix kBeta = prod(kBasis,beta);
	double error = 0;
	for(std::size_t i = 0; i != outputs; ++i){
		error += 0.5*inner_prod(column(beta,i),column(kBeta,i));
		error -= inner_prod(column(linear,i),column(beta,i));
	}
	return error;
	
	
}