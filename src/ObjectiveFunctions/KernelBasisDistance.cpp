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

void KernelBasisDistance::setupAndSolve(RealMatrix& beta, RealVector const& input, RealMatrix& Kz, RealMatrix& linear)const{
	//get access to the internal variables of the expansion
	Data<RealVector> const& expansionBasis = mep_expansion->basis();
	AbstractKernelFunction<RealVector> const& kernel = *mep_expansion->kernel();
	RealMatrix const& alpha = mep_expansion->alpha();
	std::size_t dim = dataDimension(expansionBasis);
	std::size_t outputs = mep_expansion->outputSize();
	

	//set up system of equations
	RealMatrix z = adapt_matrix(m_numApproximatingVectors,dim,&input(0));
	Kz = kernel(z,z);
	//construct the linear part = K_xz \alpha!
	//we do this batch wise for every batch in the basis of the kernel expansion
	linear.resize(m_numApproximatingVectors,outputs);
	linear.clear();
	std::size_t start = 0;
	for(std::size_t i = 0; i != expansionBasis.numberOfBatches(); ++i){
		RealMatrix const& batch = expansionBasis.batch(i);
		RealMatrix kernelBlock = kernel(z,batch);
		noalias(linear) += prod(kernelBlock,rows(alpha,start,start+batch.size1()));
		start +=batch.size1();
	}

	//solve for the optimal combination of kernel vectors beta
	beta = linear;
	solveSymmSemiDefiniteSystemInPlace<SolveAXB>(Kz,beta);
}

double KernelBasisDistance::errorOfSolution(RealMatrix const& beta, RealMatrix const& Kz, RealMatrix const& linear)const{
	RealMatrix kBeta = prod(Kz,beta);
	double error = 0;
	for(std::size_t i = 0; i != beta.size2(); ++i){
		error += 0.5*inner_prod(column(beta,i),column(kBeta,i));
		error -= inner_prod(column(linear,i),column(beta,i));
	}
	return error;
}

RealMatrix KernelBasisDistance::findOptimalBeta(RealVector const& input)const{
	RealMatrix Kz,beta,linear;
	setupAndSolve(beta,input,Kz,linear);
	return beta;
}

double KernelBasisDistance::eval(RealVector const& input) const{
	SIZE_CHECK(input.size() == numberOfVariables());
	
	RealMatrix Kz,beta,linear;
	setupAndSolve(beta,input,Kz,linear);
	return errorOfSolution(beta,Kz,linear);
}

KernelBasisDistance::ResultType KernelBasisDistance::evalDerivative( const SearchPointType & input, FirstOrderDerivative & derivative ) const {
	SIZE_CHECK(input.size() == numberOfVariables());

	//get access to the internal variables of the expansion
	Data<RealVector> const& expansionBasis = mep_expansion->basis();
	AbstractKernelFunction<RealVector> const& kernel = *mep_expansion->kernel();
	RealMatrix const& alpha = mep_expansion->alpha();
	std::size_t dim = dataDimension(expansionBasis);
	std::size_t outputs = mep_expansion->outputSize();
	RealMatrix basis = adapt_matrix(m_numApproximatingVectors,dim,&input(0));

	//set up system of equations and store the kernel states at the same time
	// (we assume here that everything fits into memory, which is the case as long as the number of
	// vectors to approximate is quite small)
	boost::shared_ptr<State> KzState = kernel.createState();
	RealMatrix Kz;
	kernel.eval(basis,basis,Kz,*KzState);
	//construct the linear part
	std::vector<boost::shared_ptr<State> > KzxState(expansionBasis.numberOfBatches());
	RealMatrix linear(m_numApproximatingVectors,outputs,0);
	std::size_t start = 0;
	for(std::size_t i = 0; i != expansionBasis.numberOfBatches(); ++i){
		RealMatrix const& batch = expansionBasis.batch(i);
		RealMatrix KzxBlock;
		KzxState[i] = kernel.createState();
		kernel.eval(basis,batch,KzxBlock,*KzxState[i]);
		noalias(linear) += prod(KzxBlock,rows(alpha,start,start+batch.size1()));
		start += batch.size1();
	}

	//solve for the optimal combination of kernel vectors beta
	RealMatrix beta = linear;
	solveSymmSemiDefiniteSystemInPlace<SolveAXB>(Kz,beta);

	//compute derivative
	// the derivative for z_l is given by
	// beta_l sum_i beta_i d/dz_l k(z_l,z_i)
	// - beta_l sum_j alpha_j d/dz_l k(z_l,x_i)
	derivative.resize(input.size());
	//compute first term by using that we can write beta_l * beta_i is an outer product
	//thus when using more than one output point it gets to a set of outer products which
	//can be written as product beta beta^T which are the weights of the derivative
	RealMatrix baseDerivative(m_numApproximatingVectors,dim);
	kernel.weightedInputDerivative(basis,basis,prod(beta,trans(beta)),*KzState,baseDerivative);
	noalias(derivative) = adapt_vector(input.size(), &baseDerivative(0,0));
	
	//compute the second term in the same way, taking the block structure into account.
	start = 0;
	for(std::size_t i = 0; i != expansionBasis.numberOfBatches(); ++i){
		RealMatrix const& batch = expansionBasis.batch(i);
		kernel.weightedInputDerivative(
			basis,batch,
			prod(beta,trans(rows(alpha,start,start+batch.size1()))),
			*KzxState[i],
			baseDerivative
		);
		noalias(derivative) -= adapt_vector(input.size(), &baseDerivative(0,0));
		start +=batch.size1();
	}
	
	return errorOfSolution(beta,Kz,linear);
}
