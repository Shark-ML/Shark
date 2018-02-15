//===========================================================================
/*!
 * 
 *
 * \brief       Implements the approximation of a kernel expansion
 * 
 * 
 *
 * \author      O.Krause
 * \date        2017
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

#include <shark/Algorithms/ApproximateKernelExpansion.h>
#include <shark/Algorithms/KMeans.h>
#include <shark/Algorithms/GradientDescent/LBFGS.h>

using namespace shark;
using namespace blas;

namespace{
/// \brief Computes the squared distance between the optimal point in a basis to the point represented by a KernelExpansion.
///
/// Assume we are given a kernel expansion \f$ w = \sum_i \alpha_i k(x_i, \cdot) \f$. The KernelBasisDistance takes
/// a new set of basis vectors \f$ z_i \f$ and finds the linear combination in that space which is closest
/// to \f$ w \f$ . More formally the function measures the squared distance in the kernel-induced feature space:
/// \f[ f(z) = \min_{\beta} \frac 1 2 \| \sum_j \beta_j k(z_j, \cdot) - w \|^2 . \f]
/// In vector notation with \f$ (K_x)_{i,j} = k(x_i,x_j) \f$, \f$ (K_z)_{i,j} = k(z_i,z_j) \f$ and \f$ (K_{zx})_{i,j} = k(z_i,x_j) \f$ it computes:
///\f[ f(z) = \min_{\beta} \frac 1 2  \beta^T K_z  \beta - \beta^T K_{zx} \alpha + \frac 1 2 \alpha^TK_x \alpha . \f]
/// The last term is independent of \f$ z_i \f$. Thus it is omitted in the actual computation. That is, the value is offset by a constant and the minimum is not 0.
/// The input of the function consists of a vector which is the concatenation \f$ v=[z_1, z_2,\dots,z_k] \f$ of all basis vectors.
///
/// The target point \f$ w \f$ is set as a KernelExpansion in the constructor. If the kernel is differentiable
/// with respect to the input point then this objective function is differentiable as well.
///
/// The kernel expansion can represent more than one single point, in this case the error is the sum of approximation errors.
class KernelBasisDistance : public SingleObjectiveFunction
{
public:
	/// \brief Constructs the objective function.
	///
	/// This functions calls sparsify on the kernel expansion to save computation time in the case of sparse bases.
	///
	/// \param kernelExpansion a pointer to the kernel expansion to approximate
	/// \param numApproximatingVectors the number of vectors used to approximate the point - the basis size
	KernelBasisDistance(KernelExpansion<RealVector> const* kernelExpansion,std::size_t numApproximatingVectors)
	:mep_expansion(kernelExpansion),m_numApproximatingVectors(numApproximatingVectors){
		SHARK_RUNTIME_CHECK(kernelExpansion != NULL, "KernelExpansion must not be NULL");
		SHARK_RUNTIME_CHECK(kernelExpansion->kernel() != NULL, "KernelExpansion must have a kernel");

		if(mep_expansion->kernel() -> hasFirstInputDerivative())
			m_features|=HAS_FIRST_DERIVATIVE;
	}

	/// \brief Returns the name of the class
	std::string name() const
	{ return "KernelBasisDistance"; }
	/// \brief Returns the number of variables of the function.
	std::size_t numberOfVariables()const{
		return m_numApproximatingVectors  * dataDimension(mep_expansion->basis());
	}
		
	/// \brief Given an input basis, returns the point with the minimum error.
	RealMatrix findOptimalBeta(RealVector const& input)const{
		RealMatrix Kz,beta,linear;
		std::vector<boost::shared_ptr<State> > KzxState;
		boost::shared_ptr<State> KzState;
		setupAndSolve(beta,input,Kz,linear, KzState, KzxState);
		return beta;
	}

	/// \brief Evaluate the (sum of) squared distance(s) between the closes point in the basis to the point(s) represented by the kernel expansion.
	///
	/// See the class description for more details on this computation.
	double eval(RealVector const& input) const{
		SIZE_CHECK(input.size() == numberOfVariables());
		RealMatrix Kz,beta,linear;
		std::vector<boost::shared_ptr<State> > KzxState;
		boost::shared_ptr<State> KzState;
		setupAndSolve(beta,input,Kz,linear, KzState, KzxState);
		return errorOfSolution(beta,Kz,linear);
	}

	/// \brief computes the derivative of the function with respect to the supplied basis.
	///
	/// Assume \f$ \beta \f$ to be the optimal value. Then the derivative with respect to the basis vectors is:
	/// \f[	\frac{ \partial f}{\partial z_l} = \beta_l \sum_i \beta_i \frac{ \partial f}{\partial z_l} k(z_l,z_i) - \beta_l \sum_i \alpha_i \frac{ \partial f}{\partial z_l} k(z_l, x_i) \f]
	ResultType evalDerivative( SearchPointType const& input, FirstOrderDerivative & derivative ) const{
		SIZE_CHECK(input.size() == numberOfVariables());
		
		//set kernel matrices and problem and store intermediate state and optimal beta
		RealMatrix Kz,beta,linear;
		std::vector<boost::shared_ptr<State> > KzxState;
		boost::shared_ptr<State> KzState;
		setupAndSolve(beta,input,Kz,linear, KzState, KzxState);
		
		//make kernele xpansion parameters more accessible
		Data<RealVector> const& expansionBasis = mep_expansion->basis();
		AbstractKernelFunction<RealVector> const& kernel = *mep_expansion->kernel();
		RealMatrix const& alpha = mep_expansion->alpha();
		std::size_t dim = dataDimension(expansionBasis);
		auto basis = to_matrix(input, m_numApproximatingVectors,dim);
		
		//compute derivative
		// the derivative for z_l is given by
		// beta_l sum_i beta_i d/dz_l k(z_l,z_i)
		// - beta_l sum_j alpha_j d/dz_l k(z_l,x_i)
		derivative.resize(input.size());
		//compute first term by using that we can write beta_l * beta_i is an outer product
		//thus when using more than one output point it gets to a set of outer products which
		//can be written as product beta beta^T which are the weights of the derivative
		RealMatrix baseDerivative;
		kernel.weightedInputDerivative(basis, basis, beta % trans(beta),*KzState, baseDerivative);
		noalias(derivative) = to_vector(baseDerivative);
		
		//compute the second term in the same way, taking the block structure into account.
		std::size_t start = 0;
		for(std::size_t i = 0; i != expansionBasis.numberOfBatches(); ++i){
			RealMatrix const& batch = expansionBasis.batch(i);
			kernel.weightedInputDerivative(
				basis,batch,
				beta % trans(rows(alpha,start,start+batch.size1())),
				*KzxState[i],
				baseDerivative
			);
			noalias(derivative) -= to_vector(baseDerivative);
			start +=batch.size1();
		}
		
		return errorOfSolution(beta,Kz,linear);
	}

private:
	/// \brief Sets up and solves the regression problem for the base z.
	///
	/// calculates K_z, the linear part of the system of equations and solves for beta.
	void setupAndSolve(
		RealMatrix& beta, RealVector const& input, RealMatrix& Kz, RealMatrix& linear, 
		boost::shared_ptr<State>& KzState, std::vector<boost::shared_ptr<State> >& KzxState
	)const{
		//get access to the internal variables of the expansion
		Data<RealVector> const& expansionBasis = mep_expansion->basis();
		AbstractKernelFunction<RealVector> const& kernel = *mep_expansion->kernel();
		RealMatrix const& alpha = mep_expansion->alpha();
		std::size_t dim = dataDimension(expansionBasis);
		std::size_t outputs = mep_expansion->outputShape().numElements();
		auto basis = to_matrix(input, m_numApproximatingVectors,dim);

		//set up system of equations and store the kernel states at the same time
		// (we assume here that everything fits into memory, which is the case as long as the number of
		// vectors to approximate is quite small)
		KzState = kernel.createState();
		kernel.eval(basis,basis,Kz,*KzState);
		//construct the linear part
		linear = blas::repeat(0.0,m_numApproximatingVectors,outputs);
		std::size_t start = 0;
		for(std::size_t i = 0; i != expansionBasis.numberOfBatches(); ++i){
			RealMatrix const& batch = expansionBasis.batch(i);
			RealMatrix KzxBlock;
			KzxState.emplace_back(std::move(kernel.createState()));
			kernel.eval(basis,batch,KzxBlock,*KzxState[i]);
			noalias(linear) += KzxBlock % rows(alpha,start,start+batch.size1());
			start += batch.size1();
		}

		//solve for the optimal combination of kernel vectors beta
		beta.resize(m_numApproximatingVectors, outputs);
		noalias(beta) = inv(Kz,blas::symm_semi_pos_def()) % linear;
	}

	/// \brief Returns the error of the solution found
	double errorOfSolution(RealMatrix const& beta, RealMatrix const& Kz, RealMatrix const& linear)const{
		RealMatrix kBeta = Kz % beta;
		return 0.5 * frobenius_prod(kBeta,beta) - frobenius_prod(linear,beta);
	}

	KernelExpansion<RealVector> const* mep_expansion;     ///< kernel expansion to approximate
	std::size_t m_numApproximatingVectors; ///< number of vectors in the basis
};
	
}

KernelExpansion<RealVector> shark::approximateKernelExpansion(
	random::rng_type& rng,
	KernelExpansion<RealVector> const& model,
	std::size_t k,
	double precision
){
	SHARK_RUNTIME_CHECK(model.kernel() != nullptr, "Supplied model needs a kernel");
	SHARK_RUNTIME_CHECK(model.kernel()->hasFirstInputDerivative(), "Kernel must be differentiable wrt its input");
	
	std::size_t dim = dataDimension(model.basis());
		
	//create initial choice of parameters using k-means
	RealVector parameters(k * dim);
	{
		Centroids initialClustering;
		kMeans(model.basis(),k,initialClustering);
		noalias(to_matrix(parameters,k,dim)) = createBatch<RealVector>(initialClustering.centroids().elements());
	}
	
	//optimize the basis iteratively to find a basis with small residual to the optimized vector
	KernelBasisDistance distance(&model,k);
	LBFGS<> optimizer;
	optimizer.init(distance,parameters);
	while(norm_sqr(optimizer.derivative()) > precision){
		RealVector paramOld = optimizer.solution().point;
		optimizer.step(distance);
		if(norm_inf(paramOld - optimizer.solution().point) == 0) break;
	}
	//transform into basis layout
	Data<RealVector> basis;
	basis.push_back(to_matrix(optimizer.solution().point,k,dim));
	RealMatrix beta = distance.findOptimalBeta(optimizer.solution().point);
	
	//create and return kernel expansion
	KernelExpansion<RealVector> expansion = model;
	expansion.basis() = basis;
	expansion.alpha() = beta;
	return expansion;
}
