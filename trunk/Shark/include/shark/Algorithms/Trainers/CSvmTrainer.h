//===========================================================================
/*!
 *  \brief Support Vector Machine Trainer for the standard C-SVM
 *
 *
 *  \par
 *  This file collects trainers for the various types of support
 *  vector machines. The trainers carry the hyper-parameters of
 *  SVM training, which includes the kernel parameters.
 *
 *
 *  \author  T. Glasmachers
 *  \date    2007-2012
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


#ifndef SHARK_ALGORITHMS_CSVMTRAINER_H
#define SHARK_ALGORITHMS_CSVMTRAINER_H


#include <shark/Algorithms/Trainers/AbstractSvmTrainer.h>
#include <shark/Algorithms/QP/BoxConstrainedProblems.h>
#include <shark/Algorithms/QP/SvmProblems.h>
#include <shark/Algorithms/QP/QpBoxLinear.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>

namespace shark {


///
/// \brief Training of C-SVMs for binary classification.
///
/// The C-SVM is the "standard" support vector machine for
/// binary (two-class) classification. Given are data tuples
/// \f$ (x_i, y_i) \f$ with x-component denoting input and
/// y-component denoting the label +1 or -1 (see the tutorial on
/// label conventions; the implementation uses values 0/1),
/// a kernel function k(x, x') and a regularization
/// constant C > 0. Let H denote the kernel induced
/// reproducing kernel Hilbert space of k, and let \f$ \phi \f$
/// denote the corresponding feature map.
/// Then the SVM classifier is the function
/// \f[
///     h(x) = \mathop{sign} (f(x))
/// \f]
/// \f[
///     f(x) = \langle w, \phi(x) \rangle + b
/// \f]
/// with coefficients w and b given by the (primal)
/// optimization problem
/// \f[
///     \min \frac{1}{2} \|w\|^2 + C \sum_i L(y_i, f(x_i)),
/// \f]
/// where \f$ L(y, f(x)) = \max\{0, 1 - y \cdot f(x)\} \f$
/// denotes the hinge loss.
///
/// For details refer to the paper:<br/>
/// <p>Support-Vector Networks. Corinna Cortes and Vladimir Vapnik,
/// Machine Learning, vol. 20 (1995), pp. 273-297.</p>
/// or simply to the Wikipedia article:<br/>
/// http://en.wikipedia.org/wiki/Support_vector_machine
///
template <class InputType, class CacheType = float>
class CSvmTrainer : public AbstractSvmTrainer<InputType, unsigned int>
{
public:

	/// \brief Convenience typedefs:
	/// this and many of the below typedefs build on the class template type CacheType.
	/// Simply changing that one template parameter CacheType thus allows to flexibly
	/// switch between using float or double as type for caching the kernel values.
	/// The default is float, offering sufficient accuracy in the vast majority
	/// of cases, at a memory cost of only four bytes. However, the template
	/// parameter makes it easy to use double instead, (e.g., in case high
	/// accuracy training is needed).
	typedef CacheType QpFloatType;

	typedef KernelMatrix<InputType, QpFloatType> KernelMatrixType;
	typedef CachedMatrix< KernelMatrixType > CachedMatrixType;
	typedef PrecomputedMatrix< KernelMatrixType > PrecomputedMatrixType;

	typedef AbstractModel<InputType, RealVector> ModelType;
	typedef AbstractKernelFunction<InputType> KernelType;
	typedef AbstractSvmTrainer<InputType, unsigned int> base_type;

	//! Constructor
	//! \param  kernel         kernel function to use for training and prediction
	//! \param  C              regularization parameter - always the 'true' value of C, even when unconstrained is set
	//! \param  unconstrained  when a C-value is given via setParameter, should it be piped through the exp-function before using it in the solver?
	//! \param  computeDerivative  should the derivative of b with respect to C be computed?
	CSvmTrainer(KernelType* kernel, double C, bool unconstrained = false, bool computeDerivative = true)
	: base_type(kernel, C, unconstrained), m_computeDerivative(computeDerivative), m_useIterativeBiasComputation(false)
	{ }
	
	//! Constructor
	//! \param  kernel         kernel function to use for training and prediction
	//! \param  negativeC   regularization parameter of the negative class (label 0)
	//! \param  positiveC    regularization parameter of the positive class (label 1)
	//! \param  unconstrained  when a C-value is given via setParameter, should it be piped through the exp-function before using it in the solver?
	CSvmTrainer(KernelType* kernel, double negativeC, double positiveC, bool unconstrained = false)
	: base_type(kernel,negativeC, positiveC, unconstrained), m_computeDerivative(false), m_useIterativeBiasComputation(false)
	{ }

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "CSvmTrainer"; }
	
	void setUseIterativeBiasComputation(bool state){
		m_useIterativeBiasComputation = state;
	}
	
	/// for the rare case that there are only bounded SVs and no free SVs, this gives access to the derivative of b w.r.t. C for external use. Derivative w.r.t. C is last.
	RealVector const& get_db_dParams() {
		return m_db_dParams;
	}


	/// \brief Train the C-SVM.
	/// \note This code is almost verbatim present in the MissingFeatureSvmTrainer. If you change here, please also change there.
	void train(KernelExpansion<InputType>& svm, LabeledData<InputType, unsigned int> const& dataset)
	{
		SHARK_CHECK(svm.outputSize() == 1, "[CSvmTrainer::train] wrong number of outputs in the kernel expansion");
		
		svm.setKernel(base_type::m_kernel);
		svm.setBasis(dataset.inputs());
		
		KernelMatrixType km(*base_type::m_kernel, dataset.inputs());
		if (QpConfig::precomputeKernel())
		{
			PrecomputedMatrixType matrix(&km);
			CSVMProblem<PrecomputedMatrixType> svmProblem(matrix,dataset.labels(),base_type::m_regularizers);
			optimize(svm,svmProblem,dataset);
		}
		else
		{
			CachedMatrixType matrix(&km);
			CSVMProblem<CachedMatrixType> svmProblem(matrix,dataset.labels(),base_type::m_regularizers);
			optimize(svm,svmProblem,dataset);
		}
		base_type::m_accessCount = km.getAccessCount();
		if (base_type::sparsify()) svm.sparsify();

	}

private:
	
	template<class SVMProblemType>
	void optimize(KernelExpansion<InputType>& svm, SVMProblemType& svmProblem, LabeledData<InputType, unsigned int> const& dataset){
		if (svm.hasOffset() && !m_useIterativeBiasComputation)
		{
			typedef SvmShrinkingProblem<SVMProblemType> ProblemType;
			ProblemType problem(svmProblem,base_type::m_shrinking);
			QpSolver< ProblemType > solver(problem);
			solver.solve(base_type::stoppingCondition(), &base_type::solutionProperties());
			column(svm.alpha(),0)= problem.getUnpermutedAlpha();
			svm.offset(0) = computeBias(problem,dataset);
		}
		else
		{
			typedef BoxConstrainedShrinkingProblem<SVMProblemType> ProblemType;
			ProblemType problem(svmProblem,base_type::m_shrinking);
			QpSolver< ProblemType> solver(problem);
			solver.solve(base_type::stoppingCondition(), &base_type::solutionProperties());
			
			
			//iteratively solve for the bias if necessary
			if(svm.hasOffset()){
				double bias = 0;//current bias value
				double stepMultiplier = 1;//current length of step
				double alphaSum = 0;//gradient at current position
				for(std::size_t i = 0; i != problem.dimensions(); ++i){
					alphaSum+= problem.alpha(i);
				}
				
				//take step in gradient direction
				double signAlphaSum = alphaSum > 0? 1:-1;
				double deltaBias = signAlphaSum*stepMultiplier;
				bias += deltaBias;
				//update problem using the new estimate for the bias and solve it
				for(std::size_t i = 0; i != problem.dimensions(); ++i){
					problem.setLinear(i,problem.linear(i)-deltaBias);
				}
				
				double epsilon = base_type::stoppingCondition().minAccuracy;
				do{
					
					
					solver.solve(base_type::stoppingCondition(), &base_type::solutionProperties());
					//check whether the step was beneficial
					double newAlphaSum = 0;
					for(std::size_t i = 0; i != problem.dimensions(); ++i){
						newAlphaSum+= problem.alpha(i);
					}
					//prolong steps when they are in the same direction
					if(newAlphaSum*alphaSum > 0){
						stepMultiplier *= 1.2;
						alphaSum = newAlphaSum;
					}
					else{
						stepMultiplier *=0.5;
						alphaSum = newAlphaSum;
					}
					//~ std::cout<<newAlphaSum<<" "<<stepMultiplier<<" "<<bias<<std::endl;
					
					//take step in gradient direction
					double signAlphaSum = alphaSum > 0? 1:-1;
					double deltaBias = signAlphaSum*stepMultiplier;
					bias += deltaBias;
					//update problem using the new estimate for the bias and solve it
					for(std::size_t i = 0; i != problem.dimensions(); ++i){
						problem.setLinear(i,problem.linear(i)-deltaBias);
					}
				}while(problem.checkKKT() > epsilon);
				svm.offset(0) = bias; 
			}
			column(svm.alpha(),0) = problem.getUnpermutedAlpha();
		}
	}
	RealVector m_db_dParams; ///< in the rare case that there are only bounded SVs and no free SVs, this will hold the derivative of b w.r.t. the hyperparameters. Derivative w.r.t. C is last.

	bool m_computeDerivative;
	bool m_useIterativeBiasComputation;

	template<class Problem>
	double computeBias(Problem const& problem, LabeledData<InputType, unsigned int> const& dataset){
		std::size_t nkp = base_type::m_kernel->numberOfParameters();
		m_db_dParams=RealZeroVector( nkp+1); //in the rare case that there are only bounded SVs and no free SVs, we provide the derivative of b w.r.t. hyperparameters for external use

		std::size_t ic = problem.dimensions();

		// compute the offset from the KKT conditions
		double lowerBound = -1e100;
		double upperBound = 1e100;
		double sum = 0.0;
		std::size_t freeVars = 0;
		std::size_t lower_i = 0;
		std::size_t upper_i = 0;
		for (std::size_t i=0; i<ic; i++)
		{
			double value = problem.gradient(i);
			if (problem.alpha(i) == problem.boxMin(i))
			{
				if (value > lowerBound) { //in case of no free SVs, we are looking for the largest gradient of all alphas at the lower bound
					lowerBound = value;
					lower_i = i;
				}
			}
			else if (problem.alpha(i) == problem.boxMax(i))
			{
				if (value < upperBound) { //in case of no free SVs, we are looking for the smallest gradient of all alphas at the upper bound
					upperBound = value;
					upper_i = i;
				}
			}
			else
			{
				sum += value;
				freeVars++;
			}
		}
		if (freeVars > 0)
			return sum / freeVars;		//stabilized (averaged) exact value

		if(!m_computeDerivative)
			return 0.5 * (lowerBound + upperBound);	//best estimate
		
		SHARK_CHECK(base_type::m_regularizers.size() == 1, "derivative only implemented for SVM with one C" );
		
		// We next compute the derivative of lowerBound and upperBound wrt C, in order to then get that of b wrt C.
		// The equation at the foundation of this simply is g_i = y_i - \sum_j \alpha_j K_{ij} .
		double dlower_dC = 0.0;
		double dupper_dC = 0.0;
		// At the same time, we also compute the derivative of lowerBound and upperBound wrt the kernel parameters.
		// The equation at the foundation of this simply is g_i = y_i - \sum_j \alpha_j K_{ij} .
		RealVector dupper_dkernel( nkp,0 );
		RealVector dlower_dkernel( nkp,0 );
		//state for eval and evalDerivative of the kernel
		boost::shared_ptr<State> kernelState = base_type::m_kernel->createState();
		RealVector der(nkp ); //derivative storage helper
		//todo: O.K.: here kernel single input derivative would be usefull
		//also it can be usefull to use here real batch processing and use batches of size 1 for lower /upper
		//and instead of singleInput whole batches.
		//what we do is, that we use the batched input versions with batches of size one.
		typename Batch<InputType>::type singleInput = Batch<InputType>::createBatch( dataset.element(0).input, 1 );
		typename Batch<InputType>::type lowerInput = Batch<InputType>::createBatch( dataset.element(lower_i).input, 1 );
		typename Batch<InputType>::type upperInput = Batch<InputType>::createBatch( dataset.element(upper_i).input, 1 );
		get( lowerInput, 0 ) = dataset.element(lower_i).input; //copy the current input into the batch
		get( upperInput, 0 ) = dataset.element(upper_i).input; //copy the current input into the batch
		RealMatrix one(1,1,1); //weight of input
		RealMatrix result(1,1); //stores the result of the call

		for (std::size_t i=0; i<ic; i++) {
			double cur_alpha = problem.alpha(i);
			if ( cur_alpha != 0 ) {
				int cur_label = ( cur_alpha>0.0 ? 1 : -1 );
				get( singleInput, 0 ) = dataset.element(i).input; //copy the current input into the batch
				// treat contributions of largest gradient at lower bound
				base_type::m_kernel->eval( lowerInput, singleInput, result, *kernelState );
				dlower_dC += cur_label * result(0,0);
				base_type::m_kernel->weightedParameterDerivative( lowerInput, singleInput,one, *kernelState, der );
				for ( std::size_t k=0; k<nkp; k++ ) {
					dlower_dkernel(k) += cur_label * der(k);
				}
				// treat contributions of smallest gradient at upper bound
				base_type::m_kernel->eval( upperInput, singleInput,result, *kernelState );
				dupper_dC += cur_label * result(0,0);
				base_type::m_kernel->weightedParameterDerivative( upperInput, singleInput, one, *kernelState, der );
				for ( std::size_t k=0; k<nkp; k++ ) {
					dupper_dkernel(k) += cur_label * der(k);
				}
			}
		}
		// assign final values to derivative of b wrt hyperparameters
		m_db_dParams( nkp ) = -0.5 * ( dlower_dC + dupper_dC );
		for ( std::size_t k=0; k<nkp; k++ ) {
			m_db_dParams(k) = -0.5 * this->C() * ( dlower_dkernel(k) + dupper_dkernel(k) );
		}
		if ( base_type::m_unconstrained ) {
			m_db_dParams( nkp ) *= this->C();
		}
		
		return 0.5 * (lowerBound + upperBound);	//best estimate
	}
};


class LinearCSvmTrainer : public AbstractLinearSvmTrainer
{
public:
	typedef AbstractLinearSvmTrainer base_type;

	LinearCSvmTrainer(double C, double accuracy = 0.001) : AbstractLinearSvmTrainer(C, accuracy)
	{ }

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "LinearCSvmTrainer"; }

	void train(LinearModel<CompressedRealVector, RealVector>& model, LabeledData<CompressedRealVector, unsigned int> const& dataset)
	{
		std::size_t dim = model.inputSize();
		SHARK_CHECK(model.outputSize() == 1, "[LinearCSvmTrainer::train] wrong number of outputs in the linear model");
		SHARK_CHECK(! model.hasOffset(), "[LinearCSvmTrainer::train] models with offset are not supported (yet).");
		QpBoxLinear solver(dataset, dim);
		RealMatrix w(1, dim, 0.0);
		column(w, 0) = solver.solve(C(), m_stoppingcondition, &m_solutionproperties, m_verbosity > 0);
		model.setStructure(w);
	}
};


template <class InputType, class CacheType = float>
class SquaredHingeCSvmTrainer : public AbstractSvmTrainer<InputType, unsigned int>
{
public:
	typedef CacheType QpFloatType;

	typedef RegularizedKernelMatrix<InputType, QpFloatType> KernelMatrixType;
	typedef CachedMatrix< KernelMatrixType > CachedMatrixType;
	typedef PrecomputedMatrix< KernelMatrixType > PrecomputedMatrixType;

	typedef AbstractModel<InputType, RealVector> ModelType;
	typedef AbstractKernelFunction<InputType> KernelType;
	typedef AbstractSvmTrainer<InputType, unsigned int> base_type;

	//! Constructor
	//! \param  kernel         kernel function to use for training and prediction
	//! \param  C              regularization parameter - always the 'true' value of C, even when unconstrained is set
	//! \param  unconstrained  when a C-value is given via setParameter, should it be piped through the exp-function before using it in the solver??
	SquaredHingeCSvmTrainer(KernelType* kernel, double C, bool unconstrained = false)
	: base_type(kernel, C, unconstrained)
	{ }
	
	//! Constructor
	//! \param  kernel         kernel function to use for training and prediction
	//! \param  negativeC   regularization parameter of the negative class (label 0)
	//! \param  positiveC    regularization parameter of the positive class (label 1)
	//! \param  unconstrained  when a C-value is given via setParameter, should it be piped through the exp-function before using it in the solver?
	SquaredHingeCSvmTrainer(KernelType* kernel, double negativeC, double positiveC, bool unconstrained = false)
	: base_type(kernel,negativeC, positiveC, unconstrained)
	{ }

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "SquaredHingeCSvmTrainer"; }

	/// \brief Train the C-SVM.
	void train(KernelExpansion<InputType>& svm, LabeledData<InputType, unsigned int> const& dataset)
	{
		SHARK_CHECK(svm.outputSize() == 1, "[CSvmTrainer::train] wrong number of outputs in the kernel expansion");
		SHARK_CHECK(numberOfClasses(dataset) == 2, "[CSvmTrainer::train] trainer can only solve binary problems");
		
		svm.setKernel(base_type::m_kernel);
		svm.setBasis(dataset.inputs());
		
		RealVector diagonalModifier(dataset.numberOfElements(),0.5/base_type::m_regularizers(0));
		if(base_type::m_regularizers.size() != 1){
			for(std::size_t i = 0; i != diagonalModifier.size();++i){
				diagonalModifier(i) = 0.5/base_type::m_regularizers(dataset.element(i).label);
			}
		}
		
		KernelMatrixType km(*base_type::m_kernel, dataset.inputs(),diagonalModifier);
		if (QpConfig::precomputeKernel())
		{
			PrecomputedMatrixType matrix(&km);
			optimize(svm,matrix,diagonalModifier,dataset);
		}
		else
		{
			CachedMatrixType matrix(&km);
			optimize(svm,matrix,diagonalModifier,dataset);
		}
		base_type::m_accessCount = km.getAccessCount();
		if (base_type::sparsify()) svm.sparsify();

	}

private:
	
	template<class Matrix>
	void optimize(KernelExpansion<InputType>& svm, Matrix& matrix,RealVector const& diagonalModifier, LabeledData<InputType, unsigned int> const& dataset){
		typedef CSVMProblem<Matrix> SVMProblemType;
		SVMProblemType svmProblem(matrix,dataset.labels(),1e100);
		if (svm.hasOffset())
		{
			typedef SvmShrinkingProblem<SVMProblemType> ProblemType;
			ProblemType problem(svmProblem,base_type::m_shrinking);
			QpSolver< ProblemType > solver(problem);
			solver.solve(base_type::stoppingCondition(), &base_type::solutionProperties());
			column(svm.alpha(),0)= problem.getUnpermutedAlpha();
			//compute the bias
			double sum = 0.0;
			std::size_t freeVars = 0;
			for (std::size_t i=0; i < problem.dimensions(); i++)
			{
				if(problem.alpha(i) > problem.boxMin(i) && problem.alpha(i) < problem.boxMax(i)){
					sum += problem.gradient(i) - problem.alpha(i)*2*diagonalModifier(i);
					freeVars++;
				}
			}
			if (freeVars > 0)
				svm.offset(0) =  sum / freeVars;		//stabilized (averaged) exact value
			else
				svm.offset(0) = 0;
		}
		else
		{
			typedef BoxConstrainedShrinkingProblem<SVMProblemType> ProblemType;
			ProblemType problem(svmProblem,base_type::m_shrinking);
			QpSolver< ProblemType > solver(problem);
			solver.solve(base_type::stoppingCondition(), &base_type::solutionProperties());
			column(svm.alpha(),0) = problem.getUnpermutedAlpha();
			
		}
	}
};


}
#endif
