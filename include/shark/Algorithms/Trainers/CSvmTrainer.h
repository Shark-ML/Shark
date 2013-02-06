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
#include <shark/Algorithms/QP/QpBoxLinear.h>


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
	typedef blas::matrix<QpFloatType> QpMatrixType;
	typedef blas::matrix_row<QpMatrixType> QpMatrixRowType;
	typedef blas::matrix_column<QpMatrixType> QpMatrixColumnType;

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
	CSvmTrainer(KernelType* kernel, double C, bool unconstrained = false)
	: base_type(kernel, C, unconstrained)
	{
		base_type::m_name = "CSvmTrainer";
	}


	/// \brief Train the C-SVM.
	/// \note This code is almost verbatim present in the MissingFeatureSvmTrainer. If you change here, please also change there.
	void train(KernelExpansion<InputType>& svm, const LabeledData<InputType, unsigned int>& dataset)
	{
		SHARK_CHECK(svm.outputSize() == 1, "[CSvmTrainer::train] wrong number of outputs in the kernel expansion");
		std::size_t nkp = base_type::m_kernel->numberOfParameters();
		m_db_dParams = RealZeroVector( nkp+1 ); //in the rare case that there are only bounded SVs and no free SVs, we provide the derivative of b w.r.t. hyperparameters for external use

		// prepare the quadratic program description
		std::size_t i, ic = dataset.numberOfElements();
		RealVector linear(ic);
		RealVector lower(ic);
		RealVector upper(ic);
		RealVector alpha = RealZeroVector(ic);
		for (i=0; i<ic; i++)
		{
			if (dataset.element(i).label == 0)
			{
				linear(i) = -1.0;
				lower(i) = -base_type::m_C;
				upper(i) = 0.0;
			}
			else
			{
				SHARK_CHECK(dataset.element(i).label == 1, "C-SVMs are for binary classification");
				linear(i) = 1.0;
				lower(i) = 0.0;
				upper(i) = base_type::m_C;
			}
		}

		svm.setKernel(base_type::m_kernel);
		svm.setBasis(dataset.inputs());

		KernelMatrixType km(*base_type::m_kernel, dataset.inputs());
		if (svm.hasOffset())
		{
			RealVector gradient;

			// solve the problem with equality constraint
			if (QpConfig::precomputeKernel())
			{
				PrecomputedMatrixType matrix(&km);
				QpSvmDecomp< PrecomputedMatrixType > solver(matrix);
				QpSolutionProperties& prop = base_type::m_solutionproperties;
				solver.setShrinking(base_type::m_shrinking);
				solver.solve(linear, lower, upper, alpha, base_type::m_stoppingcondition, &prop);
				gradient = solver.getGradient();
			}
			else
			{
				CachedMatrixType matrix(&km, base_type::m_cacheSize );
				QpSvmDecomp< CachedMatrixType > solver(matrix);
				QpSolutionProperties& prop = base_type::m_solutionproperties;
				solver.setShrinking(base_type::m_shrinking);
				solver.solve(linear, lower, upper, alpha, base_type::m_stoppingcondition, &prop);
				gradient = solver.getGradient();
			}

			RealVector param(ic + 1);
			RealVectorRange(param, Range(0, ic)) = alpha;

			// compute the offset from the KKT conditions
			double lowerBound = -1e100;
			double upperBound = 1e100;
			double sum = 0.0;
			std::size_t freeVars = 0;
			std::size_t lower_i = 0; //no reason to init to 0, but avoid compiler warnings
			std::size_t upper_i = 0; //no reason to init to 0, but avoid compiler warnings
			for (i=0; i<ic; i++)
			{
				double value = gradient(i);
				if (alpha(i) == lower(i))
				{
					if (value > lowerBound) { //in case of no free SVs, we are looking for the largest gradient of all alphas at the lower bound
						lowerBound = value;
						lower_i = i;
					}
				}
				else if (alpha(i) == upper(i))
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
			if (freeVars > 0) {
				param(ic) = sum / freeVars;		//stabilized (averaged) exact value
			} else {
				param(ic) = 0.5 * (lowerBound + upperBound);	//best estimate
				// We next compute the derivative of lowerBound and upperBound wrt C, in order to then get that of b wrt C.
				// The equation at the foundation of this simply is g_i = y_i - \sum_j \alpha_j K_{ij} .
				double dlower_dC = 0.0;
				double dupper_dC = 0.0;
				// At the same time, we also compute the derivative of lowerBound and upperBound wrt the kernel parameters.
				// The equation at the foundation of this simply is g_i = y_i - \sum_j \alpha_j K_{ij} .
				RealVector dupper_dkernel = RealZeroVector( nkp );
				RealVector dlower_dkernel = RealZeroVector( nkp );
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
				RealMatrix one(1,1); one(0,0) = 1; //weight of input
				RealMatrix result(1,1); //stores the result of the call

				for (std::size_t i=0; i<ic; i++) {
					double cur_alpha = alpha(i);
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
					m_db_dParams(k) = -0.5 * base_type::m_C * ( dlower_dkernel(k) + dupper_dkernel(k) );
				}
				if ( base_type::m_unconstrained ) {
					m_db_dParams( nkp ) *= base_type::m_C;
				}
			}
			svm.setParameterVector(param);
		}
		else
		{
			if (base_type::precomputeKernel())
			{
				PrecomputedMatrixType matrix(&km);
				QpSolutionProperties& prop = base_type::m_solutionproperties;
				QpBoxDecomp< PrecomputedMatrixType > solver(matrix);
				solver.setShrinking(base_type::m_shrinking);
				solver.solve(linear, lower, upper, alpha, base_type::m_stoppingcondition, &prop);
			}
			else
			{
				CachedMatrixType matrix(&km, base_type::m_cacheSize );
				QpSolutionProperties& prop = base_type::m_solutionproperties;
				QpBoxDecomp< CachedMatrixType > solver(matrix);
				solver.setShrinking(base_type::m_shrinking);
				solver.solve(linear, lower, upper, alpha, base_type::m_stoppingcondition, &prop);
			}

			svm.setParameterVector(alpha);
		}

		base_type::m_accessCount = km.getAccessCount();
		if (base_type::sparsify()) svm.sparsify();

	}

	/// for the rare case that there are only bounded SVs and no free SVs, this gives access to the derivative of b w.r.t. C for external use. Derivative w.r.t. C is last.
	const RealVector& get_db_dParams() {
		return m_db_dParams;
	}

protected:
	RealVector m_db_dParams; ///< in the rare case that there are only bounded SVs and no free SVs, this will hold the derivative of b w.r.t. the hyperparameters. Derivative w.r.t. C is last.
};


class LinearCSvmTrainer : public AbstractLinearSvmTrainer
{
public:
	typedef AbstractLinearSvmTrainer base_type;

	LinearCSvmTrainer(double C, double accuracy = 0.001) : AbstractLinearSvmTrainer(C, accuracy)
	{
		base_type::m_name = "LinearCSvmTrainer";
	}

	void train(LinearModel<CompressedRealVector, RealVector>& model, LabeledData<CompressedRealVector, unsigned int> const& dataset)
	{
		std::size_t dim = model.inputSize();
		SHARK_CHECK(model.outputSize() == 1, "[LinearCSvmTrainer::train] wrong number of outputs in the linear model");
		SHARK_CHECK(! model.hasOffset(), "[LinearCSvmTrainer::train] models with offset are not supported (yet).");
		QpBoxLinear solver(dataset, dim);
		RealMatrix w(1, dim, 0.0);
		row(w, 0) = solver.solve(C(), m_stoppingcondition, &m_solutionproperties, m_verbosity > 0);
		model.setStructure(w);
	}
};


}
#endif
