//===========================================================================
/*!
 *  \brief Trainer for a Regularization Network or a Gaussian Process
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


#ifndef SHARK_ALGORITHMS_REGULARIZATIONNETWORKTRAINER_H
#define SHARK_ALGORITHMS_REGULARIZATIONNETWORKTRAINER_H


#include <shark/Algorithms/Trainers/AbstractSvmTrainer.h>
#include <shark/Models/Kernels/KernelHelpers.h>
#include <shark/LinAlg/solveSystem.h>


namespace shark {


///
/// \brief Training of a regularization network.
///
/// A regularization network is a kernel-based model for
/// regression problems. Given are data tuples
/// \f$ (x_i, y_i) \f$ with x-component denoting input and
/// y-component denoting a real-valued label (see the tutorial on
/// label conventions; the implementation uses RealVector),
/// a kernel function k(x, x') and a regularization
/// constant C > 0. Let H denote the kernel induced
/// reproducing kernel Hilbert space of k, and let \f$ \phi \f$
/// denote the corresponding feature map.
/// Then the SVM regression function is of the form
/// \f[
///     (x) = \langle w, \phi(x) \rangle + b
/// \f]
/// with coefficients w and b given by the (primal)
/// optimization problem
/// \f[
///     \min \frac{1}{2} \|w\|^2 + C \sum_i L(y_i, f(x_i)),
/// \f]
/// where the simple quadratic loss is employed:
/// \f[
///     L(y, f(x)) = (y - f(x))^2 \}
/// \f]
/// Regularization networks can be interpreted as a special
/// type of support vector machine (for regression, with
/// squared loss, and thus with non-sparse weights).
///
template <class InputType>
class RegularizationNetworkTrainer : public AbstractSvmTrainer<InputType, RealVector>
{
public:
	typedef AbstractModel<InputType, RealVector> ModelType;
	typedef AbstractKernelFunction<InputType> KernelType;
	typedef AbstractSvmTrainer<InputType, RealVector> base_type;

	RegularizationNetworkTrainer(KernelType* kernel, double gamma, bool unconstrained = false)
	: base_type(kernel, 1.0 / gamma, unconstrained)
	{ }

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "RegularizationNetworkTrainer"; }

	double gamma() const
	{ return 1.0 / this->C(); }
	void setGamma(double gamma)
	{ this->C() = 1.0 / gamma; }

	void train(KernelExpansion<InputType>& svm, const LabeledData<InputType, RealVector>& dataset)
	{
		// Setup the kernel matrix

		svm.setKernel(base_type::m_kernel);
		svm.setBasis(dataset.inputs());

		SHARK_CHECK(! svm.hasOffset(), "[RegularizationNetworkTrainer::train] training of models with offset is not supported");
		SHARK_CHECK(svm.outputSize() == 1, "[RegularizationNetworkTrainer::train] wrong number of outputs in the kernel expansion");

		
		
// 		RealSymmetricMatrix M(ic, ic);
		//~ RealVector v(ic);
		//~ RealMatrix M(ic, ic);
		//~ for (std::size_t i=0; i<ic; i++)
		//~ {
			//~ for (std::size_t j=0; j<i; j++)
			//~ {
				//~ double k = base_type::m_kernel->eval(dataset(i).input, dataset(j).input);
				//~ M(i, j) = M(j, i) = k;
			//~ }
			//~ double k = base_type::m_kernel->eval(dataset(i).input, dataset(i).input);
			//~ M(i, i) = k + gamma;
			//~ v(i) = dataset(i).label(0);
		//~ }
		//~ RealMatrix invM;
		//~ invertSymmPositiveDefinite(invM, M);

		//~ RealVector param = prod(invM, v);
		//~ svm.setParameterVector(param);
		
		//O.K: I think this is what the code should look like
		RealMatrix M = calculateRegularizedKernelMatrix(*(this->m_kernel),dataset.inputs(), gamma());
		RealVector v = column(createBatch<RealVector>(dataset.labels().elements()),0);
		//~ blas::approxSolveSymmSystemInPlace(M,v); //try this later instad the below
		blas::solveSymmSystemInPlace<blas::SolveAXB>(M,v);
		svm.setParameterVector(v);
	}
};


// A reguarization network can be interpreted as a Gaussian
// process, with the same trainer:
#define GaussianProcessTrainer RegularizationNetworkTrainer


}
#endif
