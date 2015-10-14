//===========================================================================
/*!
 * 
 *
 * \brief       Trainer for One-versus-all (one-versus-rest) Multi-class Support Vector Machines
 * 
 * 
 * 
 *
 * \author      T. Glasmachers
 * \date        -
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


#ifndef SHARK_ALGORITHMS_MCSVMOVATRAINER_H
#define SHARK_ALGORITHMS_MCSVMOVATRAINER_H


#include <shark/Algorithms/Trainers/AbstractSvmTrainer.h>
#include <shark/Algorithms/Trainers/CSvmTrainer.h>


namespace shark {


///
/// \brief Training of a multi-category SVM by the one-versus-all (OVA) method.
///
/// This is a special support vector machine variant for
/// classification of more than two classes. Given are data
/// tuples \f$ (x_i, y_i) \f$ with x-component denoting input
/// and y-component denoting the label 1, ..., d (see the tutorial on
/// label conventions; the implementation uses values 0 to d-1),
/// a kernel function k(x, x') and a regularization
/// constant C > 0. Let H denote the kernel induced
/// reproducing kernel Hilbert space of k, and let \f$ \phi \f$
/// denote the corresponding feature map.
/// Then the SVM classifier is the function
/// \f[
///     h(x) = \arg \max (f_c(x))
/// \f]
/// \f[
///     f_c(x) = \langle w_c, \phi(x) \rangle + b_c
/// \f]
/// \f[
///     f = (f_1, \dots, f_d)
/// \f]
/// with class-wise coefficients w_c and b_c obtained by training
/// a standard C-SVM (see CSvmTrainer) with class c as the positive
/// and the union of all other classes as the negative class.
/// This is often a strong baseline method, and it is usually much
/// faster to train than other multi-category SVMs.
///
template <class InputType, class CacheType = float>
class McSvmOVATrainer : public AbstractSvmTrainer<InputType, unsigned int>
{
public:
	typedef CacheType QpFloatType;

	typedef AbstractModel<InputType, RealVector> ModelType;
	typedef AbstractKernelFunction<InputType> KernelType;
	typedef AbstractSvmTrainer<InputType, unsigned int> base_type;

	//! Constructor
	//! \param  kernel         kernel function to use for training and prediction
	//! \param  C              regularization parameter - always the 'true' value of C, even when unconstrained is set
	//! \param offset  whether to train offset/bias parameter
	//! \param  unconstrained  when a C-value is given via setParameter, should it be piped through the exp-function before using it in the solver?
	McSvmOVATrainer(KernelType* kernel, double C, bool offset, bool unconstrained = false)
	: base_type(kernel, C, offset, unconstrained)
	{ }

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "McSvmOVATrainer"; }

	/// \brief Train a kernelized SVM.
	void train(KernelClassifier<InputType>& svm, const LabeledData<InputType, unsigned int>& dataset)
	{
		std::size_t classes = numberOfClasses(dataset);
		svm.decisionFunction().setStructure(this->m_kernel,dataset.inputs(),this->m_trainOffset,classes);
		
		base_type::m_solutionproperties.type = QpNone;
		base_type::m_solutionproperties.accuracy = 0.0;
		base_type::m_solutionproperties.iterations = 0;
		base_type::m_solutionproperties.value = 0.0;
		base_type::m_solutionproperties.seconds = 0.0;
		for (std::size_t c=0; c<classes; c++)
		{
			LabeledData<InputType, unsigned int> bindata = oneVersusRestProblem(dataset, c);
			KernelClassifier<InputType> binsvm;
// TODO: maybe build the quadratic programs directly,
//       in order to profit from cached and
//       in particular from precomputed kernel
//       entries!
			CSvmTrainer<InputType, QpFloatType> bintrainer(base_type::m_kernel, this->C(),this->m_trainOffset);
			bintrainer.setCacheSize( base_type::m_cacheSize );
			bintrainer.sparsify() = false;
			bintrainer.stoppingCondition() = base_type::stoppingCondition();
			bintrainer.precomputeKernel() = base_type::precomputeKernel();		// sub-optimal!
			bintrainer.shrinking() = base_type::shrinking();
			bintrainer.s2do() = base_type::s2do();
			bintrainer.verbosity() = base_type::verbosity();
			bintrainer.train(binsvm, bindata);
			base_type::m_solutionproperties.iterations += bintrainer.solutionProperties().iterations;
			base_type::m_solutionproperties.seconds += bintrainer.solutionProperties().seconds;
			base_type::m_solutionproperties.accuracy = std::max(base_type::solutionProperties().accuracy, bintrainer.solutionProperties().accuracy);
			column(svm.decisionFunction().alpha(), c) = column(binsvm.decisionFunction().alpha(), 0);
			if (this->m_trainOffset)
				svm.decisionFunction().offset(c) = binsvm.decisionFunction().offset(0);
			base_type::m_accessCount += bintrainer.accessCount();
		}

		if (base_type::sparsify()) 
			svm.decisionFunction().sparsify();
	}
};


template <class InputType>
class LinearMcSvmOVATrainer : public AbstractLinearSvmTrainer<InputType>
{
public:
	typedef AbstractLinearSvmTrainer<InputType> base_type;

	LinearMcSvmOVATrainer(double C, bool unconstrained = false)
	: AbstractLinearSvmTrainer<InputType>(C, unconstrained){ }

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "LinearMcSvmOVATrainer"; }

	void train(LinearClassifier<InputType>& model, const LabeledData<InputType, unsigned int>& dataset)
	{
		base_type::m_solutionproperties.type = QpNone;
		base_type::m_solutionproperties.accuracy = 0.0;
		base_type::m_solutionproperties.iterations = 0;
		base_type::m_solutionproperties.value = 0.0;
		base_type::m_solutionproperties.seconds = 0.0;

		std::size_t dim = inputDimension(dataset);
		std::size_t classes = numberOfClasses(dataset);
		RealMatrix w(classes, dim);
		for (std::size_t c=0; c<classes; c++)
		{
			LabeledData<InputType, unsigned int> bindata = oneVersusRestProblem(dataset, c);
			QpBoxLinear<InputType> solver(bindata, dim);
			QpSolutionProperties prop;
			row(w, c) = solver.solve(this->C(), base_type::m_stoppingcondition, &prop, base_type::m_verbosity > 0);
			base_type::m_solutionproperties.iterations += prop.iterations;
			base_type::m_solutionproperties.seconds += prop.seconds;
			base_type::m_solutionproperties.accuracy = std::max(base_type::solutionProperties().accuracy, prop.accuracy);
		}
		model.decisionFunction().setStructure(w);
	}
};


}
#endif
