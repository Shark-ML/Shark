//===========================================================================
/*!
 *  \brief Trainer for the Maximum Margin Regression Multi-class Support Vector Machine
 *
 *
 *  \author  T. Glasmachers
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


#ifndef SHARK_ALGORITHMS_MCSVMMMRTRAINER_H
#define SHARK_ALGORITHMS_MCSVMMMRTRAINER_H


#include <shark/Algorithms/Trainers/AbstractSvmTrainer.h>
#include <shark/Algorithms/QP/QpMcDecomp.h>
#include <shark/Algorithms/QP/QpMcLinear.h>


namespace shark {


///
/// \brief Training of the maximum margin regression (MMR) multi-category SVM.
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
/// with class-wise coefficients w_c and b_c given by the
/// (primal) optimization problem
/// \f[
///     \min \frac{1}{2} \sum_c \|w_c\|^2 + C \sum_i L(y_i, f(x_i)),
/// \f]
/// The special property of the so-called MMR-machine is its
/// loss function, which measures the self-component of the
/// absolute margin violation.
/// Let \f$ h(m) = \max\{0, 1-m\} \f$ denote the hinge loss
/// as a function of the margin m, then the MMR loss is given
/// by
/// \f[
///     L(y, f(x)) = h(f_y(x))
/// \f]
///
/// For more details see the report:<br/>
/// <p>Learning via linear operators: Maximum margin regression.
/// S. Szedmak, J. Shawe-Taylor, and E. Parado-Hernandez,
/// PASCAL, 2006.</p>
///
template <class InputType, class CacheType = float>
class McSvmMMRTrainer : public AbstractSvmTrainer<InputType, unsigned int>
{
public:
	typedef CacheType QpFloatType;

	typedef AbstractModel<InputType, RealVector> ModelType;
	typedef AbstractKernelFunction<InputType> KernelType;
	typedef AbstractSvmTrainer<InputType, unsigned int> base_type;

	//! Constructor
	//! \param  kernel         kernel function to use for training and prediction
	//! \param  C              regularization parameter - always the 'true' value of C, even when unconstrained is set
	//! \param offset    whether to train with offset/bias parameter or not
	//! \param  unconstrained  when a C-value is given via setParameter, should it be piped through the exp-function before using it in the solver?
	McSvmMMRTrainer(KernelType* kernel, double C, bool offset, bool unconstrained = false)
	: base_type(kernel, C, offset, unconstrained)
	{ }

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "McSvmMMRTrainer"; }

	void train(KernelClassifier<InputType>& svm, const LabeledData<InputType, unsigned int>& dataset)
	{
		std::size_t ic = dataset.numberOfElements();
		unsigned int classes = numberOfClasses(dataset);
		
		// prepare the problem description
		RealVector alpha(ic,0.0);
		RealVector bias(classes,0.0);
		RealMatrix gamma(classes, 1,1.0);
		UIntVector rho(1,0);
		
		QpSparseArray<QpFloatType> nu(classes, classes, classes);
		for (unsigned int y=0; y<classes; y++) 
			nu.add(y, y, 1.0);

		QpSparseArray<QpFloatType> M(classes * classes, 1, classes);
		QpFloatType mood = (QpFloatType)(-1.0 / (double)classes);
		QpFloatType val = (QpFloatType)1.0 + mood;
		for (unsigned int r=0, yv=0; yv<classes; yv++)
		{
			for (unsigned int yw=0; yw<classes; yw++, r++)
			{
				M.setDefaultValue(r, mood);
				if (yv == yw) M.add(r, 0, val);
			}
		}
		
		typedef KernelMatrix<InputType, QpFloatType> KernelMatrixType;
		typedef CachedMatrix< KernelMatrixType > CachedMatrixType;
		typedef PrecomputedMatrix< KernelMatrixType > PrecomputedMatrixType;
		
		KernelMatrixType km(*base_type::m_kernel, dataset.inputs());

		// solve the problem
		if (base_type::precomputeKernel())
		{
			PrecomputedMatrixType matrix(&km);
			QpMcDecomp< PrecomputedMatrixType > solver(matrix, gamma, rho, nu, M, true);
			QpSolutionProperties& prop = base_type::m_solutionproperties;
			solver.setShrinking(base_type::m_shrinking);
			if (base_type::m_s2do) 
				solver.solve(dataset.labels(), this->C(), alpha, base_type::m_stoppingcondition, &prop, (this->m_trainOffset ? &bias : NULL));
			else 
				solver.solveSMO(dataset.labels(), this->C(), alpha, base_type::m_stoppingcondition, &prop, (this->m_trainOffset ? &bias : NULL));
		}
		else
		{
			CachedMatrixType matrix(&km, base_type::m_cacheSize);
			QpMcDecomp< CachedMatrixType > solver(matrix, gamma, rho, nu, M, true);
			QpSolutionProperties& prop = base_type::m_solutionproperties;
			solver.setShrinking(base_type::m_shrinking);
			if (base_type::m_s2do)
				solver.solve(dataset.labels(), this->C(), alpha, base_type::m_stoppingcondition, &prop, (this->m_trainOffset ? &bias : NULL));
			else
				solver.solveSMO(dataset.labels(), this->C(), alpha, base_type::m_stoppingcondition, &prop, (this->m_trainOffset ? &bias : NULL));
		}
		
		svm.decisionFunction().setStructure(this->m_kernel,dataset.inputs(),this->m_trainOffset,classes);

		// write the solution into the model
		for (std::size_t i=0; i<ic; i++)
		{
			unsigned int y = dataset.element(i).label;
			for (unsigned int c=0; c<classes; c++)
			{
				svm.decisionFunction().alpha(i,c) = nu(y, c) * alpha(i);
			}
		}
		if (this->m_trainOffset) 
			svm.decisionFunction().offset() = bias;

		base_type::m_accessCount = km.getAccessCount();
		if (this->sparsify()) 
			svm.decisionFunction().sparsify();
	}
};


template <class InputType>
class LinearMcSvmMMRTrainer : public AbstractLinearSvmTrainer<InputType>
{
public:
	typedef AbstractLinearSvmTrainer<InputType> base_type;

	LinearMcSvmMMRTrainer(double C, bool unconstrained = false)
	: AbstractLinearSvmTrainer<InputType>(C, unconstrained){ }

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "LinearMcSvmMMRTrainer"; }

	void train(LinearClassifier<InputType>& model, const LabeledData<InputType, unsigned int>& dataset)
	{
		std::size_t dim = inputDimension(dataset);
		std::size_t classes = numberOfClasses(dataset);

		QpMcLinearMMR<InputType> solver(dataset, dim, classes);
		RealMatrix w = solver.solve(this->C(), this->stoppingCondition(), &this->solutionProperties(), this->verbosity() > 0);
		model.decisionFunction().setStructure(w);
	}
};


}
#endif
