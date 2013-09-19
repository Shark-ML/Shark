//===========================================================================
/*!
 *  \brief Trainer for the Multi-class Support Vector Machine by Crammer and Singer
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


#ifndef SHARK_ALGORITHMS_MCSVMCSTRAINER_H
#define SHARK_ALGORITHMS_MCSVMCSTRAINER_H


#include <shark/Algorithms/Trainers/AbstractSvmTrainer.h>
#include <shark/Algorithms/QP/QpMcDecomp.h>
#include <shark/Algorithms/QP/QpMcLinear.h>


namespace shark {


///
/// \brief Training of the multi-category SVM by Crammer and Singer (CS).
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
/// The special property of the so-called CS-machine is its
/// loss function, which measures the maximal relative margin
/// violation.
/// Let \f$ h(m) = \max\{0, 1-m\} \f$ denote the hinge loss
/// as a function of the margin m, then the CS loss is given
/// by
/// \f[
///     L(y, f(x)) = \max_{c} h(f_y(x) - f_c(x))
/// \f]
///
/// For details refer to the paper:<br/>
/// <p>On the algorithmic implementation of multiclass kernel-based vector machines. K. Crammer and Y. Singer, Journal of Machine Learning Research, 2002.</p>
///
template <class InputType, class CacheType = float>
class McSvmCSTrainer : public AbstractSvmTrainer<InputType, unsigned int>
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
	McSvmCSTrainer(KernelType* kernel, double C, bool offset, bool unconstrained = false)
	: base_type(kernel, C, offset, unconstrained)
	{ }

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "McSvmCSTrainer"; }

	void train(KernelClassifier<InputType>& svm, const LabeledData<InputType, unsigned int>& dataset)
	{
		std::size_t ic = dataset.numberOfElements();
		unsigned int classes = numberOfClasses(dataset);

		// prepare the problem description
		RealVector alpha((classes-1) * ic,0.0);
		RealVector bias(classes,0.0);

		RealMatrix gamma(classes, classes-1,1.0);
		UIntVector rho(classes-1,0);
		
		QpSparseArray<QpFloatType> nu(classes * (classes-1), classes, 2*classes*(classes-1));
		for (unsigned int r=0, y=0; y<classes; y++)
		{
			for (unsigned int p=0, pp=0; p<classes-1; p++, pp++, r++)
			{
				if (pp == y) pp++;
				if (y < pp)
				{
					nu.add(r, y, 0.5);
					nu.add(r, pp, -0.5);
				}
				else
				{
					nu.add(r, pp, -0.5);
					nu.add(r, y, 0.5);
				}
			}
		}
		
		QpSparseArray<QpFloatType> M(classes * (classes-1) * classes, classes-1, 2 * classes * (classes-1) * (classes-1));
		for (unsigned int r=0, yv=0; yv<classes; yv++)
		{
			for (unsigned int pv=0, ppv=0; pv<classes-1; pv++, ppv++)
			{
				if (ppv == yv) ppv++;
				for (unsigned int yw=0; yw<classes; yw++, r++)
				{
					QpFloatType baseM = (yv == yw ? (QpFloatType)0.25 : (QpFloatType)0.0) - (ppv == yw ? (QpFloatType)0.25 : (QpFloatType)0.0); //4 casts are for compiler warnings
					M.setDefaultValue(r, baseM);
					if (yv == yw)
					{
						M.add(r, ppv - (ppv >= yw ? 1 : 0), baseM + (QpFloatType)0.25);
					}
					else if (ppv == yw)
					{
						M.add(r, yv - (yv >= yw ? 1 : 0), baseM - (QpFloatType)0.25);
					}
					else
					{
						unsigned int pw = ppv - (ppv >= yw ? 1 : 0);
						unsigned int pw2 = yv - (yv >= yw ? 1 : 0);
						if (pw < pw2)
						{
							M.add(r, pw, baseM + (QpFloatType)0.25);
							M.add(r, pw2, baseM - (QpFloatType)0.25);
						}
						else
						{
							M.add(r, pw2, baseM - (QpFloatType)0.25);
							M.add(r, pw, baseM + (QpFloatType)0.25);
						}
					}
				}
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
			// solver.setShrinking(base_type::m_shrinking);
			solver.setShrinking(false);   // hack to avoid shrinking-related bug
			solver.solve(dataset.labels(), this->C(), alpha, base_type::m_stoppingcondition, &prop, (this->m_trainOffset? &bias : NULL));
		}
		else
		{
			CachedMatrixType matrix(&km, base_type::m_cacheSize);
			QpMcDecomp< CachedMatrixType > solver(matrix, gamma, rho, nu, M, true);
			QpSolutionProperties& prop = base_type::m_solutionproperties;
			// solver.setShrinking(base_type::m_shrinking);
			solver.setShrinking(false);   // hack to avoid shrinking-related bug
			solver.solve(dataset.labels(), this->C(), alpha, base_type::m_stoppingcondition, &prop, (this->m_trainOffset? &bias : NULL));
		}
		
		svm.decisionFunction().setStructure(this->m_kernel,dataset.inputs(),this->m_trainOffset,classes);

		// write the solution into the model
		for (std::size_t a=0, i=0; i<ic; i++)
		{
			unsigned int y = dataset.element(i).label;
			for (std::size_t c=0; c<classes; c++)
			{
				double sum = 0.0;
				unsigned int r = (classes-1) * y;
				for (std::size_t p=0; p<classes-1; p++, r++)
					sum += nu(r, c) * alpha(a + p);
				svm.decisionFunction().alpha(i,c) = sum;
			}
			a += classes - 1;
		}
		if (this->m_trainOffset) 
			svm.decisionFunction().offset() = bias;

		base_type::m_accessCount = km.getAccessCount();
		if (this->sparsify()) 
			svm.decisionFunction().sparsify();
	}
};


template <class InputType>
class LinearMcSvmCSTrainer : public AbstractLinearSvmTrainer<InputType>
{
public:
	typedef AbstractLinearSvmTrainer<InputType> base_type;

	LinearMcSvmCSTrainer(double C, bool unconstrained = false)
	: AbstractLinearSvmTrainer<InputType>(C, unconstrained){ }

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "LinearMcSvmCSTrainer"; }

	void train(LinearClassifier<InputType>& model, const LabeledData<InputType, unsigned int>& dataset)
	{
		std::size_t dim = inputDimension(dataset);
		std::size_t classes = numberOfClasses(dataset);

		QpMcLinearCS<InputType> solver(dataset, dim, classes);
		RealMatrix w = solver.solve(this->C(), this->stoppingCondition(), &this->solutionProperties(), this->verbosity() > 0);
		model.decisionFunction().setStructure(w);
	}
};


// shorthands for unified naming scheme; we resort to #define
// statements since old c++ does not support templated typedefs
#define McSvmRDMTrainer McSvmCSTrainer
#define LinearMcSvmRDMTrainer LinearMcSvmCSTrainer


}
#endif
