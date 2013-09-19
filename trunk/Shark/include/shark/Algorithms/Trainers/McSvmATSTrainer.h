//===========================================================================
/*!
 *  \brief Trainer for the ATS Multi-class Support Vector Machine
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


#ifndef SHARK_ALGORITHMS_MCSVMATSTRAINER_H
#define SHARK_ALGORITHMS_MCSVMATSTRAINER_H


#include <shark/Algorithms/Trainers/AbstractSvmTrainer.h>
#include <shark/Algorithms/QP/QpMcDecomp.h>
#include <shark/Algorithms/QP/QpMcLinear.h>


namespace shark {


///
/// \brief Training of ATS-SVMs for multi-category classification.
///
/// The ATS-SVM is a special support vector machine variant for
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
///     \min \frac{1}{2} \sum_c \|w_c\|^2 + C \sum_i L(y_i, f(x_i))
/// \f]
/// \f[
///     \text{s.t. } \sum_c f_c = 0
/// \f]
/// The special property of the so-called ATS machine is its
/// loss function, which arises from the application of the
/// total sum operator to absolute margin violations.
/// Let \f$ h(m) = \max\{0, 1-m\} \f$ denote the hinge loss
/// as a function of the margin m, then the ATS loss is given
/// by
/// \f[
///     L(y, f(x)) = \sum_c h((2 \cdot \delta_{c,y} - 1) \cdot f_c(x))
/// \f]
/// where the Kronecker delta is one if its arguments agree and
/// zero otherwise.
///
/// For more details refer to the technical report:<br/>
/// <p>Fast Training of Multi-Class Support Vector Machines. &Uuml; Dogan, T. Glasmachers, and C. Igel, Technical Report 2011/3, Department of Computer Science, University of Copenhagen, 2011.</p>
///
template <class InputType, class CacheType = float>
class McSvmATSTrainer : public AbstractSvmTrainer<InputType, unsigned int>
{
public:

	typedef CacheType QpFloatType;
	typedef AbstractModel<InputType, RealVector> ModelType;
	typedef AbstractKernelFunction<InputType> KernelType;
	typedef AbstractSvmTrainer<InputType, unsigned int> base_type;

	//! Constructor
	//! \param  kernel         kernel function to use for training and prediction
	//! \param  C              regularization parameter - always the 'true' value of C, even when unconstrained is set
	//! \param  unconstrained  when a C-value is given via setParameter, should it be piped through the exp-function before using it in the solver?
	McSvmATSTrainer(KernelType* kernel, double C, bool unconstrained = false)
	: base_type(kernel, C, unconstrained)
	{ }

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "McSvmATSTrainer"; }

	void train(KernelClassifier<InputType>& svm, const LabeledData<InputType, unsigned int>& dataset)
	{
		std::size_t ic = dataset.numberOfElements();
		unsigned int classes = numberOfClasses(dataset);

		// prepare the problem description
		RealVector alpha(classes * ic,0.0);
		RealVector bias(classes,0.0);

		RealMatrix gamma(classes, classes,1.0);
		UIntVector rho(classes);
		for (unsigned int p=0; p<classes; p++) 
			rho(p) = p;
		
		QpSparseArray<QpFloatType> nu(classes*classes, classes, classes*classes);
		for (unsigned int r=0, y=0; y<classes; y++)
		{
			for (unsigned int p=0; p<classes; p++, r++)
			{
				nu.add(r, p, (QpFloatType)((p == y) ? 1.0 : -1.0));
			}
		}
		
		QpSparseArray<QpFloatType> M(classes * classes * classes, classes, 2 * classes * classes * classes);
		QpFloatType c_ne = (QpFloatType)(-1.0 / (double)classes);
		QpFloatType c_eq = (QpFloatType)1.0 + c_ne;
		for (unsigned int r=0, yv=0; yv<classes; yv++)
		{
			for (unsigned int pv=0; pv<classes; pv++)
			{
				QpFloatType sign = QpFloatType((yv == pv) ? -1 : 1);//cast to keep MSVC happy...
				for (unsigned int yw=0; yw<classes; yw++, r++)
				{
					M.setDefaultValue(r, sign * c_ne);
					if (yw == pv)
					{
						M.add(r, pv, -sign * c_eq);
					}
					else
					{
						M.add(r, pv, sign * c_eq);
						M.add(r, yw, -sign * c_ne);
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
			solver.setShrinking(base_type::m_shrinking);
			if (base_type::m_s2do) solver.solve(dataset.labels(), this->C(), alpha, base_type::m_stoppingcondition, &prop, (this->m_trainOffset ? &bias : NULL));
			else solver.solveSMO(dataset.labels(), this->C(), alpha, base_type::m_stoppingcondition, &prop, (this->m_trainOffset ? &bias : NULL));
		}
		else
		{
			CachedMatrixType matrix(&km, base_type::m_cacheSize);
			QpMcDecomp< CachedMatrixType > solver(matrix, gamma, rho, nu, M, true);
			QpSolutionProperties& prop = base_type::m_solutionproperties;
			solver.setShrinking(base_type::m_shrinking);
			if (base_type::m_s2do) solver.solve(dataset.labels(), this->C(), alpha, base_type::m_stoppingcondition, &prop, (this->m_trainOffset ? &bias : NULL));
			else solver.solveSMO(dataset.labels(), this->C(), alpha, base_type::m_stoppingcondition, &prop, (this->m_trainOffset ? &bias : NULL));
		}
		
		svm.decisionFunction().setStructure(this->m_kernel,dataset.inputs(),this->m_trainOffset,classes);

		// write the solution into the model
		for (unsigned int a=0, i=0; i<ic; i++)
		{
			unsigned int y = dataset.element(i).label;
			for (unsigned int c=0; c<classes; c++)
			{
				double sum = 0.0;
				unsigned int r = classes * y;
				for (unsigned int p=0; p<classes; p++, r++)
					sum += nu(r, c) * alpha(a + p);
				svm.decisionFunction().alpha(i,c) = sum;
			}
			a += classes;
		}
		if (this->m_trainOffset) 
			svm.decisionFunction().offset() = bias;

		base_type::m_accessCount = km.getAccessCount();
		if (this->sparsify()) 
			svm.decisionFunction().sparsify();
	}
};


template <class InputType>
class LinearMcSvmATSTrainer : public AbstractLinearSvmTrainer<InputType>
{
public:
	typedef AbstractLinearSvmTrainer<InputType> base_type;

	LinearMcSvmATSTrainer(double C, bool unconstrained = false)
	: AbstractLinearSvmTrainer<InputType>(C, unconstrained){ }

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "LinearMcSvmATSTrainer"; }

	void train(LinearClassifier<InputType>& model, const LabeledData<InputType, unsigned int>& dataset)
	{
		std::size_t dim = inputDimension(dataset);
		std::size_t classes = numberOfClasses(dataset);

		QpMcLinearATS<InputType> solver(dataset, dim, classes);
		RealMatrix w = solver.solve(this->C(), this->stoppingCondition(), &this->solutionProperties(), this->verbosity() > 0);
		model.decisionFunction().setStructure(w);
	}
};


}
#endif
