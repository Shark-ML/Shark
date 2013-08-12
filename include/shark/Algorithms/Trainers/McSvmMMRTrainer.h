//===========================================================================
/*!
 *  \brief Trainer for the Maximum Margin Regression Multi-class Support Vector Machine
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
	McSvmMMRTrainer(KernelType* kernel, double C, bool unconstrained = false)
	: base_type(kernel, C, unconstrained)
	{ }

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "McSvmMMRTrainer"; }

	void train(KernelExpansion<InputType>& svm, const LabeledData<InputType, unsigned int>& dataset)
	{
		std::size_t ic = dataset.numberOfElements();
		unsigned int classes = numberOfClasses(dataset);
		// the following test is "<=" rather than "=" to account for the rare case that one fold doesn't contain all classes due to sample scarcity
		SHARK_CHECK(classes <= svm.outputSize(), "[McSvmMMRTrainer::train] invalid number of outputs in the kernel expansion");
		svm.setKernel(base_type::m_kernel);
		svm.setBasis(dataset.inputs());
		classes = svm.outputSize();
		
		RealVector param = svm.parameterVector();

		// prepare the problem description
		RealVector alpha(ic,0.0);
		RealVector bias(classes,0.0);
		RealMatrix gamma(classes, 1,1.0);
		UIntVector rho(1,0);
		QpSparseArray<QpFloatType> nu(classes, classes, classes);

		for (unsigned int y=0; y<classes; y++) 
			nu.add(y, y, 1.0);

		QpSparseArray<QpFloatType> M(classes * classes, 1, classes);
		{
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
		}
		KernelMatrixType km(*base_type::m_kernel, dataset.inputs());

		// solve the problem
		if (base_type::precomputeKernel())
		{
			PrecomputedMatrixType matrix(&km);
			QpMcDecomp< PrecomputedMatrixType > solver(matrix, gamma, rho, nu, M, true);
			QpSolutionProperties& prop = base_type::m_solutionproperties;
			solver.setShrinking(base_type::m_shrinking);
			if (base_type::m_s2do) solver.solve(dataset.labels(), this->C(), alpha, base_type::m_stoppingcondition, &prop, (svm.hasOffset() ? &bias : NULL));
			else solver.solveSMO(dataset.labels(), this->C(), alpha, base_type::m_stoppingcondition, &prop, (svm.hasOffset() ? &bias : NULL));
		}
		else
		{
			CachedMatrixType matrix(&km, base_type::m_cacheSize);
			QpMcDecomp< CachedMatrixType > solver(matrix, gamma, rho, nu, M, true);
			QpSolutionProperties& prop = base_type::m_solutionproperties;
			solver.setShrinking(base_type::m_shrinking);
			if (base_type::m_s2do)
				solver.solve(dataset.labels(), this->C(), alpha, base_type::m_stoppingcondition, &prop, (svm.hasOffset() ? &bias : NULL));
			else
				solver.solveSMO(dataset.labels(), this->C(), alpha, base_type::m_stoppingcondition, &prop, (svm.hasOffset() ? &bias : NULL));
		}

		// write the solution into the model
		std::size_t e = 0;
		for (std::size_t i=0; i<ic; i++)
		{
			unsigned int y = dataset.element(i).label;
			for (unsigned int c=0; c<classes; c++, e++)
			{
				param(e) = nu(y, c) * alpha(i);
			}
		}
		if (svm.hasOffset()) 
			subrange(param, e, e + classes) = bias;
		svm.setParameterVector(param);

		base_type::m_accessCount = km.getAccessCount();
		if (base_type::sparsify()) svm.sparsify();
	}
};


class LinearMcSvmMMRTrainer : public AbstractLinearSvmTrainer
{
public:
	LinearMcSvmMMRTrainer(double C, double accuracy = 0.001) : AbstractLinearSvmTrainer(C, accuracy)
	{ }

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "LinearMcSvmMMRTrainer"; }

	void train(LinearModel<CompressedRealVector, RealVector>& model, const LabeledData<CompressedRealVector, unsigned int>& dataset)
	{
		SHARK_CHECK(! model.hasOffset(), "[LinearMcSvmMMRTrainer::train] models with offset are not supported (yet).");

		std::size_t dim = model.inputSize();
		std::size_t classes = model.outputSize();
/*
		CompressedRealMatrix w(classes, dim);
		std::vector<CompressedRealMatrixRow> w_s;
		for (std::size_t c=0; c<classes; c++) w_s.push_back(CompressedRealMatrixRow(w, c));
		typedef McPegasos<CompressedRealVector> PegasosType;
		PegasosType::solve(
				dataset,
				PegasosType::emAbsolute,
				PegasosType::elNaiveHinge,
				true,
				C(),
				w_s,
				std::min((std::size_t)1000, dataset.numberOfElements()),
				accuracy());
*/
		QpMcLinearMMR solver(dataset, dim, classes);
		RealMatrix w = solver.solve(C(), m_stoppingcondition, &m_solutionproperties, m_verbosity > 0);
		model.setStructure(w);
	}
};


}
#endif
