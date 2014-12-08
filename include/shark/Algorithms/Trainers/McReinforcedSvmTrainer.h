//===========================================================================
/*!
 * 
 *
 * \brief       Trainer for the Reinforced Multi-class Support Vector Machine
 * 
 * 
 * 
 *
 * \author      T. Glasmachers
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
//===========================================================================


#ifndef SHARK_ALGORITHMS_MCREINFORCEDSVMTRAINER_H
#define SHARK_ALGORITHMS_MCREINFORCEDSVMTRAINER_H


#include <shark/Algorithms/Trainers/AbstractSvmTrainer.h>
#include <shark/Algorithms/QP/QpMcBoxDecomp.h>

#include <shark/LinAlg/KernelMatrix.h>
#include <shark/LinAlg/CachedMatrix.h>
#include <shark/LinAlg/PrecomputedMatrix.h>


namespace shark {


///
/// \brief Training of reinforced-SVM for multi-category classification.
///
/// The reinforces SVM was introduced in the article<br/>
/// <p>Reinforced multicategory support vector machines. Liu, Yufeng, and Ming Yuan. Journal of Computational and Graphical Statistics 20.4 (2011): 901-919.</p>
/// Its loss function has a parameter gamma which is fixed in this
/// implementation to its default value of 1/2.
///
template <class InputType, class CacheType = float>
class McReinforcedSvmTrainer : public AbstractSvmTrainer<InputType, unsigned int>
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
	McReinforcedSvmTrainer(KernelType* kernel, double C, bool unconstrained = false)
	: base_type(kernel, C, unconstrained)
	{ }

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "McReinforcedSvmTrainer"; }

	void train(KernelClassifier<InputType>& svm, const LabeledData<InputType, unsigned int>& dataset)
	{
		std::size_t ic = dataset.numberOfElements();
		unsigned int classes = numberOfClasses(dataset);

		// prepare the problem description
		RealMatrix linear(ic, classes, 1.0);
		{
			typename LabeledData<InputType, unsigned int>::LabelContainer const& labels = dataset.labels();
			std::size_t i=0;
			for (std::size_t b=0; b<labels.numberOfBatches(); b++)
			{
				typename LabeledData<InputType, unsigned int>::LabelContainer::const_batch_reference batch = labels.batch(b);
				for (std::size_t e=0; e<boost::size(batch); e++)
				{
					unsigned int const& l = shark::get(batch, e);
					linear(i, l) = classes - 1.0;   // self-margin target value of reinforced SVM loss
					i++;
				}
			}
		}

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
		
		RealMatrix alpha(ic,classes,0.0);
		RealVector bias(classes,0.0);
		// solve the problem
		if (base_type::precomputeKernel())
		{
			PrecomputedMatrixType matrix(&km);
			QpMcBoxDecomp< PrecomputedMatrixType> problem(matrix, M, dataset.labels(), linear, this->C());
			QpSolutionProperties& prop = base_type::m_solutionproperties;
			problem.setShrinking(base_type::m_shrinking);
			if(this->m_trainOffset){
				BiasSolver<PrecomputedMatrixType> biasSolver(&problem);
				biasSolver.solve(bias,base_type::m_stoppingcondition,nu);
			}
			else{
				QpSolver<QpMcBoxDecomp< PrecomputedMatrixType> > solver(problem);
				solver.solve( base_type::m_stoppingcondition, &prop);
			}
			alpha = problem.solution();
		}
		else
		{
			CachedMatrixType matrix(&km, base_type::m_cacheSize);
			QpMcBoxDecomp< CachedMatrixType> problem(matrix, M, dataset.labels(), linear, this->C());
			QpSolutionProperties& prop = base_type::m_solutionproperties;
			problem.setShrinking(base_type::m_shrinking);
			if(this->m_trainOffset){
				BiasSolver<CachedMatrixType> biasSolver(&problem);
				biasSolver.solve(bias,base_type::m_stoppingcondition,nu);
			}
			else{
				QpSolver<QpMcBoxDecomp< CachedMatrixType> > solver(problem);
				solver.solve( base_type::m_stoppingcondition, &prop);
			}
			alpha = problem.solution();
		}
		
		svm.decisionFunction().setStructure(this->m_kernel,dataset.inputs(),this->m_trainOffset,classes);

		// write the solution into the model
		for (std::size_t i=0; i<ic; i++)
		{
			unsigned int y = dataset.element(i).label;
			for (unsigned int c=0; c<classes; c++)
			{
				double sum = 0.0;
				unsigned int r = classes * y;
				for (unsigned int p=0; p<classes; p++, r++)
					sum += nu(r, c) * alpha(i, p);
				svm.decisionFunction().alpha(i,c) = sum;
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
class LinearMcSvmReinforcedTrainer : public AbstractLinearSvmTrainer<InputType>
{
public:
	typedef AbstractLinearSvmTrainer<InputType> base_type;

	LinearMcSvmReinforcedTrainer(double C, bool unconstrained = false)
	: AbstractLinearSvmTrainer<InputType>(C, unconstrained){ }

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "LinearMcSvmReinforcedTrainer"; }

	void train(LinearClassifier<InputType>& model, const LabeledData<InputType, unsigned int>& dataset)
	{
		std::size_t dim = inputDimension(dataset);
		std::size_t classes = numberOfClasses(dataset);

		QpMcLinearReinforced<InputType> solver(dataset, dim, classes);
		RealMatrix w = solver.solve(this->C(), this->stoppingCondition(), &this->solutionProperties(), this->verbosity() > 0);
		model.decisionFunction().setStructure(w);
	}
};


}
#endif
