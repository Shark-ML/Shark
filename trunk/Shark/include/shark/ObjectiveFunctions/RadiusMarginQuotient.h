/*!
 *
 *  \brief Radius Margin Quotient for SVM model selection
 *
 *  \author T.Glasmachers, O.Krause
 *  \date 2012
 *
 *  \par Copyright (c) 2007-2011:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_RADIUSMARGINQUOTIENT_H
#define SHARK_OBJECTIVEFUNCTIONS_RADIUSMARGINQUOTIENT_H


#include <shark/ObjectiveFunctions/DataObjectiveFunction.h>
#include <shark/Algorithms/QP/QpSvmDecomp.h>
#include <shark/Models/Kernels/KernelHelpers.h>

namespace shark {


///
/// \brief radius margin quotions for binary SVMs
///
/// \par
/// The RadiusMarginQuotient is the quotient \f$ R^2 / \rho^2 \f$
/// of the radius R of the smallest sphere containing the
/// training data and the margin \f$\rho\f$ of a binary hard margin
/// support vector machine. Both distances depend on the
/// kernel function, and thus on its parameters.
/// The radius margin quotient is a common objective
/// function for the adaptation of the kernel parameters
/// of a binary hard-margin SVM.
///
template<class InputType, class CacheType = float>
class RadiusMarginQuotient : public SupervisedObjectiveFunction<InputType, unsigned int>
{
public:

	//////////////////////////////////////////////////////////////////
	// The types below define the type used for caching kernel values. The default is float,
	// since this type offers sufficient accuracy in the vast majority of cases, at a memory
	// cost of only four bytes. However, the type definition makes it easy to use double instead
	// (e.g., in case high accuracy training is needed).
	typedef CacheType QpFloatType;
	typedef blas::matrix<QpFloatType> QpMatrixType;
	typedef blas::matrix_row<QpMatrixType> QpMatrixRowType;
	typedef blas::matrix_column<QpMatrixType> QpMatrixColumnType;

	typedef KernelMatrix<InputType, QpFloatType> KernelMatrixType;
	typedef CachedMatrix< KernelMatrixType > CachedMatrixType;

	typedef SupervisedObjectiveFunction<InputType, unsigned int> base_type;
	typedef LabeledData<InputType, unsigned int> DatasetType;
	typedef VectorSpace<double>::PointType PointT;
	typedef AbstractKernelFunction<InputType> KernelType;
	typedef typename base_type::FirstOrderDerivative FirstOrderDerivative;

	/// \brief Constructor.
	///
	/// \par
	/// Don't forget to call setDataset before using the object.
	RadiusMarginQuotient(KernelType* kernel)
	: mep_kernel(kernel)
	{
		this->m_name = "RadiusMarginQuotient";
		this->m_features |= base_type::HAS_VALUE;
		if (mep_kernel->hasFirstParameterDerivative())
			this->m_features |= base_type::HAS_FIRST_DERIVATIVE;
	}

	/// \brief Constructor.
	RadiusMarginQuotient(DatasetType const& dataset, KernelType* kernel)
	: mep_kernel(kernel)
	{
		setDataset(dataset);

		this->m_name = "RadiusMarginQuotient";
		this->m_features |= base_type::HAS_VALUE;
		if (mep_kernel->hasFirstParameterDerivative())
			this->m_features |= base_type::HAS_FIRST_DERIVATIVE;
	}


	/// \brief Make labeled data known to the radius margin objective function.
	void setDataset(DatasetType const& dataset)
	{ m_dataset = dataset; }
	
	std::size_t numberOfVariables()const{
		return mep_kernel->numberOfParameters();
	}

	/// \brief Evaluate the radius margin quotient.
	///
	/// \par
	/// The parameters are passed into the kernel, and the
	/// radius-margin quotient is computed w.r.t. the
	/// kernel-induced metric.
	double eval(PointT const& parameters) const{
		SIZE_CHECK(parameters.size() == mep_kernel->numberOfParameters());
		SHARK_CHECK(! m_dataset.empty(), "[RadiusMarginQuotient::eval] call setDataset first");
		this->m_evaluationCounter++;
		
		
		mep_kernel->setParameterVector(parameters);

		std::size_t ell = m_dataset.numberOfElements();
		KernelMatrixType km(*mep_kernel, m_dataset.inputs());
		CachedMatrixType cache(&km);
		QpSvmDecomp< CachedMatrixType > solver(cache);
		QpStoppingCondition stop;
		RealVector linear(ell);
		RealVector lower(ell);
		RealVector upper(ell);
		double w2, R2;
		{
			// create and solve a quadratic program with offset
			RealVector alpha(ell, 0.0);
			for (std::size_t i=0; i<ell; i++)
			{
				linear(i) = (m_dataset.element(i).label == 0) ? +1.0 : -1.0;
				lower(i) = (m_dataset.element(i).label == 0) ? 0.0 : -1e100;
				upper(i) = (m_dataset.element(i).label == 0) ? +1e100 : 0.0;
			}
			QpSolutionProperties prop;
			solver.solve(linear, lower, upper, alpha, stop, &prop);
			w2 = 2.0 * prop.value;
		}
		{
			// create and solve the radius problem (also a quadratic program)
			RealVector beta(ell, 1.0 / (double)ell);
			for (std::size_t i=0; i<ell; i++)
			{
				linear(i) = 0.5 * km(i, i);
				lower(i) = 0.0;
				upper(i) = 1.0;
			}
			QpSolutionProperties prop;
			solver.solve(linear, lower, upper, beta, stop, &prop);
			R2 = 2.0 * prop.value;
		}

		return (w2 * R2);
	}

	/// \brief Evaluate the radius margin quotient and its first derivative.
	///
	/// \par
	/// The parameters are passed into the kernel, and the
	/// radius-margin quotient and its derivative are computed
	/// w.r.t. the kernel-induced metric.
	double evalDerivative(PointT const& parameters, FirstOrderDerivative& derivative) const{
		SHARK_CHECK(! m_dataset.empty(), "[RadiusMarginQuotient::evalDerivative] call setDataset first");
		SIZE_CHECK(parameters.size() == mep_kernel->numberOfParameters());
		this->m_evaluationCounter++;

		//~ std::size_t kc = mep_kernel->numberOfParameters();
		mep_kernel->setParameterVector(parameters);

		std::size_t ell = m_dataset.numberOfElements();
		KernelMatrixType km(*mep_kernel, m_dataset.inputs());
		CachedMatrixType cache(&km);
		QpSvmDecomp< CachedMatrixType > solver(cache);
		QpStoppingCondition stop;
		RealVector linear(ell);
		RealVector lower(ell);
		RealVector upper(ell);
		RealVector alpha(ell, 0.0);
		RealVector beta(ell, 1.0 / (double)ell);
		double w2, R2;
		{
			// create and solve a quadratic program with offset
			for (std::size_t i=0; i<ell; i++)
			{
				linear(i) = (m_dataset.element(i).label == 0) ? +1.0 : -1.0;
				lower(i) = (m_dataset.element(i).label == 0) ? 0.0 : -1e100;
				upper(i) = (m_dataset.element(i).label == 0) ? +1e100 : 0.0;
			}
			QpSolutionProperties prop;
			solver.solve(linear, lower, upper, alpha, stop, &prop);
			w2 = 2.0 * prop.value;
		}
		{
			// create and solve the radius problem (also a quadratic program)
			for (std::size_t i=0; i<ell; i++){
				linear(i) = 0.5 * km(i, i);
				lower(i) = 0.0;
				upper(i) = 1.0;
			}
			QpSolutionProperties prop;
			solver.solve(linear, lower, upper, beta, stop, &prop);
			R2 = 2.0 * prop.value;
		}

		//~ RealVector dw2(kc, 0.0);
		//~ RealVector dR2(kc, 0.0);
		//~ RealVector dkv(kc, 0.0);
		//~ boost shared_ptr<State> state = mep_kernel->createState();
		//~ for (std::size_t i=0; i<ell; i++){
			//~ double ai = alpha(i);
			//~ double bi = beta(i);
			//~ for (std::size_t j=0; j<ell; j++){
				//~ double aj = alpha(j);
				//~ double bj = beta(j);
			
				//~ mep_kernel->eval(m_dataset(i).input, m_dataset(j).input,*state);
				//~ mep_kernel->parameterDerivative(m_dataset(i).input, m_dataset(j).input,*state,dkv);
				//~ if (i == j){
					//~ for (std::size_t k=0; k<kc; k++) 
						//~ dR2(k) += bi * dkv(k);
				//~ }
				//~ for (std::size_t k=0; k<kc; k++)
				//~ {
					//~ dw2(k) -= ai * aj * dkv(k);
					//~ dR2(k) -= bi * bj * dkv(k);
				//~ }
			//~ }
		//~ }

		//~ derivative.m_gradient.resize(kc);
		//~ for (std::size_t k=0; k<kc; k++) 
			//~ derivative.m_gradient(k) = R2 * dw2(k) + w2 * dR2(k);
		
		RealDiagonalMatrix diagBeta(ell);
		for(std::size_t i = 0; i != ell; ++i){
			diagBeta(i,i) = beta(i);
		}
		derivative.m_gradient = calculateKernelMatrixParameterDerivative(
			*mep_kernel,
			m_dataset.inputs(),
			w2*(diagBeta-outer_prod(beta,beta))-R2*outer_prod(alpha,alpha)
		);
		
		
		return (w2 * R2);
	}

protected:
	DatasetType m_dataset;                  ///< labeled data for radius and (hard) margin computation
	KernelType* mep_kernel;            ///< underlying parameterized kernel object
};


}
#endif
