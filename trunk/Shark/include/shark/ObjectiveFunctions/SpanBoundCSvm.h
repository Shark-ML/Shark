/*!
 *
 *  \brief Approximate version of the span-bound for C-SVMs
 *
 *  \author T.Glasmachers
 *  \date 2010-2012
 *
 *  \par Copyright (c) 2010-2012:
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_SPANBOUND_CSVM_H
#define SHARK_OBJECTIVEFUNCTIONS_SPANBOUND_CSVM_H


#include <shark/ObjectiveFunctions/DataObjectiveFunction.h>
#include <shark/Algorithms/Trainers/SvmTrainer.h>


namespace shark {

///
/// \brief Approximate version of the span-bound for C-SVMs
///
template<class InputType = RealVector>
class SpanBoundCSvm : public SupervisedObjectiveFunction<InputType, unsigned int>
{
protected:
	typedef SupervisedObjectiveFunction<InputType, unsigned int> base_type;
	typedef LabeledData<InputType, unsigned int> DatasetType;
	typedef CSvmTrainer<InputType> TrainerType;
	typedef AbstractKernelFunction<InputType> KernelType;

	const DatasetType* mep_dataset;
	IParameterizable* mep_meta;
	TrainerType* mep_trainer;
	bool m_withOffset;

public:
	/// \brief Constructor.
	///
	/// \par
	/// Don't forget to call setDataset before using the object.
	SpanBoundCSvm(TrainerType* trainer, bool withOffset)
	: mep_dataset(NULL)
	, mep_trainer(trainer)
	, m_withOffset(withOffset)
	{
		base_type::m_features |= base_type::HAS_VALUE;
		base_type::m_features |= base_type::HAS_FIRST_DERIVATIVE;
	}

	/// \brief Constructor.
	SpanBoundCSvm(DatasetType const& dataset, TrainerType* trainer, bool withOffset)
	: mep_dataset(&dataset)
	, mep_trainer(trainer)
	, m_withOffset(withOffset)
	{
		base_type::m_features |= base_type::HAS_VALUE;
		base_type::m_features |= base_type::HAS_FIRST_DERIVATIVE;
	}


	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "SpanBoundCSvm"; }

	/// inherited from SupervisedObjectiveFunction
	void setDataset(DatasetType const& dataset)
	{ mep_dataset = &dataset; }

	/// Evaluate the span bound approximation.
	double eval() const
	{
		SHARK_ASSERT(mep_dataset != NULL);
		this->m_evaluationCounter++;

		double C = mep_trainer->C();
		const KernelType* kernel = mep_trainer->kernel();

		// train the basic SVM
		KernelExpansion<InputType> svm;
		mep_trainer->train(&svm, *mep_dataset);

		// find free support vectors, count bounded support vectors
		std::size_t examples = mep_dataset->size();
		std::size_t free_sv = 0;
		std::size_t bounded_sv = 0;
		std::vector<double> alpha;
		std::vector<std::size_t> map;
		for (std::size_t i=0; i<examples; i++)
		{
			double a = svm.alpha(i, 0);
			if (a != 0.0)
			{
				if (a == C || a == -C) bounded_sv++;
				else
				{
					alpha.push_back(a);
					map.push_back(i);
					free_sv++;
				}
			}
		}

		// compute the squared span S2
		RealVector S2(free_sv);
		RealMatrix tildeK;
		if (m_withOffset) tildeK.resize(free_sv + 1, free_sv + 1);
		else tildeK.resize(free_sv, free_sv);
		for (i=0; i<free_sv; i++)
		{
			InputType& input_i = mep_dataset->input(map[i]);
			for (j=0; j<i; j++)
			{
				InputType& input_j = mep_dataset->input(map[j]);
				double k = kernel->eval(input_i, input_j);
				tildeK(i, j) = tildeK(j, i) = k;
			}
			double k = kernel->eval(input_i, input_i);
			tildeK(i, i) = k;
		}
		if (m_withOffset)
		{
			for (i=0; i<free_sv; i++) tildeK(i, free_sv) = tildeK(free_sv, i) = 1.0;
			tildeK(free_sv, free_sv) = 0.0;
		}

		RealMatrix tildeKinv = invert(tildeK);
		for (i=0; i<sv; i++) S2(i) = 1.0 / tildeKinv(i, i);

		// return the span bound value
		double sum = bounded_sv;
		for (std::size_t i=0; i<free_sv; i++) sum += alpha[i] * S2(i);
		return (sum / examples);
	}
	
	std::size_t numberOfVariables()const{
		return mep_trainer->numberOfParameters();
	}

	/// Evaluate the span bound approximation for the given
	/// parameters, passed to the trainer object. These
	/// parameters describe the regularization constant
	/// and the kernel parameters.
	double eval(const RealVector& parameters) const {
		SHARK_ASSERT(mep_meta != NULL);
		mep_trainer->setParameterVector(parameters);
		return eval();
	}

	/// Compute the derivative of the span bound approximation.
	/// Note that the span bound is only piece-wise differentiable,
	/// and that the jumps between the differentialbe regions may
	/// render gradient information useless.
	double evalDerivative(const RealVector& parameters, FirstOrderDerivative& derivative) const
	{
		SHARK_ASSERT(mep_dataset != NULL);
		this->m_evaluationCounter++;

		double C = mep_trainer->C();
		const KernelType* kernel = mep_trainer->kernel();
		std::size_t p, pc = kernel->numberOfParameters();

		// train the basic SVM
		KernelExpansion<InputType> svm;
		mep_trainer->train(&svm, *mep_dataset);

		// find free support vectors, count bounded support vectors
		std::size_t examples = mep_dataset->size();
		std::size_t free_sv = 0;
		std::size_t bounded_sv = 0;
		std::vector<double> alpha;
		std::vector<std::size_t> map;
		for (std::size_t i=0; i<examples; i++)
		{
			double a = svm.alpha(i, 0);
			if (a != 0.0)
			{
				if (a == C || a == -C) bounded_sv++;
				else
				{
					alpha.push_back(a);
					map.push_back(i);
					free_sv++;
				}
			}
		}

		// compute the squared span S2
		// and remember the kernel derivatives
		RealVector S2(free_sv);
		std::size_t matdim = m_withOffset ? free_sv + 1 : free_sv;
		RealMatrix tildeK(matdim, matdim);
		std::vector<RealVector> der(free_sv * (free_sv+1) / 2, RealVector(pc));
		std::size_t d = 0;
		for (i=0; i<free_sv; i++)
		{
			InputType& input_i = mep_dataset->input(map[i]);
			for (j=0; j<i; j++)
			{
				InputType& input_j = mep_dataset->input(map[j]);
				double k = kernel->eval(input_i, input_j);
				tildeK(i, j) = tildeK(j, i) = k;

				kernel->parameterDerivative(input_i, input_j, der[d]);
				d++;
			}
			double k = kernel->eval(input_i, input_i);
			tildeK(i, i) = k;

			kernel->parameterDerivative(input_i, input_i, der[d]);
			d++;
		}
		if (m_withOffset)
		{
			for (i=0; i<free_sv; i++) tildeK(i, free_sv) = tildeK(free_sv, i) = 1.0;
			tildeK(free_sv, free_sv) = 0.0;
		}

		RealMatrix tildeKinv = invert(tildeK);
		for (i=0; i<sv; i++) S2(i) = 1.0 / tildeKinv(i, i);

		// compute the derivatives of the squared span
		// w.r.t. all kernel parameters
		RealMatrix D(matdim, matdim);
		RealMatrix S2der(free_sv, pc);
		if (m_withOffset)
		{
			for (std::size_t i=0; i<matdim; i++) D(free_sv, i) = D(i, free_sv) = 0.0;
		}
		for (p=0; p<pc; p++)
		{
			std::size_t d = 0;
			for (std::size_t i=0; i<free_sv; i++)
			{
				for (std::size_t j=0; j<i; j++)
				{
					D(i, j) = D(j, i) = der[d](p);
					d++;
				}
				D(i, i) = der[d](p);
				d++;
			}
			Matrix tmp = prod(prod(tildeKinv, D), tildeKinv);
			for (std::size_t i=0; i<free_sv; i++) S2der(i, p) = S2(i) * S2(i) * tmp(i, i);
		}





// ***BEGIN*** NOT PORTED YET
		// compute the derivative of the alpha parameters
		// w.r.t. the kernel parameters and C
		const Array<double>& alphaDer = csvm->PrepareDerivative();

		// compose the derivative
		derivative.resize(pc + 1, false);
		for (p=0; p<pc; p++)
		{
			double sum = 0.0;
			for (i=0; i<sv; i++) sum += alpha[i] * S2der(i, p) + alphaDer(map[i], p + 1) * S2(i);
			derivative(p + 1) = sum;
		}
		double sum = 0.0;
		for (i=0; i<sv; i++) sum += alphaDer(map[i], 0) * S2(i);
		derivative(0) = sum;
		derivative /= (double)examples;
// ***END*** NOT PORTED YET






		// return the span bound value
		double sum = bounded_sv;
		for (std::size_t i=0; i<free_sv; i++) sum += alpha[i] * S2(i);
		return (sum / examples);
	}
};


}
#endif
