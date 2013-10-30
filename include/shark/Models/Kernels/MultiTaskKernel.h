//===========================================================================
/*!
*
*  \brief Special kernel classes for multi-task and transfer learning.
*
*  \author  T. Glasmachers, O.Krause
*  \date    2012
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

#ifndef SHARK_MODELS_KERNELS_MULTITASKKERNEL_H
#define SHARK_MODELS_KERNELS_MULTITASKKERNEL_H

#include <shark/Models/Kernels/DiscreteKernel.h>
#include <shark/Models/Kernels/ProductKernel.h>
#include <shark/Data/Dataset.h>
#include "Impl/MklKernelBase.h"

namespace shark {

///
/// \brief Aggregation of input data and task index.
///
/// \par
/// Generic data structure for augmenting arbitrary data
/// with an integer. This integer is typically used as a
/// task identifier in multi-task and transfer learning.
///
template <class InputTypeT>
struct MultiTaskSample : public ISerializable
{
	typedef InputTypeT InputType;
	/// \brief Default constructor.
	MultiTaskSample()
	{ }

	/// \brief Construction from an input and a task index
	MultiTaskSample(InputType const& i, std::size_t t)
	: input(i), task(t)
	{ }

	void read(InArchive& ar){
		ar >> input;
		ar >> task;
	}

	void write(OutArchive& ar) const{
		ar << input;
		ar << task;
	}

	InputType input;                ///< input data
	std::size_t task;               ///< task index

};
}


#ifndef DOXYGEN_SHOULD_SKIP_THIS

    BOOST_FUSION_ADAPT_TPL_STRUCT(
        (InputType),
        (shark::MultiTaskSample) (InputType),
        (InputType, input)(std::size_t, task)
    )

#endif /* DOXYGEN_SHOULD_SKIP_THIS */


namespace shark{


template<class InputType>
struct Batch< MultiTaskSample<InputType> >{
	SHARK_CREATE_BATCH_INTERFACE(
		MultiTaskSample<InputType>,
		(InputType, input)(std::size_t, task)
	)
};


///
/// \brief Special "Gaussian-like" kernel function on tasks.
///
/// \par
/// See<br/>
/// Learning Marginal Predictors: Transfer to an Unlabeled Task.
/// G. Blanchard, G. Lee, C. Scott.
///
/// \par
/// This class computes a Gaussian kernel based on the distance
/// of empirical distributions in feature space induced by yet
/// another kernel. This is useful for multi-task and transfer
/// learning. It reduces the definition of a kernel on tasks to
/// that of a kernel on inputs, plus a single bandwidth parameter
/// for the Gaussian kernel of distributions.
///
/// \par
/// Given unlabaled data \f$ x_i, t_i \f$ where the x-component
/// is an input and the t-component is a task index, the kernel
/// on tasks t and t' is defined as
/// \f[
///     k(t, t') = \exp \left( -\gamma \cdot \left\| \frac{1}{\ell_{t}\ell{t'}} \sum_{i | t_i = t}\sum_{j | t_j = t'} k'(x_i, x_j) \right\|^2 \right)
/// \f]
/// where k' is an arbitrary kernel on inputs.
///
template <class InputTypeT >
class GaussianTaskKernel : public DiscreteKernel
{
private:
	typedef DiscreteKernel base_type;
public:
	typedef InputTypeT InputType;
	typedef MultiTaskSample<InputType> MultiTaskSampleType;
	typedef AbstractKernelFunction<InputType> KernelType;

	/// \brief Construction of a Gaussian kernel on tasks.
	///
	/// \param  data         unlabeled data from multiple tasks
	/// \param  tasks        number of tasks in the problem
	/// \param  inputkernel  kernel on inputs based on which task similarity is defined
	/// \param  gamma        Gaussian bandwidth parameter (also refer to the member functions setGamma and setSigma).
	GaussianTaskKernel(
			Data<MultiTaskSampleType> const& data,
			std::size_t tasks,
			KernelType& inputkernel,
			double gamma)
	: DiscreteKernel(RealMatrix(tasks, tasks,0.0))
	, m_data(data)
	, m_inputkernel(inputkernel)
	, m_gamma(gamma){
		computeMatrix();
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "GaussianTaskKernel"; }

	RealVector parameterVector() const
	{
		const std::size_t n = m_inputkernel.numberOfParameters();
		RealVector ret(n + 1);
		init(ret)<<parameters(m_inputkernel),m_gamma;
		return ret;
	}

	void setParameterVector(RealVector const& newParameters){
		init(newParameters)>>parameters(m_inputkernel),m_gamma;
		computeMatrix();
	}

	std::size_t numberOfParameters() const{
		return m_inputkernel.numberOfParameters() + 1;
	}

	std::size_t numberOfTasks() const
	{ return size(); }

	/// \brief Kernel bandwidth parameter.
	double gamma() const
	{ return m_gamma; }

	/// \brief Kernel width parameter, equivalent to the bandwidth parameter.
	///
	/// The bandwidth gamma and the width sigma are connected: \f$ gamma = 1 / (2 \cdot sigma^2) \f$.
	double sigma() const
	{ return (1.0 / std::sqrt(2 * m_gamma)); }

	// \brief Set the kernel bandwidth parameter.
	void setGamma(double gamma)
	{
		SHARK_ASSERT(gamma > 0.0);
		m_gamma = gamma;
	}

	/// \brief Set the kernel width (equivalent to setting the bandwidth).
	///
	/// The bandwidth gamma and the width sigma are connected: \f$ gamma = 1 / (2 \cdot sigma^2) \f$.
	void setWidth(double sigma)
	{
		SHARK_ASSERT(sigma > 0.0);
		m_gamma = 1.0 / (2.0 * sigma * sigma);
	}

	/// From ISerializable.
	void read(InArchive& ar)
	{
		base_type::read(ar);
		ar >> m_gamma;
	}

	/// From ISerializable.
	void write(OutArchive& ar) const
	{
		base_type::write(ar);
		ar << m_gamma;
	}

protected:

	/// \brief Compute the Gram matrix of the task kernel.
	///
	/// \par
	/// Here is the real meat. This function implements the
	/// kernel function defined in<br/>
	/// Learning Marginal Predictors: Transfer to an Unlabeled Task.
	/// G. Blanchard, G. Lee, C. Scott.
	///
	/// \par
	/// In a first step the function computes the inner products
	/// of the task-wise empirical distributions, represented by
	/// their mean elements in the kernel-induced feature space.
	/// In a second step this information is used for the computation
	/// of squared distances between empirical distribution, which
	/// allows for the straightforward computation of a Gaussian
	/// kernel.
	void computeMatrix()
	{
		// count number of examples for each task
		const std::size_t tasks = numberOfTasks();
		std::size_t elements = m_data.numberOfElements();
		std::vector<std::size_t> ell(tasks, 0);
		for (std::size_t i=0; i<elements; i++)
			ell[m_data.element(i).task]++;

		// compute inner products between mean elements of empirical distributions
		for (std::size_t i=0; i<elements; i++)
		{
			const std::size_t task_i = m_data.element(i).task;
			for (std::size_t j=0; j<i; j++)
			{
				const std::size_t task_j = m_data.element(j).task;
				const double k = m_inputkernel.eval(m_data.element(i).input, m_data.element(j).input);
				base_type::m_matrix(task_i, task_j) += k;
				base_type::m_matrix(task_j, task_i) += k;
			}
			const double k = m_inputkernel.eval(m_data.element(i).input, m_data.element(i).input);
			base_type::m_matrix(task_i, task_i) += k;
		}
		for (std::size_t i=0; i<tasks; i++)
		{
			if (ell[i] == 0) continue;
			for (std::size_t j=0; j<tasks; j++)
			{
				if (ell[j] == 0) continue;
				base_type::m_matrix(i, j) /= (double)(ell[i] * ell[j]);
			}
		}

		// compute Gaussian kernel
		for (std::size_t i=0; i<tasks; i++)
		{
			const double norm2_i = base_type::m_matrix(i, i);
			for (std::size_t j=0; j<i; j++)
			{
				const double norm2_j = base_type::m_matrix(j, j);
				const double dist2 = norm2_i + norm2_j - 2.0 * base_type::m_matrix(i, j);
				const double k = std::exp(-m_gamma * dist2);
				base_type::m_matrix(i, j) = base_type::m_matrix(j, i) = k;
			}
		}
		for (std::size_t i=0; i<tasks; i++) base_type::m_matrix(i, i) = 1.0;
	}


	Data<MultiTaskSampleType > const& m_data;  ///< multi-task data
	KernelType& m_inputkernel;            ///< kernel on inputs
	double m_gamma;                        ///< bandwidth of the Gaussian task kernel
};


///
/// \brief Special kernel function for multi-task and transfer learning.
///
/// \par
/// This class is a convenience wrapper for the product of an
/// input kernel and a kernel on tasks. It also encapsulates
/// the projection from multi-task learning data (see class
/// MultiTaskSample) to inputs and task indices.
///
template <class InputTypeT>
class MultiTaskKernel
: private detail::MklKernelBase<MultiTaskSample<InputTypeT> >
, public ProductKernel< MultiTaskSample<InputTypeT> >
{
private:
	typedef detail::MklKernelBase<MultiTaskSample<InputTypeT> > base_type1;
	typedef ProductKernel< MultiTaskSample<InputTypeT> > base_type2;
public:
	typedef AbstractKernelFunction<InputTypeT> InputKernelType;
	/// \brief Constructor.
	///
	/// \param  inputkernel  kernel on inputs
	/// \param  taskkernel   kernel on task indices
	MultiTaskKernel(
		InputKernelType* inputkernel,
		DiscreteKernel* taskkernel)
	:base_type1(boost::fusion::make_vector(inputkernel,taskkernel))
	,base_type2(base_type1::makeKernelVector())
	{}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "MultiTaskKernel"; }
};

} // namespace shark {

#endif
