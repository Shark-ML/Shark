//===========================================================================
/*!
 * 
 *
 * \brief       Kernel matrix for SVM ranking.
 * 
 * 
 * \par
 * 
 * 
 *
 * \author      T. Glasmachers
 * \date        2016
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://shark-ml.org/>
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


#ifndef SHARK_LINALG_DIFFERENCEKERNELMATRIX_H
#define SHARK_LINALG_DIFFERENCEKERNELMATRIX_H

#include <shark/Data/Dataset.h>
#include <shark/Data/DataView.h>
#include <shark/LinAlg/Base.h>

#include <vector>
#include <utility>
#include <cmath>


namespace shark {


///
/// \brief SVM ranking matrix
///
/// \par
/// The DifferenceKernelMatrix class is kernel matrix with entries of
/// the form \f$ K_{i,j} = k(g_i, g_j) - k(g_i, s_j) - k(s_i, g_j) + k(s_i, s_j) \f$
/// where for data consisting of pairs of point \f$ (g_i, s_i) \f$.
/// This matrix form is needed in SVM ranking problems.
///
template <class InputType, class CacheType>
class DifferenceKernelMatrix
{
public:
    typedef CacheType QpFloatType;

	/// Constructor.
	DifferenceKernelMatrix(
				AbstractKernelFunction<InputType> const& kernel,
				Data<InputType> const& dataset,
				std::vector<std::pair<std::size_t, std::size_t>> const& pairs)
	: m_kernel(kernel)
	, m_dataset(dataset)
	, m_indices(pairs.size())
	{
		DataView< Data<InputType> const > view(dataset);
		for (std::size_t i=0; i<pairs.size(); i++)
		{
			std::pair<std::size_t, std::size_t> const& p = pairs[i];
			m_indices[i] = std::make_tuple(
					view.batch(p.first), view.positionInBatch(p.first),
					view.batch(p.second), view.positionInBatch(p.second));
		}
	}


	/// return a single matrix entry
	QpFloatType operator () (std::size_t i, std::size_t j) const
	{ return entry(i, j); }

	/// return a single matrix entry
	QpFloatType entry(std::size_t i, std::size_t j) const
	{
		std::tuple<std::size_t, std::size_t, std::size_t, std::size_t> const& pi = m_indices[i];
		std::tuple<std::size_t, std::size_t, std::size_t, std::size_t> const& pj = m_indices[j];
		std::size_t batch_si = std::get<0>(pi);
		std::size_t index_si = std::get<1>(pi);
		std::size_t batch_gi = std::get<2>(pi);
		std::size_t index_gi = std::get<3>(pi);
		std::size_t batch_sj = std::get<0>(pj);
		std::size_t index_sj = std::get<1>(pj);
		std::size_t batch_gj = std::get<2>(pj);
		std::size_t index_gj = std::get<3>(pj);
		typedef typename Data<InputType>::const_element_reference reference;
		reference si = getBatchElement(m_dataset.batch(batch_si), index_si);
		reference gi = getBatchElement(m_dataset.batch(batch_gi), index_gi);
		reference sj = getBatchElement(m_dataset.batch(batch_sj), index_sj);
		reference gj = getBatchElement(m_dataset.batch(batch_gj), index_gj);
		double k_gi_gj = m_kernel(gi, gj);
		double k_gi_sj = m_kernel(gi, sj);
		double k_si_gj = m_kernel(si, gj);
		double k_si_sj = m_kernel(si, sj);
		return (k_gi_gj - k_gi_sj - k_si_gj + k_si_sj);
	}

	/// \brief Computes the i-th row of the kernel matrix.
	///
	/// The entries start,...,end of the i-th row are computed and stored in storage.
	/// There must be enough room for this operation preallocated.
	void row(std::size_t i, std::size_t start, std::size_t end, QpFloatType* storage) const {
		for (std::size_t j = start; j < end; j++) storage[j-start] = entry(i, j);
	}

	/// \brief Computes the kernel-matrix
	template<class M>
	void matrix(blas::matrix_expression<M, blas::cpu_tag>& storage) const {
		for (std::size_t i = 0; i != size(); ++i) {
			for (std::size_t j = 0; j != size(); ++j) {
				storage()(i, j) = entry(i, j);
			}
		}
	}

	/// swap two variables
	void flipColumnsAndRows(std::size_t i, std::size_t j)
	{
		using namespace std;
		swap(m_indices[i], m_indices[j]);
	}

    /// return the size of the quadratic matrix
    std::size_t size() const
    { return m_indices.size(); }

protected:
	/// underlying kernel function
	AbstractKernelFunction<InputType> const& m_kernel;

	/// underlying set of points
	Data<InputType> const& m_dataset;

	/// pairs of points defining the matrix components
	std::vector<std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>> m_indices;
};

}
#endif
