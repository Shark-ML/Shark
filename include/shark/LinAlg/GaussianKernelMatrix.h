//===========================================================================
/*!
 *
 *
 * \brief       Efficient special case if the kernel is gaussian and the inputs are sparse vectors
 *
 *
 * \par
 *
 *
 *
 * \author      T. Glasmachers
 * \date        2007-2012
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


#ifndef SHARK_LINALG_GAUSSIANKERNELMATRIX_H
#define SHARK_LINALG_GAUSSIANKERNELMATRIX_H

#include <shark/Data/Dataset.h>
#include <shark/Data/DataView.h>
#include <shark/LinAlg/Base.h>

#include <vector>
#include <cmath>


namespace shark {


///\brief Efficient special case if the kernel is Gaussian and the inputs are sparse vectors
template <class T, class CacheType>
class GaussianKernelMatrix {
public:
	typedef CacheType QpFloatType;
	typedef T InputType;

	/// Constructor
	/// \param gamma   bandwidth parameter of Gaussian kernel
	/// \param data    data evaluated by the kernel function
	GaussianKernelMatrix(
		double gamma,
		Data<InputType> const &data
	):m_data(data)
	, m_squaredNorms(m_data.size())
	, m_gamma(gamma)
	, m_accessCounter(0) {
		for (std::size_t i = 0; i != m_data.size(); ++i) {
			m_squaredNorms(i) = norm_sqr(m_data[i]); //precompute the norms
		}
	}

	/// return a single matrix entry
	QpFloatType operator()(std::size_t i, std::size_t j) const
	{ return entry(i, j); }

	/// return a single matrix entry
	QpFloatType entry(std::size_t i, std::size_t j) const {
		++m_accessCounter;
		double distance = m_squaredNorms(i)-2*inner_prod(m_data[i], m_data[j])+m_squaredNorms(j);
		return (QpFloatType)std::exp(- m_gamma * distance);
	}

	/// \brief Computes the i-th row of the kernel matrix.
	///
	///The entries start,...,end of the i-th row are computed and stored in storage.
	///There must be enough room for this operation preallocated.
	void row(std::size_t i, std::size_t start, std::size_t end, QpFloatType *storage) const {
		auto const& xi = m_data[i];
		m_accessCounter +=end-start;
		SHARK_PARALLEL_FOR(int j = start; j < (int) end; j++) {
			double distance = m_squaredNorms(i)-2*inner_prod(xi, m_data[j])+m_squaredNorms(j);
			storage[j-start] = std::exp(- m_gamma * distance);
		}
	}

	/// \brief Computes the kernel-matrix
	template<class M>
	void matrix(
		blas::matrix_expression<M, blas::cpu_tag> &storage
	) const {
		for (std::size_t i = 0; i != size(); ++i) {
			row(i, 0, size(), &storage()(i, 0));
		}
	}

	/// swap two variables
	void flipColumnsAndRows(std::size_t i, std::size_t j) {
		using std::swap;
		m_data.swapElements(i,j);
		swap(m_squaredNorms[i], m_squaredNorms[j]);
	}

	/// return the size of the quadratic matrix
	std::size_t size() const
	{ return m_data.size(); }

	/// query the kernel access counter
	unsigned long long getAccessCount() const
	{ return m_accessCounter; }

	/// reset the kernel access counter
	void resetAccessCount()
	{ m_accessCounter = 0; }

protected:

	DataView<Data<InputType> const> m_data;
	RealVector m_squaredNorms;

	double m_gamma;

	/// counter for the kernel accesses
	mutable unsigned long long m_accessCounter;
};

}
#endif
