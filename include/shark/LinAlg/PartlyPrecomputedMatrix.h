//===========================================================================
/*!
 *
 *
 * \brief       Partly Precomputed version of a matrix for quadratic programming
 *
 *
 * \par
 *
 *
 *
 * \author      T. Glasmachers, A. Demircioglu
 * \date        2007-2014
 *
 *
 * \par Copyright 1995-2015 Shark Development Team
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


#ifndef SHARK_LINALG_PARTLYPRECOMPUTEDMATRIX_H
#define SHARK_LINALG_PARTLYPRECOMPUTEDMATRIX_H

#include <shark/Data/Dataset.h>
#include <shark/LinAlg/Base.h>

#include <vector>
#include <cmath>


namespace shark
{

///
/// \brief Partly Precomputed version of a matrix for quadratic programming
///
/// \par
/// The PartlyPrecomputedMatrix class computes all pairs of kernel
/// evaluations that fits the given cachesize in its constructor and
/// stores them im memory.
///
/// \par
/// Use of this class may be beneficial for certain model
/// selection strategies, where the whole matrix does not fit into
/// memory, and the LRU cache will produce too much hit rates,
/// so that at least partially caching the kernel matrix will help.
/// In particular this will help the KernelSGD/Pegasos algorithm.
///
template <class Matrix>
class PartlyPrecomputedMatrix
{
public:
	typedef typename Matrix::QpFloatType QpFloatType;

	/// Constructor
	/// \param[in]  base    matrix to be cached. it is assumed that this matrix is not precomputed,
	///                                 but the (costy) computation takes place every time an entry is queried.
	/// \param[in]  cachesize       size of the cache to use in bytes. the size of the cached matrix will
	//                                  depend on this value.
	PartlyPrecomputedMatrix(Matrix* base, std::size_t cachesize = 0x4000000)
		: m_cacheSize(cachesize)
		, m_baseMatrix(base)
	{
		if((m_baseMatrix == NULL) || (m_baseMatrix ->size() == 0))
			throw SHARKEXCEPTION("Cannot cache a NULL matrix!");

		// remember the original size of the matrix
		m_originalNumberOfRows = m_baseMatrix -> size();

		// determine how many bytes we need for a single row
		size_t rowSizeBytes = m_originalNumberOfRows * sizeof(QpFloatType);

		// how many rows fit into our cache?
		size_t m_nRows = (size_t) m_cacheSize / rowSizeBytes;
		if(m_nRows < 1)
			throw SHARKEXCEPTION("Cache size is smaller than the size of a row!");

		// if we have more space than needed, well, we do not need it.
		if(m_nRows > m_originalNumberOfRows)
			m_nRows = m_originalNumberOfRows ;

		// resize matrix
		m_cachedMatrix.resize(m_nRows, m_baseMatrix ->size());

		// copy the rows
		for(std::size_t r = 0; r < m_cachedMatrix.size1(); r++)
		{
			for(std::size_t j = 0; j < m_baseMatrix->size(); j++)
			{
				m_cachedMatrix(r, j) = (*m_baseMatrix)(r, j);
			}
		}
	}



	/// return, if a given row is cached
	/// \param[in]  k       row to check
	/// \return     is given row in cached matrix or not?
	bool isCached(std::size_t k) const
	{
		if(k < m_cachedMatrix.size1())
			return true;
		return false;
	}



	/// return a complete row of the matrix.
	/// if the row is cached, it will be returned from there, if not, it will
	/// be recomputed on-the-fly and not stored.
	/// param[in]  k       row to compute
	/// param[in]  storage     vector to store the row. must be the same size as a row!
	void row(std::size_t k, blas::vector<QpFloatType> &storage) const
	{
		RANGE_CHECK(k < m_originalNumberOfRows);
		RANGE_CHECK(0 <= k);
		SIZE_CHECK(storage.size() == m_cachedMatrix.size2());
		if(isCached(k) == true)
		{
			for(std::size_t j = 0; j < m_cachedMatrix.size2(); j++)
			{
				storage[j] = m_cachedMatrix(k, j);
			}
		}
		else
		{
			for(std::size_t j = 0; j < m_cachedMatrix.size2(); j++)
			{
				storage[j] = (*m_baseMatrix)(k, j);
			}
		}
	}



	/// return a single matrix entry
	/// param[in]  i       row of entry
	/// param[in]  j       column entry
	/// @return     value of matrix at given position
	QpFloatType operator()(std::size_t i, std::size_t j) const
	{
		return entry(i, j);
	}



	/// return a single matrix entry
	/// param[in]  i       row of entry
	/// param[in]  j       column entry
	/// @return     value of matrix at given position
	QpFloatType entry(std::size_t i, std::size_t j) const
	{
		// check if we have to compute that or not
		if(isCached(i))
			return m_cachedMatrix(i, j);

		// ok, need to compute that element
		return (*m_baseMatrix)(i, j);
	}



	/// return the number of cached rows
	/// @return     number of rows that are cached
	std::size_t size() const
	{
		return m_cachedMatrix.size();
	}



	/// return size of cached matrix in QpFloatType units
	/// @return     the capacity of the cached matrix in QpFloatType units
	std::size_t getMaxCacheSize()
	{
		return m_cachedMatrix.size() * m_cachedMatrix.size2();
	}



	/// return the dimension of a row in the cache (as we do not shorten our
	/// rows, this must be the same as the dimension of a row in the original kernel matrix).
	/// @return     dimension of any cached row
	std::size_t getCacheRowSize() const
	{
		return m_cachedMatrix.size2();
	}



protected:
	/// container for precomputed values
	blas::matrix<QpFloatType> m_cachedMatrix;

	// maximal size of cache
	size_t m_cacheSize;

	// original kernel matrix, will be accessed if entries outsied the cache are requested
	Matrix* m_baseMatrix;

	// remember how big the original matrix was to prevent access errors
	size_t m_originalNumberOfRows;
};

}
#endif
