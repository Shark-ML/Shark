//===========================================================================
/*!
 * 
 *
 * \brief       Efficient quadratic matrix cache
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


#ifndef SHARK_LINALG_CACHEDMATRIX_H
#define SHARK_LINALG_CACHEDMATRIX_H

#include <shark/Algorithms/QP/LRUCache.h>
#include <shark/Data/Dataset.h>
#include <shark/LinAlg/Base.h>

#include <vector>
#include <cmath>


namespace shark {


///
/// \brief Efficient quadratic matrix cache
///
/// \par
/// The access operations of the CachedMatrix class
/// are specially tuned towards the iterative solution
/// of quadratic programs resulting in sparse solutions.
///
/// \par
/// The kernel cache is probably one of the most intricate
/// or mind-twisting parts of Shark. In order to fully understand
/// it, reading the source code is essential and this description
/// naturally not sufficient. However, the general ideas are as
/// follows:
///
/// \par
/// A CachedMatrix owns a pointer to a regular (non-cached)
/// kernel matrix, the exact type of which is a template
/// parameter. Through it, the actions of requesting an entry
/// and propagating column-/row-flips are carried out. Even
/// though a CachedMatrix offers some methods also offered
/// by the general KernelMatrix, note that it does not inherit
/// from it in order to offer greater flexibility.
///
/// \par
/// The CachedMatrix defines a struct tCacheEntry, which
/// represents one row of varying length of a stored kernel matrix.
/// The structure aggregates a pointer to the kernel values (stored
/// as values of type CacheType); the number of stored values; and
/// the indices of the next older and newer cache entries. The latter
/// two indices pertain to the fact that the CachedMatrix maintains
/// two different "orders" of the examples: one related to location
/// in memory, and the other related to last usage time, see below.
/// During the lifetime of a CachedMatrix, it will hold a vector of
/// the length of the number of examples of tCacheEntry: one entry
/// for each example. When an example has no kernel values in the cache,
/// its tCacheEntry.length will be 0, its tCacheEntry.data will be NULL,
/// and its older and newer variables will be SHARK_CACHEDMATRIX_NULL_REFERENCE.
/// Otherwise, the entries take their corresponding meaningful values.
/// In particular for tCacheEntry.data, memory is dynamically allocated
/// via malloc, reallocated via realloc, and freed via free as fit.
///
/// \par
/// A basic operation the CachedMatrix must support is throwing away
/// old stored values to make room for new values, because it is very
/// common that not all values fit into memory (otherwise, consider the
/// PrecomputedMatrix class). When a new row is requested via row(..),
/// but no room to store it, row(..) has two options for making space:
///
/// \par
/// First option: first, the row method checks if the member index
/// m_truncationColumnIndex is lower than the overall number of examples.
/// If so, it goes through all existing rows and shortens them to a length
/// of m_truncationColumnIndex. Through this shortening, memory becomes
/// available. In other words, m_truncationColumnIndex can be used to
/// indicate to the CachedMatrix that every row longer than
/// m_truncationColumnIndex can be clipped at the end. By default,
/// m_truncationColumnIndex is equal to the number of examples and not
/// changed by the CachedMatrix, so no clipping will occur if the
/// CachedMatrix is left to its own devices. However, m_truncationColumnIndex
/// can be set from externally via setTruncationIndex(..) [this might be
/// done after a shrinking event, for example]. Now imagine a situation
/// where the cache is full, and the possibility exists to free some
/// memory by truncating longer cache rows to length m_truncationColumnIndex.
/// As soon as enough rows have been clipped for a new row to fit in, the
/// CachedMatrix computes the new row and passes back control. Most likely,
/// the next time a new, uncached row is requested, more rows will have to
/// get clipped. In order not to start checking if rows can be clipped from
/// the very first row over again, the member variable m_truncationRowIndex
/// internally stores where the chopping-procedure left off the last time.
/// When a new row is requested and it's time to clear out old entries, it
/// will start looking for choppable rows at this index to save time. In
/// general, any chopping would happen via cacheResize(..) internally.
///
/// \par
/// Second option: if all rows have been chopped of at the end, or if this
/// has never been an option (due to all rows being shorter or equal to
/// m_truncationColumnIndex anyways), entire rows will get discarded as
/// the second option. This will probably be the more common case. In
/// general, row deletions will happen via cacheDelete(..) internally.
/// The CachedMatrix itself only resorts to a very simple heuristic in
/// order to determine which rows to throw away to make room for new ones.
/// Namely, the CachedMatrix keeps track of the "age" or "oldness" of all
/// cached rows. This happens via the so-to-speak factually doubly-linked
/// list of indices in the tCacheEntry.older/newer entries, plus two class
/// members m_cacheNewest and m_cacheOldest, which point to the beginning
/// and end of this list. When row(..) wants to delete a cached row, it
/// always does so on the row with index m_cacheOldest, and this index is
/// then set to the next oldest row. Likewise, whenever a new row is requested,
/// m_cacheNewest is set to point to that one. In order to allow for smarter
/// heuristics, external classes may intervene with the deletion order via
/// the methods cacheRedeclareOldest and cacheRedeclareNewest, which move
/// an example to be deleted next or as late as possible, respectively.
///
/// \par
/// Two more drastic possibilites to influence the cache behaviour are
/// cacheRowResize and cacheRowRelease, which both directly resize (e.g.,
/// chop off) cached row values or even discard the row altogether.
/// In general however, it is preferred that the external application
/// only indicate preferences for deletion, because it will usually not
/// have information on the fullness of the cache (although this functionality
/// could easily be added).
///
template <class Matrix>
class CachedMatrix
{
public:
    typedef typename Matrix::QpFloatType QpFloatType;

    /// Constructor
    /// \param base       Matrix to cache
    /// \param cachesize  Main memory to use as a kernel cache, in QpFloatTypes. Default is 256MB if QpFloatType is float, 512 if double.
    CachedMatrix(Matrix* base, std::size_t cachesize = 0x4000000)
    : mep_baseMatrix(base), m_cache( base->size(),cachesize ){}
        
    /// \brief Copies the range [start,end) of the k-th row of the matrix in external storage
    ///
    /// This call regards the access to the line as out-of-order, thus the cache is not influenced.
    /// \param k the index of the row
    /// \param start the index of the first element in the range
    /// \param end the index of the last element in the range
    /// \param storage the external storage. must be big enough capable to hold the range
    void row(std::size_t k, std::size_t start,std::size_t end, QpFloatType* storage) const{
        SIZE_CHECK(start <= end);
        SIZE_CHECK(end <= size());
        std::size_t cached= m_cache.lineLength(k);
        if ( start < cached)//copy already available data into the temporary storage
        {
            QpFloatType const* line = m_cache.getLinePointer(k);
            std::copy(line + start, line+cached, storage);
        }
        //evaluate the remaining entries
        mep_baseMatrix->row(k,cached,end,storage+(cached-start));
    }

    /// \brief Return a subset of a matrix row
    ///
    /// \par
    /// This method returns an array of QpFloatType with at least
    /// the entries in the interval [begin, end[ filled in.
    ///
    /// \param k      matrix row
    /// \param start  first column to be filled in
    /// \param end    last column to be filled in +1
    QpFloatType* row(std::size_t k, std::size_t start, std::size_t end){
        (void)start;//unused
        //Save amount of entries already cached
        std::size_t cached= m_cache.lineLength(k);
        //create or extend cache line
        QpFloatType* line = m_cache.getCacheLine(k,end);
        if (end > cached)//compute entries not already cached
            mep_baseMatrix->row(k,cached,end,line+cached);
        return line;
    }

    /// return a single matrix entry
    QpFloatType operator () (std::size_t i, std::size_t j) const{ 
        return entry(i, j);
    }

    /// return a single matrix entry
    QpFloatType entry(std::size_t i, std::size_t j) const{
        return mep_baseMatrix->entry(i, j);
    }

    ///
    /// \brief Swap the rows i and j and the columns i and j
    ///
    /// \par
    /// It may be advantageous for caching to reorganize
    /// the column order. In order to keep symmetric matrices
    /// symmetric the rows are swapped, too. This corresponds
    /// to swapping the corresponding variables.
    ///
    /// \param  i  first row/column index
    /// \param  j  second row/column index
    ///
    void flipColumnsAndRows(std::size_t i, std::size_t j)
    {
        if(i == j)
            return;
        if (i > j)
            std::swap(i,j);

        // exchange all cache row entries
        for (std::size_t  k = 0; k < size(); k++)
        {
            std::size_t length = m_cache.lineLength(k);
            if(length <= i) continue;
            QpFloatType* line = m_cache.getLinePointer(k);//do not affect caching
            if (j < length)
                std::swap(line[i], line[j]);
            else // only one element is available from the cache
                line[i] = mep_baseMatrix->entry(k, j);
        }
        m_cache.swapLineIndices(i,j);
        mep_baseMatrix->flipColumnsAndRows(i, j);
    }

    /// return the size of the quadratic matrix
    std::size_t size() const
    { return mep_baseMatrix->size(); }

    /// return the size of the kernel cache (in "number of QpFloatType-s")
    std::size_t getMaxCacheSize() const
    { return m_cache.maxSize(); }

    /// get currently used size of kernel cache (in "number of QpFloatType-s")
    std::size_t getCacheSize() const
    { return m_cache.size(); }

    /// get length of one specific currently cached row
    std::size_t getCacheRowSize(std::size_t k) const
    { return m_cache.lineLength(k); }
    
    bool isCached(std::size_t k) const
    { return m_cache.isCached(k); }
    
    ///\brief Restrict the cached part of the matrix to the upper left nxn sub-matrix
    void setMaxCachedIndex(std::size_t n){
        SIZE_CHECK(n <=size());
        
        //truncate lines which are too long
        //~ m_cache.restrictLineSize(n);//todo: we can do that better, only resize if the memory is actually needed
        //~ for(std::size_t i = 0; i != n; ++i){
            //~ if(m_cache.lineLength(i) > n)
                //~ m_cache.resizeLine(i,n);
        //~ }
        for(std::size_t i = n; i != size(); ++i){//mark the lines for deletion which are not needed anymore
            m_cache.markLineForDeletion(i);
        }
    }

    /// completely clear/purge the kernel cache
    void clear()
    { m_cache.clear(); }

protected:
    Matrix* mep_baseMatrix; ///< matrix to be cached

    LRUCache<QpFloatType> m_cache; ///< cache of the matrix lines
};

}
#endif
