//===========================================================================
/*!
 * 
 *
 * \brief       Cache implementing an Least-Recently-Used Strategy
 * 
 * 
 *
 * \author      O. Krause
 * \date        2013
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
#ifndef SHARK_ALGORITHMS_QP_LRUCACHE_H
#define SHARK_ALGORITHMS_QP_LRUCACHE_H

#include <shark/Core/Exception.h>
#include <boost/intrusive/list.hpp>
#include <vector>


namespace shark{

/// \brief Implements an LRU-Caching Strategy for arbitrary Cache-Lines.
///
/// Low Level Cache which stores cache lines, arrays of T[size] where size is a variable length for every cache line. 
/// Every line is associated with some index 0<i < max. It is assumed that not all cache lines can be 
/// stored at the same time, but  only a (small) subset. The size of the cache is bounded by the summed length of all
/// cache lines, that means that when the lines are very short, the cache can store more lines.
/// If the cache is full and another line needs to be accessed, or an existing line needs to be resized,
/// cache lines need to be freed. This cache uses an Least-Recently-Used strategy. The cache maintains
/// a list. Everytime a cacheline is accessed, it moves to the front of the list. When a line is freed
/// the end of the list is chosen.
template<class T>
class LRUCache{
	/// cache data held for every example
	struct CacheEntry:  public boost::intrusive::list_base_hook<>
	{
		T* data; ///< pointer to the beginning of the matrix row
		std::size_t length; ///< length of the currently calculated strip of variables
		CacheEntry():length(0){}
	};
public:
	/// \brief Creates a cache with a given maximum index "lines" and a given maximum cache size.
	LRUCache(std::size_t lines, std::size_t cachesize = 0x4000000)
	: m_cacheEntry(lines)
	, m_cacheSize( 0 )
	, m_maxSize( cachesize ){}
	
	~LRUCache(){
		clear();
	}
	
	///\brief Returns true if the line is cached.
	bool isCached(std::size_t i)const{
		return m_cacheEntry[i].length != 0;
	}
	///\brief Returns the size of the cached line.
	std::size_t lineLength(std::size_t i)const{
		return m_cacheEntry[i].length;
	}
	
	/// \brief Returns the number of lines currently allocated.
	std::size_t cachedLines()const{
		return m_lruList.size();
	}
	
	///\brief Returns the line with index i with the correct size.
	///
	///If the line is not cached, it is created with the exact size. if it is cached
	///and is at least as big, it is returned unchanged. otherwise it is
	///resized to the proper size and the old values are kept.
	T* getCacheLine(std::size_t i, std::size_t size){
		CacheEntry& entry = m_cacheEntry[i];
		//if the is cached, we push it to the front
		if(!isCached(i))
			cacheCreateRow(entry,size);
		else{
			if(entry.length >= size)
				cacheRedeclareNewest(entry);
			else
				resizeLine(entry,size);
		}
		return entry.data;
	}
	
	///\brief Just returns the pointer to the i-th line without affcting cache at all.
	T* getLinePointer(std::size_t i){
		return m_cacheEntry[i].data;
	}
	
	///\brief Just returns the pointer to the i-th line without affcting cache at all.
	T const* getLinePointer(std::size_t i)const{
		return m_cacheEntry[i].data;
	}
	
	/// \brief Resizes a line while retaining the data stored inside it.
	///
	/// if the new size is smaller than the old, only the first size entries are saved.
	void resizeLine(std::size_t i ,std::size_t size){
		resizeLine(m_cacheEntry[i],size);
	}
	
	///\brief Marks cache line i for deletion, that is the next time memory is needed, this line will be freed.
	void markLineForDeletion(std::size_t i){
		if(!isCached(i)) return;
		CacheEntry& block = m_cacheEntry[i];
		m_lruList.erase(m_lruList.iterator_to(block));
		m_lruList.push_back(block);
	}
	
	///\brief swaps index of lines i and j.
	void swapLineIndices(std::size_t i, std::size_t j){
		typedef typename boost::intrusive::list<CacheEntry>::iterator Iterator;
		//SHARK_ASSERT( i <= j );
		//nothing to do if lines are not cached or indizes are the same
		if( i == j || (!isCached(i) && !isCached(j)))  return; 
		
		CacheEntry& cachei = m_cacheEntry[i];
		CacheEntry& cachej = m_cacheEntry[j];
		
		//correct list to point to the exchanged values
		if(isCached(i) && !isCached(j)){
			Iterator pos = m_lruList.iterator_to( cachei );
			m_lruList.insert(pos,cachej);
			m_lruList.erase(pos);
		}else if(!isCached(i) && isCached(j)){
			Iterator pos = m_lruList.iterator_to( cachej );
			m_lruList.insert(pos,cachei);
			m_lruList.erase(pos);
		}else if(isCached(i) && isCached(j)){
			Iterator posi = m_lruList.iterator_to( cachei );
			Iterator posj = m_lruList.iterator_to( cachej );
			//increment to the next position in the list so that we have
			//a stable position in case we ned to remove one. Also note that insert(incposi,elem) now
			//inserts directly before the position of elem i
			Iterator incposi = posi;++incposi;
			Iterator incposj = posj;++incposj;
			//there is one important edge-case: that is the two elements are next in the list
			//in this case, we can just remove  and insert it before i again
			if(incposi == posj){
				m_lruList.erase( posj );
				m_lruList.insert(posi,cachej);
			} else if(incposj == posi){
				m_lruList.erase( posi );
				m_lruList.insert(posj,cachei);
			}
			else{
				//erase elements, this does not affect the incremented iterators
				m_lruList.erase( m_lruList.iterator_to( cachei ) );
				m_lruList.erase( m_lruList.iterator_to( cachej ) );
				//insert at correct positions
				m_lruList.insert(incposi,cachej);
				m_lruList.insert(incposj,cachei);
			}
		}
		
		//exchange entries
		std::swap(cachei.length,cachej.length);
		std::swap(cachei.data,cachej.data);
	}
	
	std::size_t size()const{
		return m_cacheSize;
	}
	
	std::size_t listIndex(std::size_t i)const{
		typename boost::intrusive::list<CacheEntry>::const_iterator iter = m_lruList.begin();
		std::advance(iter,i);
		return &(*iter)-&m_cacheEntry[0];
	}
	std::size_t maxSize()const{
		return m_maxSize;
	}
	
	///\brief empty cache
	void clear(){
		ensureFreeMemory(m_maxSize);
	}
private:
	/// \brief Pushes a cached entry to the bginning of the lru-list
	void cacheRedeclareNewest(CacheEntry& block){
		m_lruList.erase(m_lruList.iterator_to(block));
		m_lruList.push_front(block);
	}
	
	///\brief Creates a new row with a certain size > 0.
	void cacheCreateRow(CacheEntry& block,std::size_t size){
		SIZE_CHECK(size > 0);
		ensureFreeMemory(size);
		block.length = size;
		block.data = new T[size];
		m_lruList.push_front(block);
		m_cacheSize += size;
	}
	/// \brief Removes a cached row.
	void cacheRemoveRow(CacheEntry& block){
		m_cacheSize -= block.length;
		m_lruList.erase( m_lruList.iterator_to( block ) );
		delete[] block.data;
		block.length = 0;
	}
	/// \brief Resizes a line and copies all old values into it.
	void resizeLine(CacheEntry& block,std::size_t size){
		
		//salvage block data
		T* newLine  = new T[size];
		std::copy(block.data,block.data+std::min(size,block.length),newLine);
		
		//remove old data
		cacheRemoveRow(block);
		//free space for the new block
		ensureFreeMemory(size);
		
		//add new block
		block.data = newLine;
		block.length = size;
		m_cacheSize += size;
		m_lruList.push_front(block);
	}
	
	///\brief Frees enough memory until a given amount of T can be allocated
	void ensureFreeMemory(std::size_t size){
		SIZE_CHECK(size <= m_maxSize);
		while(m_maxSize-m_cacheSize < size){
			cacheRemoveRow(m_lruList.back());//remove the oldest row
		}
	}
	
	std::vector<CacheEntry> m_cacheEntry; ///< cache entry description
	boost::intrusive::list<CacheEntry> m_lruList;
	
	std::size_t m_cacheSize;//current size of cache in T
	std::size_t m_maxSize;//maximum size of cache in T

	
};
}
#endif