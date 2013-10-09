/*!
 *
 *  \brief Vector container prepared for data sharing.
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
#ifndef SHARK_CORE_SHAREDVECTOR_INL
#define SHARK_CORE_SHAREDVECTOR_INL

#include <shark/Core/ISerializable.h>
#include <algorithm>
#include <boost/shared_ptr.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <vector>
#include <iterator>
namespace shark{

///
/// \brief Vector container prepared for data sharing.
///
/// \par
/// This class is a thin wrapper around a shared array
/// to mimic more std::vector like behaviour.
///
///it also supports lazy subsets of another shared vector e.g if only 
///a specific range is needed
template<class T>
class SharedVector : public ISerializable{
private:
	boost::shared_ptr<std::vector<T> > m_array;
	std::size_t m_start;
	std::size_t m_end;
	
public:
	//std Container definitions...
	typedef typename std::vector<T>::iterator iterator;
	typedef typename std::vector<T>::const_iterator const_iterator;
	typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
    typedef std::reverse_iterator<iterator> reverse_iterator;
	typedef T value_type;
	typedef T& reference;
	typedef T const& const_reference;
	typedef T* pointer;
	typedef T const* const_pointer;
	typedef int difference_type;
	typedef std::size_t size_type;
	
	
	/// constructs an empty SharedVector
	SharedVector()
	:m_array(new std::vector<T>()),m_start(0),m_end(0){}

	/// constructcs a SharedVector with size instances of objects
	SharedVector(size_t size)
	:m_array(new std::vector<T>(size)),m_start(0),m_end(size){}

	/// constructs a SharedVector from the data of a std::vector
	SharedVector(const std::vector<T>& vec)
	:m_array(new std::vector<T>(vec)),m_start(0),m_end(vec.size()){}
	
	///constructs a SharedVector as a lazy subset from another SharedVector
	SharedVector(SharedVector const& data,std::size_t start,std::size_t end)
	:m_array(data.m_array),m_start(start+data.m_start),m_end(end){}

	/// returns a reference to the i-th element
	T& operator[](size_t i){
		return (*m_array)[i+m_start];
	}

	/// returns a const reference to the i-th element
	const T& operator[](size_t i)const{
		return (*m_array)[i+m_start];
	}

	/// begin iterator
	iterator begin(){
		return m_array->begin()+m_start;
	}
	/// end iterator
	iterator end(){
		return m_array->begin()+m_end;
	}
	/// const begin iterator
	const_iterator begin()const{
		return m_array->begin()+m_start;
	}
	/// const end iterator
	const_iterator end()const{
		return m_array->begin()+m_end;
	}
	
	///reverse iterators
	/// begin iterator
	reverse_iterator rbegin(){
		return reverse_iterator(begin());
	}
	/// end iterator
	reverse_iterator rend(){
		return reverse_iterator(end());
	}
	/// const begin iterator
	const_reverse_iterator rbegin()const{
		return const_reverse_iterator(begin());
	}
	/// const end iterator
	const_reverse_iterator rend()const{
		return const_reverse_iterator(end());
	}

	///maximum size of vector allowed
	size_type max_size () const {
		return std::numeric_limits<size_type>::max ();
	}
	/// size of the vector
	size_t size()const{
		return m_end - m_start;
	}

	/// true if (size() == 0)
	bool empty()const{
		return m_end == 0;
	}

	/// This function resizes its internal data container.
	/// If necessary, it makes the container independent
	/// from its sibling containers. This operations
	/// invalidates the contents of the container.
	void resize(size_t newSize){
		if(newSize==size()){
			return;
		}
		if(!isIndependent() || m_start != 0) {
			m_array.reset(new std::vector<T>(newSize));
		}
		else{
			m_array->resize(newSize);
		}
		m_start = 0;
		m_end = newSize;
	}

	/// return true when the contents of this vector are not shared
	bool isIndependent()const{
		return m_array.unique();
	}

	/// if the vector is not independent then its contents are copied
	/// so that changes of this vector do not affect sibling vectors
	void makeIndependent(){
		if(!isIndependent()){
			SharedVector copy(*m_array);
			m_array.swap(copy.m_array);
		}
	}

	/// make this vector independent and clear its state
	void clear(){
		m_array.reset();
	}

	/// from ISerializable
	void read(InArchive& archive)
	{
		archive & m_array;
		archive & m_start;
		archive & m_end;
	}

	/// from ISerializable
	void write(OutArchive& archive) const
	{
		archive & m_array;
		archive & m_start;
		archive & m_end;
	}
	template<class U>
	friend bool operator == (const SharedVector<U>& op1, const SharedVector<U>& op2);
};
template<class T>
bool operator == (const SharedVector<T>& op1, const SharedVector<T>& op2) {
	return op1.m_array == op2.m_array;
}
template<class T>
bool operator != (const SharedVector<T>& op1, const SharedVector<T>& op2) {
	return !(op1 == op2);
}

typedef SharedVector<double> Intermediate;

} // namespace shark
#endif
