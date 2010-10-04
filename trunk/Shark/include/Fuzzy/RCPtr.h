/**
 * \file RCPtr.h
 *
 * \brief Template class for Reference Counted Pointers
 * 
 * \author Marc Nunkesser
 * 
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 */

/* $log$ */

#ifndef RCPTR_H
#define RCPTR_H

/**
 * \brief Template class for Reference Counted Pointers.
 * 
 * When we create objects like FuzzySets for instance it is hard to decide who 
 * should be owner of the object and responsible for freeing the memory again, 
 * since the FuzzySet can be used at many different locations, e.g. as part of a
 * ComposedFS again. Problems like these are solved by the concept of Reference
 * Counted Pointers (see: <i>Scott Meyers</i>, <i>More Effective C++</i>). RCPtr 
 * objects keep track of the number of references to the object and frees it 
 * when the last reference is deleted.<br>
 * <br>
 * But unfortunately there are disadvantages as well. For instance, this concept
 * prevents us from using the usual polymorphism:
 * 
 * <pre><code>RCPtr<FuzzySet> fs( new BellFS( 1, 0 ) );</code></pre>
 * 
 * So, in this case <pre><code>fs->defuzzify()</code></pre> would mean the 
 * method from FuzzySet, not the one from BellFS.
 * 
 * <b>Note:</b> T must inherit from RCObject to be handled by RCPtr.
 *  
 */

template<class T>
class RCPtr {
	
public:
	
	/**
	 * \brief Constructor with pointer to the handled object.
	 * 
	 * @param realPtr Pointer to handled object. Must be derived from RCObject.
	 */
	RCPtr(T* realPtr = 0);
	
	/**
	 * \brief Copy Constructor
	 * 
 	 * @param rhs reference to the object to be copied
	 */
	RCPtr(const RCPtr& rhs);
	
	/// Destructor
	~RCPtr();

	/// assignment operator
	RCPtr& operator=(const RCPtr& rhs);

	/**
	 * \brief Equality operator
	 * 
	 * \return Boolean indicating if the compared RCPtr objects are handling the same RCObject 
	 */
	bool operator==(const RCPtr& rhs) const;

	/**
	 * \brief Arrow operator
	 * 
	 * \return pointer to handled RCObject
	 */
	T* operator->() const;
	
	/**
	 * \brief Indirection operator
	 * 
	 * \return reference to handled RCObject 
	 */
	T& operator*() const;

	/**
	 * \brief Logical negation operator
	 * 
	 * Use this operator to test for null:
	 * 
	 * <pre><code>if( !myRCPtr ) { ... } else { ... }</code></pre>
	 * 
	 * \return Boolean indicating if handled pointer is null
	 */
	bool operator!() const;

	/**
	 * \brief Creates a unique copy the handled RCObject.
	 * 
	 * The RCPtr object will point to the new copy afterwards and the copy will
	 * be marked as unshareable.
	 */
	//void freeze();


	template< class newType >
	operator RCPtr< newType >() {
		return RCPtr<newType>(pointee);
	};

private:
	T *pointee;
	void init();
};


/////////////////////////////////////////////////////////////
//// end of class definition
//// template methods follow
////////////////////////////////////////////////////////////

template<class T>
void RCPtr<T>::init() {
	if (pointee == 0) return;
	//if (pointee->isShareable() == false) {
		// pointee = new T(*pointee);
	//};
	pointee->addReference();
}

template<class T>
RCPtr<T>::RCPtr(T * realPtr)
		: pointee(realPtr) {
	init();
}


template<class T>
RCPtr<T>::RCPtr(const RCPtr& rhs) : pointee(rhs.pointee) {
	init();
}


template<class T>
RCPtr<T>::~RCPtr() {
	if (pointee) pointee->removeReference();
}

template<class T>
RCPtr<T>& RCPtr<T>::operator= (const RCPtr& rhs) {
	if (pointee != rhs.pointee) {
		if (pointee) pointee->removeReference();
		// cout<<"in operator ="<<endl;
		pointee = rhs.pointee;
		init();
	};
	return *this;
}

template<class T>
bool RCPtr<T>::operator== (const RCPtr& rhs) const {
	return(pointee == rhs.pointee);
}


template<class T>
T* RCPtr<T>::operator->() const {
	return pointee;
}

template <class T>
T& RCPtr<T>::operator*() const {
	return *pointee;
}

template <class T>
bool RCPtr<T>::operator!() const {
	return (pointee==0);
}

//template <class T>
//void RCPtr<T>::freeze() {
//	T* unique = new T(*pointee); // call copy constructor
//	pointee->removeReference();
//	pointee = unique;
//	init();
//	pointee->markUnshareable();
//}

#endif
