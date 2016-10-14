//===========================================================================
/*!
 * 
 *
 * \brief       This file defines a class for managing objects.
 * 
 * 
 *
 * \author      T. Glasmachers
 * \date        2016
 *
 *
 * \par Copyright 1995-2016 Shark Development Team
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

#ifndef SHARK_OPENML_POOLEDENTRY_H
#define SHARK_OPENML_POOLEDENTRY_H

#include "Base.h"
#include "Entity.h"
#include <map>
#include <memory>


namespace shark {
namespace openML {


/// \brief Base class for managing objects in a pool.
///
/// The purpose of this class is to avoid the existence of two distinct
/// C++ objects representing the same OpenML entity, e.g., a data set.
/// Objects can be obtained from their ID.
///
/// As the name suggests, pooled objects live in a managed "pool", so
/// they can be reused at need. Still, objects are not owned by the pool.
/// They are deleted when they are not needed any longer, i.e., when all
/// shared pointers to the object go out of scope. This is achieved by
/// the pool holding a weak pointer to the object.
///
/// Inherit the PooledEntity template class like this:
///    class MyClass : public PooledEntity<MyClass> { ... };
///
/// Ideally, the constructors of the object should be private to force
/// creation of objects on the heap, e.g., through the get method.
/// The pool gives shared pointers to the outside, so objects get
/// deleted automatically when they are not needed any longer.
///
/// The class offers a default constructor. This constructor does NOT
/// register the object in the pool, because it is assumed that the
/// object ID is not yet known or even defined. An existing object
/// must be added to the pool as soon as it obtains an ID from OpenML.
///
SHARK_EXPORT_SYMBOL template <class T> class PooledEntity
: public Entity
, public std::enable_shared_from_this<PooledEntity<T>>
{
public:
	/// \brief Obtain a shared pointer to an object by its ID.
	///
	/// Conceptually, this function returns an object corresponding to
	/// an already existing OpenML entity.
	static std::shared_ptr<T> get(IDType id)
	{
		auto it = m_all.find(id);
		if (it == m_all.end() || it->second.expired())
		{
//			auto ret = std::make_shared<T>(id);  // cannot use make_shared since the constructor is private
			auto ret = std::shared_ptr<T>(new T(id));
			m_all[id] = ret;
			return ret;
		}
		else return std::shared_ptr<T>(it->second);
	}

	/// \brief Construct a new object.
	///
	/// Conceptually, this function returns an object corresponding to
	/// a newly created OpenML entity.
	template <typename ... Args>
	static std::shared_ptr<T> create(Args ... args)
	{
//		return std::make_shared<T>(std::forward<Args>(args)...);  // cannot use make_shared since the constructor is private
		return std::shared_ptr<T>(new T(std::forward<Args>(args)...));
	}

protected:
	/// \brief Default constructor.
	PooledEntity()
	: Entity()
	{ }

	/// \brief Construction from ID, registering the object.
	PooledEntity(IDType id)
	: Entity(id)
	{ }


	/// \brief Set the object's ID (inherited from Entity).
	///
	/// This call registers the object. This means that all call of the
	/// Entity's setID function (through the virtual function interface)
	/// will automatically register the object in the pool.
	virtual void setID(IDType id)
	{
		SHARK_ASSERT(id != invalidID);
		Entity::setID(id);
		m_all[id] = std::static_pointer_cast<T, PooledEntity<T>>(
					std::enable_shared_from_this<PooledEntity<T>>::shared_from_this()
			);
	}

	/// \brief Destructor; unregister the object.
	virtual ~PooledEntity()
	{
		IDType id = Entity::id();
		if (id != invalidID) m_all.erase(id);
	}

private:
	static std::map<IDType, std::weak_ptr<T>> m_all;   ///< map for accessing existing objects by ID
};


template <class T>
std::map<IDType, std::weak_ptr<T>> PooledEntity<T>::m_all;


};  // namespace openML
};  // namespace shark
#endif
