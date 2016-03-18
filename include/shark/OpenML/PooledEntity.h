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
/// Inherit like this:
///    class MyClass : public Pool<MyClass> { ... };
///
/// Ideally, the constructors of the object should be private to force
/// creation of objects on the heap, e.g., through the get method.
/// The pool gives shared pointers to the outside, so objects get
/// deleted automatically when they are not needed any longer.
///
SHARK_EXPORT_SYMBOL template <class T> class PooledEntity : public Entity
{
public:
//	/// \brief Obtain a non-owning pointer to an newly created object.
//	template <class... Args>
//	static std::shared_ptr<T> create(Args&& args)
//	{
//		T* object = new T(std::forward<Args>(args));
//		m_all[id] = object;
//		return std::shared_ptr<T>(object);
//	}

	/// \brief Obtain a non-owning pointer to an object by ID.
	static std::shared_ptr<T> get(IDType id)
	{
		typename std::map<IDType, T*>::iterator it = m_all.find(id);
		if (it == m_all.end())
		{
//			return create<IDType>(id);
			T* object = new T(id);
			m_all[id] = object;
			return std::shared_ptr<T>(object);
		}
		else return std::shared_ptr<T>(it->second);
	}

//	using Entity::id;
//	using Entity::tags;
//	using Entity::tag;
//	using Entity::untag;
//	using Entity::print;

protected:
//	using Entity::setID;
//	using Entity::setTags;

	/// \brief Default constructor.
	///
	/// Calling the default constructor makes it OBLIGATORY to call
	/// registerObject later - ideally before the constructor returns.
	PooledEntity()
	: Entity()
	{ }

	/// \brief Constructor, register the object.
	PooledEntity(IDType id)
	: Entity(id)
	{
		registerObject((T*)this);
	}

	static void registerObject(T* object)
	{
		m_all[object->id()] = object;
	}

	/// \brief Destructor; unregister the object.
	virtual ~PooledEntity()
	{
		m_all.erase(id());
	}

private:
	static std::map<IDType, T*> m_all;   ///< map for accessing existing objects by ID
};


template <class T>
std::map<IDType, T*> PooledEntity<T>::m_all;


};  // namespace openML
};  // namespace shark
#endif
