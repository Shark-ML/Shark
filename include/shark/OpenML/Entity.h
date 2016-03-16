//===========================================================================
/*!
 * 
 *
 * \brief       This file defines several handy types, as well as a base class for all OpenML entities.
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

#ifndef SHARK_OPENML_ENTITY_H
#define SHARK_OPENML_ENTITY_H

#include "detail/Json.h"
#include <shark/Core/Exception.h>

#include <boost/filesystem.hpp>

#include <cstdint>
#include <memory>
#include <vector>
#include <set>
#include <iostream>


namespace shark {
namespace openML {


typedef boost::filesystem::path PathType;                        ///< \brief  Path type, e.g., for specifying the cache directory.
typedef std::uint64_t IDType;                                    ///< \brief  An ID is an unsigned integer.

static const IDType invalidID = 0;                               ///< \brief Invalid ID, marker for default constructed objects.


/// \brief Super class of all OpenML entities, providing an ID and tags.
class Entity
{
public:
	/// \brief Default construct an entity, the ID is invalid
	Entity()
	: m_id(invalidID)
	{ }

	/// \brief Construct an entity with a given ID.
	Entity(IDType id)
	: m_id(id)
	{ }

	/// \brief Obtain the ID of the entity.
	IDType id() const
	{ return m_id; }

	/// \brief Obtain the set of all tags attributed to the entity.
	std::set<std::string> const& tags() const
	{ return m_tags; }

	/// \brief Add (set) a tag to the entity.
//	virtual void addTag(std::string const& tag) = 0;

	/// \brief Remove a tag from the entity.
//	virtual void removeTag(std::string const& tag) = 0;

	/// \brief Print a human readable summary of the entity.
	virtual void print(std::ostream& os = std::cout) const
	{
		os << " ID: " << m_id << std::endl;
		os << " tags:";
		for (std::set<std::string>::const_iterator it = m_tags.begin(); it != m_tags.end(); ++it)
		{
			os << " " << *it;
		}
		os << std::endl;
	}

protected:
	/// \brief Set the ID of a default-constructed entity.
	void setID(IDType id)
	{
		if (m_id != invalidID) throw SHARKEXCEPTION("The entity already has a valid ID.");
		m_id = id;
	}

	/// \brief Acquire tags from a Json array of strings.
	void setTags(detail::Json tags)
	{
		m_tags.clear();
		for (std::size_t i=0; i<tags.size(); i++) m_tags.insert(tags[i].asString());
	}

private:
	IDType m_id;                       ///< ID of the entity
	std::set<std::string> m_tags;      ///< tags of the entity
};


};  // namespace openML
};  // namespace shark
#endif
