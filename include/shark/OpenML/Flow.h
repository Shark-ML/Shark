//===========================================================================
/*!
 * 
 *
 * \brief       Definition of an OpenML Flow.
 * 
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

#ifndef SHARK_OPENML_FLOW_H
#define SHARK_OPENML_FLOW_H


#include "Base.h"
#include "PooledEntity.h"
#include <shark/Core/INameable.h>
#include <shark/Core/Exception.h>
#include <vector>
#include <map>
#include <string>


namespace shark {
namespace openML {


/// \brief Representation of an OpenML flow.
SHARK_EXPORT_SYMBOL class Flow : public PooledEntity<Flow>
{
private:
	friend class PooledEntity<Flow>;

	/// \brief Construct an existing OpenML flow from its ID.
	Flow(IDType id);

	/// \brief Construct a new OpenML flow.
	Flow(std::string const& name, std::string const& description, std::vector<Hyperparameter> const& hyperparameters, std::map<std::string, std::string> const& properties = std::map<std::string, std::string>());

public:
	using PooledEntity<Flow>::get;
	using PooledEntity<Flow>::create;

	/// \brief Obtain the version string used for creating new flows.
	static std::string sharkVersion();

	/// \brief Add a tag to the entity.
	void tag(std::string const& tagname);

	/// \brief Remove a tag from the entity.
	void untag(std::string const& tagname);

	/// \brief Print a human readable summary of the entity.
	void print(std::ostream& os = std::cout) const;

	/// \brief Name of the flow (acts as a key in OpenML).
	std::string const& name() const
	{ return m_name; }

	/// \brief Version of the flow (acts as a key in OpenML).
	std::string const& version() const
	{ return m_version; }

	/// \brief Short description of the flow.
	std::string const& description() const
	{ return m_description; }

	/// \brief Obtain the number of hyperparameters of the flow.
	std::size_t numberOfHyperparameters() const
	{ return m_hyperparameter.size(); }

	/// \brief Obtain a hyperparameter description by index.
	Hyperparameter const& hyperparameter(std::size_t index) const
	{ return m_hyperparameter[index]; }

	/// \brief Obtain the index of a hyperparameter by name.
	std::size_t hyperparameterIndex(std::string const& name) const
	{
		for (std::size_t i=0; i<m_hyperparameter.size(); i++)
		{
			if (m_hyperparameter[i].name == name) return i;
		}
		throw SHARKEXCEPTION("[hyperparameterIndex] hyperparameter " + name + " not found");
	}

private:
	/// \brief Obtain an existing flow from the OpenML server.
	void obtainFromServer();

	std::string m_name;                                ///< name of the flow (key)
	std::string m_description;                         ///< short description of the flow
	std::string m_version;                             ///< version of the flow (key)
	std::vector<Hyperparameter> m_hyperparameter;      ///< description of hyperparameters of the flow, accessed by index (ordered collection)
	std::map<std::string, std::string> m_properties;   ///< additional non-mandatory properties of the flow, fields must comply with the OpenML flow XML schema
};


};  // namespace openML
};  // namespace shark
#endif
