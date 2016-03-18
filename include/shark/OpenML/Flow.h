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
#include <vector>
#include <map>
#include <string>


namespace shark {
namespace openML {


/// \brief Representation of an OpenML flow.
SHARK_EXPORT_SYMBOL class Flow : public PooledEntity<Flow>
{
public:
	/// \brief Construct an existing OpenML flow from its ID.
	Flow(IDType id);

	/// \brief Construct a new OpenML flow.
	Flow(std::string const& name, std::string const& description, std::vector<Hyperparameter> const& hyperparameters, std::map<std::string, std::string> const& properties = std::map<std::string, std::string>());

	/// \brief Construct a new OpenML flow named after a Shark object (usually a trainer).
	Flow(INameable const& method, std::string const& description, std::vector<Hyperparameter> const& hyperparameters, std::map<std::string, std::string> const& properties = std::map<std::string, std::string>());

public:
	/// \brief Construct a new OpenML flow.
	static std::shared_ptr<Flow> get(std::string const& name, std::string const& description, std::vector<Hyperparameter> const& hyperparameters, std::map<std::string, std::string> const& properties = std::map<std::string, std::string>());

	/// \brief Construct a new OpenML flow named after a Shark object (usually a trainer).
	static std::shared_ptr<Flow> get(INameable const& method, std::string const& description, std::vector<Hyperparameter> const& hyperparameters, std::map<std::string, std::string> const& properties = std::map<std::string, std::string>());

	/// \brief Obtain the version string used for creating new flows.
	static std::string sharkVersion();

	/// \brief Add a tag to the entity.
	void tag(std::string const& tagname);

	/// \brief Remove a tag from the entity.
	void untag(std::string const& tagname);

	/// \brief Print a human readable summary of the entity.
	void print(std::ostream& os = std::cout) const;

	std::string const& name() const
	{ return m_name; }

	std::string const& version() const
	{ return m_version; }

	std::string const& description() const
	{ return m_description; }

	std::size_t numberOfHyperparameters() const
	{ return m_hyperparameter.size(); }

	Hyperparameter const& hyperparameter(std::size_t index) const
	{ return m_hyperparameter[index]; }

private:
	void create(std::string const& name, std::string const& description, std::vector<Hyperparameter> const& hyperparameters, std::map<std::string, std::string> const& properties = std::map<std::string, std::string>());
	void obtainFromServer();

	std::string m_name;
	std::string m_description;
	std::string m_version;
	std::vector<Hyperparameter> m_hyperparameter;
	std::map<std::string, std::string> m_properties;
};


};  // namespace openML
};  // namespace shark
#endif
