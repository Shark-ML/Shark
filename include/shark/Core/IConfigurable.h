/*!
 * 
 *
 * \brief       IConfigurable interface
 * 
 * 
 *
 * \author      T.Voss, T. Glasmachers, O.Krause
 * \date        2010-2011
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
#ifndef SHARK_CORE_ICONFIGURABLE_H
#define SHARK_CORE_ICONFIGURABLE_H

#include <boost/property_tree/ptree.hpp>

namespace shark {

  /**
   * \brief Type of a property tree.
   */
typedef boost::property_tree::ptree PropertyTree;

/**
 * \brief Interface that abstracts a configurable component.
 */
class IConfigurable {
public:
  /**
   * \brief Virtual d'tor.
   */
	virtual ~IConfigurable() {}

	/**
	 * \brief Configures the component given a property tree.
	 * \param [in] node The root of the property tree.
	 */
	virtual void configure( const PropertyTree & node )
	{ }
};

}

#endif // SHARK_CORE_ICONFIGURABLE_H
