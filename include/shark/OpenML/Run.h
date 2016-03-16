//===========================================================================
/*!
 * 
 *
 * \brief       Definition of an OpenML Run.
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

#ifndef SHARK_OPENML_RUN_H
#define SHARK_OPENML_RUN_H


#include "Entity.h"
#include <shark/Data/Dataset.h>


namespace shark {
namespace openML {


/// \brief Representation of an OpenML run.
//template <typename OutputType>
class Run : public Entity
{
public:
	bool upload() const;

	double namedProperty(std::string const& property) const;

	void print(std::ostream& os = std::cout) const;

private:
//	Data<OutputType> m_predictions;
};


};  // namespace openML
};  // namespace shark
#endif
