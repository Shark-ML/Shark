//===========================================================================
/*!
*
*  \brief Class which externalizes the state of an Object.
*
*  \author  O. Krause
*  \date    2012
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
//===========================================================================

#ifndef SHARK_CORE_STATE_H
#define SHARK_CORE_STATE_H

#include <boost/cast.hpp>

namespace shark{


/// \brief Represents the State of an Object.
///
/// Often the State of an object is changed during usage. 
/// This, however, makes it impossible to use the object in a
/// multithreaded environment in parallel. The solution is to externalize
/// the state, so that every thread can have it's own storage.
struct State{

	///\brief prevents that this class can be instantiated
	virtual ~State(){};

	///\brief Safely downcast State to it's derived type.
	///
	///Tries to do a safe cast from State to it's derived type.
	///The program is terminated in debug mode, if the wrong Type
	///was used.
	template<class DerivedStateType>
	DerivedStateType const& toState()const{
		return *boost::polymorphic_downcast<DerivedStateType const*>(this);
	}

	template<class DerivedStateType>
	DerivedStateType & toState(){
		return *boost::polymorphic_downcast<DerivedStateType*>(this);
	}
};

///\brief Default State of an Object which does not need a State
struct EmptyState:public State{
};
}
#endif
