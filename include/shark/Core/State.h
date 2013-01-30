//===========================================================================
/*!
*
*  \brief Class which externalises the state of an Object
*
*  \author  O. Krause
*  \date    2012
*
*  \par Copyright (c) 1999-2012:
*      Institut f&uuml;r Neuroinformatik<BR>
*      Ruhr-Universit&auml;t Bochum<BR>
*      D-44780 Bochum, Germany<BR>
*      Phone: +49-234-32-27974<BR>
*      Fax:   +49-234-32-14209<BR>
*      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
*      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
*
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


///\brief Represents the State of an Object.
///
///Often the State of an object is changed during usage. 
///This, however, makes it impossible to use the obejct in a
///multithreaded environment in parallel. The solution is to externbalize
///the state such, that every thread can have it's own storage.
///This Interface 
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
