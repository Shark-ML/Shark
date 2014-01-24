//===========================================================================
/*!
 * 
 * \file        Oracle.cpp
 *
 * \brief       Demonstration of Shark's factory pattern.
 * 
 * 
 *
 * \author      T. Voﬂ
 * \date        2011
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
//===========================================================================

#include <shark/Core/Factory.h>

#include <iostream>
#include <string>
#include <cstdlib>

// convenience macro for registering sub-classes with their names as keys to the factory
#define ANNOUNCE_ORACLE( Oracle, Factory ) \
	namespace Oracle ## _detail {\
		typedef TypeErasedAbstractFactory< Oracle, Factory > abstract_factory_type;\
		typedef FactoryRegisterer< Factory > factory_registerer_type;\
		static factory_registerer_type FACTORY_REGISTERER = factory_registerer_type( #Oracle, new abstract_factory_type() );\
	}\


namespace shark {

	// opaque super class
	class Oracle {
	public:
		virtual ~Oracle() {}
		virtual unsigned int answer() const = 0;
	};

	// concrete sub-class 1
	class OracleOfDelphi : public Oracle {
	public:
		unsigned int answer() const { return( 41 ); }
	};

	// concrete sub-class 2
	class HitchhikersGuideToTheGalaxy : public Oracle {
	public:
		unsigned int answer() const { return( 42 ); }
	};

	// factory type for decendants of the super class, identified by string keys
	typedef Factory< Oracle, std::string > OracleFactory;

	// register sub-classes to the factory
	ANNOUNCE_ORACLE( OracleOfDelphi, OracleFactory );
	ANNOUNCE_ORACLE( HitchhikersGuideToTheGalaxy, OracleFactory );
}


using namespace shark;


int main( int argc, char ** argv )
{
	// create sub-class instances from class name strings
	Oracle * o1 = OracleFactory::instance()[ "Delphi" ];
	if( !o1 )
	std::cout << "Delphi says: " << o1->answer() << std::endl;
	Oracle * o2 = OracleFactory::instance()[ "HitchhikersGuideToTheGalaxy" ];
	if( !o2 )
	std::cout << "HitchhikersGuideToTheGalaxy says: " << o2->answer() << std::endl;

	return( EXIT_SUCCESS );
}
