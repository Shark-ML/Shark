#ifndef SHARK_FUZZY_FUZZY_H
#define SHARK_FUZZY_FUZZY_H

#include <string>

namespace shark {

	enum Connective {
		AND, 
		OR, 
		PROD, 
		PROBOR
	}; //order is important cf parsing

	static std::string connective_to_name( Connective c ) {
		std::string s;
		switch( c ) {
			case AND:		s = "AND"; break;
			case OR:		s = "OR"; break;
			case PROD:		s = "PROD"; break;
			case PROBOR:	s = "PROBOR"; break;
		}
		return( s );
	}

}
#endif // SHARK_FUZZY_FUZZY_H