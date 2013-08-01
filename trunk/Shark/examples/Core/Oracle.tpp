#include <shark/Core/Factory.h>

#include <cstdlib>

#include <iostream>
#include <string>

#define ANNOUNCE_ORACLE( Oracle, Factory ) \
  namespace Oracle ## _detail {\
    typedef TypeErasedAbstractFactory< Oracle, Factory > abstract_factory_type;\
    typedef FactoryRegisterer< Factory > factory_registerer_type;\
    static factory_registerer_type FACTORY_REGISTERER = factory_registerer_type( #Oracle, new abstract_factory_type() );\
}\

namespace shark {

    /**
     * \brief Abstract base class for oracles.
     */
    class Oracle {
    public:
	virtual ~Oracle() {}

	virtual unsigned int answer() const = 0;
    };

    /**
     * \brief Implements the oracle of Delphi.
     */
    class OracleOfDelphi : public Oracle {
    public:

	unsigned int answer() const { return( 41 ); }
    };

    /**
     * \brief Implements the oracle interface in terms of the 
     * answers given by the Hitchhiker's Guide to the Galaxy.
     */
    class HitchhikersGuideToTheGalaxy : public Oracle {
	unsigned int answer() const { return( 42 ); }
    };

    typedef Factory< Oracle, std::string > OracleFactory;

    ANNOUNCE_ORACLE( OracleOfDelphi, OracleFactory );
    ANNOUNCE_ORACLE( HitchhikersGuideToTheGalaxy, OracleFactory );
}

int main( int argc, char ** argv ) {
    
    shark::Oracle * o1 = shark::OracleFactory::instance()[ "OracleOfDelphi" ];
    if( o1 )
	std::cout << "Delphi says: " << o1->answer() << std::endl;
    shark::Oracle * o2 = shark::OracleFactory::instance()[ "HitchhikersGuideToTheGalaxy" ];
    if( o2 )
	std::cout << "HitchhikersGuideToTheGalaxy says: " << o2->answer() << std::endl;
    
    return( EXIT_SUCCESS );

}


