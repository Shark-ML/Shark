#include <shark/Core/Factory.h>

#include <iostream>
#include <string>

#define ANNOUNCE_ORACLE( Oracle, Factory ) \
  namespace Oracle ## _detail {\
    typedef TypeErasedAbstractFactory< Oracle, Factory > abstract_factory_type;\
    typedef FactoryRegisterer< Factory > factory_registerer_type;\
    static factory_registerer_type FACTORY_REGISTERER = factory_registerer_type( #Oracle, new abstract_factory_type() );\
}\

namespace shark {

    class Oracle {
    public:
	virtual ~Oracle() {}

	virtual unsigned int answer() const = 0;
    };

    class OracleOfDelphi : public Oracle {
    public:

	unsigned int answer() const { return( 41 ); }
    };

    class HitchhikersGuideToTheGalaxy : public Oracle {
	unsigned int answer() const { return( 42 ); }
    };

    typedef Factory< Oracle, std::string > OracleFactory;

    ANNOUNCE_ORACLE( OracleOfDelphi, OracleFactory );
    ANNOUNCE_ORACLE( HitchhikersGuideToTheGalaxy, OracleFactory );
}

int main( int argc, char ** argv ) {
    
    Oracle * o1 = OracleFactory::instance()[ "Delphi" ];
    if( !o1 )
	std::cout << "Delphi says: " << o1->answer() << std::endl;
    Oracle * o2 = OracleFactory::instance()[ "HitchhikersGuideToTheGalaxy" ];
    if( !o2 )
	std::cout << "HitchhikersGuideToTheGalaxy says: " << o2->answer() << std::endl;
    
    return( EXIT_SUCCESS );

}


