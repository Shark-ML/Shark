#define BOOST_TEST_MODULE FuzzySets
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Fuzzy/FCL/FuzzyControlLanguageParser.h>

#include <shark/Fuzzy/FuzzySets/BellFS.h>
#include <shark/Fuzzy/FuzzySets/ComposedFS.h>
// #include <shark/Fuzzy/FuzzySets/ComposedNDimFS.h>
#include <shark/Fuzzy/FuzzySets/ConstantFS.h>
#include <shark/Fuzzy/FuzzySets/CustomizedFS.h>
#include <shark/Fuzzy/FuzzySets/GeneralizedBellFS.h>
// #include <shark/Fuzzy/FuzzySets/HomogenousNDimFS.h>
#include <shark/Fuzzy/FuzzySets/InfinityFS.h>
//#include <shark/Fuzzy/FuzzySets/NDimFS.h>
#include <shark/Fuzzy/FuzzySets/SigmoidalFS.h>
#include <shark/Fuzzy/FuzzySets/SingletonFS.h>
#include <shark/Fuzzy/FuzzySets/TrapezoidFS.h>
#include <shark/Fuzzy/FuzzySets/TriangularFS.h>

#include <fstream>

BOOST_AUTO_TEST_CASE( FuzzySets ) {
    typedef shark::FuzzySetFactory::class_type fuzzy_set_type;

    shark::FuzzySetFactory::instance().print( std::cout );

    shark::FuzzySetFactory::const_iterator it;
    for( it = shark::FuzzySetFactory::instance().begin();
            it != shark::FuzzySetFactory::instance().end();
            ++it
    ) {

        std::cout << "Considering fuzzy sets: " << it->first << std::endl;
        BOOST_CHECK( it->second != NULL );
        boost::shared_ptr< fuzzy_set_type > fuzzySet( it->second->create() );

        BOOST_CHECK( fuzzySet );

    }
}
