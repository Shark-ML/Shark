#define BOOST_TEST_MODULE LinguisticTerms
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Fuzzy/FCL/FuzzyControlLanguageParser.h>

#include <shark/Fuzzy/LinguisticTerms/BellLT.h>
#include <shark/Fuzzy/LinguisticTerms/ComposedLT.h>
#include <shark/Fuzzy/LinguisticTerms/ConstantLT.h>
#include <shark/Fuzzy/LinguisticTerms/CustomizedLT.h>
#include <shark/Fuzzy/LinguisticTerms/GeneralizedBellLT.h>
#include <shark/Fuzzy/LinguisticTerms/InfinityLT.h>
#include <shark/Fuzzy/LinguisticTerms/SigmoidalLT.h>
#include <shark/Fuzzy/LinguisticTerms/SingletonLT.h>
#include <shark/Fuzzy/LinguisticTerms/TrapezoidLT.h>
#include <shark/Fuzzy/LinguisticTerms/TriangularLT.h>

#include <fstream>

BOOST_AUTO_TEST_CASE( LinguisticTerms ) {
    typedef shark::LinguisticTermFactory::class_type linguistic_term_type;

    shark::LinguisticTermFactory::instance().print( std::cout );

    shark::LinguisticTermFactory::const_iterator it;
    for( it = shark::LinguisticTermFactory::instance().begin();
            it != shark::LinguisticTermFactory::instance().end();
            ++it
    ) {

        std::cout << "Considering linguistic term: " << it->first << std::endl;
        BOOST_CHECK( it->second != NULL );
        boost::shared_ptr< linguistic_term_type > lt( it->second->create() );

        BOOST_CHECK( lt );

    }
}
