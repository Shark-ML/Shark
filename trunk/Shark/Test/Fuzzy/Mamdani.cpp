#define BOOST_TEST_MODULE FuzzySets
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Fuzzy/RuleBase.h>
#include <shark/Fuzzy/LinguisticVariable.h>
#include <shark/Fuzzy/LinguisticTerms/TrapezoidLT.h>
#include <shark/Fuzzy/LinguisticTerms/TriangularLT.h>

#include <shark/Fuzzy/MamdaniIM.h>

/**
  *
  * Simple test case for a fan controller:
  *   Input variables: Temperature (Cold, Warm, Hot), Humidity (dry, moist, wet)
  *   Output variable: Fan speed (Low, Med, High)
  *
  * Rule base:
  *   IF H = WET                THEN FS = HIGH
  *   IF T = COOL AND H = DRY   THEN FS = MED
  *   IF T = COOL AND H = MOIST THEN FS = HIGH
  *   IF T = WARM AND H = DRY   THEN FS = LOW
  *   IF T = WARM AND H = MOIST THEN FS = MED
  *   IF T = HOT  AND H = DRY   THEN FS = MED
  *   IF T = HOT  AND H = MOIST THEN FS = HIGH
  *
  */

BOOST_AUTO_TEST_CASE( Mamdani ) {

    // Input variables
    //define linguistic variable "Temperature"
    boost::shared_ptr<
            shark::LinguisticVariable
    > temperature( new shark::LinguisticVariable( "Temperature", 0. ) );
    
    //define corresponding lingusitic terms
    boost::shared_ptr<
            shark::LinguisticTerm
    > cold( new shark::TrapezoidLT( "Cold", temperature, -1, 0, 40, 80 ) );
    boost::shared_ptr<
            shark::LinguisticTerm
    > warm( new shark::TriangularLT( "Warm", temperature, 60, 80, 100 ) );
    boost::shared_ptr<
            shark::LinguisticTerm
    > hot( new shark::TrapezoidLT( "Hot", temperature, 80, 100, 120, 200 ) );

    temperature->setBounds( 0, 150 );
    temperature->addTerm( cold );
    temperature->addTerm( warm );
    temperature->addTerm( hot );

    BOOST_CHECK_EQUAL( temperature->size(), 3 );
    BOOST_CHECK( cold->linguisticVariable() == temperature );
    BOOST_CHECK( warm->linguisticVariable() == temperature );
    BOOST_CHECK( hot->linguisticVariable() == temperature );

	
	//define linguistic variable "Humidity"
    boost::shared_ptr<
            shark::LinguisticVariable
    > humidity( new shark::LinguisticVariable( "Humidity", 0. ) );
    //define corresponding lingusitic terms
    boost::shared_ptr<
            shark::LinguisticTerm
    > dry( new shark::TrapezoidLT( "Dry", humidity, -1, 0, 30, 40 ) );
    boost::shared_ptr<
            shark::LinguisticTerm
    > moist( new shark::TriangularLT( "Moist", humidity, 40, 60, 80 ) );
    boost::shared_ptr<
            shark::LinguisticTerm
    > wet( new shark::TrapezoidLT( "Wet", humidity, 60, 70, 120, 200 ) );

    humidity->setBounds( 0, 200 );
    humidity->addTerm( dry );
    humidity->addTerm( moist );
    humidity->addTerm( wet );

    BOOST_CHECK_EQUAL( humidity->size(), 3 );
    BOOST_CHECK( dry->linguisticVariable() == humidity );
    BOOST_CHECK( moist->linguisticVariable() == humidity );
    BOOST_CHECK( wet->linguisticVariable() == humidity );

    // Output variables
    //define linguistic variable "FanSpeed"
    boost::shared_ptr<
            shark::LinguisticVariable
    > fanSpeed( new shark::LinguisticVariable( "FanSpeed", 0. ) );
    //define corresponding lingusitic terms
    boost::shared_ptr<
            shark::LinguisticTerm
    > low( new shark::TrapezoidLT( "Low", fanSpeed, -1, 250, 700, 750 ) );
    boost::shared_ptr<
            shark::LinguisticTerm
    > med( new shark::TriangularLT( "Med", fanSpeed, 500, 750, 1000 ) );
    boost::shared_ptr<
            shark::LinguisticTerm
    > high( new shark::TrapezoidLT( "High", fanSpeed, 750, 800, 1250, 1500 ) );

    fanSpeed->setBounds( 250, 1400 );
    fanSpeed->addTerm( low );
    fanSpeed->addTerm( med );
    fanSpeed->addTerm( high );

	//define the rules
    boost::shared_ptr< shark::Rule > r1( new shark::Rule() );
    r1->premise().push_back( wet );
    r1->addConclusion( low );

    boost::shared_ptr< shark::Rule > r2( new shark::Rule() );
    r2->premise().push_back( cold );
    r2->premise().push_back( dry );
    r2->addConclusion( med );

    boost::shared_ptr< shark::Rule > r3( new shark::Rule() );
    r3->premise().push_back( cold );
    r3->premise().push_back( moist );
    r3->addConclusion( high );

    boost::shared_ptr< shark::Rule > r4( new shark::Rule() );
    r4->premise().push_back( warm );
    r4->premise().push_back( dry );
    r4->addConclusion( low );

    boost::shared_ptr< shark::Rule > r5( new shark::Rule() );
    r5->premise().push_back( warm );
    r5->premise().push_back( moist );
    r5->addConclusion( med );

    boost::shared_ptr< shark::Rule > r6( new shark::Rule() );
    r6->premise().push_back( hot );
    r6->premise().push_back( dry );
    r6->addConclusion( med );

    boost::shared_ptr< shark::Rule > r7( new shark::Rule() );
    r7->premise().push_back( hot);
    r7->premise().push_back( moist );
    r7->addConclusion( high );

	//define the rule base
    boost::shared_ptr< shark::RuleBase > rb( new shark::RuleBase() );

	//add variables to rule base
    rb->addToInputFormat( temperature );
    rb->addToInputFormat( humidity );
    rb->addToOutputFormat( fanSpeed );

	//add rules to rule base
    rb->addRule( r1 );
    rb->addRule( r2 );
    rb->addRule( r3 );
    rb->addRule( r4 );
    rb->addRule( r5 );
    rb->addRule( r6 );
    rb->addRule( r7 );

	//define inference machine
    shark::MamdaniIM im( rb );
    //plot the defuzzification results
    im.characteristicCurve( "curve.dat", 100 );
}
