/*
	Read in a mime structure, parse it, and write it back to a file
	with the same name as the input file, but with "-Results" appended to the name

	We don't just write to stdout, because we want to read/write binary data,
	and stdout on some systems eats CRLF, and turns them into newlines.
	
	Returns 0 for success, non-zero for failure
	
*/

#include <boost/mime.hpp>
#include <boost/bind.hpp>

#include <boost/test/included/unit_test.hpp>

#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <algorithm>
#include <exception>

namespace {

    std::string readfile ( const char *fileName ) {
		std::ifstream in ( fileName );
		if ( !in ) {
			std::cerr << std::string ( "Can't open file: " ) + fileName << std::endl;
			throw std::runtime_error ( std::string ( "Can't open file: " ) + fileName );
			}
		
        std::istreambuf_iterator<char> src(in);
        std::istreambuf_iterator<char> eof;
        std::string retVal;

		in >> std::noskipws;
        std::copy(src, eof, std::back_inserter(retVal));
        return retVal;
    }

	struct my_traits {
		typedef	std::string string_type;
	//	typedef std::pair < std::string, string_type > header_type;
		typedef std::string body_type;
		};
	
	//using namespace boost::mime;
	typedef boost::mime::basic_mime<my_traits>	mime_part;
	typedef boost::shared_ptr<mime_part> 		smp;
	
	smp to_mime ( const char *fileName ) {
		std::ifstream in ( fileName );
		if ( !in ) {
			std::cerr << std::string ( "Can't open file: " ) + fileName << std::endl;
			throw std::runtime_error ( std::string ( "Can't open file: " ) + fileName );
			}
		
		in >> std::noskipws;
		return mime_part::parse_mime ( in );
		}
	
	std::string from_mime ( smp mp ) {
		std::ostringstream oss;
		oss << *mp;
		return oss.str ();
		}
	
	void test_roundtrip ( const char *fileName ) {
		smp mp;
		BOOST_REQUIRE_NO_THROW( mp = to_mime ( fileName ));
		BOOST_CHECK_EQUAL ( readfile ( fileName ), from_mime ( mp ));
		}
	
	void test_expected_parse_fail ( const char *fileName ) {
		}
	
}


using namespace boost::unit_test;

test_suite*
init_unit_test_suite( int argc, char* argv[] ) 
{
    framework::master_test_suite().add ( BOOST_TEST_CASE( boost::bind ( test_roundtrip, "TestMessages/00000001" )));
    framework::master_test_suite().add ( BOOST_TEST_CASE( boost::bind ( test_roundtrip, "TestMessages/00000019" )));
    framework::master_test_suite().add ( BOOST_TEST_CASE( boost::bind ( test_roundtrip, "TestMessages/00000431" )));
    framework::master_test_suite().add ( BOOST_TEST_CASE( boost::bind ( test_roundtrip, "TestMessages/00000975" )));
    framework::master_test_suite().add ( BOOST_TEST_CASE( boost::bind ( test_roundtrip, "TestMessages/00001136" )));

//	test cases that fail
//  framework::master_test_suite().add ( BOOST_TEST_CASE( boost::bind ( test_roundtrip, "TestMessages/0019-NoBoundary" )));
    return 0;
}
