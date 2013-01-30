
// Copyright Dean Michael Berris 2009.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)


//[ uri_main
/*`
  This is a simple program that validates a URI.
 */
#include <boost/network/uri.hpp>
#include <boost/network/uri/http/uri.hpp>
#include <string>
#include <iostream>


int main(int argc, char * argv[]) {
    std::string input;
    std::cout << "Please enter a URI to parse: ";
    std::cin >> input;

    /*<< Create a `boost::network::uri::uri` object from the input.  >>*/
    boost::network::uri::uri uri_(input);
    /*<< Check if it's a valid URI. >>*/
    std::cout << "You've entered "
              << (boost::network::uri::valid(uri_)?
                  std::string("a valid") : std::string("an invalid"))
              << " URI!" << std::endl;

    /*<< Create a `boost::network::http::uri` object from the input. >>*/
    boost::network::uri::http::uri http_uri_(input);
    /*<< Check if it's a valid HTTP URI. >>*/
    std::cout << "It's also "
              << (boost::network::uri::valid(http_uri_)?
                  std::string("a valid HTTP URI") : std::string("an invalid HTTP URI."))
              << "!" << std::endl;

    return EXIT_SUCCESS;
}
//]
