
//          Copyright Dean Michael Berris 2008.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

//[ http_client_main
/*`
  This application takes a URL as a command line argument and prints
  the resource to the console.
*/
#include <boost/program_options.hpp>
#include <boost/network/protocol/http.hpp>
#include <boost/function_output_iterator.hpp>
#include <string>
#include <iostream>

namespace po = boost::program_options;
using namespace std;

struct header_printer {
    std::ostream & os;
    header_printer(std::ostream & os_) : os(os_) {}
    template <class Pair>
    void operator()(Pair const & p) {
        os << p.first << ": " << p.second << endl;
    }
};

int main(int argc, char * argv[]) {
    po::options_description options("Allowed options");
    string output_filename, source;
    bool show_headers;
    options.add_options()
        ("help,h", "produce help message")
        ("headers,H", "print headers")
        ("source,s", po::value<string>(&source), "source URL")
        ;

    po::positional_options_description positional_options;
    positional_options.add("source", 1);
    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(options).positional(positional_options).run(),
            vm);
        po::notify(vm);
    } catch(exception & e) {
        cout << "Error: " << e.what() << endl;
        cout << options << endl;
        return EXIT_FAILURE;
    };

    if (vm.count("help")) {
        cout << options << endl;
        return EXIT_SUCCESS;
    };

    if (vm.count("source") < 1) {
        cout << "Error: Source URL required." << endl;
        cout << options << endl;
        return EXIT_FAILURE;
    };

    show_headers = vm.count("headers") ? true : false ;

    using namespace boost::network;

    typedef http::basic_client<http::tags::http_async_8bit_tcp_resolve, 1, 0>
        http_client;
    
    http_client::request request(source);
    http_client::string_type destination_ = host(request);
    
    request << ::boost::network::header("Connection", "close");
    http_client client(http::_follow_redirects=true);
    http_client::response response = client.get(request);

    if (show_headers) {
        headers_range<http_client::response>::type headers_ = response.headers();
        std::copy(headers_.begin(), headers_.end(),
            boost::make_function_output_iterator(header_printer(cout)));
        cout << endl;
    };
    
    body_range<http_client::response>::type body_ = body(response).range();
    boost::copy(body_, std::ostream_iterator<char_<http_client::request::tag>::type>(cout));
    
    return EXIT_SUCCESS;
}
//]
