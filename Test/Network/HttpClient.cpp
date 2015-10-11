#include <boost/network/protocol/http.hpp>
#include <boost/progress.hpp>

namespace shark {

    template< unsigned int N >
    void request(const std::string & url) {
	for( unsigned int i = 0; i < N; i++ ) {
	    try {
		boost::network::http::client::request request_( url );
	
		request_ << boost::network::header("Connection", "close");

		boost::network::http::client client_;
		boost::network::http::client::response response_ = client_.get(request_);
	
		std::cout << boost::network::http::body( response_ ) << std::endl;
	    } catch( ... ) {
	    }
	}
    }
}

int main( int argc, char ** argv ) {

    std::string url( "http://localhost:9090/About" );
    if( argc > 1 )
	url = argv[1];
    
    boost::progress_timer pt;

    boost::thread t1( boost::bind( shark::request<100>, url ) );
    boost::thread t2( boost::bind( shark::request<100>, url ) );
    boost::thread t3( boost::bind( shark::request<100>, url ) );
    
    t1.join();
    t2.join();
    t3.join();

    std::cout << pt.elapsed() / 30000 << " seconds per request." << std::endl;

    return( EXIT_SUCCESS );
}

BOOST_AUTO_TEST_SUITE_END()
