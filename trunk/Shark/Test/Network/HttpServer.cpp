#define BOOST_TEST_MODULE Http_Server
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <iostream>

#include <shark/Core/LogFormatters/PrintfLogFormatter.h>
#include <shark/Core/LogHandlers/StreamHandler.h>

#include <shark/Network/HttpServer.h>

#include <shark/Network/Handlers/AboutHandler.h>
#include <shark/Network/Handlers/FileHandler.h>
#include <shark/Network/Handlers/RestHandler.h>

#include <shark/Rng/GlobalRng.h>

#include <boost/network/protocol/http.hpp>
#include <boost/progress.hpp>

namespace shark {

    struct Producer : public ProbeProvider< Producer > {

	Producer() : ProbeProvider<Producer>( "Producer" ) {
	    for( unsigned int i = 0; i < 30; i++ )
		m_probes.push_back( 
		    registerProbe( 
			( boost::format( "Probe_%1%" ) % i ).str(), 
			"An example probe" 
			) 
		    );
	}

	void start() {
	    m_thread = boost::move( boost::thread( boost::bind( &Producer::run, this ) ) );
	}

	void stop() {
	    m_thread.join();
	}

	void run() {
			
	    while( true ) {
		m_probes[ shark::Rng::discrete( 0, m_probes.size()-1 ) ]->setValue( shark::Rng::gauss() );

		boost::this_thread::sleep( boost::posix_time::milliseconds( 100 ) );
	    }

	}

	std::vector< 
	    boost::shared_ptr< 
		shark::Probe
		>
	    > m_probes;

    boost::thread m_thread;
};

    template< unsigned int N >
    void request(const std::string & url) {
	for( unsigned int i = 0; i < N; i++ ) {
	    boost::network::http::client::request request_( url );
	    
	    request_ << boost::network::header("Connection", "close");
	    
	    boost::network::http::client client_;
	    boost::network::http::client::response response_ = client_.get(request_);
	    
	    std::cout << boost::network::http::body( response_ ) << std::endl;
	}
    }
}

BOOST_AUTO_TEST_SUITE (Network_HttpServer)

BOOST_AUTO_TEST_CASE( HttpServer ) {

    shark::Producer producer;
    producer.start();

    shark::HttpServer httpServer( "0.0.0.0", "9090" );
    
    httpServer.registerHandler( 
	"/About", 
	boost::shared_ptr< shark::HttpServer::AbstractRequestHandler >( new shark::SharkAboutHandler() ) 
	);
    httpServer.registerHandler( 
	"/ProbeManager", 
	boost::shared_ptr< shark::HttpServer::AbstractRequestHandler >( new shark::RestHandler() ) 
	);
    httpServer.setFallbackHandler(
	boost::shared_ptr< shark::HttpServer::AbstractRequestHandler >( new shark::FileHandler( boost::filesystem::initial_path() ) ) 
	);

    try {
	boost::thread serverThread( boost::bind( &shark::HttpServer::start, boost::ref( httpServer ) ) );

	std::string url( "http://localhost:9090/About" );
	boost::progress_timer pt;

	boost::thread t1( boost::bind( shark::request<100>, url ) );
	boost::thread t2( boost::bind( shark::request<100>, url ) );
	boost::thread t3( boost::bind( shark::request<100>, url ) );
    
	t1.join();
	t2.join();
	t3.join();

	std::cout << pt.elapsed() / 3000 << " seconds per request." << std::endl;
    } catch( ... ) {
    }
}

BOOST_AUTO_TEST_SUITE_END()
