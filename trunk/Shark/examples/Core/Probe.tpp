#include <shark/Core/Probe.h>

#include <shark/Rng/GlobalRng.h>
#include <shark/Statistics/Statistics.h>

#include <boost/foreach.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/smart_ptr/scoped_ptr.hpp>

namespace shark {

    /**
     * \brief Produces normally distributed values and reports them via a probe.
     */
    struct Producer : public ProbeProvider< Producer > {

	Producer() : ProbeProvider<Producer>( "Producer" ) {
	    m_probe = registerProbe( "Probe", "An example probe" );
	}

	void start() {
		m_thread.reset(new boost::thread(boost::bind(&Producer::run, this)));
	}

	void stop() {
		if (m_thread)
			m_thread->join();
	}

	void run() {
	    for( unsigned int i = 0; i < 100000; i++ )
		m_probe->setValue( shark::Rng::gauss() );
	}

	boost::shared_ptr<
	    shark::Probe
	    > m_probe;

	boost::scoped_ptr<boost::thread> m_thread;
    };

    /**
     * \brief Consumes floating point values via a probe.
     */
    struct Consumer {
	void operator()( shark::Probe::time_type ts, const shark::Probe::variant_type & value ) {
	    std::cout << "ts:\t" << ts << ", value:\t" << value << std::endl;
	    try {
		m_stats( boost::get< double >( value ) );
	    } catch( boost::bad_get & e ) {
		std::cerr << "Problem casting probe value: " << e.what() << std::endl;
	    }
	}

	shark::Statistics m_stats;
    };
}

int main( int argc, char ** argv ) {

    // Create Producer, register a probe
    shark::Producer producer;

    // Create consumer
    shark::Consumer consumer;

    // Find the probes of all producer instances;
    shark::ProbeManager::Path p;
    p /= shark::ProbeManager::Path( producer.context() ) / shark::ProbeManager::Path( ".*" ) / "Probe";

    std::list<
	boost::shared_ptr<
	    shark::Probe
	    >
	> probes = shark::ProbeManager::instance().find( boost::regex( p.str() ) );

    shark::Statistics stats;

// Register consumer with all probes.
    BOOST_FOREACH( boost::shared_ptr< shark::Probe > & probe, probes ) {
	probe->signalUpdated().connect(
	    boost::bind(
		&shark::Consumer::operator(),
		&consumer,
		_1,
		_2
		)
	    );
    }

// Start the producer
    producer.start();
// Stop the producer
    producer.stop();

// Output statistics
    std::cout << consumer.m_stats( shark::Statistics::Mean() ) << ", " << consumer.m_stats( shark::Statistics::Variance() ) << std::endl;


    return( EXIT_SUCCESS );
}
