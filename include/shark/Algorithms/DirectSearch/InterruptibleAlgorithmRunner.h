/**
 *
 *  \brief Executes multi-objective optimizers.
 *
 *  \author T.Voss
 *  \date 2010
 *
 *
 *  <BR><HR>
 *  This file is part of Shark. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 3, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef INTERRUPTIBLE_ALORITHM_RUNNER_H
#define INTERRUPTIBLE_ALORITHM_RUNNER_H

#include <shark/Algorithms/DirectSearch/Experiments/Experiment.h>

#include <shark/Core/Traits/OptimizerTraits.h>

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>

#include <shark/Core/Exception.h>
#include <shark/Core/SignalTrap.h>

#include <shark/Core/Logger.h>
#include <shark/Core/LogHandlers/StreamHandler.h>
#include <shark/Core/LogFormatters/PrintfLogFormatter.h>

#include <shark/Algorithms/DirectSearch/Experiments/FrontStore.h>

#include <shark/Rng/GlobalRng.h>

#include <boost/bind.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/optional.hpp>
#include <boost/progress.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/signals.hpp>

#include <fstream>
#include <string>

namespace shark {

namespace moo {

/**
 * \brief Executes one trial of a multi-objective optimizer for a given
 * multi-objective fitness function.
 *
 * \tparam Algo Models the type of the optimizer.
 * \tparam Function Models the type of the objective function.
 */
template<typename Algo, typename Function>
class InterruptibleAlgorithmRunner {
public:

	/** \brief Make the algorithm type known to the outside world.*/
	typedef Algo algo_type;

	/** \brief Result type announced from this class. */
	typedef typename algo_type::SolutionSetType result_type;

	/** \brief Make the function type known to the outside world.*/
	typedef Function function_type;

public:

	/** \brief Metadata describing actual result data. */
	struct ResultMetaData {
		ResultMetaData(
				const std::string & optimizerName,
				const std::string & objectiveFunctionName,
				std::size_t seed,
				std::size_t searchSpaceDimension,
				std::size_t objectiveSpaceDimension,
				std::size_t evaluationCounter,
				double timeStamp,
				bool isFinal)
		: m_optimizerName( optimizerName ),
			m_objectiveFunctionName( objectiveFunctionName ),
			m_seed( seed ),
			m_searchSpaceDimension( searchSpaceDimension ),
			m_objectiveSpaceDimension( objectiveSpaceDimension ),
			m_evaluationCounter( evaluationCounter ),
			m_timeStamp( timeStamp ),
			m_isFinal( isFinal )
		{ }

		std::string m_optimizerName;
		std::string m_objectiveFunctionName;
		std::size_t m_seed;
		std::size_t m_searchSpaceDimension;
		std::size_t m_objectiveSpaceDimension;
		std::size_t m_evaluationCounter;
		double m_timeStamp;
		bool m_isFinal;
	};

	/**
	 * \brief Signal for delivering results to the outside world.
	 */
	typedef boost::signal<
		void
		(
			SHARK_ARGUMENT( const result_type &, "Actual optimization results" ),
			SHARK_ARGUMENT( const ResultMetaData & , "Meta data for actual optimization results" )
		)
	> event_type;

	/**
	 * \brief C'tor. 
	 */
	InterruptibleAlgorithmRunner( 
		boost::shared_ptr< Algo > algo = boost::shared_ptr< Algo >( new Algo ), 
		boost::shared_ptr< Function > function = boost::shared_ptr< Function >( new Function() ) )
	: mep_algorithm( algo )
	, mep_function( function )
	{
		SignalTrap::instance().signalTrapped().connect(
		boost::bind( &InterruptibleAlgorithmRunner< Algo, Function >::signalTrap,
				this, _1 ) );
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "InterruptibleAlgorithmRunner<" + mep_algorithm->name() + "," + mep_function->name() + ">"; }

	/**
	 * \brief Triggered when new results are available.
	 */
	event_type & signalResultsAvailable() {
		return( m_signalResultsAvailable );
	}

	/**
	* \brief Executes the optimizer trial for the given parameters.
	*/
	void run( unsigned int seed, 
			unsigned int interval, 
			unsigned int n, 
			unsigned int m, 
			unsigned int g, 
			unsigned int timeLimit, 
			const boost::optional< boost::property_tree::ptree > & configNode = boost::optional< boost::property_tree::ptree >() )
	{ 
		shark::Rng::seed( seed );

		m_n = n;
		m_m = m;
		m_seed = seed;
		m_algoName = mep_algorithm->name();
		m_functionName = mep_function->name();

		
		mep_function->setNumberOfVariables( n );
		//mep_function->setNoObjectives( m );	
		mep_function->init();

		if( configNode ) {
			mep_algorithm->configure( *configNode );
		}

		mep_algorithm->init( *mep_function.get() );

		boost::progress_display pd( g, std::clog );
		m_pt.restart();

		std::size_t currentEvaluationCount = 0;

		while( mep_function->evaluationCounter() <= g && m_pt.elapsed() / 3600  < timeLimit ) {
			mep_algorithm->step( *mep_function.get() );
			m_front = mep_algorithm->solution();

			if( mep_function->evaluationCounter() % interval == 0 ) {

				m_signalResultsAvailable(
					m_front, 
					ResultMetaData(
						m_algoName, 
						m_functionName, 
						m_seed, 
						m_n, 
						m_m, 
						mep_function->evaluationCounter(), 
						m_pt.elapsed(), 
						false));
			}
			pd += mep_function->evaluationCounter() - currentEvaluationCount;
			currentEvaluationCount = mep_function->evaluationCounter();
		}

		m_signalResultsAvailable(
			m_front, 
			ResultMetaData(
				m_algoName,
				m_functionName,
				m_seed,
				m_n,
				m_m,
				mep_function->evaluationCounter(),
				m_pt.elapsed(),
				true ) );
	}

	/**
	 * \brief Reports known fitness functions.
	 */
	DEFINE_SIMPLE_OPTION( ReportFitnessFunctionsTag, reportFitnessFunctions );


	static int main( int argc, char ** argv ) {
		// Set up logging.
		shark::Shark::logger()->setLogLevel( shark::Logger::DEBUG_LEVEL );
		boost::shared_ptr<
			shark::Logger::AbstractHandler
		> plaintTextLogHandler( shark::LogHandlerFactory::instance()[ "ClogLogHandler" ] );

		plaintTextLogHandler->setFormatter(
		boost::shared_ptr<
			shark::Logger::AbstractFormatter
		>( shark::LogFormatterFactory::instance()[ "PlainTextLogFormatter" ] ) );

		shark::Shark::logger()->registerHandler( plaintTextLogHandler );

		shark::moo::Experiment::Options options;
		options.addDefaultOptions();
		options.addOption( ReportFitnessFunctionsTag() );

		if( !options.parse( argc, argv ) ) {
			SHARK_LOG_ERROR( shark::Shark::logger(), "Problem parsing command line", "InterruptibleAlgorithmRunner::main" );
			options.printOptions( std::cerr );
			return( EXIT_FAILURE );
		}

		if( options.hasValue( ReportFitnessFunctionsTag() ) ) {
			shark::moo::RealValuedObjectiveFunctionFactory::instance().print( std::cout );
			return( EXIT_SUCCESS );
		}

		if(options.hasValue( shark::moo::Experiment::Options::DefaultAlgorithmUsageTag()) ) {
			boost::property_tree::ptree pt;
			OptimizerTraits<algo_type>::defaultConfig(pt);
			boost::property_tree::write_json(std::cout, pt);
			return( EXIT_SUCCESS );
		}

		if( options.value( shark::moo::Experiment::Options::ObjectiveFunctionTag() ).empty() ) {
			SHARK_LOG_ERROR( shark::Shark::logger(), "Missing objective function, aborting now.", "InterruptibleAlgorithmRunner::main" );
			options.printOptions( std::cerr );
			return( EXIT_FAILURE );
		}

		boost::optional< boost::property_tree::ptree > configurationTree;

		if( !options.value( shark::moo::Experiment::Options::AlgorithmConfigFile() ).empty() ) {
			try {
				boost::property_tree::ptree pt;
				boost::property_tree::read_json( options.value( shark::moo::Experiment::Options::AlgorithmConfigFile() ), pt );
				configurationTree = pt;
			} catch( ... ) {
				SHARK_LOG_ERROR( shark::Shark::logger(), "Problem reading algorithm configuration file.", "InterruptibleAlgorithmRunner::main" );
				return( EXIT_FAILURE );
			}
		}

		std::string resultFormat = options.value( shark::moo::Experiment::Options::ResultFormatTag());

		std::string objectiveFunction = options.value( shark::moo::Experiment::Options::ObjectiveFunctionTag() );

		SHARK_LOG_DEBUG( shark::Shark::logger(), "Considering objective function: " + objectiveFunction, "InterruptibleAlgorithmRunner::main" );

		typedef shark::moo::InterruptibleAlgorithmRunner< Algo, Function > runner_type;
		typedef shark::FrontStore< typename runner_type::result_type, typename runner_type::ResultMetaData > front_store_type;

		front_store_type frontStore;    
		frontStore.m_resultDir = options.value( shark::moo::Experiment::Options::ResultDirTag() );
		if(resultFormat == "JSON")
			frontStore.m_format = front_store_type::JSON_FORMAT;
		else if(resultFormat == "RawText")
			frontStore.m_format = front_store_type::RAW_TEXT_FORMAT;

		SHARK_LOG_DEBUG(
			shark::Shark::logger(),
			"Storing results to: " + frontStore.m_resultDir,
			"InterruptibleAlgorithmRunner::main" );

		runner_type abstractRunner( 
			boost::shared_ptr< 
				Algo 
			>( new Algo() ),
			boost::shared_ptr< 
				Function
			>( shark::Factory< Function, std::string >::instance()[ objectiveFunction ] ) );

		abstractRunner.signalResultsAvailable().connect(
			boost::bind( 
				&front_store_type::onNewResult, 
				boost::ref( frontStore ),
				_1,
				_2 ) );

		try {
			abstractRunner.run( 
			options.value( shark::moo::Experiment::Options::SeedTag() ),
			options.value( shark::moo::Experiment::Options::StorageIntervalTag() ),
			options.value( shark::moo::Experiment::Options::SearchSpaceDimensionTag() ),
			options.value( shark::moo::Experiment::Options::ObjectiveSpaceDimensionTag() ),
			options.value( shark::moo::Experiment::Options::MaxNoEvaluationsTag() ),
			options.value( shark::moo::Experiment::Options::TimeLimitTag() ),
			configurationTree
						  ); 
		} catch( const shark::Exception & e ) {
			SHARK_LOG_ERROR( shark::Shark::logger(), ( boost::format( "Exception while running: %1% [%2%@%3%]" ) % e.what() % e.file() % e.line() ).str(), "InterruptibleAlgorithmRunner::main" );
		}
		return( EXIT_SUCCESS );
	}

protected:

	/**
	 * \brief Stores the current front.
	 */
	typename algo_type::SolutionSetType m_front;	

	/**
	 * \brief Reference to the optimizer.
	 */
	boost::shared_ptr< Algo > mep_algorithm;

	/**
	 * \brief Reference to the objective function.
	 */
	boost::shared_ptr< Function > mep_function;

	/**
	 * \brief Stores the name of the optimizer.
	 */
	std::string m_algoName;

	/**
	 * \brief Stores the name of the objective function.
	 */
	std::string m_functionName;

	/**
	 * \brief Stores the initial seed of the trial.
	 */
	std::size_t m_seed;

	/**
	 * \brief Stores the dimension of the search space n.
	 */
	std::size_t m_n;

	/**
	 * \brief Store the number of objectives m.
	 */
	std::size_t m_m;

	/**
	 * \brief Timer instance for time-limited experiments.
	 */
	boost::timer m_pt;

	/**
	 * \brief Signal for delivering results to the outside world.
	 */
	event_type m_signalResultsAvailable;

	/**
	 * \brief Slot that is called when a signal is emitted from the 
	 * signal trap. Stores the current front.
	 *
	 * \param [in] signal The emitted signal
	 */
	void signalTrap( SignalTrap::SignalType signal ) {
		if( m_front.empty() ) return;

		m_signalResultsAvailable(
			m_front, 
			ResultMetaData(
				m_algoName, 
				m_functionName, 
				m_seed, 
				m_n, 
				m_m, 
				mep_function->evaluationCounter(), 
				m_pt.elapsed(), 
				true));
		exit( EXIT_SUCCESS );
	}
};


} // namespace moo

namespace soo {

/**
 * \brief Executes one trial of a single-objective optimizer for a given
 * single-objective fitness function.
 *
 * \tparam Algo Models the type of the optimizer.
 * \tparam Function Models the type of the objective function.
 */
template<typename Algo, typename Function>
class InterruptibleAlgorithmRunner {
 public:

  /** \brief Make the algorithm type known to the outside world.*/
  typedef Algo algo_type;

  /** \brief Result type announced from this class. */
  typedef typename algo_type::SolutionSetType result_type;

  /** \brief Make the function type known to the outside world.*/
  typedef Function function_type;

 public:

  /** \brief Metadata describing actual result data. */
  struct ResultMetaData {

    ResultMetaData(
        const std::string & optimizerName,
        const std::string & objectiveFunctionName,
        std::size_t seed,
        std::size_t searchSpaceDimension,
        std::size_t evaluationCounter,
        double timeStamp,
        bool isFinal
                   ) : m_optimizerName( optimizerName ),
                       m_objectiveFunctionName( objectiveFunctionName ),
                       m_seed( seed ),
                       m_searchSpaceDimension( searchSpaceDimension ),
                       m_evaluationCounter( evaluationCounter ),
                       m_timeStamp( timeStamp ),
                       m_isFinal( isFinal ) {
    }


    std::string m_optimizerName;
    std::string m_objectiveFunctionName;
    std::size_t m_seed;
    std::size_t m_searchSpaceDimension;
    std::size_t m_evaluationCounter;
    double m_timeStamp;
    bool m_isFinal;
  };

  /**
   * \brief Signal for delivering results to the outside world.
   */
  typedef boost::signal< 
    void
    ( 
        SHARK_ARGUMENT( const result_type &, "Actual optimization results" ),
        SHARK_ARGUMENT( const ResultMetaData & , "Meta data for actual optimization results" )
      )
    > event_type;

  /**
   * \brief C'tor. 
   */
  InterruptibleAlgorithmRunner( 
      boost::shared_ptr< Algo > algo = boost::shared_ptr< Algo >( new Algo ), 
      boost::shared_ptr< Function > function = boost::shared_ptr< Function >( new Function() ) ) : mep_algorithm( algo ),
                                                                                                   mep_function( function ) {
    SignalTrap::instance().signalTrapped().connect( boost::bind( &InterruptibleAlgorithmRunner< Algo, Function >::signalTrap, this, _1 ) );
  }

  /**
   * \brief Triggered when new results are available.
   */
  event_type & signalResultsAvailable() {
    return( m_signalResultsAvailable );
  }
  /**
   * \brief Executes the optimizer trial for the given parameters.
   */
  void run( unsigned int seed, 
            unsigned int interval, 
            unsigned int n, 
            unsigned int g, 
            unsigned int timeLimit, 
            double fitnessLimit = 1E-10,
            const boost::optional< boost::property_tree::ptree > & configNode = boost::optional< boost::property_tree::ptree >() 
            ) { 
    SHARK_LOG_DEBUG( 
        shark::Shark::logger(), 
        ( boost::format( "run( seed=%1%, interval=%2%, n=%3%, g=%4%, timeLimit=%5%, fitnessLimit=%6% )" ) % seed % interval % n % g % timeLimit % fitnessLimit ).str(),
        "shark::soo::InterruptibleAlgorithmRunner::run()"
                     );
    shark::Rng::seed( seed );

    m_n = n;
    m_seed = seed;
    m_algoName = mep_algorithm->name();
    m_functionName = mep_function->name();

    //ObjectiveFunctionTraits::setNumberOfVariables( *(mep_function.get()), n )
    mep_function->setNumberOfVariables( n );
    mep_function->init();

    if( configNode ) {
      mep_algorithm->configure( *configNode );
    }

    mep_algorithm->init( *mep_function.get() );

    boost::progress_display pd( g, std::clog );
    m_pt.restart();

    std::size_t currentEvaluationCount = 0;

    while( mep_function->evaluationCounter() <= g && m_pt.elapsed() / 3600  < timeLimit && mep_algorithm->solution().value > fitnessLimit ) {
      mep_algorithm->step( *mep_function.get() );
      m_result = mep_algorithm->solution();

      if( mep_function->evaluationCounter() % interval == 0 ) {

        m_signalResultsAvailable(
            m_result, 
            ResultMetaData(
                m_algoName, 
                m_functionName, 
                m_seed, 
                m_n, 
                mep_function->evaluationCounter(), 
                m_pt.elapsed(), 
                false
                           )
                                 );
      }
      pd += mep_function->evaluationCounter() - currentEvaluationCount;
      currentEvaluationCount = mep_function->evaluationCounter();
    }

    m_signalResultsAvailable(
        m_result, 
        ResultMetaData(
            m_algoName, 
            m_functionName, 
            m_seed, 
            m_n, 
            mep_function->evaluationCounter(), 
            m_pt.elapsed(), 
            true
                       )
                             );
  }

  /**
   * \brief Reports known fitness functions.
   */
  DEFINE_SIMPLE_OPTION( ReportFitnessFunctionsTag, reportFitnessFunctions );

  static int main( int argc, char ** argv ) {
    // Set up logging.
    shark::Shark::logger()->setLogLevel( shark::Logger::DEBUG_LEVEL );
    boost::shared_ptr<
        shark::Logger::AbstractHandler
        > plaintTextLogHandler( 
            shark::LogHandlerFactory::instance()[ "CoutLogHandler" ] 
                                );
    plaintTextLogHandler->setFormatter(
        boost::shared_ptr<
        shark::Logger::AbstractFormatter
        >( shark::LogFormatterFactory::instance()[ "PlainTextLogFormatter" ] )
                                       );
    shark::Shark::logger()->registerHandler( plaintTextLogHandler );

    shark::soo::Experiment::Options options;
    options.addDefaultOptions();
    options.addOption( ReportFitnessFunctionsTag() );

    if( !options.parse( argc, argv ) ) {
      SHARK_LOG_ERROR( shark::Shark::logger(), "Problem parsing command line", "InterruptibleAlgorithmRunner::main" );
      options.printOptions( std::cerr );
      return( EXIT_FAILURE );
    }

    if( options.hasValue( ReportFitnessFunctionsTag() ) ) {
      shark::soo::RealValuedObjectiveFunctionFactory::instance().print( std::cout );
      return( EXIT_SUCCESS );
    }

    if(options.hasValue(
           shark::soo::Experiment::Options::DefaultAlgorithmUsageTag())) {
      boost::property_tree::ptree pt;
      OptimizerTraits<algo_type>::defaultConfig(pt);
      boost::property_tree::write_json(std::cout, pt);
      return( EXIT_SUCCESS );
    }

    if( options.value( shark::moo::Experiment::Options::ObjectiveFunctionTag() ).empty() ) {
      SHARK_LOG_ERROR( shark::Shark::logger(), "Missing objective function, aborting now.", "InterruptibleAlgorithmRunner::main" );
      options.printOptions( std::cerr );
      return( EXIT_FAILURE );
    }	

    boost::optional< boost::property_tree::ptree > configurationTree;

    if( !options.value( shark::moo::Experiment::Options::AlgorithmConfigFile() ).empty() ) {
      try {
        boost::property_tree::ptree pt;
        boost::property_tree::read_json( options.value( shark::moo::Experiment::Options::AlgorithmConfigFile() ), pt );
        configurationTree = pt;
      } catch( ... ) {
        SHARK_LOG_ERROR( shark::Shark::logger(), "Problem reading algorithm configuration file.", "InterruptibleAlgorithmRunner::main" );
        return( EXIT_FAILURE );
      }

    }

    std::string objectiveFunction = options.value( shark::moo::Experiment::Options::ObjectiveFunctionTag() );

    SHARK_LOG_DEBUG( shark::Shark::logger(), "Considering objective function: " + objectiveFunction, "InterruptibleAlgorithmRunner::main" );

    typedef shark::soo::InterruptibleAlgorithmRunner<
        Algo,
        Function
        > runner_type;

    /*
      typedef shark::FrontStore< 
      typename runner_type::result_type,
      typename runner_type::ResultMetaData
      > front_store_type;

      front_store_type frontStore;
      frontStore.m_resultDir = options.value( shark::moo::Experiment::Options::ResultDirTag() );

      SHARK_LOG_DEBUG( shark::Shark::logger(), "Storing results to: " + frontStore.m_resultDir, "InterruptibleAlgorithmRunner::main" );
    */

    runner_type abstractRunner( 
        boost::shared_ptr< 
        Algo 
        >( new Algo() ),
        boost::shared_ptr< 
        Function
        >( shark::Factory< Function, std::string >::instance()[ objectiveFunction ] )
                                );

    /*
      abstractRunner.signalResultsAvailable().connect( 
      boost::bind( 
      &front_store_type::onNewResult, 
      boost::ref( frontStore ),
      _1,
      _2
      )
      );*/



    try {
      abstractRunner.run( 
          options.value( shark::moo::Experiment::Options::SeedTag() ),
          options.value( shark::moo::Experiment::Options::StorageIntervalTag() ),
          options.value( shark::moo::Experiment::Options::SearchSpaceDimensionTag() ),
          options.value( shark::moo::Experiment::Options::MaxNoEvaluationsTag() ),
          options.value( shark::moo::Experiment::Options::TimeLimitTag() ),
          options.value( shark::moo::Experiment::Options::FitnessLimitTag() ),
          configurationTree
                          ); 
    } catch( const shark::Exception & e ) {
      SHARK_LOG_ERROR( shark::Shark::logger(), ( boost::format( "Exception while running: %1% [%2%@%3%]" ) % e.what() % e.file() % e.line() ).str(), "InterruptibleAlgorithmRunner::main" );
    }
    return( EXIT_SUCCESS );
  }

 protected:

  /**
   * \brief Stores the current front.
   */
  result_type m_result;	

  /**
   * \brief Reference to the optimizer.
   */
  boost::shared_ptr< Algo > mep_algorithm;

  /**
   * \brief Reference to the objective function.
   */
  boost::shared_ptr< Function > mep_function;

  /**
   * \brief Stores the name of the optimizer.
   */
  std::string m_algoName;

  /**
   * \brief Stores the name of the objective function.
   */
  std::string m_functionName;

  /**
   * \brief Stores the initial seed of the trial.
   */
  std::size_t m_seed;

  /**
   * \brief Stores the dimension of the search space n.
   */
  std::size_t m_n;

  /**
   * \brief Timer instance for time-limited experiments.
   */
  boost::timer m_pt;

  /**
   * \brief Target fitness function value.
   */
  double m_fitnessLimit;

  /**
   * \brief Signal for delivering results to the outside world.
   */
  event_type m_signalResultsAvailable;

  /**
   * \brief Slot that is called when a signal is emitted from the 
   * signal trap. Stores the current front.
   *
   * \param [in] signal The emitted signal
   */
  void signalTrap( SignalTrap::SignalType signal ) {

    m_signalResultsAvailable(
        m_result, 
        ResultMetaData(
            m_algoName, 
            m_functionName, 
            m_seed, 
            m_n, 
            mep_function->evaluationCounter(), 
            m_pt.elapsed(), 
            true
                       )
                             );
    exit( EXIT_SUCCESS );
  }
};
}
}

#endif
