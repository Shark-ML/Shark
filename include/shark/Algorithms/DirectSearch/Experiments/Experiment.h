/**
 *
 *  \brief Single-objective and multi-objective experiments for evolutionary
 *	algorithms.
 *
 *  \author T.Voss
 *  \date 2010
 *
 *  \par Copyright (c) 1998-2007:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
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
#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_EXPERIMENTS_EXPERIMENT_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_EXPERIMENTS_EXPERIMENT_H

#include <boost/mpl/if.hpp>
#include <boost/program_options.hpp>

/**
 * \brief Defines a command line option.
 * \param TagName Tag that identifies the option.
 * \param OptionName Actual name of the option.
 */
#define DEFINE_SIMPLE_OPTION( TagName, OptionName )             \
  struct TagName {                                              \
    typedef bool type;                                          \
    BOOST_STATIC_CONSTANT( bool, HAS_VALUE = false );           \
    static const char * name() { return( #OptionName ); };      \
    static const char * description() { return( "" ); };        \
    static type defaultValue() { return( false ); }             \
  };                                                            \

/**
 * \brief Defines a command line option carrying a value.
 * \param TagName Tag that identifies the option.
 * \param OptionName Actual name of the option.
 * \param TypeName Type of the value.
 * \param DefaultValue Default value of the option.
 */
#define DEFINE_OPTION( TagName, OptionName, TypeName, DefaultValue )    \
  struct TagName {                                                      \
    typedef TypeName type;                                              \
    BOOST_STATIC_CONSTANT( bool, HAS_VALUE = true );                    \
    static const char * name() { return( #OptionName ); };              \
    static const char * description() { return( "" ); };                \
    static type defaultValue() { return( DefaultValue ); }              \
  };                                                                    \


namespace shark {

/**
 *  \brief Single-objective and multi-objective experiments for evolutionary
 *	algorithms.
 */
class ExperimentBase {
 public:

  /**
   * \brief Collection of an extensible set of options for a
   * multi-objective experiment
   */
  class Options {
   protected:
    /** \brief Stores known options. */
    boost::program_options::options_description m_options;
    /** \brief Stores parsed options. */
    boost::program_options::variables_map m_variablesMap;
   public:

    /** \brief Virtual d'tor */
    virtual ~Options() {}

    /**
     * \brief Reads the value of the option specified by the tag.
     * \tparam Tag Tag structure. Needs to provide:
     *   - static const bool HAS_VALUE
     *   - static const char * name();
     *   - static const char * description();
     *	- typename type;
     *   - static type defaultValue();
     */
    template<typename Tag>
    typename Tag::type value( const Tag & t ) {
      return( m_variablesMap[ Tag::name() ].template as< typename Tag::type >() );
    }

    /**
     * \brief Checks whether the option specified by tag has been parsed.
     * \tparam Tag Tag structure. Needs to provide:
     *   - static const bool HAS_VALUE
     *   - static const char * name();
     *   - static const char * description();
     *	- typename type;
     *   - static type defaultValue();
     */
    template<typename Tag>
    bool hasValue( const Tag & t ) {
      return( m_variablesMap.count( Tag::name() ) > 0 );
    }

    /**
     * \brief Adds the supplied option tag.
     * \tparam Tag Tag structure. Needs to provide:
     *   - static const bool HAS_VALUE
     *   - static const char * name();
     *   - static const char * description();
     *	- typename type;
     *   - static type defaultValue();
     */
    template<typename Tag>
    void addOption( const Tag & t ) {
      if( Tag::HAS_VALUE ) {
        m_options.add_options()
            ( 
                Tag::name(), 
                boost::program_options::value< typename Tag::type >()->default_value( Tag::defaultValue() ),
                Tag::description()
              );
      } else {
        m_options.add_options()
            ( 
                Tag::name(), 
                Tag::description()
              );
      }
    }

    /**
     * \brief Parses the command line.
     * \returns true if parsing was successful, false otherwise.
     */
    bool parse( int argc, char ** argv ) {
      try {		
        boost::program_options::store( boost::program_options::parse_command_line(argc, argv, m_options ), m_variablesMap );
        boost::program_options::notify( m_variablesMap );
      } catch( ... ) {
        return( false );
      }
      return( true );
    }

    /**
     * \brief Inserts default options of the experiment.
     */
    virtual void addDefaultOptions() {
      addOption( ObjectiveFunctionTag() );
      addOption( SeedTag() );
      addOption( StorageIntervalTag() );
      addOption( SearchSpaceDimensionTag() );
      addOption( MaxNoEvaluationsTag() );
      addOption( TimeLimitTag() );
      addOption( FitnessLimitTag() );
      addOption( ResultDirTag() );
      addOption( ResultFormatTag() );
      addOption( AlgorithmConfigFile() );
      addOption( AlgorithmUsageTag() );
      addOption( DefaultAlgorithmUsageTag() );
    }

    /**
     * \brief Outputs the configured options to the supplied stream.
     * \tparam Stream Type of thream, needs to provide operator<<.
     * \param [in,out] s Instance of the stream.
     */
    template<typename Stream>
    void printOptions( Stream & s ) const {
      s << m_options;
    }

    /** \brief Name of the objective function, passed to factory. */
    DEFINE_OPTION( ObjectiveFunctionTag, objectiveFunction, std::string, "" );
    /** \brief Seed for the RNG */
    DEFINE_OPTION( SeedTag, seed, unsigned int, 1 );
    /** \brief Storage frequency for optimizer results. */
    DEFINE_OPTION( StorageIntervalTag, storageInterval, unsigned int, 100 );
    /** \brief Dimension of the search space. */
    DEFINE_OPTION( SearchSpaceDimensionTag, searchSpaceDimension, unsigned int, 10 );
    /** \brief Termination criterion, maximum number of evaluations. */
    DEFINE_OPTION( MaxNoEvaluationsTag, maxNoEvaluations, unsigned int, 50000 );
    /** \brief Termination criterion, maximum runtime in hours. */
    DEFINE_OPTION( TimeLimitTag, timeLimit, unsigned int, 1000 );
    /** \brief Termination criterion, fitness limit. */
    DEFINE_OPTION( FitnessLimitTag, fitnessLimit, double, 1E-10 );
    /** \brief Directory to place results in. */
    DEFINE_OPTION( ResultDirTag, resultDir, std::string, "." );
    /** \brief Configuration file for the optimizer, JSON syntax. */
    DEFINE_OPTION( AlgorithmConfigFile, algorithmConfigFile, std::string, "" );
    /** \brief Result format, either JSON or RawText */
    DEFINE_OPTION( ResultFormatTag, resultFormat, std::string, "JSON" );
    /** \brief Reports configuration options of the optimizer. */
    DEFINE_SIMPLE_OPTION( AlgorithmUsageTag, algorithmUsage );
    /** \brief Generates default algorithm configuration file in JSON syntax. */
    DEFINE_SIMPLE_OPTION( DefaultAlgorithmUsageTag, defaultAlgorithmUsage );

  };
};

namespace soo {

/**
 *  \brief Single-objective and multi-objective experiments for evolutionary
 *	algorithms.
 */
class Experiment {
 public:
  typedef ExperimentBase::Options Options;
};

		
}

namespace moo {

/**
 *  \brief Single-objective and multi-objective experiments for evolutionary
 *	algorithms.
 */
class Experiment {
 public:

  //typedef SearchSpaceType search_space_type;
  /**
   * \brief Options specialization for multi-objective experiments.
   */
  class Options : public ExperimentBase::Options {
   public:
			
			
    /**
     * \brief Inserts default options of the experiment.
     */
    void addDefaultOptions() {
      ExperimentBase::Options::addDefaultOptions();
      ExperimentBase::Options::addOption( ObjectiveSpaceDimensionTag() );
    }
			
    /** \brief Dimension of the objective space. */
    DEFINE_OPTION( ObjectiveSpaceDimensionTag, objectiveSpaceDimension, unsigned int, 2 );
  };
		    
};
		
		
}
}

#endif 
