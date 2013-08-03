/**
 *
 * \brief Probing framework providing for profiling of algorithms.
 *
 *  \author tvoss
 *  
 * <BR><HR>
 * This file is part of Shark. This library is free software;
 * you can redistribute it and/or modify it under the terms of the
 * GNU General Public License as published by the Free Software
 * Foundation; either version 3, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this library; if not, see <http://www.gnu.org/licenses/>.
 *  
 */
#ifndef SHARK_CORE_PROBE_H
#define SHARK_CORE_PROBE_H

// we should move to Boost.Signals2
#define BOOST_SIGNALS_NO_DEPRECATION_WARNING 1

#include <shark/Core/Timer.h>
#include <shark/LinAlg/BLAS/ublas.h>
#include <boost/algorithm/string.hpp>
#include <boost/bimap.hpp>
#include <boost/bimap/multiset_of.hpp>
#include <boost/flyweight.hpp>
#include <boost/foreach.hpp>
#include <boost/regex.hpp>
#include <boost/serialization/extended_type_info.hpp>
#include <boost/signals.hpp>
#include <boost/thread.hpp>
#include <boost/units/make_scaled_unit.hpp>
#include <boost/units/systems/si.hpp>
#include <boost/units/io.hpp>
#include <boost/unordered_map.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/variant.hpp>

#include <complex>
#include <list>
#include <string>


namespace shark {

/**
 * \brief Implements a probe for profiling and debugging purposes.
 */
class Probe /** \cond */ : public boost::noncopyable 
/** \endcond */ {
 public:

  /** \brief Make the mutex type known to the outside world. */
  typedef boost::timed_mutex mutex_type;

  /**
   * \brief Models microseconds as default unit of time.
   */
  typedef boost::units::make_scaled_unit<
    boost::units::si::time, 
    boost::units::scale<
      10, 
      boost::units::static_rational<-6> 
      > 
    >::type time_unit_type;

  /**
   * \brief Default time quantity.
   */
  typedef boost::units::quantity< time_unit_type > time_type;

  /**
   * \brief The probe framework supports the following types.
   */
  typedef boost::variant<
    bool,
    boost::int_fast8_t,
    boost::uint_fast8_t,
    boost::int_fast16_t,
    boost::uint_fast16_t,
    boost::int_fast32_t,
    boost::uint_fast32_t,
    boost::int_fast64_t,
    boost::uint_fast64_t,
    float,
    double,
    std::complex< double >,
    shark::blas::vector< double >,
    shark::blas::matrix< double >,
    std::string
    > variant_type;

  /**
   * \brief Models a timestamped value.
   */
  typedef std::pair< variant_type, time_type > timestamped_value_type;

  /**
   * \brief Signal type that is emitted when probe values change.
   */
  typedef boost::signal<
    void ( 
        time_type,
        const variant_type &
           ) 
    > signal_type;

		

  /**
   * \brief Accesses the name of the probe.
   */
  const std::string & name() const {
    return( m_name );
  }

  /**
   * \brief Accesses the name of the probe.
   */
  const std::string & description() const {
    return( m_description );
  }

  /**
   * \brief Accesses the value of the probe.
   */
  void value( variant_type & value ) const {
    boost::lock_guard< mutex_type > lg( m_valueGuard );
    value = m_value;
  }

  /**
   * \brief Accesses the value of the probe.
   */
  variant_type value() const {
    boost::lock_guard< mutex_type > lg( m_valueGuard );
    variant_type value( m_value );
    return( value );
  }

  /**
   * \brief Accesses the timestamp of the last value update.
   */
  void timestamp( time_type & timestamp ) const {
    boost::lock_guard< mutex_type > lg( m_valueGuard );
    timestamp = m_timestamp;
  }

  /**
   * \brief Accesses the timestamp of the last value update.
   */
  time_type timestamp() const {
    boost::lock_guard< mutex_type > lg( m_valueGuard );
    time_type timestamp( m_timestamp );
    return( timestamp );
  }

  /**
   * \brief Access the value of the probe together with the timestamp
   * of the last update.
   */
  timestamped_value_type timestampedValue() const {
    boost::lock_guard< mutex_type > lg( m_valueGuard );
    return( std::make_pair( m_value, m_timestamp ) );
  }

  /**
   * \brief Access the value of the probe together with the timestamp
   * of the last update.
   */
  void timestampedValue( timestamped_value_type & value ) const {
    boost::lock_guard< mutex_type > lg( m_valueGuard );
    value.first = m_value;
    value.second = m_timestamp;
  }

  /**
   * \brief Updates the value of the probe. Triggers the signalUpdated.
   * \param [in] value The new value.
   * \param [in] timeOut Wait for at most timeout milliseconds.
   */
  bool setValue( 
      const variant_type & value, 
      const boost::posix_time::milliseconds & timeOut = boost::posix_time::milliseconds( 1000 )
                 ) {
    boost::unique_lock< mutex_type > lg( m_valueGuard, boost::defer_lock_t() );

    if( !lg.timed_lock( timeOut ) )
      return( false );

    m_value = value;
    m_timestamp = time_type::from_value( Timer::now() );
    m_signalUpdated( m_timestamp, m_value );

    return( true );
  }

  /**
   * \brief Accesses a signal that is triggered when the value is updated.
   */
  signal_type & signalUpdated() {
    return( m_signalUpdated );
  }

  /**
   * \brief Serializes the probe instance to the supplied archive.
   * \tparam Archive Type of archive type to use for (de)serialization.
   * \param [in,out] archive The archive instance to write to, read from.
   * \param [in] version Currently unused.
   */
  template<typename Archive>
  void serialize( Archive & archive, const unsigned int version ) {
    boost::unique_lock< mutex_type > lg(
        m_valueGuard,
        boost::adopt_lock_t() );
                  
    archive & m_name;
    archive & m_description;
    archive & m_value;
  }

 protected:

  /**
   * \brief Only the probe manager is allowed to instantiate probes.
   */
  friend class ProbeManager;

  /**
   * \brief C'tor.
   * \param [in] name Name of the probe.
   * \param [in] description Description of the probe.
   */
  Probe( 
      const std::string & name = std::string(), 
      const std::string & description = std::string() 
         ) : m_name( name ),
             m_description( description ),
             m_timestamp( time_type::from_value( shark::Timer::now() ) ) {}

  /**
   * \brief Adjusts the name of the probe.
   */
  void setName( const std::string & name ) {
    m_name = name;
  }

  /**
   * \brief Adjusts the description of the probe.
   */
  void setDescription( const std::string & description ) {
    m_description = description;
  }

  /** \brief Name of the probe. */
  boost::flyweight<
    std::string
    > m_name;

  /** \brief Description of the probe */
  boost::flyweight<
    std::string
    > m_description;

  /** \brief Value of the probe */
  variant_type m_value;

  /** \brief Timestamp of last value update. */
  time_type m_timestamp;

  /** \brief Signals that the probe's value has been updated. */
  signal_type m_signalUpdated;

  /** \brief Guards the value. */
  mutable mutex_type m_valueGuard;
};

/**
 * \brief ProbeManager for handling program-wide probes. Implemented as a singleton.
 */
class ProbeManager : public boost::noncopyable {
 protected:

  /**
   * \brief Default c'tor.
   */
  ProbeManager() {}
  /**
   * \brief D'tor.
   */
  ~ProbeManager() {};

 public:

  /** \brief Make the mutex type known to the outside world. */
  typedef boost::timed_mutex mutex_type;

  /**
   * \brief Models a path in the tree of registered probes.
   */
  class Path {
   public:

    /**
     * \brief Default c'tor.
     */
    Path() {}

    /**
     * \brief Construct from supplied string.
     * \param p Initial path, parsed for defaultPathSeparator().
     */
    Path( const std::string & p ) {
      std::list< std::string > tokens;
      boost::split( tokens, p, boost::is_any_of( ProbeManager::defaultPathSeparator() ) );
      m_tokens.insert( m_tokens.end(), tokens.begin(), tokens.end() );
    }

    /**
     * \brief Copy c'tor.
     */
    Path( const Path & p ) : m_tokens( p.m_tokens ) {}

    /**
     * \brief Joins this path and the supplied path.
     */
    Path operator/( const Path & p ) const {
      Path result( *this );
      result.m_tokens.insert( result.m_tokens.end(), p.m_tokens.begin(), p.m_tokens.end() );
      return( result );
    }

    /**
     * \brief Appends the supplied path to this path.
     */
    Path & operator/=( const Path & p ) {
      m_tokens.insert( m_tokens.end(), p.m_tokens.begin(), p.m_tokens.end() );
      return( *this );
    }

    /**
     * \brief Joins this path and the path constructed from the supplied string.
     */
    Path operator/( const std::string & p ) const {
      Path result( *this );
      std::list< std::string > tokens;
      boost::split( tokens, p, boost::is_any_of( ProbeManager::defaultPathSeparator() ) );
      result.m_tokens.insert( result.m_tokens.end(), tokens.begin(), tokens.end() );
      return( result );
    }

    /**
     * \brief Appends the supplied path to this path.
     */
    Path & operator/=( const std::string & p ) {
      std::list< std::string > tokens;
      boost::split( tokens, p, boost::is_any_of( ProbeManager::defaultPathSeparator() ) );
      m_tokens.insert( m_tokens.end(), tokens.begin(), tokens.end() );
      return( *this );
    }

    /**
     * \brief Generates a string representation for this path.
     */
    std::string str() const {
      std::stringstream ss;
      for( std::list< std::string >::const_iterator it = m_tokens.begin(); it != m_tokens.end(); ++it )
        ss << ProbeManager::defaultPathSeparator() << *it;
      return( ss.str() );
    }

   protected:
    /** \brief Stores the tokens of this path. */
    std::list< std::string > m_tokens;
  };

  /** \brief Marks a managed pointer to a probe. */
  typedef boost::shared_ptr<
    Probe
    > ProbePtr;

  /**
   * \brief Models a visitor for iterating registered probes in a 
   * thread-safe manner.
   */
  typedef boost::function< 
    void ( const Path &, const ProbePtr & ) 
    > AbstractVisitor;

  /** \brief Registry type for managing the probes. */
  typedef boost::bimap< 
    boost::bimaps::multiset_of< 
      std::string
      >, 
    ProbePtr
    > registry_type;

  /**
   * \brief Accesses the default path separator.
   */
  static const char * defaultPathSeparator() {
    return( "/" );
  }

  /**
   * \brief Accesses the global instance of the ProbeManager.
   */
  static ProbeManager & instance() {
    static ProbeManager probeManager;
    return( probeManager );
  }

  /**
   * \brief Visit registered probe and delegate handling to supplied visitor instance.
   */
  void visit( AbstractVisitor visitor ) const {
    boost::lock_guard< mutex_type > lg( m_registryGuard );
    BOOST_FOREACH( const registry_type::left_value_type & value, m_registry.left ) {
      visitor( value.first, value.second );
    }
  }

  /**
   * \brief Registers a new probe for the given path with the 
   * supplied name and description.
   *
   * \param [in] path Path to the probe.
   * \param [in] name Name of the new probe.
   * \param [in] description Description of the new probe.
   */
  ProbePtr registerProbe( const std::string & path, const std::string & name, const std::string & description ) {
    boost::lock_guard< mutex_type > lg( m_registryGuard );
    std::string uniqueProbeName = path + ProbeManager::defaultPathSeparator() + name;
			
    ProbePtr probe( new Probe( name, description ) );
    m_registry.insert( registry_type::value_type( uniqueProbeName, probe ) );

    return( probe );
  }

  /**
   * \brief Removes the supplied probe from the registry.
   */
  void unregisterProbe( ProbePtr p ) {
    boost::lock_guard< mutex_type > lg( m_registryGuard );
    m_registry.right.erase( p );
  }

  std::pair<
    registry_type::left_const_iterator,
    registry_type::left_const_iterator
    > equalRange( const std::string & path ) {
    boost::lock_guard< mutex_type > lg( m_registryGuard );
    return( m_registry.left.equal_range( path ) );
  }

  /**
   * \brief Finds all probes matching the supplied regular expression.
   */
  std::list< 
    ProbePtr
    > find( const boost::regex & regEx ) {
    boost::lock_guard< mutex_type > lg( m_registryGuard );

    std::list< 
        ProbePtr 
        > result;

    for( registry_type::left_const_iterator it = m_registry.left.begin(); it != m_registry.left.end(); ++it ) {
      if( boost::regex_match( it->first, regEx  ) )
        result.push_back( it->second );
    }

    return( result );
  }

  /**
   * \brief Prints the content of the registry to the supplied stream.
   */
  template<typename Stream>
  void print( Stream & s ) {
    boost::lock_guard< mutex_type > lg( m_registryGuard );
    registry_type::left_const_iterator it = m_registry.left.begin();
    for( ; it != m_registry.left.end(); ++it ) {
      s << it->first << " -> " << it->second->name() << std::endl;
    }
  }

 protected:
  /** \brief The actual registry. */
  registry_type m_registry; 

  /** \brief Guards the registry. */
  mutable mutex_type m_registryGuard;

		
};

/**
 * \brief Eases registration of probe's for classes.
 * \tparam Base Base class that wants to register probes with the manager.
 */ 
template<typename Base>
class ProbeProvider {
 public:

  /**
   * \brief Makes the base type known to the outside world.
   */
  typedef Base base_type;

  /**
   * \brief C'tor.
   * \param [in] context Class context, defaults to the class name of the base class.
   */
  ProbeProvider( const std::string & context ) : m_context( context ) {
    m_instanceId = boost::uuids::random_generator()();
  }

  /**
   * \brief Virtual d'tor.
   */
  virtual ~ProbeProvider() {}

  /**
   * \brief Accesses the uuid identifying the class instance.
   */
  const boost::uuids::uuid & instanceId() const {
    return( m_instanceId );
  }

  /**
   * \brief Accesses the class context.
   */
  const std::string & context() const {
    return( m_context );
  }

  /**
   * \brief Registers a probe with the given name and description.
   * \param [in] name The name of the new probe.
   * \param [in] description The description of the new probe.
   *
   * Generates a unique name for the probe with the supplied context and 
   * the class instance id.
   *
   * \returns The new probe or an invalid shared_ptr.
   */
  boost::shared_ptr< Probe > registerProbe( const std::string & name, const std::string & description ) {
    std::stringstream ss;
    ss << m_instanceId;

    ProbeManager::Path p;
    p = p / m_context / ss.str();

    /*
      std::string path = ProbeManager::defaultPathSeparator() + 
      m_context +
      ProbeManager::defaultPathSeparator() +
      ss.str();*/
				
    ProbeManager::ProbePtr probe = ProbeManager::instance().registerProbe( p.str(), name, description );
    m_registeredProbes[ name ] = probe;
    return( probe );
  }

  /**
   * \brief Returns a probe or an empty pointer for the supplied name.
   */
  ProbeManager::ProbePtr operator[]( const std::string & name ) const {
    registry_type::const_iterator it = m_registeredProbes.find( name );
    return( it == m_registeredProbes.end() ? ProbeManager::ProbePtr() : it->second );
  }

 protected:
  typedef boost::unordered_map< 
   std::string, 
   boost::shared_ptr< Probe > 
   > registry_type;

  /** \brief The uuid uniquely marking the instance. */
  boost::uuids::uuid m_instanceId;
  /** \brief Class context marking the class-specific context. */
  std::string m_context;
  /** \brief Stores all of the registered probes. */
  registry_type m_registeredProbes;
};
}

#endif
