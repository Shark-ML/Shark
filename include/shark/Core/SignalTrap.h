/**
 *
 * \brief Class that interferes with signals and allows for graceful shutdowns.
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
 */
#ifndef SHARK_CORE_SIGNALTRAP_H
#define SHARK_CORE_SIGNALTRAP_H

// we should move to Boost.Signals2
#define BOOST_SIGNALS_NO_DEPRECATION_WARNING 1

#include <shark/Core/Singleton.h>
#include <boost/signal.hpp>

#include <signal.h>

namespace shark {

    /**
     * \brief Class that interferes with signals and allows for a graceful shutdown.
     * 
     * \sa InterruptibleAlgorithmRunner.
     */
    class SignalTrap : public Singleton< SignalTrap > {
    public:

	/**
	 * \brief Models the type of the trapped signal.
	 */
	enum SignalType {
	    SIGNAL_INTERRUPT = SIGINT ,					///< interrupt
	    SIGNAL_ILLEGAL_INSTRUCTION = SIGILL,		///< illegal instruction - invalid function image
	    SIGNAL_FLOATING_POINT_EXCEPTION = SIGFPE,	///< floating point exception
	    SIGNAL_SEGMENT_VIOLATION = SIGSEGV,			///< segment violation
	    SIGNAL_TERMINATION = SIGTERM,				///< Software termination signal from kill
	    SIGNAL_ABORT = SIGABRT						///< abnormal termination triggered by abort call
	};

	/**
	 * \brief Accesses the signal that delegates can connect to.
	 *
	 * The signal is emitted whenever a low level signal has been caught.
	 */
	boost::signal< void ( SignalType ) > & signalTrapped() {
	    return( m_signalTrapped );
	}

    protected:
	friend class Singleton< SignalTrap >;
	static void signal_handler( int signal ) {
	    SignalTrap::instance().signalTrapped()( static_cast< SignalType >( signal ) );
	}

	SignalTrap() {
	    signal( SIGINT, SignalTrap::signal_handler );
	    signal( SIGABRT, SignalTrap::signal_handler );
	    signal( SIGTERM, SignalTrap::signal_handler );
	}

	boost::signal< void ( SignalType ) > m_signalTrapped;
    };
}

#endif
