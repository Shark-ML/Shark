/**
 *
 *  \brief Timer abstraction with microsecond resolution
 *
 *  \author  T. Voss, M. Tuma
 *  \date    2010
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

#ifndef SHARK_CORE_TIMER_H
#define SHARK_CORE_TIMER_H


#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#else
#include <sys/resource.h>
#endif 


namespace shark {


/// \brief Timer abstraction with microsecond resolution///
///
/// \par
/// use start() to start the timer and stop() to retrive the
/// elapsed time in seconds (guaranteed/forced to be >= 0 ).
/// use now() to get the current time (may in rare cases give decreasing values).
class Timer
{
public:
	Timer() 
	: m_lastLap( 0.0 ),
	  m_startTime( 0.0 )
	  { }

	/// \brief stores the current time in m_startTime.
	void start() {
#ifdef _WIN32
		LARGE_INTEGER tick, tps;
		QueryPerformanceFrequency(&tps);
		QueryPerformanceCounter(&tick);
		m_startTime = static_cast<double>( tick.QuadPart ) / static_cast<double>( tps.QuadPart );
#else
		rusage res;
		getrusage(RUSAGE_SELF, &res);
		m_startTime = res.ru_utime.tv_sec + res.ru_stime.tv_sec
			+ 1e-6 * (res.ru_utime.tv_usec + res.ru_stime.tv_usec);
#endif
	}

	/// \brief returns the difference between current time and m_startTime (but a minimum of 0)
	double stop() {
#ifdef _WIN32
		LARGE_INTEGER tick, tps;
		QueryPerformanceFrequency(&tps);
		QueryPerformanceCounter(&tick);
		double stop = static_cast<double>( tick.QuadPart ) / static_cast<double>( tps.QuadPart );
#else
		rusage res;
		getrusage(RUSAGE_SELF, &res);
		double stop = res.ru_utime.tv_sec + res.ru_stime.tv_sec
			+ 1e-6 * (res.ru_utime.tv_usec + res.ru_stime.tv_usec);
#endif
		m_lastLap = stop - m_startTime;

		// avoid rare cases of non-increasing timer values (cf. eg. http://www.linuxmisc.com/8-freebsd/d4c6ddc8fbfbd523.htm)

		if ( m_lastLap < 0.0 ) {
			m_lastLap = 0.0;
		}

		return m_lastLap;
	}

	/// \brief returns the last value of stop()
	double lastLap() {
		return m_lastLap;
	}

	/// \brief Returns the current time in a microsecond resolution. Att: may in rare cases give decreasing values.
	static double now() {
#ifdef _WIN32
		LARGE_INTEGER tick, tps;
		QueryPerformanceFrequency(&tps);
		QueryPerformanceCounter(&tick);
		return( static_cast<double>( tick.QuadPart ) / static_cast<double>( tps.QuadPart ) );
#else
		rusage res;
		getrusage(RUSAGE_SELF, &res);
		return(res.ru_utime.tv_sec + res.ru_stime.tv_sec)
			+ 1e-6 * (res.ru_utime.tv_usec + res.ru_stime.tv_usec);
#endif
	}

private:
	double m_lastLap;
	double m_startTime;
};


}
#endif

