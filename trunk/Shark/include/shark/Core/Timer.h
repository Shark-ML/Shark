/*!
 * 
 *
 * \brief       Timer abstraction with microsecond resolution
 * 
 * 
 *
 * \author      T. Voss, M. Tuma
 * \date        2010
 *
 *
 * \par Copyright 1995-2014 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://image.diku.dk/shark/>
 * 
 * Shark is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Shark is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with Shark.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef SHARK_CORE_TIMER_H
#define SHARK_CORE_TIMER_H


#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <ctime>
#else
#include <sys/resource.h>
#include <sys/time.h>
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
	Timer(bool measureWallclockTime = true) 
	: m_lastLap( 0.0 )
	, m_startTime( 0.0 )
	, m_measureWallclockTime(measureWallclockTime)

	  { }
	  
	/// \brief Returns the current time in a microsecond resolution. Att: may in rare cases give decreasing values.
	static double now(bool measureWallclockTime = true) {
#ifdef _WIN32
		if(measureWallclockTime){
			return static_cast<double>(std::clock()) / CLOCKS_PER_SEC;
		}
		else{
			LARGE_INTEGER tick, tps;
			QueryPerformanceFrequency(&tps);
			QueryPerformanceCounter(&tick);
			return( static_cast<double>( tick.QuadPart ) / static_cast<double>( tps.QuadPart ) );
		}
#else
		if(measureWallclockTime){
			timeval time;
			if (gettimeofday(&time,0)){
				//  Handle error
				return 0;
			}
			return time.tv_sec +1e-6 *time.tv_usec;
		}
		else
		{
			rusage res;
			getrusage(RUSAGE_SELF, &res);
			return(res.ru_utime.tv_sec + res.ru_stime.tv_sec)
				+ 1e-6 * (res.ru_utime.tv_usec + res.ru_stime.tv_usec);
		}
#endif
	}

	/// \brief stores the current time in m_startTime.
	void start() {
		m_startTime = now(m_measureWallclockTime);
	}

	/// \brief returns the difference between current time and m_startTime (but a minimum of 0)
	double stop() {
		double stop = now(m_measureWallclockTime);
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

private:
	double m_lastLap;
	double m_startTime;
	bool m_measureWallclockTime;
};


}
#endif

