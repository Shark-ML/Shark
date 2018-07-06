/*!
 * \brief       Implements the Synchronous Queue
 * \author      O.Krause
 * \date        2018
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://shark-ml.org/>
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
#ifndef SHARK_CORE_THREADING_SYNCQUEUE_H
#define SHARK_CORE_THREADING_SYNCQUEUE_H


#include <deque>//std::deque
#include <mutex>//std::mutex, std::unique_lock
#include <condition_variable>//std::condition_variable
#include <utility>//std::move
namespace shark{
namespace threading{
	
enum class queue_status{
	success = 0, 
	empty,
	full,
	closed,
	busy
};
	
template <class ValueType, class Container = std::deque<ValueType> >
class SyncQueue{
public:
	typedef ValueType value_type;

	// push functions
	/// \brief Push a new element into the queue
	///
	/// Tries to push a new element into the queue and blocks until it is done.
	/// To prevent deadlocks, if the queue is full, push does not wait for empty space in the queue.
	/// If the push was not successful, the element provided is not changed. 
	/// Otherwise contents are moved into the queue.
	/// Returns:
	/// queue_status::success - element is pushed
	/// queue_status::full - queue is full, try again later
	/// queue_status::closed - queue is closed for newly added elements
	queue_status push(ValueType&& elem){
		std::unique_lock<std::mutex> lock(m_mutex);
		return push(std::move(elem),lock);
	}
	/// \brief Push a new element into the queue without blocking the current thread
	///
	/// Tries to push a new element into the queue.
	/// If the push was not successful, the element provided is not changed. 
	/// Otherwise contents are moved into the queue.
	/// Returns:
	/// queue_status::success - element is pushed
	/// queue_status::busy - queue is busy, try again later
	/// queue_status::full - queue is full, try again later
	/// queue_status::closed - queue is closed for newly added elements
	queue_status nonblocking_push(ValueType&& elem){
		std::unique_lock<std::mutex> lock(m_mutex, std::try_to_lock);
		if (!lock.owns_lock()){
			return queue_status::busy;
		}
		return push(std::move(elem), lock);
	}
	
	//pull functions
	/// \brief Blocks until an element is pulled or the queue is closed
	///
	/// This function tries to pull the next element. On success, returns
	/// queue_status::success, otherwise queue_status::closed
	queue_status pull(value_type& elem){
		std::unique_lock<std::mutex> lock(m_mutex);
		wait_until_not_empty_or_closed(lock);
		return pull(elem, lock);
	}
	
	/// \brief Pulls an element without blocking.
	///
	/// This function tries to pull the next element if possible.
	/// Returns:
	/// queue_status::success - element got pulled
	/// queue_status::busy - queue is busy, try again later
	/// queue_status::empty - queue is empty but not closed
	/// queue_status::closed - queue is empty and there won't be more elements
	queue_status nonblocking_pull(value_type& elem){
		std::unique_lock<std::mutex> lock(m_mutex, std::try_to_lock);
		if (!lock.owns_lock()){
			return queue_status::busy;
		}
		if(m_data.empty()){
			return queue_status::empty;
		}
		return pull(elem, lock);
	}
	
	
	/// \brief closes the queue. no more pushs are possible.
	///
	/// push will return queue_status::closed, pull will return
	/// queue_status::closed as soon as the queue is empty.
	void close(){
		std::unique_lock<std::mutex> lock(m_mutex);
		m_closed = true;
		//once we are closed, we have to notify all waiting threads that there is not more coming
		m_state_change.notify_all();
	}
	
	/// \brief block the current thread until either there are elements to process or the queue is closed.
	bool wait_until_not_empty_or_closed(){
		std::unique_lock<std::mutex> lock;
		return wait_until_not_empty_or_closed(lock);
	}
	
private:
	Container m_data;
	std::mutex m_mutex;
	std::condition_variable m_state_change;
	bool m_closed;

	queue_status pull(value_type& elem, std::unique_lock<std::mutex>&){
		if(m_data.empty()){
			return m_closed? queue_status::closed: queue_status::empty;
		}
		elem = std::move(m_data.front());
		m_data.pop_front();
		return queue_status::success;
	}
	queue_status push(value_type&& elem, std::unique_lock<std::mutex>& lock){
		if(m_closed){
			return queue_status::closed;
		}
		//~ if(m_full){//current queue can not be full.
			//~ return queue_status::full;
		//~ }
		m_data.push_back(std::move(elem));
		m_state_change.notify_one();
		return queue_status::success;
	}
	
	bool wait_until_not_empty_or_closed(std::unique_lock<std::mutex>& lock){
		//We have a lock, so we can
		//quick check if the condition is fulfilled
		if(m_closed || !m_data.empty())
			return m_closed;
		//otherwise wait for condition to be fulfilled
		m_state_change.wait(lock, [this]{return m_closed || !m_data.empty();});
		return m_closed;
	}
};

}}

#endif