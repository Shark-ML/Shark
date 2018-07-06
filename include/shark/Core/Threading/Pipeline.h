/*!
 * \brief       Implements the Pipeline class
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
#ifndef SHARK_CORE_THREADING_PIPELINE_H
#define SHARK_CORE_THREADING_PIPELINE_H

#include <shark/Core/Threading/ThreadPool.h>
#include <shark/Core/utility/KeyValuePair.h>
#include <queue>
#include <mutex>
namespace shark{
namespace threading{

/// \brief a Pipeline computes an ordered set of tasks using a threadpool.
///
/// For a set of tasks T1...TN computes Xi=f(Ti). the computations are assumed to be independent
/// and are performed in parallel. However, the results Xi are kept in the correct order. In this sense,
/// the pipeline is a true FIFO-Queue.
/// Note that pushing and pulling elements in the pipeline is not threadsafe.
template<class T>
class Pipeline{
public:
	typedef T value_type;
	Pipeline(std::size_t maxUnreadElements, ThreadPool& threadpool)
	: m_threadpool(threadpool)
	, m_maxUnreadElements(maxUnreadElements)
	, m_task_start_counter(0)
	, m_task_read_counter(0){}
	
	bool empty(){
		return m_task_start_counter == m_task_read_counter;
	}
	
	
	/// \brief Pushes task into the queue.
	///
	/// On return:
	/// queue_status::success: task is moved into the pool and evaluated in parallel to all
	/// other tasks pushed before (assuming there are enough worker threads available).
	/// Once ready, the tasks can be read in order of their push by calling pull.
	////
	/// queue_status::full: Pipeline is full, i.e. the number of tasks started + the number of
	/// tasks read is equal to the suplied maximum in the constructor.
	/// In this case, task is elft unchanged.
	template<class Task>
	queue_status push(Task&& task){
		//check whether we are at max load
		std::size_t loadSize = m_task_start_counter - m_task_read_counter;
		if(loadSize >= m_maxUnreadElements){
			return queue_status::full;
		}
		
		//get the id of the current task in the pipeline, increment the counter
		std::size_t id = m_task_start_counter++;
		
		//the task computes the function and then waits until all tasks with lower id are done
		auto taskPackage = [this, id](Task& task){
			//compute work package
			T results = task();

			//write results in queue
			std::lock_guard<std::mutex> lock(m_results_mutex);
			m_ordered_results.emplace(id, std::move(results));
		};
		m_threadpool.submit(std::bind(taskPackage, std::move(task)));
		
		return queue_status::success;
	};
	/// \brief Pulls the enxt task from the queue
	///
	/// Checks if the result of next task in the order of pushed tasks is ready.
	/// If it is available, it is stored in elem.
	/// Returns:
	/// queue_status::success: elem contains the result of the next task
	/// queue_status::empty: all pushed tasks are pulled
	/// queue_status::busy: the next item is not done processing.
	queue_status pull(value_type& elem){
		//check if we expect any more work in the queue
		if(empty()){
			return queue_status::empty;
		}
		std::lock_guard<std::mutex> lock(m_results_mutex);
		//check whether the next work item in the sequence is done processing
		if(m_ordered_results.empty() || (m_ordered_results.top().key != m_task_read_counter)){
			return queue_status::busy;
		}
		//extract item
		elem = std::move(m_ordered_results.top().value);
		++m_task_read_counter;
		m_ordered_results.pop();
		return queue_status::success;
	};
	
private:
	typedef KeyValuePair<std::size_t, value_type> result_type;
	std::priority_queue<result_type, std::vector<result_type>, std::greater<result_type> > m_ordered_results;
	std::mutex m_results_mutex;
	ThreadPool& m_threadpool;
	std::size_t m_maxUnreadElements;
	std::size_t m_task_start_counter;
	std::size_t m_task_read_counter;
};

}}

#endif