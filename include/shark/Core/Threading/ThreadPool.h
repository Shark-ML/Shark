/*!
 * \brief       Implements Threadpool facilities
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
#ifndef SHARK_CORE_THREADING_THREADPOOL_H
#define SHARK_CORE_THREADING_THREADPOOL_H


#include <shark/Core/Threading/SyncQueue.h>
#include <vector>
#include <thread>
#include <future>
#include <functional>
#include <cstdlib>
#include <chrono>


namespace shark{
namespace threading{
	
template<class F>
class is_future: public std::false_type{};
	
template<class T>
class is_future<std::future<T> >: public std::true_type{};

/// \brief Implements a threadpool class that computes work asynchronously in parallel
///
/// The threadpool is designed such that worker threads are allowed to submit work themselves and lend
/// their compute power back to the pool in case they have to wait. This allows to implement threading functionality where 
/// functions that are computed in parallel can themselves queue more work. Look at Threading/Algorithms.h for a set
/// of possible algorithms.
class ThreadPool{
private:
	/// the worker m_workers
	std::vector<std::thread> m_workers;
	/// the thread safe work queue
	SyncQueue<std::function<void()> > m_tasks;

	void worker_thread() {
		try {
			for (;;) {
				std::function<void()> task;
				if (m_tasks.pull(task) == queue_status::closed) {
					return;
				}
				task();
			}
		} catch (...) {
			std::terminate();
			return;
		}
	}

public:
	/// \brief Creates a thread pool that runs work on threadCount m_workers.
	ThreadPool(unsigned const threadCount) {
		for (unsigned i = 0; i < threadCount; ++i) {
			m_workers.emplace_back(&ThreadPool::worker_thread, this);
		}
	}

	~ThreadPool() {
		//close the queue and wait until all work is done
		m_tasks.close();
		for (auto& thread: m_workers){
			thread.join();
		}
	}
	
	/// \brief Returns the number of workers in the Pool.
	///
	/// This is the maximum number of threads.
	std::size_t numWorkers() const{
		return m_workers.size();
	}
	
	/// \brief Closes the queue.
	///
	/// Threads will empty the queue and newly added work via submit() 
	/// is performed by the thread calling submit(), i.e. all task are performed serial, like
	/// in a siungle-threaded programm.
	void close(){
		m_tasks.close();
	}
	
	/// \brief Lends the current thread as additional worker to the work pool
	void yield(){
		std::function<void()> task;
		if (m_tasks.nonblocking_pull(task) == queue_status::success)
			task();
		else
			std::this_thread::yield();
	}
	
	/// \brief Lends the current thread as additional worker to the work pool until future is ready
	template<class T>
	void wait(std::future<T>& future){
		if(!future.valid()) return;
		
		while(future.wait_for(std::chrono::seconds(0)) != std::future_status::ready){
			yield();
		}
	}
	
	/// \brief Lends the current thread as additional worker to the work pool until all futures are ready
	template<class T, class... Futures>
	void wait_for_all(std::future<T>& future, Futures&... futures){
		bool dummy[] = {(wait(future), true), (wait(futures), true)...};
		(void)dummy; //prevent warning
	}
	
	/// \brief Lends the current thread as additional worker to the work pool until all futures are ready
	///
	/// Overload for ranges
	template<class Iterator, class = typename std::enable_if<!is_future<Iterator>::value, void >::type >
	void wait_for_all(Iterator begin, Iterator end){
		for(Iterator pos = begin; pos != end; ++pos){
			wait(*pos);
		}
	}

	///  \brief Schedules a task for execution in the thread pool.
	///
	/// Tries to submit the task to the work queue. the task is processed asynchronously.
	/// If this is not possible for whatever reason - for example the pool is full or closed,
	/// the task is immediately computed synchronously
	template<class Functor>
	void submit(Functor task) {
		std::function<void()> f(std::move(task));
		if(m_tasks.push(std::move(f)) != queue_status::success){
			f();
		}
	}
	
	///  \brief Calls submit on the task but returns a future holding the return value
	///
	/// the return value of the task is returned via a future. If the return type
	/// of the fucntion is void, the future just signals when the task is done processing
	///
	/// See submit for further information
	template<class Function>
	std::future<decltype(std::declval<Function>()())> execute_async( Function task){
		//obtain return type of the task
		typedef decltype(task()) return_type;
		//Create a task package from the task
		//in a perfect world, we would create it and move it inside an std::function. however this is not
		//possible, because std::function requires the Functor to be copyable. hint: packaged_task is not.
		auto task_package = std::make_shared<std::packaged_task<return_type()> >(task);
		std::future<return_type> future = task_package->get_future();
		
		//submit package and return future
		submit([task_package]{(*task_package)();});
		return std::move(future);
	}
};

/// \brief Returns a reference to the global Threadpool.
///
/// On first call, the threadpool is initialized. For this, the following information is used:
/// sizeHint: if not zero, the threadpool uses that amount of threads
/// environment variable "SHARK_NUM_THREADS" 
/// otherwise, the number of threads is number_of_cores+1
/// \param sizeHint optional. On first call, it is used to initialize the threadpool to a certain size
inline ThreadPool& globalThreadPool(std::size_t sizeHint = 0){
	static unsigned threadCount = 0;
	if(threadCount == 0){
		threadCount = sizeHint;
	}
	if(threadCount == 0){
		char const* m_workers = std::getenv("SHARK_NUM_THREADS");
		if(m_workers && std::atoi(m_workers) > 0)
			threadCount = std::atoi(m_workers);
		else
			threadCount = std::thread::hardware_concurrency()+1;
	}
	static ThreadPool pool(threadCount);
	return pool;
}
}}

#endif