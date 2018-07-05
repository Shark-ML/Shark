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
	
	std::size_t numWorkers() const{
		return m_workers.size();
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
	void submit(Functor&& task) {
		std::function<void()> f(std::move(task));
		if(m_tasks.push(std::move(f)) != queue_status::success){
			task();
		}
	}
	
	///  \brief Calls submit on the task but returns a future holding the return value
	///
	/// the return value of the task is returned via a future. If the return type
	/// of the fucntion is void, the future just signals when the task is done processing
	///
	/// See submit for further information
	template<class Function>
	std::future<decltype(std::declval<Function>()())> execute_async( Function&& task){
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

inline ThreadPool& globalThreadPool(){
	static unsigned threadCount = 0;
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