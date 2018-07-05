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
	typedef KeyValuePair<unsigned int, value_type> result_type;
	std::priority_queue<result_type, std::vector<result_type>, std::greater<result_type> > m_ordered_results;
	std::mutex m_results_mutex;
	ThreadPool& m_threadpool;
	std::size_t m_maxUnreadElements;
	unsigned int m_task_start_counter;
	unsigned int m_task_read_counter;
};

}}

#endif