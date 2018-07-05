#ifndef SHARK_CORE_THREADING_ALGORITHMS_H
#define SHARK_CORE_THREADING_ALGORITHMS_H


#include <shark/Core/Threading/ThreadPool.h>
#include <shark/Core/Threading/Pipeline.h>
#include <initializer_list>
#include <array>
namespace shark{
namespace threading{
	
// this type can be initialized using {1,2, .., N} and only exactly this.
// it can be used to make calls to function with varying number of arguments unique.
template<std::size_t N>
struct size_arg{
	size_arg(){}	
	template<class ...Args, class = typename std::enable_if<sizeof...(Args) == N,void>::type >
	size_arg(Args... args):m_array({static_cast<std::size_t>(args)...}){}
	std::size_t operator[](std::size_t i) const{
		return m_array[i];
	}
private:
	std::array<std::size_t, N> m_array;
};


/// \brief Runs jobs on a number of function evaluations in parallel. Tasks are grouped in larger task packages
///
/// Runs the function f(i) for i in {0,...,rangeSize[0]-1}. for efficiency reasons, the range is split up in packages of
/// size workSize[0]. The evaluations are evaluated in parallel and no order is guarantueed.
template<class Functor>
void parallelND(
	size_arg<1> rangeSize,
	size_arg<1> taskSize,
	Functor f,
	ThreadPool& pool
){
	if(rangeSize[0] == 0) return;
	std::size_t numThreads = pool.numWorkers();
	std::size_t taskSize1 = taskSize[0]? taskSize[0]: ((rangeSize[0] + numThreads - 1)/numThreads);
	auto task = [=](std::size_t id){
		std::size_t end = std::min((id + 1)* taskSize1, rangeSize[0]);
		for(std::size_t i = id * taskSize1; i != end; ++i)
			f(i);
	};
	
	std::size_t numTasks = (rangeSize[0] + taskSize1 -1)/taskSize1;
	
	//vector to store the events in so we can wait for them later
	std::vector<std::future<void> > events;
	events.reserve(numTasks);
	
	for(std::size_t t = 0; t != numTasks; ++t){
		std::future<void> event = pool.execute_async(std::bind(task, t));
		events.push_back(std::move(event));
	}
	//lend the computation time of the current thread to compute this step faster
	pool.wait_for_all(events.begin(), events.end());
}

/// \brief Runs jobs on a number of function evaluations in parallel. Tasks are grouped in larger task packages
///
/// Runs the function f(i,j) for (i,j) in {0,...,rangeSize[0]-1}x{0,...,rangeSize[1]-1}. for efficiency reasons, the range is split up in packages of
/// size workSize[0].x workSize[1] The evaluations are evaluated in parallel and no order is guarantueed.
template<class Functor>
void parallelND(
	size_arg<2> const& rangeSize,
	size_arg<2> const& taskSize,
	Functor f,
	ThreadPool& pool
){
	if(rangeSize[0] == 0 || rangeSize[1] == 0) return;
	std::size_t numThreads = pool.numWorkers();
	std::size_t taskSize1 = taskSize[0]? taskSize[0]: ((rangeSize[0] + numThreads - 1)/numThreads);
	std::size_t taskSize2 = taskSize[1]? taskSize[1]: ((rangeSize[1] + numThreads - 1)/numThreads);
	auto task = [=](std::size_t id1, std::size_t id2){
		std::size_t end1 = std::min((id1 + 1)* taskSize1, rangeSize[0]);
		std::size_t end2 = std::min((id2 + 1)* taskSize2, rangeSize[1]);
		for(std::size_t i = id1 * taskSize1; i != end1; ++i)
			for(std::size_t j = id2 * taskSize2; j != end2; ++j)
				f(i,j);
	};
	
	std::size_t numTasks1 = (rangeSize[0]+taskSize1 -1)/taskSize1;
	std::size_t numTasks2 = (rangeSize[1]+taskSize2 -1)/taskSize2;
	
	//vector to store the events in so we can wait for them later
	std::vector<std::future<void> > events;
	events.reserve(numTasks1 * numTasks2);
	
	for(std::size_t id1 = 0; id1 != numTasks1; ++id1){
		for(std::size_t id2= 0; id2 != numTasks2; ++id2){
			std::future<void> event = pool.execute_async(std::bind(task, id1, id2));
			events.push_back(std::move(event));
		}
	}
	//lend the computation time of the current thread to compute this step faster
	pool.wait_for_all(events.begin(), events.end());
}

/// \brief performs elementsTo[i] = f(elementsFrom[i]) in parallel for all elements in the range.
template<class Functor, class RangeFrom, class RangeTo>
void transform(
	RangeFrom const& elementsFrom, 
	RangeTo& elementsTo, 
	Functor f,
	ThreadPool& pool
){
	auto task = [f, &elementsFrom, &elementsTo](std::size_t i){elementsTo[i] = f(elementsFrom[i]);};
	parallelND({elementsFrom.size()}, {1}, task, pool);
};

template<
	class Functor, class RangeFrom, class RangeTo, 
	class = typename std::enable_if<!std::is_reference<RangeTo>::Value, void>::type//gcc 4.8 bug with rvalue references requires this.
>
void transform(
	RangeFrom const& elementsFrom, 
	RangeTo&& elementsTo, 
	Functor f,
	ThreadPool& pool
){
	threading::transform(elementsFrom, static_cast<RangeTo&>(elementsTo), std::move(f), pool);
};
	

///\brief Maps all elements in the given range using the function map(elem and runs a function apply on the result
///
/// map is performed in parallel, apply is performed in order of elements in range in a threadsafem sequential manner.
template<class FMap, class FApply, class Range>
void mapApply(
	Range const& elements, 
	FMap map, FApply apply, 
	ThreadPool& pool
){
	typedef decltype(map(elements.front())) return_type;
	return_type temp;
	//assign work to pipeline. make sure not to put too many tasks in the pipeline
	Pipeline<return_type> pipeline(pool.numWorkers() *2, pool);
	for(auto pos = elements.begin(); pos != elements.end(); ++pos){
		auto test = [map, pos]{return map(*pos);};
		
		//try to push into the pipeline
		while(pipeline.push(test) == queue_status::full){
			//pipeline full, do a reduce-step
			if(pipeline.pull(temp) == queue_status::success)
				apply(std::move(temp));
			else //otherwise yield thread to the pool to do another round of processing
				pool.yield();
		}
	}
	//now pull until the queue is empty
	while(!pipeline.empty()){
		queue_status status = pipeline.pull(temp);
		if(status == queue_status::success)
			apply(std::move(temp));
		else //otherwise yield thread to the pool to do another round of processing
			pool.yield();
	}
};

///\brief MapApply for binary functions taking a pair of ranges.
///
/// map is performed in parallel, apply is performed in order of elements in range in a threadsafem sequential manner.
template<class FMap, class FApply, class Range1, class Range2>
void mapApply(
	Range1 const& elements1,
	Range2 const& elements2,
	FMap map, FApply apply, 
	ThreadPool& pool
){
	typedef decltype(map(elements1.front(), elements2.front())) return_type;
	return_type temp;
	//assign work to pipeline. make sure not to put too many tasks in the pipeline
	Pipeline<return_type> pipeline(pool.numWorkers() *2, pool);
	auto pos2 = elements2.begin();
	for(auto pos1 = elements1.begin(); pos1 != elements1.end(); ++pos1, ++pos2){
		//try to push into the pipeline
		auto test = [map, pos1, pos2]{return map(*pos1, *pos2);};
		while(pipeline.push(test) == queue_status::full){
			//pipeline full, do a reduce-step
			if(pipeline.pull(temp) == queue_status::success)
				apply(std::move(temp));
			else //otherwise yield thread to the pool to do another round of processing
				pool.yield();
		}
	}
	//now pull until the queue is empty
	while(!pipeline.empty()){
		if(pipeline.pull(temp) == queue_status::success)
			apply(std::move(temp));
		else //otherwise yield thread to the pool to do another round of processing
			pool.yield();
	}
};

/// \brief maps the elements in range and reduces the result using the provided reduce function.
///
/// map is a function map(elem) and reduce is called as: accumulator = function(std::move(accumulator), map(elem1,elem2)).
/// accumulator is initialized with init.
/// map is performed in parallel, reduce is performed in order of elements in range in a threadsafem sequential manner.
template<class T, class FMap, class FReduce, class Range>
T mapReduce(
	Range const& elements, T init,
	FMap map, FReduce reduce, 
	ThreadPool& pool
){
	auto apply = [reduce, &init](T arg){
		init = reduce(std::move(init), std::move(arg));
	};
	mapApply(elements, map, apply, pool);
	return std::move(init);
};

/// \brief maps the elements in range for binary functions and reduces the result using the provided reduce function.
///
/// map is a function map(elem,elem1 and reduce is called as: accumulator = function(std::move(accumulator), map(elem1,elem2)).
/// accumulator is initialized with init.
/// map is performed in parallel, reduce is performed in order of elements in range in a threadsafem sequential manner.
template<class T, class FMap, class FReduce, class Range1, class Range2>
T mapReduce(
	Range1 const& elements1,
	Range2 const& elements2,
	T init,
	FMap map, FReduce reduce, 
	ThreadPool& pool
){
	auto apply = [reduce, &init](T arg){
		init = reduce(std::move(init), std::move(arg));
	};
	mapApply(elements1, elements2, map, apply, pool);
	return std::move(init);
};

/// \brief computes init + map(range_1) + ... + map(range_N) in parallel.
///
/// Addition is implemented as accumulator += map(elem).
template<class T, class FMap, class Range>
T mapAccumulate(
	Range const& elements, T init,
	FMap map, ThreadPool& pool
){
	auto reduce =  [](T a, T const& b){a +=b; return std::move(a);};
	return mapReduce(elements, init, map, reduce, pool);
}

/// \brief computes init + map(range1_1, range2_1) + ... + map(range1_N, range2_N) in parallel.
///
/// Addition is implemented as accumulator += map(elem1, elem2).
template<class T, class FMap, class Range1, class Range2>
T mapAccumulate(
	Range1 const& elements1,
	Range2 const& elements2,
	T init,
	FMap map, ThreadPool& pool
){
	auto reduce =  [](T a, T const& b){a +=b; return std::move(a);};
	return mapReduce(elements1, elements2, init, map, reduce, pool);
}

}}
#endif