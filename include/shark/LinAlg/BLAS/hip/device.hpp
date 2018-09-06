//===========================================================================
/*!
 * 
 *
 * \brief       Handling of the hip device
 *
 * \author      O. Krause
 * \date        2018
 *
 *
 * \par Copyright 1995-2015 Shark Development Team
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
//===========================================================================

#ifndef REMORA_HIP_DEVICE_HPP
#define REMORA_HIP_DEVICE_HPP

#include "exception.hpp"
#include <hip/hip_runtime.h>
#include <vector>
namespace remora{namespace hip{
	
class device{
public:
	device(int id)
	:m_id(id), m_use_stream_per_thread(true){
		check_hip(hipGetDeviceProperties(&m_device_properties, m_id));
	}
	device(device const&) = delete;
	device(device&& other){
		m_id = other.m_id;
		m_use_stream_per_thread = other.m_use_stream_per_thread;
		m_device_properties = other.m_device_properties;
	}
	
	device& operator=(device const&) = delete;
	device& operator=(device&&) = delete;
	
	void set_device(){
		hipSetDevice(m_id);
	}
	
	void use_stream_per_thread(bool val){
		m_use_stream_per_thread = val;
	}
	
	bool use_stream_per_thread()const{
		return m_use_stream_per_thread;
	}
	
	int device_id() const{
		return m_id;
	}
	
	int warp_size() const{
		return m_device_properties.warpSize;
	}
	
private:
	int m_id;
	bool m_use_stream_per_thread;
	hipDeviceProp_t m_device_properties;
};

class device_manager{
public:
	friend device_manager& devices();
	hip::device const& device(std::size_t i) const{
		return m_devices[i];
	}
	hip::device& device(std::size_t i){
		return m_devices[i];
	}
	std::size_t num_devices()const{
		return m_devices.size();
	}
private:
	device_manager(){
		int count;
		check_hip(hipGetDeviceCount(&count));
		for(int i = 0; i != count; ++i){
			m_devices.emplace_back(i);
		}
	}
	std::vector<hip::device> m_devices;
};
device_manager& devices(){
	static device_manager mgr;
	return mgr;
}

class stream{
public:
	stream(bool defaultStream = true)
	:m_stream(0){
		if(!defaultStream){
			check_hip(hipStreamCreate(&m_stream));
		}
	}
	stream(stream const&) = delete;
	stream(stream&& other){
		m_stream = other.m_stream;
		other.m_stream = 0;
	}
	
	stream& operator=(stream const&) = delete;
	stream& operator=(stream&& other){
		if(!is_default()){
			hipStreamDestroy(m_stream);
		}
		m_stream = other.m_stream;
		other.m_stream = 0;
		return *this;
	}
	
	void synchronize() const{
		hipStreamSynchronize(m_stream);
	}
	
	bool is_default() const{
		return m_stream == 0;
	}
	
	hipStream_t handle() const{
		return m_stream;
	}
	~stream(){
		if(!is_default()){
			check_hip(hipStreamDestroy(m_stream));
		}
	}
	
private:
 	hipStream_t m_stream;
	bool m_owns_handle;
};

inline stream& get_default_stream(){
	static stream stream;
	return stream;
}
inline stream& get_stream(device& device){
	if(!device.use_stream_per_thread())
		return get_default_stream();
	thread_local static std::vector<std::pair<int, stream> > streams;
	for(std::size_t i = 0; i != streams.size(); ++i){
		if(streams[i].first == device.device_id())
			return streams[i].second;
	}
	device.set_device();
	streams.emplace_back(device.device_id(), stream(false));
	return streams.back().second;
}

inline void synchronize_stream(device& device){
	get_stream(device).synchronize();
}


}}
#endif