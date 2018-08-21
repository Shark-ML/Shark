/*!
 * 
 *
 * \brief       Implements support for Zip-Archives
 * 
 * 
 *
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

#include <shark/Core/ZipSupport.h>
#include <shark/Core/Exception.h>
#include <mutex>
#include <unordered_map>

extern "C"{
#include <zip.h>
}
using namespace shark;

struct ZipReader::ZipImpl{
	mutable std::mutex m_mutex;
	zip_t* m_archive;
	struct Info{
		std::string name;
		std::size_t size;
		zip_uint64_t index;
	};
	std::vector<Info> m_fileList;
	std::unordered_map<std::string, std::size_t> m_nameLookup;
	
	ZipImpl(std::string const& pathToArchive){
		int errorCode;
		m_archive = zip_open(pathToArchive.c_str(), ZIP_RDONLY, &errorCode);
		if(m_archive == nullptr){
			zip_error_t zipError;
			zip_error_init_with_code(&zipError, errorCode); 
			std::string message = std::string("Error opening ZipArchive. Error returned was:") + zip_error_strerror(&zipError);
			zip_error_fini(&zipError);
			throw SHARKEXCEPTION(message);
		}
		
		//enumerate list of all files
		zip_uint64_t numEntries = zip_get_num_entries(m_archive, 0);
		zip_stat_t stat;
		for(zip_uint64_t i = 0; i != numEntries; ++i){
			if(zip_stat_index(m_archive, i, 0, &stat) == -1){
				handleError();
			}
			std::string fileName = stat.name;
			if(fileName.back() != '/'){
				m_fileList.push_back({fileName, stat.size, i});
				m_nameLookup.emplace(fileName, m_fileList.size() -1);
			}
		}
	}
	
	~ZipImpl(){
		zip_close(m_archive);
	}
	void handleError() const{
		zip_error_t* zipError = zip_get_error(m_archive);
		std::string message =  std::string("Error processing zip-archive. Error returned was:") + zip_error_strerror(zipError);
		throw SHARKEXCEPTION(message);
	}
	
	std::vector<unsigned char> readFile(Info const& info) const{
		//lock for thread-safety
		std::lock_guard<std::mutex> lock(m_mutex);
		
		//open file for reading
		zip_file_t* file = zip_fopen_index(m_archive, info.index, ZIP_FL_UNCHANGED);
		if(file == nullptr){
			handleError();
		}

		//read file contents chunk by chunk
		std::vector<unsigned char> buffer(info.size);
		unsigned char* pos = buffer.data();
		unsigned char* end = pos + buffer.size();
		while(pos != end){
			zip_int64_t  len = zip_fread(file, (void*) pos, end - pos);
			if(len == -1){
				std::string message =  std::string("Error reading from zip-archive. Error returned was:") + zip_file_strerror(file);
				zip_fclose(file);
				throw SHARKEXCEPTION(message);
			}
			pos += len;
		}
		zip_fclose(file);
		return buffer;
	}
};

ZipReader::ZipReader(std::string const& pathToArchive):m_impl(new ZipImpl(pathToArchive)){}

//explicit implementation in the cpp so that the destructor of ZipImpl is defined!
ZipReader::~ZipReader() = default;

std::size_t ZipReader::numFiles() const{
	return m_impl->m_fileList.size();
}
std::string const& ZipReader::fileName(std::size_t i) const{
	SIZE_CHECK(i < m_impl->m_fileList.size());
	return m_impl->m_fileList[i].name;
}

std::vector<std::string> ZipReader::fileNames() const{
	std::vector<std::string> names(m_impl->m_fileList.size());
	for(std::size_t i = 0; i != names.size(); ++i){
		names[i] = m_impl->m_fileList[i].name;
	}
	return names;
}


std::vector<unsigned char> ZipReader::readFile(std::size_t i) const{
	return m_impl->readFile(m_impl->m_fileList[i]);
}

std::vector<unsigned char> ZipReader::readFile(std::string const& path) const{
	auto pos = m_impl->m_nameLookup.find(path);
	if(pos == m_impl->m_nameLookup.end()){
		throw SHARKEXCEPTION("FileName does not exist in archive!");
	}
	return m_impl->readFile(m_impl->m_fileList[pos->second]);
}