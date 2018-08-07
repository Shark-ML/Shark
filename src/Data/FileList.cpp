//===========================================================================
/*!
 * 
 *
 * \brief       Implementation of the FileList class
 * 
 * 
 *
 * \author      O.Krause
 * \date        2018
 *
 *
 * \par Copyright 1995-2018 Shark Development Team
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
//===========================================================================


#define SHARK_COMPILE_DLL
#include <shark/Data/DataDistribution.h>

//~ #include <regex>
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <boost/algorithm/string/replace.hpp>

using namespace shark;


FileList::FileList(std::vector<std::string> const& filePaths):DataDistribution(Shape()), m_paths(filePaths){
	m_paths = filePaths;
}

FileList::FileList(std::vector<std::string> const& filePaths, std::string const& expression):DataDistribution(Shape()){
	m_paths = filterGlob(expression, filePaths);
}

FileList::FileList(std::string const& expression):DataDistribution(Shape()){
	namespace fs = boost::filesystem;
	//we need to find the base path to do our recursive search from
	//in a path a/b/c/d*e/ we take the path before globbing, a/b/c/
	//as the base path
	fs::path globPath=expression;
	fs::path base;
	for(auto pos = globPath.begin(); pos != globPath.end(); ++pos){
		auto const& elem = pos->string();
		if(elem.find('*') != std::string::npos || elem.find('?') != std::string::npos || elem.find('{') != std::string::npos)
			break;
		base /= elem;
	}
	//if there is no base, we assume a relative path
	if(base.empty())
		base = "./";
	
	//base must exist and must be a directory, otherwise we can not iterate
	if(!fs::exists(base) || !fs::is_directory(base))
		throw SHARKEXCEPTION("Path does not exist or is no directory: "+base.string());
	
	std::vector<std::string> filePaths;
	//~ for(auto const& entry: fs::recursive_directory_iterator(base)){//unsupported of boost @ travis
	for(fs::recursive_directory_iterator pos(base); pos != fs::recursive_directory_iterator(); ++pos){
		auto const& entry = *pos;
		//skip directories
		if(fs::is_directory(entry))
			continue;
		filePaths.push_back(entry.path().string());
		
	}
	m_paths = filterGlob(expression, filePaths);
}
	

std::vector<std::string> FileList::filterGlob(std::string expression, std::vector<std::string> const& paths){
	namespace rx=boost;
	//first escape everything that might interfere with the regex we construct
	expression = regex_replace(
		expression, 
		rx::regex("[.^$()\\[\\]+\\\\]"), std::string("\\\\&"),
		rx::regex_constants::match_default | rx::regex_constants::format_sed
	);
	
	//replace "?"->".", "{"->"(?:" and "}"->")"
	boost::replace_all(expression, "?", ".");
	boost::replace_all(expression, "{", "(?:");
	boost::replace_all(expression, "}", ")");
	//replace "*"->".*"
	boost::replace_all(expression, "*", ".*");
	
	//create regex and filter strings
	rx::regex regex(expression, rx::regex::ECMAScript | rx::regex::optimize);
	std::vector<std::string> filtered;
	
	for(auto const& path: paths){
		if(rx::regex_match(path, regex)){
			filtered.push_back(path);
		}
	}
	return filtered;
}