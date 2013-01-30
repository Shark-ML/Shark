/**
 *
 *  \brief Stores Pareto-fronts.
 *
 *  \author T.Voss
 *  \date 2010
 *
 *
 *  <BR><HR>
 *  This file is part of Shark. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 3, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, see <http://www.gnu.org/licenses/>.
 *
 */
#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_EXPERIMENTS_FRONT_STORE_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_EXPERIMENTS_FRONT_STORE_H

#include <shark/Core/Exception.h>
#include <shark/Core/Shark.h>

#include <shark/Data/json_spirit_writer_template.h>

#include <boost/filesystem.hpp>
#include <boost/format.hpp>

#include <fstream>
#include <string>

namespace shark {

namespace tag {

struct JSONFormat {};

struct RawTextFormat {};

}

/**
 * \brief A store for Pareto-front approximations, for usage with
 * class InterruptibleAlgorithmRunner.
 * \tparam ResultType The front type.
 * \tparam MetaDataType Meta data type that describes the front.
 */
template<typename ResultType, typename MetaDataType>
struct FrontStore {

  enum Format {
    JSON_FORMAT,
    RAW_TEXT_FORMAT
  };

  /** \brief Make the result_type known. */
  typedef ResultType result_type;

  /** \brief Make the result_meta_data_type known. */
  typedef MetaDataType result_meta_data_type;

  FrontStore( const std::string & resultDir = "." )
      : m_format(JSON_FORMAT),
        m_resultDir(resultDir) {
  }

  /**
   * \brief Stores the supplied front.
   * \param [in] front Pareto-front approximation.
   * \param [in] meta Meta data describing the front.
   *
   * Fronts are stored in the filesystem according to the 
   * following scheme:
   *   - resultDir
   *   - algoName 
   *   - functionName
   *   - n
   *   - m
   *   - seed;
   */
  void onNewResult ( 
      const result_type & front,
      const result_meta_data_type & meta
                     ) {

    boost::format fnFormat( "%1%/%2%/%3%/%4%/%5%/%6%" );
    fnFormat = fnFormat % 
        m_resultDir % 
        meta.m_optimizerName % 
        meta.m_objectiveFunctionName % 
        meta.m_searchSpaceDimension % 
        meta.m_objectiveSpaceDimension % 
        meta.m_seed;

    boost::filesystem::path p( fnFormat.str() );
    try {
      boost::filesystem::create_directories( p );
    } catch( ... ) {
      throw SHARKEXCEPTION( "Problem creating directory" + p.string() );
    }

    if( meta.m_isFinal )
      p /= ( boost::format( "%1%.txt" ) % "Final" ).str(); 
    else
      p /= ( boost::format( "%1%.txt" ) % meta.m_evaluationCounter ).str(); 

    std::ofstream out( p.string().c_str() );
    if( !out ) {
      throw SHARKEXCEPTION( "Problem opening result file: " + p.string() );
    }

    switch(m_format) {
      case JSON_FORMAT:
        writeFront(out, front, meta, tag::JSONFormat());
        break;
      case RAW_TEXT_FORMAT:
        writeFront(out, front, meta, tag::RawTextFormat());
        break;
    }
    
  }

  template<typename Stream>
  void writeFront(
      Stream & s,
      const result_type & front,
      const result_meta_data_type & meta,
      tag::JSONFormat) const {
    json_spirit::Object result;
    result.push_back(
        json_spirit::Pair(
            "evaluationCounter",
            static_cast<int>(meta.m_evaluationCounter)));
    result.push_back(
        json_spirit::Pair(
            "timestamp",
            static_cast<int>(meta.m_timeStamp)));    
    
    json_spirit::Array array;
    typename result_type::const_iterator itf;
    for(itf = front.begin(); itf != front.end(); ++itf ) {
      json_spirit::Object point;

      typename result_type::value_type::ResultType::const_iterator itp;
      for(itp = itf->value.begin();
          itp != itf->value.end();
          ++itp) {
        boost::format f("%1%");
        f = f % std::distance(
            itf->value.begin(),
            itp);
        
        point.push_back(
            json_spirit::Pair(
                f.str(),
                *itp));
      }
              
      array.push_back(point);
    }

    result.push_back(
        json_spirit::Pair(
            "data",
            array));
    json_spirit::write_stream(
        json_spirit::Value(result),
        s,
        json_spirit::pretty_print);
  }

  template<typename Stream>
  void writeFront(
      Stream & s,
      const result_type & front,
      const result_meta_data_type & meta,
      tag::RawTextFormat) const {
    s << meta.m_evaluationCounter << "," << meta.m_timeStamp << std::endl;
    
    typename result_type::const_iterator itf;
    for( itf = front.begin(); itf != front.end(); ++itf ) {
      std::copy( 
          itf->value.begin(),
          itf->value.end(),
          std::ostream_iterator<double>( s, " " )
                 );
      s << std::endl;
    }
    s << std::endl;
  }

  Format m_format;
  std::string m_resultDir;
};
}

#endif 
