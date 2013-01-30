//===========================================================================
/*!
 *  \file HighchartRenderer.h
 *
 *  \brief HighchartRenderer.h
 *
 *  \author T.Voss
 *  \date 2011
 *
 *  \par Copyright (c) 1998-2007:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
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
//===========================================================================
#ifndef SHARK_CORE_RENDERERS_HIGHCHART_RENDERER_H
#define SHARK_CORE_RENDERERS_HIGHCHART_RENDERER_H

#include <shark/Core/Exception.h>

#include <boost/foreach.hpp>
#include <boost/format.hpp>

namespace shark {

	/**
	* \brief Models a renderer that renders charts using the 
	* Highcharts JS API (see http://www.highcharts.com).
	* \tparam Stream The type of the stream to write the resulting document to, needs to adhere to the STL stream concept.
	*/
	template<typename Stream>
	struct HighchartRenderer {
		
		static std::string type_to_name( shark::Chart::Series::Type seriesType ) {
			std::string result;

			switch( seriesType ) {
				case shark::Chart::Series::AREA_TYPE:
					result = "area";
					break;
				case shark::Chart::Series::AREA_SPLINE_TYPE: 
					result = "areaspline";
					break;
				case shark::Chart::Series::BAR_TYPE:
					result = "bar";
					break;
				case shark::Chart::Series::COLUMN_TYPE:
					result = "column";
					break;
				case shark::Chart::Series::LINE_TYPE:
					result = "line";
					break;
				case shark::Chart::Series::PIE_TYPE:
					result = "pie";
					break;
				case shark::Chart::Series::SCATTER_TYPE:
					result = "scatter";
					break;
				case shark::Chart::Series::SPLINE_TYPE:
					result = "spline";
					break;
			}
			return( result );
		}

		/**
		* \brief Thrown if the supplied stream is bad.
		*/
		struct BadStreamException : public shark::Exception {
			BadStreamException() : shark::Exception( "Bad stream" ) {}
		};

		/**
		* \brief c'tor.
		* \param [in,out] s The stream to write to.
		*/
		HighchartRenderer( Stream & s ) : m_stream( s ) {}

		/**
		* \brief Renders the supplied chart to the previously supplied stream.
		* \tparam Chart The type of the chart, needs to adhere to the concept modelled by the class shark::Chart.
		* \param [in] chart The chart to be rendered.
		* \throws BadStreamException if the previously supplied stream is bad.
		*/
		template<typename Chart>
		void render( const Chart & chart ) {
			if( !m_stream )
				throw( BadStreamException() );

			m_stream << "<html>" << std::endl; {
				m_stream << "<head>" << std::endl; {
					m_stream << "<script src=\"http://ajax.googleapis.com/ajax/libs/jquery/1.4.4/jquery.min.js\" type=\"text/javascript\">" << std::endl;
					m_stream << "</script>" << std::endl;
					m_stream << "<script src=\"js/highcharts.js\" type=\"text/javascript\">" << std::endl;
					m_stream << "</script>" << std::endl;
					m_stream << "<script src=\"js/modules/exporting.js\" type=\"text/javascript\">" << std::endl;
					m_stream << "</script>" << std::endl;
					m_stream << "<script src=\"js/themes/gray.js\" type=\"text/javascript\">" << std::endl;
					m_stream << "</script>" << std::endl;
					m_stream << "<script type=\"text/javascript\">" << std::endl; {
						boost::format f( "var chart1; // globally available\n\
										$(document).ready(function() {\n\
											chart1 = new Highcharts.Chart({\n\
												chart: {\n\
													renderTo: 'chart-container-1',\n\
													defaultSeriesType: 'bar',\n\
													zoomType: 'xy'\n\
												},\n\
												title: {\n\
													text: '%1%'\n\
												},\n\
												xAxis: {\n\
													title: {\n\
														text: '%2%'\n\
													}\n\
												},\n\
												yAxis: {\n\
													title: {\n\
														text: '%3%'\n\
													}\n\
												},\n\
												series: [\n\
													%4%\n\
													]\n\
												});\n\
											});" 
						);

						/*{\n\
name: 'Jane',\n\
data: [1, 0, 4]\n\
						},\n\
						{\n\
name: 'John',\n\
data: [5, 7, 3]\n\
						}\n\*/
						std::stringstream ss;
						for( unsigned int i = 0; i < chart.m_series.size(); i++ ) {
							ss << "{ name: '" << chart.m_series[i].m_name << "', type: '" << HighchartRenderer::type_to_name(chart.m_series.at( i ).m_type) << "', data: [";
							BOOST_FOREACH( const shark::Chart::Series::ElementType & e, chart.m_series[i].m_data ) {
								ss << "[" << e.first << "," << e.second << "],";
							}
							ss << "]}";
							if( i + 1 < chart.m_series.size() )
								ss << ",";
						}	

						f = f % 
							chart.title().m_text  % 
							chart.axis( tag::FirstXAxis() ).m_title %
							chart.axis( tag::FirstYAxis() ).m_title %
							ss.str();

						m_stream << f << std::endl;
					} m_stream << "</script>" << std::endl;
				} m_stream << "</head>" << std::endl;
				m_stream << "<body>" << std::endl; {
					m_stream << "<div id=\"chart-container-1\"></div>" << std::endl;
				} m_stream << "</body>" << std::endl;
			} m_stream << "</html>" << std::endl;
		}

		Stream & m_stream; ///< Reference to the user supplied stream.
	};

}

#endif // SHARK_CORE_RENDERERS_HIGHCHART_RENDERER_H