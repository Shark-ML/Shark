//===========================================================================
/*!
 *  \file Chart.h
 *
 *  \brief Chart.h
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
#ifndef SHARK_CORE_CHART_H
#define SHARK_CORE_CHART_H

#include <boost/property_tree/ptree.hpp>
#include <boost/variant.hpp>

namespace shark {

	namespace tag {
		/**
		* \brief Tags the first (left) x-axis of a chart.
		*/
		struct FirstXAxis	{};

		/**
		* \brief Tags the second (right) x-axis of a chart.
		*/
		struct SecondXAxis	{};

		/**
		* \brief Tags the first (bottom) y-axis of a chart.
		*/
		struct FirstYAxis	{};

		/**
		* \brief Tags the second (top) y-axis of a chart.
		*/
		struct SecondYAxis	{};
	}

	/**
	* \brief Models an abstract chart.
	*
	* Allows for programmatically "plotting" of numerical data without the
	* the need to worry about the actual rendering of the chart.
	*
	*/
	class Chart {
	public:

		/**
		* \brief Constant that marks an axis as enabled.
		*/
		static const bool AXIS_ENABLED = true;

		/**
		* \brief Constant that marks an axis as disabled.
		*/
		static const bool AXIS_DISABLED = false;

		/**
		* \brief Default c'tor.
		* 
		* Enabled the left x-axis and the bottom y-axis. All other axis are
		* disabled.
		*/
		Chart( ) : m_firstXAxis( AXIS_ENABLED ),
			m_firstYAxis( AXIS_ENABLED) {
		}

		/**
		* \brief Models the horizontal alignment of elements.
		*/
		enum HorizontalAlignment {
			LEFT,
			CENTER,
			RIGHT
		};

		/**
		* \brief Models the vertical alignment of elements.
		*/
		enum VerticalAlignment {
			TOP,
			MIDDLE,
			BOTTOM
		};

		/**
		* \brief Models an axis and its attributes like its title and its scale.
		*/
		struct Axis {

			/**
			* \brief Models the scale of the axis.
			*/
			enum Scale {
				LINEAR_SCALE, ///< Default setting.
				LOG_10_SCALE
			};
	
			/**
			* \brief C'tor.
			*
			* \param [in] isEnabled Flag that indicate whether the axis shall be enabled or not.
			*/
			Axis( bool isEnabled = false ) : m_isEnabled( isEnabled ),
				m_scale( LINEAR_SCALE ) {}

			/**
			* \brief Access the properties of the axis.
			* 
			* In addition to the attributes that are explicitly modelled 
			* arbitrary key/value pairs can be associated with the axis.
			* These key/value-pairs might be used by renderers to further customize the
			* output.
			*
			* \returns A mutable reference to the properties, allows for l-value semantics.
			*/
			boost::property_tree::ptree & properties() {
				return( m_properties );
			}

			/**
			* \brief Access the properties of the axis.
			* 
			* In addition to the attributes that are explicitly modelled 
			* arbitrary key/value pairs can be associated with the axis.
			* These key/value-pairs might be used by renderers to further customize the
			* output.
			*
			* \returns An immutable reference to the properties.
			*/
			const boost::property_tree::ptree & properties() const {
				return( m_properties );
			}

			/**
			* \brief (De-)Serializes the axis, its attributes and properties.
			* 
			* \tparam Archive The type of the archive.
			* \param [in, out] a The archive to read from/write to.
			* \param [in] version Currently unused.
			*/
			template<typename Archive>
			void serialize( Archive & a, const unsigned int version ) {
				(void) version; // Currently unused
				a & m_scale;
				a & m_title;
				a & m_properties;
			}

			bool m_isEnabled; ///< Indicates whether the axis is enabled or not.

			Scale m_scale; ///< The scaling of the axis.
			std::string m_title; ///< The title of the axis.

			boost::property_tree::ptree m_properties; ///< The properties of the axis.
		};

		/**
		* \brief Models the title of a chart.
		*/
		struct Title {
			std::string m_text; ///< The title text of the chart.
			HorizontalAlignment m_horizontalAlignment; ///< The horizontal alignment of the title text, default value: CENTER.
			VerticalAlignment m_verticalAlignment; ///< The vertical alignment of the title text, default value: MIDDLE.

			boost::property_tree::ptree m_properties; ///< Renderer specific properties of the title, modelled as kez/value pairs.

			/**
			* \brief Default c'tor.
			*/
			Title() : m_horizontalAlignment( CENTER ),
				m_verticalAlignment( MIDDLE ) {}

			/**
			* \brief Accesses the properties of the title.
			* 
			* In addition to the attributes that are explicitly modelled 
			* arbitrary key/value pairs can be associated with the axis.
			* These key/value-pairs might be used by renderers to further customize the
			* output.
			*
			* \returns A mutable reference to the properties, allows for l-value semantics.
			*/
			boost::property_tree::ptree & properties() {
				return( m_properties );
			}

			/**
			* \brief Accesses the properties of the title.
			* 
			* In addition to the attributes that are explicitly modelled 
			* arbitrary key/value pairs can be associated with the axis.
			* These key/value-pairs might be used by renderers to further customize the
			* output.
			*
			* \returns An immutable reference to the properties.
			*/
			const boost::property_tree::ptree & properties() const {
				return( m_properties );
			}

			/**
			* \brief (De-)Serializes the title, its attributes and properties.
			* 
			* \tparam Archive The type of the archive.
			* \param [in, out] a The archive to read from/write to.
			* \param [in] version Currently unused.
			*/
			template<typename Archive>
			void serialize( Archive & a, const unsigned int version ) {
				(void) version; // Currently unused
				a & m_text;
				a & m_horizontalAlignment;
				a & m_verticalAlignment;
				a & m_properties;
			}
		};

		/**
		* \brief Models a series and its underlying data.
		*/
		struct Series {

			/**
			* \brief Models a single value, might either be numeric or a category.
			*/
			typedef boost::variant< double, std::string > ValueType;

			/**
			* \brief Models a pair of x-y values.
			*/
			typedef std::pair< ValueType, ValueType > ElementType;

			/**
			* \brief Models the type of the series.
			*/
			enum Type {
				AREA_TYPE,			
				AREA_SPLINE_TYPE, 
				BAR_TYPE, 
				COLUMN_TYPE, 
				LINE_TYPE, 
				PIE_TYPE, 
				SCATTER_TYPE,
				SPLINE_TYPE
			};

			/**
			* \brief Default c'tor.
			*/
			Series() : m_type( SCATTER_TYPE ) {}

			/**
			* \brief Accesses the properties of the series.
			* 
			* In addition to the attributes that are explicitly modelled 
			* arbitrary key/value pairs can be associated with the series.
			* These key/value-pairs might be used by renderers to further customize the
			* output.
			*
			* \returns A mutable reference to the properties, allows for l-value semantics.
			*/
			boost::property_tree::ptree & properties() {
				return( m_properties );
			}

			/**
			* \brief Accesses the properties of the series.
			* 
			* In addition to the attributes that are explicitly modelled 
			* arbitrary key/value pairs can be associated with the series.
			* These key/value-pairs might be used by renderers to further customize the
			* output.
			*
			* \returns An immutable reference to the properties
			*/
			const boost::property_tree::ptree & properties() const {
				return( m_properties );
			}

			/**
			* \brief (De-)Serializes the series, its data, attributes and properties.
			* 
			* \tparam Archive The type of the archive.
			* \param [in, out] a The archive to read from/write to.
			* \param [in] version Currently unused.
			*/
			template<typename Archive>
			void serialize( Archive & a, const unsigned int version ) {
				(void) version; // Currently unused
				a & m_type;
				a & m_name;
				a & m_data;
				a & m_properties;
			}

			Type m_type; ///< Type of the series, default value: SCATTER.
			std::string m_name; ///< Name of the series.

			std::vector< ElementType > m_data; ///< Data of the series.

			boost::property_tree::ptree m_properties; ///< Optional properties of the series.
		};

		/**
		* \brief Accesses the title of the chart.
		* \returns A mutable reference to the title of the chart, allows for l-value semantic.
		*/
		Title & title() {
			return( m_title );
		}

		/**
		* \brief Accesses the title of the chart.
		* \returns An immutable reference to the title of the chart.
		*/
		const Title & title() const {
			return( m_title );
		}
		
		/**
		* \brief Accesses the first (left) x-axis.
		* \returns An mutable reference to the axis, allows for l-value semantics.
		*/
		Axis & axis( tag::FirstXAxis ) {
			return( m_firstXAxis );
		}

		/**
		* \brief Accesses the first (bottom) y-axis.
		* \returns An mutable reference to the axis, allows for l-value semantics.
		*/
		Axis & axis( tag::FirstYAxis ) {
			return( m_firstYAxis );
		}

		/**
		* \brief Accesses the second (right) x-axis.
		* \returns An mutable reference to the axis, allows for l-value semantics.
		*/
		Axis & axis( tag::SecondXAxis ) {
			return( m_secondXAxis );
		}

		/**
		* \brief Accesses the second (bottom) y-axis.
		* \returns An mutable reference to the axis, allows for l-value semantics.
		*/
		Axis & axis( tag::SecondYAxis ) {
			return( m_secondYAxis );
		}

		/**
		* \brief Accesses the first (left) x-axis.
		* \returns An immutable reference to the axis.
		*/
		const Axis & axis( tag::FirstXAxis ) const {
			return( m_firstXAxis );
		}

		/**
		* \brief Accesses the second (right) x-axis.
		* \returns An immutable reference to the axis.
		*/
		const Axis & axis( tag::SecondXAxis ) const {
			return( m_secondXAxis );
		}
		
		/**
		* \brief Accesses the first (left) y-axis.
		* \returns An immutable reference to the axis.
		*/
		const Axis & axis( tag::FirstYAxis ) const {
			return( m_firstYAxis );
		}

		/**
		* \brief Accesses the second (top) y-axis.
		* \returns An immutable reference to the axis.
		*/
		const Axis & axis( tag::SecondYAxis ) const {
			return( m_secondYAxis );
		}

		/**
		* \brief Accesses the vector of series objects.
		* \returns A mutable reference to the vector, allows for l-value semantics.
		*/
		std::vector< Series > & series() {
			return( m_series );
		}
	
		/**
		* \brief Accesses the vector of series objects.
		* \returns An immutable reference to the vector.
		*/
		const std::vector< Series > & series() const {
			return( m_series );
		}	

		/**
		* \brief Accesses the properties of the chart.
		* 
		* In addition to the attributes that are explicitly modelled 
		* arbitrary key/value pairs can be associated with the chart.
		* These key/value-pairs might be used by renderers to further customize the
		* output.
		*
		* \returns A mutable reference to the properties, allows for l-value semantics.
		*/
		boost::property_tree::ptree & properties() {
			return( m_properties );
		}

		/**
		* \brief Accesses the properties of the chart.
		* 
		* In addition to the attributes that are explicitly modelled 
		* arbitrary key/value pairs can be associated with the chart.
		* These key/value-pairs might be used by renderers to further customize the
		* output.
		*
		* \returns An immutable reference to the properties
		*/
		const boost::property_tree::ptree & properties() const {
			return( m_properties );
		}

		/**
		* \brief (De-)Serializes the title, its attributes and properties.
		* 
		* \tparam Archive The type of the archive.
		* \param [in, out] a The archive to read from/write to.
		* \param [in] version Currently unused.
		*/
		template<typename Archive>
		void serialize( Archive & a, const unsigned int version ) {
			a & m_title;

			a & m_firstXAxis;
			a & m_secondXAxis;
			a & m_firstYAxis;
			a & m_secondYAxis;

			a & m_series;
			a & m_properties;
		}

		Title m_title;

		Axis m_firstXAxis;
		Axis m_secondXAxis;
		Axis m_firstYAxis;
		Axis m_secondYAxis;

		std::vector< Series > m_series;

		boost::property_tree::ptree m_properties;
	};

}

#endif // SHARK_CORE_CHART_H
