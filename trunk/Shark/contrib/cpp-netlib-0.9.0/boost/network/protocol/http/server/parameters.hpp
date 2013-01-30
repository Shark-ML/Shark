#ifndef BOOST_NETWORK_PROTOCOL_HTTP_SERVER_PARAMETERS_HPP_20101210
#define BOOST_NETWORK_PROTOCOL_HTTP_SERVER_PARAMETERS_HPP_20101210

// Copyright 2010 Dean Michael Berris.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include <boost/network/protocol/http/parameters.hpp>

namespace boost { namespace network { namespace http {

    BOOST_PARAMETER_NAME(address)
    BOOST_PARAMETER_NAME(port)
    BOOST_PARAMETER_NAME(handler)
    BOOST_PARAMETER_NAME(thread_pool)

    BOOST_PARAMETER_NAME(reuse_address)
    BOOST_PARAMETER_NAME(report_aborted)
    BOOST_PARAMETER_NAME(receive_buffer_size)
    BOOST_PARAMETER_NAME(send_buffer_size)
    BOOST_PARAMETER_NAME(receive_low_watermark)
    BOOST_PARAMETER_NAME(send_low_watermark)
    BOOST_PARAMETER_NAME(non_blocking_io)
    BOOST_PARAMETER_NAME(linger)
    BOOST_PARAMETER_NAME(linger_timeout)
    
} /* http */
    
} /* network */
    
} /* boost */

#endif /* BOOST_NETWORK_PROTOCOL_HTTP_SERVER_PARAMETERS_HPP_20101210 */
