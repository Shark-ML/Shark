#ifndef BOOST_NETWORK_PROTOCOL_HTTP_CLIENT_20091215
#define BOOST_NETWORK_PROTOCOL_HTTP_CLIENT_20091215

//          Copyright Dean Michael Berris 2007-2010.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <boost/network/version.hpp>
#include <boost/network/traits/ostringstream.hpp>
#include <boost/network/protocol/http/message.hpp>
#include <boost/network/protocol/http/response.hpp>
#include <boost/network/protocol/http/request.hpp>

#include <boost/asio/io_service.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/foreach.hpp>
#include <ostream>
#include <istream>
#include <string>
#include <stdexcept>
#include <map>

#include <boost/network/protocol/http/client/facade.hpp>
#include <boost/network/protocol/http/client/parameters.hpp>

namespace boost { namespace network { namespace http {

    template <class Tag, unsigned version_major, unsigned version_minor>
    struct basic_client
        : basic_client_facade<Tag, version_major, version_minor>
    {
    private:
        typedef basic_client_facade<Tag, version_major, version_minor> 
            base_facade_type;
    public:
        typedef basic_request<Tag> request;
        typedef basic_response<Tag> response;
        typedef typename string<Tag>::type string_type;
        typedef Tag tag_type;

        // Constructor
        // =================================================================
        // This is a Boost.Parameter-based constructor forwarder, whose
        // implementation is actually forwarded to the base type.
        //
        // The supported parameters are:
        //      _follow_redirects : bool -- whether to follow HTTP redirect
        //                                  responses (default: false)
        //      _cache_resolved   : bool -- whether to cache the resolved
        //                                  endpoints (default: false)
        //      _io_service       : boost::asio::io_service &
        //                               -- supply an io_service to the
        //                                  client
        //      _openssl_certificate : string
        //                               -- the name of the certificate file
        //                                  to use
        //      _openssl_verify_path : string
        //                               -- the name of the directory from
        //                                  which the certificate authority
        //                                  files can be found

        BOOST_PARAMETER_CONSTRUCTOR(
            basic_client, (base_facade_type), tag,
            (optional   
                (in_out(io_service), (boost::asio::io_service))
                (follow_redirects, (bool))
                (cache_resolved, (bool))
                (openssl_certificate, (string_type))
                (openssl_verify_path, (string_type))
                ))

        //
        // =================================================================

    };

    typedef basic_client<tags::http_default_8bit_udp_resolve, 1, 0> client;

} // namespace http

} // namespace network

} // namespace boost

#endif // BOOST_NETWORK_PROTOCOL_HTTP_CLIENT_20091215
