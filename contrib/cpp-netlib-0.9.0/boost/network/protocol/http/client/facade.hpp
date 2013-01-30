#ifndef BOOST_NETWORK_PROTOCOL_HTTP_CLIENT_FACADE_HPP_20100623
#define BOOST_NETWORK_PROTOCOL_HTTP_CLIENT_FACADE_HPP_20100623

// Copyright Dean Michael Berris 2010.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include <boost/network/protocol/http/request.hpp>
#include <boost/network/protocol/http/response.hpp>
#include <boost/network/protocol/http/client/pimpl.hpp>
#include <boost/network/protocol/http/client/parameters.hpp>

namespace boost { namespace network { namespace http {

    template <class Tag>
    struct basic_request;

    template <class Tag>
    struct basic_response;

    template <class Tag, unsigned version_major, unsigned version_minor>
    struct basic_client_facade {

        typedef typename string<Tag>::type string_type;
        typedef basic_request<Tag> request;
        typedef basic_response<Tag> response;
        typedef basic_client_impl<Tag,version_major,version_minor> pimpl_type;

        template <class ArgPack>
        basic_client_facade(ArgPack const & args)
        {
            init_pimpl(args, 
                typename mpl::if_<
                    is_same<
                        typename parameter::value_type<ArgPack, tag::io_service, void>::type,
                        void
                        >,
                    no_io_service,
                    has_io_service
                    >::type());
        }

        response const head (request const & request_) {
            return pimpl->request_skeleton(request_, "HEAD", false);
        }

        response const get (request const & request_) {
            return pimpl->request_skeleton(request_, "GET", true);
        }

        response const post (request const & request_) {
            return pimpl->request_skeleton(request_, "POST", true);
        }

        response const post (request request_, string_type const & content_type, string_type const & body_) {
            if (!boost::empty(headers(request_)["Content-Type"]))
                request_ << remove_header("Content-Type");

            request_ << ::boost::network::body(body_)
                << header("Content-Type", content_type)
                << header("Content-Length", boost::lexical_cast<string_type>(body_.size()));
            return post(request_);
        }

        response const post (request const & request_, string_type const & body_) {
            string_type content_type = "x-application/octet-stream";
            typename headers_range<request>::type content_type_headers =
                headers(request_)["Content-Type"];
            if (!boost::empty(content_type_headers))
                content_type = boost::begin(content_type_headers)->second;
            return post(request_, content_type, body_);
        }

        response const put (request const & request_) {
            return pimpl->request_skeleton(request_, "PUT", true);
        }

        response const put (request const & request_, string_type const & body_) {
            string_type content_type = "x-application/octet-stream";
            typename headers_range<request>::type content_type_headers =
                headers(request_)["Content-Type"];
            if (!boost::empty(content_type_headers))
                content_type = boost::begin(content_type_headers)->second;
            return put(request_, content_type, body_);
        }

        response const put (request request_, string_type const & content_type, string_type const & body_) {
            if (!boost::empty(headers(request_)["Content-Type"]))
                request_ << remove_header("Content-Type");

            request_ << ::boost::network::body(body_)
                << header("Content-Type", content_type)
                << header("Content-Length", boost::lexical_cast<string_type>(body_.size()));
            return put(request_);
        }

        response const delete_ (request const & request_) {
            return pimpl->request_skeleton(request_, "DELETE", true);
        }

        void clear_resolved_cache() {
            pimpl->clear_resolved_cache();
        }

    protected:

        struct no_io_service {};
        struct has_io_service {};

        boost::shared_ptr<pimpl_type> pimpl;

        template <class ArgPack>
        void init_pimpl(ArgPack const & args, no_io_service) {
            pimpl.reset(
                new pimpl_type(
                    args[_cache_resolved|false]
                    , args[_follow_redirects|false]
                    , optional<string_type>(args[_openssl_certificate|optional<string_type>()])
                    , optional<string_type>(args[_openssl_verify_path|optional<string_type>()])
                    )
                );
        }

        template <class ArgPack>
        void init_pimpl(ArgPack const & args, has_io_service) {
            pimpl.reset(
                new pimpl_type(
                    args[_cache_resolved|false]
                    , args[_follow_redirects|false]
                    , args[_io_service]
                    , optional<string_type>(args[_openssl_certificate|optional<string_type>()])
                    , optional<string_type>(args[_openssl_verify_path|optional<string_type>()])
                    )
                );
        }

    };

} // namespace http

} // namespace network

} // namespace boost

#endif // BOOST_NETWORK_PROTOCOL_HTTP_CLIENT_FACADE_HPP_20100623
