//            Copyright (c) Glyn Matthews 2010.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)


#define BOOST_TEST_MODULE HTTP URL Test
#include <boost/config/warning_disable.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/network/uri/http/uri.hpp>
#include <boost/mpl/list.hpp>
#include <boost/range/algorithm/equal.hpp>

using namespace boost::network;

BOOST_AUTO_TEST_CASE(http_url_test) {
    typedef uri::basic_uri<http::tags::http_default_8bit_tcp_resolve> uri_type;
    typedef uri_type::string_type string_type;

    const std::string url("http://www.boost.org/");
    const std::string scheme("http");
    const std::string host("www.boost.org");
    const std::string path("/");

    uri_type instance(string_type(boost::begin(url), boost::end(url)));
    boost::optional<string_type> host_ = uri::host(instance);
    boost::optional<boost::uint16_t> port_ = uri::port(instance);

    BOOST_REQUIRE(uri::is_valid(instance));
    BOOST_CHECK_EQUAL(instance.raw(), url);
    BOOST_CHECK( !port_ );
    string_type scheme_ = uri::scheme(instance);
    BOOST_CHECK_EQUAL(scheme_, scheme);
    BOOST_CHECK(boost::equal(uri::scheme(instance), scheme));
    BOOST_CHECK(boost::equal(uri::host(instance), host));
    BOOST_CHECK(boost::equal(uri::path(instance), path));
}

BOOST_AUTO_TEST_CASE(full_http_url_test) {
    typedef uri::basic_uri<http::tags::http_default_8bit_tcp_resolve> uri_type;
    typedef uri_type::string_type string_type;

    const std::string url("http://user:password@www.boost.org:8000/path?query#fragment");
    const std::string scheme("http");
    const std::string user_info("user:password");
    const std::string host("www.boost.org");
    const boost::uint16_t port = 8000;
    const std::string path("/path");
    const std::string query("query");
    const std::string fragment("fragment");

    uri_type instance(string_type(boost::begin(url), boost::end(url)));
    BOOST_REQUIRE(uri::is_valid(instance));
    BOOST_CHECK(boost::equal(uri::scheme(instance), scheme));
    BOOST_CHECK(boost::equal(uri::user_info(instance), user_info));
    BOOST_CHECK(boost::equal(uri::host(instance), host));
    BOOST_CHECK_EQUAL(uri::port(instance), port);
    BOOST_CHECK(boost::equal(uri::path(instance), path));
    BOOST_CHECK(boost::equal(uri::query(instance), query));
    BOOST_CHECK(boost::equal(uri::fragment(instance), fragment));
}

BOOST_AUTO_TEST_CASE(https_url_test) {
    typedef uri::basic_uri<http::tags::http_default_8bit_tcp_resolve> uri_type;
    typedef uri_type::string_type string_type;

    const std::string url("https://www.boost.org/");
    const std::string scheme("https");
    const std::string host("www.boost.org");
    const boost::uint16_t port = 443;
    const std::string path("/");

    uri_type instance(string_type(boost::begin(url), boost::end(url)));
    BOOST_REQUIRE(uri::is_valid(instance));
    BOOST_CHECK(boost::equal(uri::scheme(instance), scheme));
    BOOST_CHECK(boost::equal(uri::host(instance), host));
    BOOST_CHECK_EQUAL(uri::port(instance), port);
    BOOST_CHECK(boost::equal(uri::path(instance), path));
}

//BOOST_AUTO_TEST_CASE(invalid_http_url_test) {
//    typedef uri::basic_uri<http::tags::http_default_8bit_tcp_resolve> uri_type;
//    typedef uri_type::string_type string_type;

//    const std::string url("ftp://www.boost.org/");

//    uri_type instance(string_type(boost::begin(url), boost::end(url)));
//    BOOST_CHECK(!uri::is_valid(instance));
//}
