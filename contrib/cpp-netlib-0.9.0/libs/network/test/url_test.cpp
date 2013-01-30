
// Copyright 2009 Dean Michael Berris, Jeroen Habraken.
// Copyright 2010 Glyn Matthews.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt of copy at
// http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_TEST_MODULE URL Test
#include <boost/config/warning_disable.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/network/uri/basic_uri.hpp>
#include <boost/network/tags.hpp>
#include <boost/mpl/list.hpp>
#include <boost/range/algorithm/equal.hpp>

using namespace boost::network;

typedef boost::mpl::list<
    tags::default_string
    , tags::default_wstring
    > tag_types;


BOOST_AUTO_TEST_CASE_TEMPLATE(uri_test, T, tag_types) {
    typedef uri::basic_uri<T> uri_type;
    typedef typename uri_type::string_type string_type;

    const std::string url("http://www.boost.org/");
    const std::string scheme("http");
    const std::string host("www.boost.org");
    const std::string path("/");

    uri_type instance(string_type(boost::begin(url), boost::end(url)));
    BOOST_REQUIRE(uri::is_valid(instance));
    BOOST_CHECK(boost::equal(uri::scheme(instance), scheme));
    BOOST_CHECK(boost::equal(uri::host(instance), host));
    BOOST_CHECK(boost::equal(uri::path(instance), path));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(full_uri_test, T, tag_types) {
    typedef uri::basic_uri<T> uri_type;
    typedef typename uri_type::string_type string_type;

    const std::string url("http://user:password@www.boost.org:8000/path?query#fragment");
    const std::string scheme("http");
    const std::string user_info("user:password");
    const std::string host("www.boost.org");
    boost::uint16_t port = 8000;
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

BOOST_AUTO_TEST_CASE_TEMPLATE(mailto_test, T, tag_types) {
    typedef uri::basic_uri<T> uri_type;
    typedef typename uri_type::string_type string_type;

    const std::string url("mailto:john.doe@example.com");
    const std::string scheme("mailto");
    // RFC 3986 interprets this as the path
    const std::string path("john.doe@example.com");

    uri_type instance(string_type(boost::begin(url), boost::end(url)));
    BOOST_REQUIRE(uri::is_valid(instance));
    BOOST_CHECK(boost::equal(uri::scheme(instance), scheme));
    BOOST_CHECK(boost::equal(uri::path(instance), path));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(file_test, T, tag_types) {
    typedef uri::basic_uri<T> uri_type;
    typedef typename uri_type::string_type string_type;

    const std::string url("file:///bin/bash");
    const std::string scheme("file");
    const std::string path("/bin/bash");

    uri_type instance(string_type(boost::begin(url), boost::end(url)));
    BOOST_REQUIRE(uri::is_valid(instance));
    BOOST_CHECK(boost::equal(uri::scheme(instance), scheme));
    BOOST_CHECK(boost::equal(uri::path(instance), path));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ipv4_address_test, T, tag_types) {
    typedef uri::basic_uri<T> uri_type;
    typedef typename uri_type::string_type string_type;

    const std::string url("http://129.79.245.252/");
    const std::string scheme("http");
    const std::string host("129.79.245.252");
    const std::string path("/");

    uri_type instance(string_type(boost::begin(url), boost::end(url)));
    BOOST_REQUIRE(uri::is_valid(instance));
    BOOST_CHECK(boost::equal(uri::scheme(instance), scheme));
    BOOST_CHECK(boost::equal(uri::host(instance), host));
    BOOST_CHECK(boost::equal(uri::path(instance), path));
}

// IPv6 is not yet supported by the parser
// BOOST_AUTO_TEST_CASE_TEMPLATE(ipv6_address_test, T, tag_types) {
//     typedef uri::basic_uri<T> uri_type;
//     typedef typename uri_type::string_type string_type;
//
//     const std::string url("http://1080:0:0:0:8:800:200C:417A/");
//     const std::string scheme("http");
//     const std::string host("1080:0:0:8:800:200C:417A");
//     const std::string path("/");
//
//     uri_type instance(string_type(boost::begin(url), boost::end(url)));
//     BOOST_REQUIRE(uri::is_valid(instance));
//     std::cout << uri::scheme(instance) << std::endl;
//     std::cout << uri::user_info(instance) << std::endl;
//     std::cout << uri::host(instance) << std::endl;
//     std::cout << uri::port(instance) << std::endl;
//     std::cout << uri::path(instance) << std::endl;
//     std::cout << uri::query(instance) << std::endl;
//     std::cout << uri::fragment(instance) << std::endl;
//     BOOST_CHECK(boost::equal(uri::scheme(instance), scheme));
//     BOOST_CHECK(boost::equal(uri::host(instance), host));
//     BOOST_CHECK(boost::equal(uri::path(instance), path));
// }

BOOST_AUTO_TEST_CASE_TEMPLATE(ftp_test, T, tag_types) {
    typedef uri::basic_uri<T> uri_type;
    typedef typename uri_type::string_type string_type;

    const std::string url("ftp://john.doe@ftp.example.com/");
    const std::string scheme("ftp");
    const std::string user_info("john.doe");
    const std::string host("ftp.example.com");
    const std::string path("/");

    uri_type instance(string_type(boost::begin(url), boost::end(url)));
    BOOST_REQUIRE(uri::is_valid(instance));
    BOOST_CHECK(boost::equal(uri::scheme(instance), scheme));
    BOOST_CHECK(boost::equal(uri::user_info(instance), user_info));
    BOOST_CHECK(boost::equal(uri::host(instance), host));
    BOOST_CHECK(boost::equal(uri::path(instance), path));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(news_test, T, tag_types) {
    typedef uri::basic_uri<T> uri_type;
    typedef typename uri_type::string_type string_type;

    const std::string url("news:comp.infosystems.www.servers.unix");
    const std::string scheme("news");
    const std::string path("comp.infosystems.www.servers.unix");

    uri_type instance(string_type(boost::begin(url), boost::end(url)));
    BOOST_REQUIRE(uri::is_valid(instance));
    BOOST_CHECK(boost::equal(uri::scheme(instance), scheme));
    BOOST_CHECK(boost::equal(uri::path(instance), path));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(tel_test, T, tag_types) {
    typedef uri::basic_uri<T> uri_type;
    typedef typename uri_type::string_type string_type;

    const std::string url("tel:+1-816-555-1212");
    const std::string scheme("tel");
    const std::string path("+1-816-555-1212");

    uri_type instance(string_type(boost::begin(url), boost::end(url)));
    BOOST_REQUIRE(uri::is_valid(instance));
    BOOST_CHECK(boost::equal(uri::scheme(instance), scheme));
    BOOST_CHECK(boost::equal(uri::path(instance), path));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(encoded_uri_test, T, tag_types) {
    typedef uri::basic_uri<T> uri_type;
    typedef typename uri_type::string_type string_type;

    const std::string url("http://www.boost.org/");
    const std::string scheme("http");
    const std::string host("www.boost.org");
    const std::string path("/");

    uri_type instance(string_type(boost::begin(url), boost::end(url)));
    BOOST_REQUIRE(uri::is_valid(instance));
    BOOST_CHECK(boost::equal(uri::scheme(instance), scheme));
    BOOST_CHECK(boost::equal(uri::host(instance), host));
    BOOST_CHECK(boost::equal(uri::path(instance), path));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(copy_constructor_test, T, tag_types) {
    typedef uri::basic_uri<T> uri_type;
    typedef typename uri_type::string_type string_type;

    const std::string url("http://www.boost.org/");

    uri_type instance(string_type(boost::begin(url), boost::end(url)));
    uri_type copy = instance;
    BOOST_CHECK(instance == copy);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(assignment_test, T, tag_types) {
    typedef uri::basic_uri<T> uri_type;
    typedef typename uri_type::string_type string_type;

    const std::string url("http://www.boost.org/");

    uri_type instance(string_type(boost::begin(url), boost::end(url)));
    uri_type copy;
    copy = instance;
    BOOST_CHECK(instance.raw() == copy.raw());
    BOOST_CHECK(instance == copy);
}
