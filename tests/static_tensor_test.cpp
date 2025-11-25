#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "arcana/tensor/static_tensor.hpp"

using namespace arcana::tensor;

TEST_CASE("StaticTensor creation", "[static_tensor]")
{
    SECTION("Zeros")
    {
        auto t = StaticTensor<float, 3, 3>::zeros();
        REQUIRE(t.size() == 9);
        REQUIRE(t(0, 0) == 0.0f);
    }

    SECTION("Ones")
    {
        auto t = StaticTensor<float, 2, 2>::ones();
        REQUIRE(t(0, 0) == 1.0f);
        REQUIRE(t(1, 1) == 1.0f);
    }

    SECTION("Access")
    {
        StaticTensor<float, 2> t;
        t(0) = 10.0f;
        t(1) = 20.0f;
        REQUIRE(t(0) == 10.0f);
        REQUIRE(t(1) == 20.0f);
    }
}

TEST_CASE("StaticTensor Arithmetic", "[static_tensor][ops]")
{
    auto a = StaticTensor<float, 2, 2>::ones(); // [[1, 1], [1, 1]]
    auto b = StaticTensor<float, 2, 2>::ones(); // [[1, 1], [1, 1]]

    SECTION("Addition")
    {
        auto c = a + b;
        REQUIRE(c(0, 0) == 2.0f);
    }

    SECTION("Scalar multiplication")
    {
        auto c = a * 5.0f;
        REQUIRE(c(0, 0) == 5.0f);
    }
}

TEST_CASE("StaticTensor Matmul", "[static_tensor][matmul]")
{
    SECTION("Matrix x Matrix (2D)")
    {
        // A [2x3], B [3x2]
        auto A = StaticTensor<float, 2, 3>::ones();
        auto B = StaticTensor<float, 3, 2>::ones();

        auto C = A.matmul(B); // Should be [2x2]

        // 1*1 + 1*1 + 1*1 = 3
        REQUIRE(C.rows() == 2);
        REQUIRE(C.cols() == 2);
        REQUIRE(C(0, 0) == 3.0f);
    }

    SECTION("Matrix x Vector (2D x 1D)")
    {
        auto A = StaticTensor<float, 2, 2>::ones();
        auto v = StaticTensor<float, 2>::ones();

        auto res = A.matmul(v);

        REQUIRE(res.size() == 2);
        REQUIRE(res(0) == 2.0f);
    }

    SECTION("Dot Product (1D x 1D)")
    {
        auto v1 = StaticTensor<float, 3>::ones();
        auto v2 = StaticTensor<float, 3>::ones();

        auto res = v1.matmul(v2);

        REQUIRE(res(0) == 3.0f);
    }
}

TEST_CASE("StaticTensor Reshape & Transpose", "[static_tensor][shape]")
{
    SECTION("Reshape")
    {
        auto t = StaticTensor<float, 4>::ones();
        auto m = t.reshape<2, 2>();

        REQUIRE(m.rows() == 2);
        REQUIRE(m.cols() == 2);
        REQUIRE(m(0, 1) == 1.0f);
    }

    SECTION("Transpose")
    {
        StaticTensor<float, 2, 3> t;
        t(0, 0) = 1;
        t(0, 1) = 2;
        t(0, 2) = 3;
        t(1, 0) = 4;
        t(1, 1) = 5;
        t(1, 2) = 6;

        auto t_T = t.t(); // [3x2]

        REQUIRE(t_T.rows() == 3);
        REQUIRE(t_T.cols() == 2);
        REQUIRE(t_T(0, 1) == 4.0f);
        REQUIRE(t_T(2, 0) == 3.0f);
    }
}