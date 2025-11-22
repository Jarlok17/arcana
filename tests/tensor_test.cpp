#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "arcana/tensor/tensor.hpp"
#include "arcana/tensor/tensor_io.hpp"

using namespace arcana::tensor;

TEST_CASE("Tensor creation", "[tensor]")
{
    SECTION("Empty tensor")
    {
        auto t = Tensor<float>::empty(3, 4);
        REQUIRE(t.shape().size() == 2);
        REQUIRE(t.shape()[0] == 3);
        REQUIRE(t.shape()[1] == 4);
        REQUIRE(t.size() == 12);
    }

    SECTION("Zeros tensor")
    {
        auto t = Tensor<float>::zeros(2, 3);
        for (size_t i = 0; i < 2; ++i)
            for (size_t j = 0; j < 3; ++j)
                REQUIRE(t(i, j) == 0.0f);
    }

    SECTION("Ones tensor")
    {
        auto t = Tensor<float>::ones(5);
        for (size_t i = 0; i < t.size(); ++i)
            REQUIRE(t(i) == 1.0f);
    }

    SECTION("Random tensor")
    {
        auto t = Tensor<float>::randn(10, 10);
        REQUIRE(t.size() == 100);

        bool has_different = false;
        float first = t(0, 0);
        for (size_t i = 0; i < 10; ++i)
            for (size_t j = 0; j < 10; ++j)
                if (t(i, j) != first)
                    has_different = true;
        REQUIRE(has_different);
    }

    SECTION("Moderate size (1000x1000)")
    {
        // 1 million floats ≈ 4 MB
        REQUIRE_NOTHROW([&]
                        {
            auto t = Tensor<float>::zeros(1000, 1000);
            REQUIRE(t.size() == 1'000'000);
            REQUIRE(t(0, 0) == 0.0f);
            REQUIRE(t(999, 999) == 0.0f); }());
    }

    SECTION("Large size (5000x5000)")
    {
        // 25 million floats ≈ 100 MB
        REQUIRE_NOTHROW([&]
                        {
            auto t = Tensor<float>::zeros(5000, 5000);
            REQUIRE(t.size() == 25'000'000);
            REQUIRE(t(0, 0) == 0.0f);
            REQUIRE(t(4999, 4999) == 0.0f); }());
    }

    /*SECTION("Very large size (10000x10000)")
    {
        // 100 million floats ≈ 400 MB
        // Its may fall on some operation systems due to memory limits
        REQUIRE_NOTHROW([&]
                        {
            auto t = Tensor<float>::empty(10000, 10000);
            REQUIRE(t.size() == 100'000'000); }());
    }*/
}

TEST_CASE("Tensor arithmetic", "[tensor][operations]")
{
    auto a = Tensor<float>::ones(3, 3);
    auto b = Tensor<float>::ones(3, 3) * 2.0f;

    SECTION("Addition")
    {
        auto c = a + b;
        REQUIRE(c(0, 0) == 3.0f);
        REQUIRE(c(1, 1) == 3.0f);
    }

    SECTION("Subtraction")
    {
        auto c = b - a;
        REQUIRE(c(0, 0) == 1.0f);
    }

    SECTION("Multiplication")
    {
        auto c = a * b;
        REQUIRE(c(0, 0) == 2.0f);
    }

    SECTION("Division")
    {
        auto c = b / a;
        REQUIRE(c(0, 0) == 2.0f);
    }
}

TEST_CASE("Tensor scalar operations", "[tensor][scalar]")
{
    auto t = Tensor<float>::ones(2, 2);

    SECTION("Add scalar")
    {
        auto result = t + 5.0f;
        REQUIRE(result(0, 0) == 6.0f);
    }

    SECTION("Multiply scalar")
    {
        auto result = t * 3.0f;
        REQUIRE(result(0, 0) == 3.0f);
    }
}

TEST_CASE("Matmul 2D", "[tensor][matmul]")
{
    SECTION("Basic matmul")
    {
        auto A = Tensor<float>::ones(2, 3);
        auto B = Tensor<float>::ones(3, 2);

        auto C = A.matmul(B);

        REQUIRE(C.shape()[0] == 2);
        REQUIRE(C.shape()[1] == 2);
        REQUIRE(C(0, 0) == 3.0f); // sum of ones
    }

    SECTION("Identity matmul")
    {
        auto I = Tensor<float>::zeros(3, 3);
        for (size_t i = 0; i < 3; ++i)
            I(i, i) = 1.0f;

        auto A = Tensor<float>::randn(3, 3);
        auto result = A.matmul(I);

        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 3; ++j)
                REQUIRE_THAT(result(i, j),
                             Catch::Matchers::WithinRel(A(i, j), 0.0001f));
    }
}

TEST_CASE("Transpose", "[tensor][transpose]")
{
    SECTION("2D transpose")
    {
        auto t = Tensor<float>::randn(3, 4);
        auto t_T = t.t();

        REQUIRE(t_T.shape()[0] == 4);
        REQUIRE(t_T.shape()[1] == 3);

        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 4; ++j)
                REQUIRE(t(i, j) == t_T(j, i));
    }

    SECTION("3D transpose")
    {
        auto t = Tensor<float>::ones(2, 3, 4);
        auto t_01 = t.transpose(0, 1);

        REQUIRE(t_01.shape()[0] == 3);
        REQUIRE(t_01.shape()[1] == 2);
        REQUIRE(t_01.shape()[2] == 4);
    }
}

TEST_CASE("Reshape operations", "[tensor][reshape]")
{
    SECTION("Reshape")
    {
        auto t = Tensor<float>::randn(2, 3, 4);
        auto r = t.reshape({6, 4});

        REQUIRE(r.shape()[0] == 6);
        REQUIRE(r.shape()[1] == 4);
        REQUIRE(r.size() == 24);
    }

    SECTION("Flatten")
    {
        auto t = Tensor<float>::randn(2, 3, 4);
        auto f = t.flatten();

        REQUIRE(f.shape().size() == 1);
        REQUIRE(f.shape()[0] == 24);
    }
}

TEST_CASE("View operations", "[tensor][view]")
{
    auto t = Tensor<float>::randn(12);
    auto v = t.view(3, 4);

    REQUIRE(v.shape()[0] == 3);
    REQUIRE(v.shape()[1] == 4);

    t(0) = 99.0f;
    REQUIRE(v(0, 0) == 99.0f);
}

TEST_CASE("Squeeze and unsqueeze", "[tensor][squeeze]")
{
    SECTION("Squeeze all")
    {
        auto t = Tensor<float>::ones(1, 3, 1, 5);
        auto s = t.squeeze();

        REQUIRE(s.shape().size() == 2);
        REQUIRE(s.shape()[0] == 3);
        REQUIRE(s.shape()[1] == 5);
    }

    SECTION("Squeeze specific dim")
    {
        auto t = Tensor<float>::ones(1, 3, 1, 5);
        auto s = t.squeeze(0);

        REQUIRE(s.shape()[0] == 3);
        REQUIRE(s.shape()[1] == 1);
        REQUIRE(s.shape()[2] == 5);
    }

    SECTION("Unsqueeze")
    {
        auto t = Tensor<float>::ones(3, 5);
        auto u = t.unsqueeze(1);

        REQUIRE(u.shape()[0] == 3);
        REQUIRE(u.shape()[1] == 1);
        REQUIRE(u.shape()[2] == 5);
    }
}

TEST_CASE("Tensor methods", "[tensor][methods]")
{
    SECTION("Mean")
    {
        auto t = Tensor<float>::ones(10, 10);
        REQUIRE(t.mean() == 1.0f);
    }

    SECTION("Sum")
    {
        auto t = Tensor<float>::ones(5, 5);
        REQUIRE(t.sum() == 25.0f);
    }

    SECTION("Max and Min")
    {
        auto t = Tensor<float>::randn(100);
        float max_val = t.max();
        float min_val = t.min();
        REQUIRE(max_val >= min_val);
    }

    SECTION("Argmax and Argmin for 1D Tensor")
    {
        auto t = Tensor<float>::randn(100);

        auto max_idx = t.argmax();
        auto min_idx = t.argmin();

        REQUIRE(max_idx < t.size());
        REQUIRE(min_idx < t.size());

        float max_v = t.data()[max_idx];
        float min_v = t.data()[min_idx];

        REQUIRE(max_v >= min_v);
    }

    SECTION("Argmax and Argmin for 2D Tensor")
    {
        auto t = Tensor<float>::randn(10, 10);

        auto max_idx = t.argmax(0); // shape: [10]
        auto min_idx = t.argmin(0); // shape: [10]

        REQUIRE(max_idx.shape().size() == 1);
        REQUIRE(max_idx.shape()[0] == 10);

        REQUIRE(min_idx.shape().size() == 1);
        REQUIRE(min_idx.shape()[0] == 10);

        for (size_t j = 0; j < 10; ++j)
        {
            auto max_i = static_cast<size_t>(max_idx(j));
            auto min_i = static_cast<size_t>(min_idx(j));

            REQUIRE(max_i < 10);
            REQUIRE(min_i < 10);

            float max_val = t(max_i, j);
            float min_val = t(min_i, j);

            REQUIRE(max_val >= min_val);
        }
    }

    SECTION("Argmax and Argmin for 3D Tensor")
    {
        auto t = Tensor<float>::randn(5, 10, 20);

        auto max_idx = t.argmax(1); // shape: [5, 20]
        auto min_idx = t.argmin(1); // shape: [5, 20]

        REQUIRE(max_idx.shape().size() == 2);
        REQUIRE(max_idx.shape()[0] == 5);
        REQUIRE(max_idx.shape()[1] == 20);

        REQUIRE(min_idx.shape().size() == 2);
        REQUIRE(min_idx.shape()[0] == 5);
        REQUIRE(min_idx.shape()[1] == 20);

        for (size_t b = 0; b < 5; ++b)
        {
            for (size_t k = 0; k < 20; ++k)
            {
                auto max_i = static_cast<size_t>(max_idx(b, k));
                auto min_i = static_cast<size_t>(min_idx(b, k));

                REQUIRE(max_i < 10);
                REQUIRE(min_i < 10);

                float max_val = t(b, max_i, k);
                float min_val = t(b, min_i, k);

                REQUIRE(max_val >= min_val);
            }
        }
    }

    SECTION("Exp")
    {
        auto t = Tensor<float>::ones(5, 5);
        auto t_exp = t.exp();
        REQUIRE(t_exp(0, 0) == 2.718281746f);
    }

    SECTION("Exp")
    {
        auto t = Tensor<float>::ones(5, 5);
        auto t_exp = t.exp();
        REQUIRE(t_exp(0, 0) == 2.718281746f);
    }

    SECTION("Log")
    {
        auto t = Tensor<float>::ones(5, 5);
        auto t_exp = t.log();
        REQUIRE(t_exp(0, 0) == 0.0f);
    }

    SECTION("Pow")
    {
        auto t = Tensor<float>::ones(5, 5);
        auto t_exp = t.pow(2);
        REQUIRE(t_exp(0, 0) == 1.0f);
    }

    SECTION("Sqrt")
    {
        auto t = Tensor<float>::ones(5, 5);
        auto t_exp = t.sqrt();
        REQUIRE(t_exp(0, 0) == 1.0f);
    }

    SECTION("ReLU")
    {
        auto t = Tensor<float>::ones(5, 5);
        auto t_exp = t.relu();
        REQUIRE(t_exp(0, 0) == 1.0f);
    }

    SECTION("Sigmoid")
    {
        auto t = Tensor<float>::ones(5, 5);
        auto t_exp = t.sigmoid();
        REQUIRE(t_exp(0, 0) == 0.731058578f);
    }

    SECTION("Tanh")
    {
        auto t = Tensor<float>::ones(5, 5);
        auto t_exp = t.tanh();
        REQUIRE(t_exp(0, 0) == 0.761594176f);
    }
}