#pragma once

#include <immintrin.h>
#include <concepts>
#include <stdexcept>

namespace arcana::tensor
{
    template <typename T>
    concept SIMDCapable = std::is_same_v<T, float> || std::is_same_v<T, double>;

    template <typename T>
    struct has_simd : std::false_type
    {
    };

    template <>
    struct has_simd<float> : std::true_type
    {
    };

    template <typename Derived, typename T>
    class TensorOps
    {
    public:
        const T *lhs_data;
        const T *rhs_data;
        T *result_data;
        size_t total_size;

        TensorOps(const T *lhs, const T *rhs, T *result, size_t size)
            : lhs_data(lhs), rhs_data(rhs), result_data(result), total_size(size)
        {
            if (total_size == 0)
            {
                throw std::runtime_error("Empty tensor");
            }
        }

        void execute()
        {
            if constexpr (has_simd<T>::value)
            {
                execute_simd(total_size);
            }
            else
            {
                for (size_t i = 0; i < total_size; ++i)
                {
                    result_data[i] = static_cast<Derived *>(this)->apply_element(
                        lhs_data[i], rhs_data[i]);
                }
            }
        }

    private:
        void execute_simd(size_t total)
        {
            size_t i = 0;
            const size_t SIMD_WIDTH = 8;

            for (; i + SIMD_WIDTH <= total; i += SIMD_WIDTH)
            {
                __m256 lhs_v = _mm256_loadu_ps(&lhs_data[i]);
                __m256 rhs_v = _mm256_loadu_ps(&rhs_data[i]);
                __m256 result_v = static_cast<Derived *>(this)->apply_simd(lhs_v, rhs_v);
                _mm256_storeu_ps(&result_data[i], result_v);
            }

            for (; i < total; ++i)
            {
                result_data[i] = static_cast<Derived *>(this)->apply_element(
                    lhs_data[i], rhs_data[i]);
            }
        }
    };

    template <typename T>
    class AddOp : public TensorOps<AddOp<T>, T>
    {
    public:
        using Base = TensorOps<AddOp<T>, T>;
        using Base::TensorOps;

        T apply_element(T a, T b) const { return a + b; }
    };

    template <>
    class AddOp<float> : public TensorOps<AddOp<float>, float>
    {
    public:
        using Base = TensorOps<AddOp<float>, float>;
        using Base::TensorOps;

        float apply_element(float a, float b) const { return a + b; }

        __m256 apply_simd(__m256 a, __m256 b) const { return _mm256_add_ps(a, b); }
    };

    template <typename T>
    class SubOp : public TensorOps<SubOp<T>, T>
    {
    public:
        using Base = TensorOps<SubOp<T>, T>;
        using Base::TensorOps;

        T apply_element(T a, T b) const { return a - b; }
    };

    template <>
    class SubOp<float> : public TensorOps<SubOp<float>, float>
    {
    public:
        using Base = TensorOps<SubOp<float>, float>;
        using Base::TensorOps;

        float apply_element(float a, float b) const { return a - b; }

        __m256 apply_simd(__m256 a, __m256 b) const { return _mm256_sub_ps(a, b); }
    };

    template <typename T>
    class MulOp : public TensorOps<MulOp<T>, T>
    {
    public:
        using Base = TensorOps<MulOp<T>, T>;
        using Base::TensorOps;

        T apply_element(T a, T b) const { return a * b; }
    };

    template <>
    class MulOp<float> : public TensorOps<MulOp<float>, float>
    {
    public:
        using Base = TensorOps<MulOp<float>, float>;
        using Base::TensorOps;

        float apply_element(float a, float b) const { return a * b; }

        __m256 apply_simd(__m256 a, __m256 b) const { return _mm256_mul_ps(a, b); }
    };

    template <typename T>
    class DivOp : public TensorOps<DivOp<T>, T>
    {
    public:
        using Base = TensorOps<DivOp<T>, T>;
        using Base::TensorOps;

        T apply_element(T a, T b) const { return a / b; }
    };

    template <>
    class DivOp<float> : public TensorOps<DivOp<float>, float>
    {
    public:
        using Base = TensorOps<DivOp<float>, float>;
        using Base::TensorOps;

        float apply_element(float a, float b) const { return a / b; }

        __m256 apply_simd(__m256 a, __m256 b) const { return _mm256_div_ps(a, b); }
    };

} // namespace arcana::tensor
