#pragma once

#include <immintrin.h>
#include <concepts>
#include <stdexcept>

namespace arcana::tensor
{
    // ==========================================
    // CONCEPTS & TRAITS
    // ==========================================

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

    template <>
    struct has_simd<double> : std::true_type
    {
    };

    // ==========================================
    // BASE TENSOR OPERATIONS (Element-wise)
    // ==========================================

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
                throw std::runtime_error("Empty tensor");
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
            if constexpr (std::is_same_v<T, float>)
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
                    result_data[i] = static_cast<Derived *>(this)->apply_element(lhs_data[i], rhs_data[i]);
                }
            }
            else if constexpr (std::is_same_v<T, double>)
            {
                size_t i = 0;
                const size_t SIMD_WIDTH = 4;

                for (; i + SIMD_WIDTH <= total; i += SIMD_WIDTH)
                {
                    __m256d lhs_v = _mm256_loadu_pd(&lhs_data[i]);
                    __m256d rhs_v = _mm256_loadu_pd(&rhs_data[i]);
                    __m256d result_v = static_cast<Derived *>(this)->apply_simd(lhs_v, rhs_v);
                    _mm256_storeu_pd(&result_data[i], result_v);
                }
                for (; i < total; ++i)
                {
                    result_data[i] = static_cast<Derived *>(this)->apply_element(lhs_data[i], rhs_data[i]);
                }
            }
        }
    };

    // ==========================================
    // TENSOR OPS: ADDITION (+)
    // ==========================================

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

    template <>
    class AddOp<double> : public TensorOps<AddOp<double>, double>
    {
    public:
        using Base = TensorOps<AddOp<double>, double>;
        using Base::TensorOps;
        double apply_element(double a, double b) const { return a + b; }
        __m256d apply_simd(__m256d a, __m256d b) const { return _mm256_add_pd(a, b); }
    };

    // ==========================================
    // TENSOR OPS: SUBTRACTION (-)
    // ==========================================

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

    template <>
    class SubOp<double> : public TensorOps<SubOp<double>, double>
    {
    public:
        using Base = TensorOps<SubOp<double>, double>;
        using Base::TensorOps;
        double apply_element(double a, double b) const { return a - b; }
        __m256d apply_simd(__m256d a, __m256d b) const { return _mm256_sub_pd(a, b); }
    };

    // ==========================================
    // TENSOR OPS: MULTIPLICATION (*)
    // ==========================================

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

    template <>
    class MulOp<double> : public TensorOps<MulOp<double>, double>
    {
    public:
        using Base = TensorOps<MulOp<double>, double>;
        using Base::TensorOps;
        double apply_element(double a, double b) const { return a * b; }
        __m256d apply_simd(__m256d a, __m256d b) const { return _mm256_mul_pd(a, b); }
    };

    // ==========================================
    // TENSOR OPS: DIVISION (/)
    // ==========================================

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

    template <>
    class DivOp<double> : public TensorOps<DivOp<double>, double>
    {
    public:
        using Base = TensorOps<DivOp<double>, double>;
        using Base::TensorOps;
        double apply_element(double a, double b) const { return a / b; }
        __m256d apply_simd(__m256d a, __m256d b) const { return _mm256_div_pd(a, b); }
    };

    // ==========================================
    // BASE SCALAR OPERATIONS (Tensor + Scalar)
    // ==========================================

    template <typename Derived, typename T>
    class ScalarOps
    {
    public:
        T *data;
        T scalar;
        size_t size;

        ScalarOps(T *d, T s, size_t sz) : data(d), scalar(s), size(sz) {}

        void execute()
        {
            if constexpr (has_simd<T>::value)
            {
                execute_simd(size);
            }
            else
            {
                for (size_t i = 0; i < size; ++i)
                {
                    data[i] = static_cast<Derived *>(this)->apply_element(data[i], scalar);
                }
            }
        }

    private:
        void execute_simd(size_t total)
        {
            if constexpr (std::is_same_v<T, float>)
            {
                size_t i = 0;
                const size_t SIMD_WIDTH = 8;
                __m256 scalar_v = _mm256_set1_ps(scalar);

                for (; i + SIMD_WIDTH <= total; i += SIMD_WIDTH)
                {
                    __m256 data_v = _mm256_loadu_ps(&data[i]);
                    __m256 result_v = static_cast<Derived *>(this)->apply_simd(data_v, scalar_v);
                    _mm256_storeu_ps(&data[i], result_v);
                }
                for (; i < total; ++i)
                {
                    data[i] = static_cast<Derived *>(this)->apply_element(data[i], scalar);
                }
            }
            else if constexpr (std::is_same_v<T, double>)
            {
                size_t i = 0;
                const size_t SIMD_WIDTH = 4;
                __m256d scalar_v = _mm256_set1_pd(scalar);

                for (; i + SIMD_WIDTH <= total; i += SIMD_WIDTH)
                {
                    __m256d data_v = _mm256_loadu_pd(&data[i]);
                    __m256d result_v = static_cast<Derived *>(this)->apply_simd(data_v, scalar_v);
                    _mm256_storeu_pd(&data[i], result_v);
                }
                for (; i < total; ++i)
                {
                    data[i] = static_cast<Derived *>(this)->apply_element(data[i], scalar);
                }
            }
        }
    };

    // ==========================================
    // SCALAR OPS IMPLEMENTATIONS
    // ==========================================

    // --- AddScalarOp ---
    template <typename T>
    class AddScalarOp : public ScalarOps<AddScalarOp<T>, T>
    {
    public:
        using Base = ScalarOps<AddScalarOp<T>, T>;
        using Base::ScalarOps;
        T apply_element(T a, T b) const { return a + b; }
    };

    template <>
    class AddScalarOp<float> : public ScalarOps<AddScalarOp<float>, float>
    {
    public:
        using Base = ScalarOps<AddScalarOp<float>, float>;
        using Base::ScalarOps;
        float apply_element(float a, float b) const { return a + b; }
        __m256 apply_simd(__m256 a, __m256 b) const { return _mm256_add_ps(a, b); }
    };

    template <>
    class AddScalarOp<double> : public ScalarOps<AddScalarOp<double>, double>
    {
    public:
        using Base = ScalarOps<AddScalarOp<double>, double>;
        using Base::ScalarOps;
        double apply_element(double a, double b) const { return a + b; }
        __m256d apply_simd(__m256d a, __m256d b) const { return _mm256_add_pd(a, b); }
    };

    // --- SubScalarOp ---
    template <typename T>
    class SubScalarOp : public ScalarOps<SubScalarOp<T>, T>
    {
    public:
        using Base = ScalarOps<SubScalarOp<T>, T>;
        using Base::ScalarOps;
        T apply_element(T a, T b) const { return a - b; }
    };

    template <>
    class SubScalarOp<float> : public ScalarOps<SubScalarOp<float>, float>
    {
    public:
        using Base = ScalarOps<SubScalarOp<float>, float>;
        using Base::ScalarOps;
        float apply_element(float a, float b) const { return a - b; }
        __m256 apply_simd(__m256 a, __m256 b) const { return _mm256_sub_ps(a, b); }
    };

    template <>
    class SubScalarOp<double> : public ScalarOps<SubScalarOp<double>, double>
    {
    public:
        using Base = ScalarOps<SubScalarOp<double>, double>;
        using Base::ScalarOps;
        double apply_element(double a, double b) const { return a - b; }
        __m256d apply_simd(__m256d a, __m256d b) const { return _mm256_sub_pd(a, b); }
    };

    // --- MulScalarOp ---
    template <typename T>
    class MulScalarOp : public ScalarOps<MulScalarOp<T>, T>
    {
    public:
        using Base = ScalarOps<MulScalarOp<T>, T>;
        using Base::ScalarOps;
        T apply_element(T a, T b) const { return a * b; }
    };

    template <>
    class MulScalarOp<float> : public ScalarOps<MulScalarOp<float>, float>
    {
    public:
        using Base = ScalarOps<MulScalarOp<float>, float>;
        using Base::ScalarOps;
        float apply_element(float a, float b) const { return a * b; }
        __m256 apply_simd(__m256 a, __m256 b) const { return _mm256_mul_ps(a, b); }
    };

    template <>
    class MulScalarOp<double> : public ScalarOps<MulScalarOp<double>, double>
    {
    public:
        using Base = ScalarOps<MulScalarOp<double>, double>;
        using Base::ScalarOps;
        double apply_element(double a, double b) const { return a * b; }
        __m256d apply_simd(__m256d a, __m256d b) const { return _mm256_mul_pd(a, b); }
    };

    // --- DivScalarOp ---
    template <typename T>
    class DivScalarOp : public ScalarOps<DivScalarOp<T>, T>
    {
    public:
        using Base = ScalarOps<DivScalarOp<T>, T>;
        using Base::ScalarOps;
        T apply_element(T a, T b) const { return a / b; }
    };

    template <>
    class DivScalarOp<float> : public ScalarOps<DivScalarOp<float>, float>
    {
    public:
        using Base = ScalarOps<DivScalarOp<float>, float>;
        using Base::ScalarOps;
        float apply_element(float a, float b) const { return a / b; }
        __m256 apply_simd(__m256 a, __m256 b) const { return _mm256_div_ps(a, b); }
    };

    template <>
    class DivScalarOp<double> : public ScalarOps<DivScalarOp<double>, double>
    {
    public:
        using Base = ScalarOps<DivScalarOp<double>, double>;
        using Base::ScalarOps;
        double apply_element(double a, double b) const { return a / b; }
        __m256d apply_simd(__m256d a, __m256d b) const { return _mm256_div_pd(a, b); }
    };

} // namespace arcana::tensor