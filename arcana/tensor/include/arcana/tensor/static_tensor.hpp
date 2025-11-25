#pragma once

#include "arcana/tensor/tensor_base.hpp"
#include "arcana/tensor/tensor_methods.hpp"
#include "arcana/tensor/tensor_generators.hpp"
#include "arcana/tensor/tensor_arithmetic.hpp"

#include <array>
#include <vector>

namespace arcana::tensor
{
    template <typename T, size_t... Dims>
    class StaticTensor : public TensorBase<StaticTensor<T, Dims...>>,
                         public ElementWiseMethods<StaticTensor<T, Dims...>>,
                         public GlobalReductionMethods<StaticTensor<T, Dims...>>,
                         public ArithmeticMixin<StaticTensor<T, Dims...>, T>
    {
    public:
        static constexpr size_t TotalSize = (Dims * ...);
        static constexpr size_t Rank = sizeof...(Dims);
        static constexpr std::array<size_t, Rank> Shape = {Dims...};
        using scalar_type = T;

        StaticTensor() = default;
        ~StaticTensor() = default;

        StaticTensor(std::initializer_list<T> list)
        {
            if (list.size() != TotalSize)
            {
                throw std::runtime_error("Initializer list size mismatch");
            }
            std::copy(list.begin(), list.end(), data_.begin());
        }

        static StaticTensor zeros()
        {
            StaticTensor t;
            gen::fill_(t, T(0));
            return t;
        }

        static StaticTensor ones()
        {
            StaticTensor t;
            gen::fill_(t, T(1));
            return t;
        }

        static StaticTensor randn()
        {
            StaticTensor t;
            gen::fill_randn(t);
            return t;
        }

        static StaticTensor uniform()
        {
            StaticTensor t;
            gen::fill_uniform(t);
            return t;
        }

        template <size_t... OtherDims>
        auto matmul(const StaticTensor<T, OtherDims...> &other) const
        {
            constexpr size_t OtherRank = sizeof...(OtherDims);
            constexpr std::array<size_t, OtherRank> OtherShape = {OtherDims...};

            // ============ CASE 1: Dot Product (1D * 1D) ============
            if constexpr (Rank == 1 && OtherRank == 1)
            {
                static_assert(Shape[0] == OtherShape[0], "Dot product dimension mismatch");

                StaticTensor<T, 1> result;
                T sum = 0;
                for (size_t i = 0; i < Shape[0]; ++i)
                    sum += data_[i] * other.data()[i];
                result(0) = sum;
                return result;
            }

            // ============ CASE 2: Matrix Mul (2D * 2D) ============
            else if constexpr (Rank == 2 && OtherRank == 2)
            {
                static_assert(Shape[1] == OtherShape[0], "Matmul inner dimension mismatch");

                constexpr size_t M = Shape[0];
                constexpr size_t K = Shape[1];
                constexpr size_t N = OtherShape[1];

                StaticTensor<T, M, N> result;
                for (size_t m = 0; m < M; ++m)
                {
                    for (size_t n = 0; n < N; ++n)
                    {
                        T sum = 0;
                        for (size_t k = 0; k < K; ++k)
                        {
                            sum += (*this)(m, k) * other(k, n);
                        }
                        result(m, n) = sum;
                    }
                }
                return result;
            }

            // ============ CASE 3: Matrix-Vector (2D * 1D) ============
            else if constexpr (Rank == 2 && OtherRank == 1)
            {
                static_assert(Shape[1] == OtherShape[0], "Matmul dimension mismatch");

                constexpr size_t M = Shape[0];
                constexpr size_t K = Shape[1];

                StaticTensor<T, M> result;
                for (size_t m = 0; m < M; ++m)
                {
                    T sum = 0;
                    for (size_t k = 0; k < K; ++k)
                    {
                        sum += (*this)(m, k) * other(k);
                    }
                    result(m) = sum;
                }
                return result;
            }
            else
            {
                return StaticTensor<T>();
            }
        }

        // ============ TRANSPOSE ============
        auto transpose() const
            requires(Rank == 2)
        {
            StaticTensor<T, Shape[1], Shape[0]> result;
            for (size_t i = 0; i < rows(); ++i)
            {
                for (size_t j = 0; j < cols(); ++j)
                {
                    result(j, i) = (*this)(i, j);
                }
            }
            return result;
        }

        auto t() const
        {
            return transpose();
        }

        template <size_t... NewDims>
        auto reshape() const
        {
            static_assert((NewDims * ...) == TotalSize, "Reshape total size mismatch");
            StaticTensor<T, NewDims...> result;
            std::copy(this->begin(), this->end(), result.begin());
            return result;
        }

        template <typename Func>
        StaticTensor map(Func f) const
        {
            StaticTensor result;
            for (size_t i = 0; i < TotalSize; ++i)
                result.data_[i] = f(data_[i]);
            return result;
        }

        T &operator()(size_t i) { return data_[i]; }
        const T &operator()(size_t i) const { return data_[i]; }

        T *data() { return data_.data(); }
        const T *data() const { return data_.data(); }

        auto begin() { return data_.begin(); }
        auto end() { return data_.end(); }
        auto begin() const { return data_.begin(); }
        auto end() const { return data_.end(); }

        static constexpr size_t size() { return TotalSize; }
        static constexpr size_t rows()
            requires(Rank >= 1)
        {
            return Shape[0];
        }
        static constexpr size_t cols()
            requires(Rank >= 2)
        {
            return Shape[1];
        }

        template <typename... Indices>
            requires(sizeof...(Indices) == Rank)
        T &operator()(Indices... indices)
        {
            return data_[compute_offset(indices...)];
        }

        template <typename... Indices>
            requires(sizeof...(Indices) == Rank)
        const T &operator()(Indices... indices) const
        {
            return data_[compute_offset(indices...)];
        }

    private:
        std::array<T, TotalSize> data_;

        template <typename... Indices>
        static constexpr size_t compute_offset(Indices... indices)
        {
            std::array<size_t, Rank> idxs = {static_cast<size_t>(indices)...};
            size_t offset = 0;
            size_t stride = 1;

            for (int i = Rank - 1; i >= 0; --i)
            {
                offset += idxs[i] * stride;
                stride *= Shape[i];
            }
            return offset;
        }
    };
}