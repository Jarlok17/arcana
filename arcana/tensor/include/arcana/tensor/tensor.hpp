#pragma once

#include <algorithm>
#include <memory>
#include <random>
#include <stdexcept>
#include <concepts>

#include "arcana/tensor/tensor_base.hpp"
#include "arcana/tensor/tensor_methods.hpp"
#include "arcana/tensor/tensor_operations.hpp"
#include "arcana/pcg_random.hpp"

namespace arcana::tensor
{
    template <Numeric T>
    class Tensor : public TensorBase<Tensor<T>>, public TensorMethods<Tensor<T>>
    {
    public:
        using scalar_type = T;

        // ============ CONSTRUCTORS ============

        Tensor() : m_data(nullptr), m_size(0) {}

        template <typename... Dims>
        Tensor(Dims... dims) : m_shape{static_cast<size_t>(dims)...}, m_offset(0)
        {
            compute_size_and_strides();
            m_storage = std::shared_ptr<T[]>(new T[m_size]());
            m_data = m_storage.get();
        }

        Tensor(std::initializer_list<size_t> shape) : m_shape(shape), m_offset(0)
        {
            compute_size_and_strides();
            m_storage = std::shared_ptr<T[]>(new T[m_size]());
            m_data = m_storage.get();
        }

        explicit Tensor(const std::vector<size_t> &shape) : m_shape(shape), m_offset(0)
        {
            compute_size_and_strides();
            m_storage = std::shared_ptr<T[]>(new T[m_size]());
            m_data = m_storage.get();
        }

        ~Tensor() = default;

        // Copy & Move
        Tensor(const Tensor &other) : m_shape(other.m_shape), m_strides(other.m_strides), m_size(other.m_size)
        {
            m_storage = std::shared_ptr<T[]>(new T[m_size]);
            m_data = m_storage.get();
            std::copy_n(other.m_data, m_size, m_data);
        }

        Tensor(Tensor &&other) noexcept
            : m_storage(std::move(other.m_storage)), m_data(other.m_data), m_shape(std::move(other.m_shape)), m_strides(std::move(other.m_strides)), m_size(other.m_size)
        {
            other.m_data = nullptr;
            other.m_size = 0;
        }

        Tensor &operator=(const Tensor &other)
        {
            if (this != &other)
            {
                m_storage = std::shared_ptr<T[]>(new T[other.m_size]);
                m_data = m_storage.get();
                m_shape = other.m_shape;
                m_strides = other.m_strides;
                m_size = other.m_size;
                m_offset = 0;
                std::copy_n(other.m_data, m_size, m_data);
            }
            return *this;
        }

        Tensor &operator=(Tensor &&other) noexcept
        {
            if (this != &other)
            {
                m_storage = std::move(other.m_storage);
                m_data = other.m_data;
                m_shape = std::move(other.m_shape);
                m_strides = std::move(other.m_strides);
                m_size = other.m_size;
                other.m_data = nullptr;
                other.m_size = 0;
            }
            return *this;
        }

        T &operator()(const std::vector<size_t> &indices) { return m_data[compute_offset(indices)]; }

        const T &operator()(const std::vector<size_t> &indices) const { return m_data[compute_offset(indices)]; }

        // ============ FACTORY METHODS ============

        static Tensor empty(const std::vector<size_t> &shape) { return Tensor(shape); }
        static Tensor empty(std::initializer_list<size_t> shape) { return Tensor(shape); }

        template <typename... Dims>
            requires(std::is_integral_v<Dims> && ...)
        static Tensor empty(Dims... dims)
        {
            return Tensor(dims...);
        }

        // --- ZEROS ---
        static Tensor zeros(const std::vector<size_t> &shape)
        {
            Tensor result(shape);
            std::fill_n(result.m_data, result.m_size, T(0));
            return result;
        }

        static Tensor zeros(std::initializer_list<size_t> shape)
        {
            return zeros(std::vector<size_t>(shape));
        }

        template <typename... Dims>
            requires(std::is_integral_v<Dims> && ...)
        static Tensor zeros(Dims... dims)
        {
            return zeros({static_cast<size_t>(dims)...});
        }

        // --- ONES ---
        static Tensor ones(const std::vector<size_t> &shape)
        {
            Tensor result(shape);
            std::fill_n(result.m_data, result.m_size, T(1));
            return result;
        }

        static Tensor ones(std::initializer_list<size_t> shape)
        {
            return ones(std::vector<size_t>(shape));
        }

        template <typename... Dims>
            requires(std::is_integral_v<Dims> && ...)
        static Tensor ones(Dims... dims)
        {
            return ones({static_cast<size_t>(dims)...});
        }

        // --- RANDN ---
        static Tensor randn(const std::vector<size_t> &shape, T mean = 0, T stddev = 1, uint64_t seed = 0)
        {
            Tensor result(shape);
            result.randn_(mean, stddev, seed);
            return result;
        }

        static Tensor randn(std::initializer_list<size_t> shape, T mean = 0, T stddev = 1, uint64_t seed = 0)
        {
            return randn(std::vector<size_t>(shape), mean, stddev, seed);
        }

        template <typename... Dims>
            requires(std::is_integral_v<Dims> && ...)
        static Tensor randn(Dims... dims)
        {
            return randn({static_cast<size_t>(dims)...}, T(0), T(1), 0);
        }

        // --- UNIFORM ---
        static Tensor uniform(const std::vector<size_t> &shape, T min = 0, T max = 1, uint64_t seed = 0)
        {
            Tensor result(shape);
            result.uniform_(min, max, seed);
            return result;
        }

        static Tensor uniform(std::initializer_list<size_t> shape, T min = 0, T max = 1, uint64_t seed = 0)
        {
            return uniform(std::vector<size_t>(shape), min, max, seed);
        }

        template <typename... Dims>
            requires(std::is_integral_v<Dims> && ...)
        static Tensor uniform(Dims... dims)
        {
            return uniform({static_cast<size_t>(dims)...}, T(0), T(1), 0);
        }

        // ============ FACTORY _LIKE METHODS ============

        static Tensor empty_like(const Tensor &other) { return Tensor(other.m_shape); }

        static Tensor zeros_like(const Tensor &other) { return zeros(other.m_shape); }

        static Tensor ones_like(const Tensor &other) { return ones(other.m_shape); }

        static Tensor randn_like(const Tensor &other, T mean = 0, T stddev = 1, uint64_t seed = 0)
        {
            return randn(other.m_shape, mean, stddev, seed);
        }

        static Tensor uniform_like(const Tensor &other, T min = 0, T max = 1, uint64_t seed = 0)
        {
            return uniform(other.m_shape, min, max, seed);
        }

        // ============ INDEXING ============

        template <typename... Indices>
        T &operator()(Indices... indices)
        {
            return m_data[compute_offset({static_cast<size_t>(indices)...})];
        }

        template <typename... Indices>
        const T &operator()(Indices... indices) const
        {
            return m_data[compute_offset({static_cast<size_t>(indices)...})];
        }

        // ============ TENSOR ARITHMETIC ============

        Tensor operator+(const Tensor &other) const
        {
            if (m_shape != other.m_shape)
            {
                throw std::runtime_error("Shape mismatch");
            }

            Tensor result(m_shape);
            arcana::tensor::AddOp<T> op(m_data, other.m_data, result.m_data, m_size);
            op.execute();
            return result;
        }

        Tensor operator-(const Tensor &other) const
        {
            if (m_shape != other.m_shape)
            {
                throw std::runtime_error("Shape mismatch");
            }

            Tensor result(m_shape);
            arcana::tensor::SubOp<T> op(m_data, other.m_data, result.m_data, m_size);
            op.execute();
            return result;
        }

        Tensor operator*(const Tensor &other) const
        {
            if (m_shape != other.m_shape)
            {
                throw std::runtime_error("Shape mismatch");
            }

            Tensor result(m_shape);
            arcana::tensor::MulOp<T> op(m_data, other.m_data, result.m_data, m_size);
            op.execute();
            return result;
        }

        Tensor operator/(const Tensor &other) const
        {
            if (m_shape != other.m_shape)
            {
                throw std::runtime_error("Shape mismatch");
            }

            Tensor result(m_shape);
            arcana::tensor::DivOp<T> op(m_data, other.m_data, result.m_data, m_size);
            op.execute();
            return result;
        }

        Tensor matmul(const Tensor &other) const
        {
            size_t ndim_a = m_shape.size();
            size_t ndim_b = other.m_shape.size();

            // ============ CASE 1: 1D × 1D (dot product) ============
            if (ndim_a == 1 && ndim_b == 1)
            {
                if (m_shape[0] != other.m_shape[0])
                    throw std::runtime_error("Dimension mismatch for dot product");

                T sum = 0;
                for (size_t i = 0; i < m_shape[0]; ++i)
                    sum += m_data[i] * other.m_data[i];

                Tensor result({1});
                result.m_data[0] = sum;
                return result;
            }

            // ============ CASE 2: 2D × 2D (matrix multiplication) ============
            if (ndim_a == 2 && ndim_b == 2)
            {
                return matmul_2d(other);
            }

            // ============ CASE 3: 2D × 1D ============
            if (ndim_a == 2 && ndim_b == 1)
            {
                // (m, k) @ (k,) → (m,)
                if (m_shape[1] != other.m_shape[0])
                {
                    throw std::runtime_error("Dimension mismatch");
                }

                size_t m = m_shape[0];
                size_t k = m_shape[1];

                Tensor result({m});
                for (size_t i = 0; i < m; ++i)
                {
                    T sum = 0;
                    for (size_t j = 0; j < k; ++j)
                    {
                        sum += (*this)(i, j) * other(j);
                    }
                    result(i) = sum;
                }
                return result;
            }

            // ============ CASE 4: Batch matmul (N-D × N-D) ============
            if (ndim_a >= 3 && ndim_b >= 3)
            {
                return matmul_batched(other);
            }

            throw std::runtime_error("Unsupported matmul dimensions");
        }

        // Alias
        Tensor mm(const Tensor &other) const { return matmul(other); }

        // ============ IN-PLACE TENSOR ARITHMETIC ============

        Tensor &operator+=(const Tensor &other)
        {
            if (m_shape != other.m_shape)
            {
                throw std::runtime_error("Shape mismatch");
            }

            for (size_t i = 0; i < m_size; ++i)
                m_data[i] += other.m_data[i];

            return *this;
        }

        Tensor &operator-=(const Tensor &other)
        {
            if (m_shape != other.m_shape)
            {
                throw std::runtime_error("Shape mismatch");
            }

            for (size_t i = 0; i < m_size; ++i)
                m_data[i] -= other.m_data[i];

            return *this;
        }

        Tensor &operator*=(const Tensor &other)
        {
            if (m_shape != other.m_shape)
            {
                throw std::runtime_error("Shape mismatch");
            }

            for (size_t i = 0; i < m_size; ++i)
                m_data[i] *= other.m_data[i];

            return *this;
        }

        Tensor &operator/=(const Tensor &other)
        {
            if (m_shape != other.m_shape)
            {
                throw std::runtime_error("Shape mismatch");
            }

            for (size_t i = 0; i < m_size; ++i)
                m_data[i] /= other.m_data[i];

            return *this;
        }

        // ============ SCALAR ARITHMETIC ============

        Tensor operator+(T scalar) const
        {
            Tensor result(m_shape);
            for (size_t i = 0; i < m_size; ++i)
                result.m_data[i] = m_data[i] + scalar;
            return result;
        }

        Tensor operator-(T scalar) const
        {
            Tensor result(m_shape);
            for (size_t i = 0; i < m_size; ++i)
                result.m_data[i] = m_data[i] - scalar;
            return result;
        }

        Tensor operator*(T scalar) const
        {
            Tensor result(m_shape);
            for (size_t i = 0; i < m_size; ++i)
                result.m_data[i] = m_data[i] * scalar;
            return result;
        }

        Tensor operator/(T scalar) const
        {
            Tensor result(m_shape);
            for (size_t i = 0; i < m_size; ++i)
                result.m_data[i] = m_data[i] / scalar;
            return result;
        }

        // ============ IN-PLACE SCALAR ARITHMETIC ============

        Tensor &operator+=(T scalar)
        {
            for (size_t i = 0; i < m_size; ++i)
                m_data[i] += scalar;
            return *this;
        }

        Tensor &operator-=(T scalar)
        {
            for (size_t i = 0; i < m_size; ++i)
                m_data[i] -= scalar;
            return *this;
        }

        Tensor &operator*=(T scalar)
        {
            for (size_t i = 0; i < m_size; ++i)
                m_data[i] *= scalar;
            return *this;
        }

        Tensor &operator/=(T scalar)
        {
            for (size_t i = 0; i < m_size; ++i)
                m_data[i] /= scalar;
            return *this;
        }

        // ============ SHAPE OPERATIONS ============

        [[nodiscard]] Tensor reshape(std::initializer_list<size_t> new_shape) const
        {
            size_t new_size = 1;
            for (auto dim : new_shape)
                new_size *= dim;

            if (new_size != m_size)
                throw std::runtime_error("Total size must remain the same");

            Tensor result(new_shape);
            std::copy_n(m_data, m_size, result.m_data);
            return result;
        }

        [[nodiscard]] Tensor flatten() const { return reshape({m_size}); }

        [[nodiscard]] Tensor view(std::initializer_list<size_t> new_shape) const
        {
            size_t new_size = 1;
            for (auto dim : new_shape)
                new_size *= dim;

            if (new_size != m_size)
                throw std::runtime_error("Total size must remain the same for view");

            Tensor result;
            result.m_storage = m_storage;
            result.m_data = m_data;
            result.m_shape = new_shape;
            result.m_size = m_size;
            result.m_offset = m_offset;
            result.compute_size_and_strides();

            return result;
        }

        template <typename... Dims>
        [[nodiscard]] Tensor view(Dims... dims) const { return view({static_cast<size_t>(dims)...}); }

        [[nodiscard]] Tensor transpose(size_t dim0, size_t dim1) const
        {
            if (dim0 >= m_shape.size() || dim1 >= m_shape.size())
            {
                throw std::runtime_error("Invalid dimensions for transpose");
            }

            if (dim0 == dim1)
            {
                return *this;
            }

            std::vector<size_t> new_shape = m_shape;
            std::swap(new_shape[dim0], new_shape[dim1]);

            Tensor result(new_shape);

            std::vector<size_t> indices(m_shape.size(), 0);
            do
            {
                T value = (*this)(indices);

                std::vector<size_t> transposed_indices = indices;
                std::swap(transposed_indices[dim0], transposed_indices[dim1]);

                result(transposed_indices) = value;
            } while (increment_indices(indices));

            return result;
        }

        [[nodiscard]] Tensor t() const
        {
            if (m_shape.size() != 2)
            {
                throw std::runtime_error("t() only works for 2D tensors, use transpose() or permute() for higher dims");
            }
            return transpose(0, 1);
        }

        [[nodiscard]] Tensor permute(std::initializer_list<size_t> order) const
        {
            std::vector<size_t> perm_order(order);

            if (perm_order.size() != m_shape.size())
            {
                throw std::runtime_error("Permutation size must match ndim");
            }

            std::vector<bool> used(m_shape.size(), false);
            for (size_t dim : perm_order)
            {
                if (dim >= m_shape.size())
                    throw std::runtime_error("Invalid dimension in permutation");
                if (used[dim])
                    throw std::runtime_error("Duplicate dimension in permutation");
                used[dim] = true;
            }

            std::vector<size_t> new_shape(m_shape.size());
            for (size_t i = 0; i < perm_order.size(); ++i)
            {
                new_shape[i] = m_shape[perm_order[i]];
            }

            Tensor result(new_shape);

            std::vector<size_t> indices(m_shape.size(), 0);

            do
            {
                T value = (*this)(indices);

                std::vector<size_t> permuted_indices(m_shape.size());
                for (size_t i = 0; i < perm_order.size(); ++i)
                {
                    permuted_indices[i] = indices[perm_order[i]];
                }

                result(permuted_indices) = value;

            } while (increment_indices(indices));

            return result;
        }

        template <typename... Dims>
        [[nodiscard]] Tensor permute(Dims... dims) { return permute({static_cast<size_t>(dims)...}); }

        [[nodiscard]] Tensor broadcast_to(std::initializer_list<size_t> new_shape) const
        {
            std::vector<size_t> target_shape(new_shape);

            int orig_idx = m_shape.size() - 1;
            int target_idx = target_shape.size() - 1;

            while (orig_idx >= 0 && target_idx >= 0)
            {
                if (m_shape[orig_idx] != target_shape[target_idx] && m_shape[orig_idx] != 1)
                {
                    throw std::runtime_error("Shapes not broadcastable");
                }
                --orig_idx;
                --target_idx;
            }

            Tensor result(new_shape);

            std::vector<size_t> indices(target_shape.size(), 0);

            do
            {
                std::vector<size_t> orig_indices;
                int offset = target_shape.size() - m_shape.size();

                for (size_t i = 0; i < m_shape.size(); ++i)
                {
                    size_t target_idx = i + offset;
                    orig_indices.push_back(m_shape[i] == 1 ? 0 : indices[target_idx]);
                }

                result(indices) = (*this)(orig_indices);

            } while (increment_indices_for_shape(indices, target_shape));

            return result;
        }

        [[nodiscard]] Tensor slice(size_t dim, size_t start, size_t end) const
        {
            if (dim >= m_shape.size())
            {
                throw std::runtime_error("Invalid dimension");
            }
            if (start >= end || end > m_shape[dim])
            {
                throw std::runtime_error("Invalid range");
            }

            std::vector<size_t> result_shape = m_shape;
            result_shape[dim] = end - start;

            Tensor result = from_shape(result_shape);

            std::vector<size_t> indices(result_shape.size(), 0);
            do
            {
                std::vector<size_t> src_indices = indices;
                src_indices[dim] += start;

                result(indices) = (*this)(src_indices);

            } while (increment_indices_for_shape(indices, result_shape));

            return result;
        }

        Tensor operator[](size_t index) const
        {
            if (m_shape.empty())
            {
                throw std::runtime_error("Cannot index scalar");
            }
            if (index >= m_shape[0])
            {
                throw std::runtime_error("Index out of bounds");
            }

            std::vector<size_t> result_shape(m_shape.begin() + 1, m_shape.end());

            if (result_shape.empty())
            {
                Tensor result(1);
                result.m_data[0] = m_data[index];
                return result;
            }

            Tensor result(result_shape);

            size_t slice_size = result.size();
            size_t offset = index * slice_size;

            std::copy_n(m_data + offset, slice_size, result.m_data);

            return result;
        }

        [[nodiscard]] Tensor squeeze(int dim = -1) const
        {
            std::vector<size_t> new_shape;

            if (dim == -1)
            {
                for (size_t d : m_shape)
                {
                    if (d != 1)
                    {
                        new_shape.push_back(d);
                    }
                }
            }
            else
            {
                if (static_cast<size_t>(dim) >= m_shape.size())
                {
                    throw std::runtime_error("Invalid dimension");
                }

                for (size_t i = 0; i < m_shape.size(); i++)
                {
                    if (i != static_cast<size_t>(dim) || m_shape[i] != 1)
                    {
                        new_shape.push_back(m_shape[i]);
                    }
                }
            }

            if (new_shape.empty())
            {
                new_shape.push_back(1);
            }

            return view_from_vector(new_shape);
        }

        [[nodiscard]] Tensor unsqueeze(size_t dim) const
        {
            if (dim > m_shape.size())
            {
                throw std::runtime_error("Invalid dimension");
            }

            std::vector<size_t> new_shape = m_shape;
            new_shape.insert(new_shape.begin() + dim, 1);

            return view_from_vector(new_shape);
        }

        // ============ IN-PLACE METHODS ============

        Tensor &zero_()
        {
            std::fill_n(m_data, m_size, T(0));
            return *this;
        }

        Tensor &ones_()
        {
            std::fill_n(m_data, m_size, T(1));
            return *this;
        }

        Tensor &fill_(T value)
        {
            std::fill_n(m_data, m_size, value);
            return *this;
        }

        Tensor &randn_(T mean = 0, T stddev = 1, uint64_t seed = 0)
        {
            if (seed == 0)
            {
                std::random_device rd;
                seed = rd();
            }
            pcg32 gen(seed);
            std::normal_distribution<T> dis(mean, stddev);

            for (size_t i = 0; i < m_size; ++i)
                m_data[i] = dis(gen);
            return *this;
        }

        Tensor &uniform_(T min = 0, T max = 1, uint64_t seed = 0)
        {
            if (seed == 0)
            {
                std::random_device rd;
                seed = rd();
            }
            pcg32 gen(seed);
            std::uniform_real_distribution<T> dis(min, max);

            for (size_t i = 0; i < m_size; ++i)
                m_data[i] = dis(gen);
            return *this;
        }

        // ============ GETTERS ============

        const std::vector<size_t> &shape() const { return m_shape; }
        const std::vector<size_t> &strides() const { return m_strides; }
        size_t size() const { return m_size; }
        size_t ndim() const { return m_shape.size(); }

        size_t dim(size_t i) const { return m_shape[i]; }

        T *data() { return m_data; }
        const T *data() const { return m_data; }

        T *begin() { return m_data; }
        T *end() { return m_data + m_size; }
        const T *begin() const { return m_data; }
        const T *end() const { return m_data + m_size; }

    private:
        std::shared_ptr<T[]> m_storage;
        T *m_data;
        std::vector<size_t> m_shape;
        std::vector<size_t> m_strides;
        size_t m_size;
        size_t m_offset;

        static Tensor from_shape(const std::vector<size_t> &shape) { return Tensor(shape); }

        void compute_size_and_strides()
        {
            m_size = 1;
            for (auto dim : m_shape)
                m_size *= dim;

            m_strides.resize(m_shape.size());
            size_t stride = 1;

            for (int i = static_cast<int>(m_shape.size()) - 1; i >= 0; --i)
            {
                m_strides[i] = stride;
                stride *= m_shape[i];
            }
        }

        size_t compute_offset(const std::vector<size_t> &indices) const
        {
            size_t offset = 0;
            for (size_t i = 0; i < indices.size(); ++i)
            {
                offset += indices[i] * m_strides[i];
            }
            return offset;
        }

        bool increment_indices(std::vector<size_t> &indices) const
        {
            for (int i = m_shape.size() - 1; i >= 0; --i)
            {
                indices[i]++;
                if (indices[i] < m_shape[i])
                    return true;
                indices[i] = 0;
            }
            return false;
        }

        bool increment_indices_for_shape(std::vector<size_t> &indices, const std::vector<size_t> &shape) const
        {
            for (int i = shape.size() - 1; i >= 0; --i)
            {
                indices[i]++;
                if (indices[i] < shape[i])
                    return true;
                indices[i] = 0;
            }
            return false;
        }

        // ============ HELPER: 2D matrix multiplication ============
        Tensor matmul_2d(const Tensor &other) const
        {
            // (m, k) @ (k, n) → (m, n)
            if (m_shape.size() != 2 || other.m_shape.size() != 2)
            {
                throw std::runtime_error("matmul_2d requires 2D tensors");
            }

            size_t m = m_shape[0];       // rows of A
            size_t k = m_shape[1];       // cols of A = rows of B
            size_t n = other.m_shape[1]; // cols of B

            if (k != other.m_shape[0])
            {
                throw std::runtime_error("Inner dimensions must match");
            }

            constexpr size_t TILE_SIZE = 64;

            Tensor result = Tensor::zeros({m, n});
            for (size_t i = 0; i < m; i += TILE_SIZE)
            {
                for (size_t j = 0; j < n; j += TILE_SIZE)
                {
                    for (size_t p = 0; p < k; p += TILE_SIZE)
                    {
                        size_t i_end = std::min(i + TILE_SIZE, m);
                        size_t j_end = std::min(j + TILE_SIZE, n);
                        size_t p_end = std::min(p + TILE_SIZE, k);

                        for (size_t ii = i; ii < i_end; ++ii)
                        {
                            for (size_t jj = j; jj < j_end; ++jj)
                            {
                                T sum = result(ii, jj);
                                for (size_t pp = p; pp < p_end; ++pp)
                                {
                                    sum += (*this)(ii, pp) * other(pp, jj);
                                }
                                result(ii, jj) = sum;
                            }
                        }
                    }
                }
            }

            return result;
        }

        Tensor matmul_batched(const Tensor &other) const
        {
            // Example: (b, m, k) @ (b, k, n) → (b, m, n)

            if (m_shape.size() < 3 || other.m_shape.size() < 3)
            {
                throw std::runtime_error("Batch matmul requires 3D+ tensors");
            }

            size_t m = m_shape[m_shape.size() - 2];
            size_t k = m_shape[m_shape.size() - 1];
            size_t n = other.m_shape[other.m_shape.size() - 1];

            if (k != other.m_shape[other.m_shape.size() - 2])
            {
                throw std::runtime_error("Inner dimensions must match");
            }

            std::vector<size_t> batch_shape_a(m_shape.begin(), m_shape.end() - 2);
            std::vector<size_t> batch_shape_b(other.m_shape.begin(), other.m_shape.end() - 2);

            std::vector<size_t> result_batch_shape = broadcast_shapes(batch_shape_a, batch_shape_b);

            std::vector<size_t> result_shape = result_batch_shape;
            result_shape.push_back(m);
            result_shape.push_back(n);

            Tensor result = from_shape(result_shape);

            size_t num_batches = 1;
            for (size_t dim : result_batch_shape)
                num_batches *= dim;

            for (size_t b = 0; b < num_batches; ++b)
            {
                std::vector<size_t> batch_idx = batch_indices_from_linear(b, result_batch_shape);

                std::vector<size_t> idx_a_batch = broadcast_index_back(batch_idx, batch_shape_a, result_batch_shape);
                std::vector<size_t> idx_b_batch = broadcast_index_back(batch_idx, batch_shape_b, result_batch_shape);

                for (size_t i = 0; i < m; ++i)
                {
                    for (size_t j = 0; j < n; ++j)
                    {
                        T sum = 0;
                        for (size_t p = 0; p < k; ++p)
                        {
                            std::vector<size_t> idx_a = idx_a_batch;
                            idx_a.push_back(i);
                            idx_a.push_back(p);

                            std::vector<size_t> idx_b = idx_b_batch;
                            idx_b.push_back(p);
                            idx_b.push_back(j);

                            sum += (*this)(idx_a)*other(idx_b);
                        }

                        std::vector<size_t> idx_result = batch_idx;
                        idx_result.push_back(i);
                        idx_result.push_back(j);

                        result(idx_result) = sum;
                    }
                }
            }

            return result;
        }

        std::vector<size_t> batch_indices_from_linear(size_t linear_idx, const std::vector<size_t> &batch_shape) const
        {
            std::vector<size_t> indices(batch_shape.size());

            for (int i = batch_shape.size() - 1; i >= 0; --i)
            {
                indices[i] = linear_idx % batch_shape[i];
                linear_idx /= batch_shape[i];
            }

            return indices;
        }

        std::vector<size_t> broadcast_shapes(const std::vector<size_t> &shape_a, const std::vector<size_t> &shape_b) const
        {
            size_t max_ndim = std::max(shape_a.size(), shape_b.size());
            std::vector<size_t> result_shape(max_ndim);

            for (int i = 0; i < static_cast<int>(max_ndim); ++i)
            {
                int idx_a = static_cast<int>(shape_a.size()) - 1 - i;
                int idx_b = static_cast<int>(shape_b.size()) - 1 - i;

                size_t dim_a = (idx_a >= 0) ? shape_a[idx_a] : 1;
                size_t dim_b = (idx_b >= 0) ? shape_b[idx_b] : 1;

                if (dim_a != dim_b && dim_a != 1 && dim_b != 1)
                {
                    throw std::runtime_error("Shapes are not broadcastable");
                }

                result_shape[max_ndim - 1 - i] = std::max(dim_a, dim_b);
            }

            return result_shape;
        }

        std::vector<size_t> broadcast_index_back(const std::vector<size_t> &broadcast_idx, const std::vector<size_t> &original_shape,
                                                 const std::vector<size_t> &broadcast_shape) const
        {
            std::vector<size_t> result(original_shape.size());

            int offset = broadcast_shape.size() - original_shape.size();

            for (size_t i = 0; i < original_shape.size(); ++i)
            {
                size_t broadcast_i = i + offset;

                if (original_shape[i] == 1)
                    result[i] = 0;
                else
                    result[i] = broadcast_idx[broadcast_i];
            }

            return result;
        }

        Tensor view_from_vector(const std::vector<size_t> &new_shape) const
        {
            size_t new_size = 1;
            for (auto dim : new_shape)
                new_size *= dim;

            if (new_size != m_size)
                throw std::runtime_error("Total size must remain the same for view");

            Tensor result;
            result.m_storage = m_storage;
            result.m_data = m_data;
            result.m_shape = new_shape;
            result.m_size = m_size;
            result.m_offset = m_offset;
            result.compute_size_and_strides();

            return result;
        }
    };
} // namespace arcana::tensor