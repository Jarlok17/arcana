#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <omp.h>
#include <stdexcept>
#include <vector>

namespace arcana::tensor
{
    // ==================================================================
    // 1. Element-Wise Methods (SAFE for static tensors)
    //    Operations, that dont change the shape of a tensor (exp, relu, normalize...)
    // ==================================================================
    template <typename Derived>
    class ElementWiseMethods
    {
    public:
        // --- Immutable (return copy) ---

        [[nodiscard]] Derived exp() const
        {
            const Derived &self = static_cast<const Derived &>(*this);
            Derived result = Derived::empty_like(self);

#pragma omp parallel for simd
            for (size_t i = 0; i < self.size(); ++i)
                result.data()[i] = std::exp(self.data()[i]);

            return result;
        }

        [[nodiscard]] Derived log() const
        {
            const Derived &self = static_cast<const Derived &>(*this);
            Derived result = Derived::empty_like(self);

#pragma omp parallel for simd
            for (size_t i = 0; i < self.size(); ++i)
                result.data()[i] = std::log(self.data()[i]);

            return result;
        }

        template <typename T>
        [[nodiscard]] Derived pow(T exponent) const
        {
            const Derived &self = static_cast<const Derived &>(*this);
            Derived result = Derived::empty_like(self);

#pragma omp parallel for simd
            for (size_t i = 0; i < self.size(); ++i)
                result.data()[i] = std::pow(self.data()[i], exponent);

            return result;
        }

        [[nodiscard]] Derived sqrt() const
        {
            const Derived &self = static_cast<const Derived &>(*this);
            Derived result = Derived::empty_like(self);

#pragma omp parallel for simd
            for (size_t i = 0; i < self.size(); ++i)
                result.data()[i] = std::sqrt(self.data()[i]);

            return result;
        }

        // --- Activation Functions ---

        [[nodiscard]] Derived relu() const
        {
            const Derived &self = static_cast<const Derived &>(*this);
            Derived result = Derived::empty_like(self);

#pragma omp parallel for simd
            for (size_t i = 0; i < self.size(); ++i)
                result.data()[i] = std::max(typename Derived::scalar_type(0), self.data()[i]);

            return result;
        }

        [[nodiscard]] Derived sigmoid() const
        {
            const Derived &self = static_cast<const Derived &>(*this);
            Derived result = Derived::empty_like(self);

#pragma omp parallel for simd
            for (size_t i = 0; i < self.size(); ++i)
                result.data()[i] = typename Derived::scalar_type(1) /
                                   (typename Derived::scalar_type(1) + std::exp(-self.data()[i]));

            return result;
        }

        [[nodiscard]] Derived tanh() const
        {
            const Derived &self = static_cast<const Derived &>(*this);
            Derived result = Derived::empty_like(self);

#pragma omp parallel for simd
            for (size_t i = 0; i < self.size(); ++i)
                result.data()[i] = std::tanh(self.data()[i]);

            return result;
        }

        // --- Mutable (In-Place) ---

        Derived &exp_()
        {
            Derived &self = static_cast<Derived &>(*this);
#pragma omp parallel for simd
            for (size_t i = 0; i < self.size(); ++i)
                self.data()[i] = std::exp(self.data()[i]);
            return self;
        }

        Derived &log_()
        {
            Derived &self = static_cast<Derived &>(*this);
#pragma omp parallel for simd
            for (size_t i = 0; i < self.size(); ++i)
                self.data()[i] = std::log(self.data()[i]);
            return self;
        }

        template <typename T>
        Derived &pow_(T exponent)
        {
            Derived &self = static_cast<Derived &>(*this);
#pragma omp parallel for simd
            for (size_t i = 0; i < self.size(); ++i)
                self.data()[i] = std::pow(self.data()[i], exponent);
            return self;
        }

        Derived &sqrt_()
        {
            Derived &self = static_cast<Derived &>(*this);
#pragma omp parallel for simd
            for (size_t i = 0; i < self.size(); ++i)
                self.data()[i] = std::sqrt(self.data()[i]);
            return self;
        }

        Derived &relu_()
        {
            Derived &self = static_cast<Derived &>(*this);
#pragma omp parallel for simd
            for (size_t i = 0; i < self.size(); ++i)
                self.data()[i] = std::max(typename Derived::scalar_type(0), self.data()[i]);
            return self;
        }

        Derived &sigmoid_()
        {
            Derived &self = static_cast<Derived &>(*this);
#pragma omp parallel for simd
            for (size_t i = 0; i < self.size(); ++i)
                self.data()[i] = typename Derived::scalar_type(1) /
                                 (typename Derived::scalar_type(1) + std::exp(-self.data()[i]));
            return self;
        }

        Derived &tanh_()
        {
            Derived &self = static_cast<Derived &>(*this);
#pragma omp parallel for simd
            for (size_t i = 0; i < self.size(); ++i)
                self.data()[i] = std::tanh(self.data()[i]);
            return self;
        }
    };

    // ==================================================================
    // 2. Global Reduction Methods (SAFE for static tensors)
    //    Operations that return a scalar (sum, mean, max...)
    // ==================================================================
    template <typename Derived>
    class GlobalReductionMethods
    {
    public:
        [[nodiscard]] auto mean() const
        {
            const Derived &self = static_cast<const Derived &>(*this);

            if (self.size() == 0)
                throw std::runtime_error("Cannot compute mean of empty tensor");

            typename Derived::scalar_type sum = 0;

#pragma omp parallel for reduction(+ : sum)
            for (size_t i = 0; i < self.size(); ++i)
                sum += self.data()[i];

            return sum / static_cast<typename Derived::scalar_type>(self.size());
        }

        [[nodiscard]] auto sum() const
        {
            const Derived &self = static_cast<const Derived &>(*this);
            typename Derived::scalar_type sum = 0;

#pragma omp parallel for reduction(+ : sum)
            for (size_t i = 0; i < self.size(); ++i)
                sum += self.data()[i];

            return sum;
        }

        [[nodiscard]] auto stddev() const
        {
            const Derived &self = static_cast<const Derived &>(*this);

            if (self.size() == 0)
                throw std::runtime_error("Cannot compute stddev of empty tensor");

            auto m = mean();
            typename Derived::scalar_type variance = 0;

#pragma omp parallel for reduction(+ : variance)
            for (size_t i = 0; i < self.size(); ++i)
            {
                auto diff = self.data()[i] - m;
                variance += diff * diff;
            }

            return std::sqrt(variance / static_cast<typename Derived::scalar_type>(self.size()));
        }

        [[nodiscard]] auto max() const
        {
            const Derived &self = static_cast<const Derived &>(*this);
            if (self.size() == 0)
                throw std::runtime_error("Cannot compute max of empty tensor");
            return *std::max_element(self.begin(), self.end());
        }

        [[nodiscard]] auto min() const
        {
            const Derived &self = static_cast<const Derived &>(*this);
            if (self.size() == 0)
                throw std::runtime_error("Cannot compute min of empty tensor");
            return *std::min_element(self.begin(), self.end());
        }

        [[nodiscard]] auto argmax() const
        {
            const Derived &self = static_cast<const Derived &>(*this);

            if (self.size() == 0)
                throw std::runtime_error("Cannot compute argmax of empty tensor");

            size_t max_index = 0;
            typename Derived::scalar_type max_value = self.data()[0];

#pragma omp parallel
            {
                size_t local_max_idx = 0;
                typename Derived::scalar_type local_max_value = max_value;

#pragma omp for nowait
                for (size_t i = 0; i < self.size(); ++i)
                {
                    if (self.data()[i] > local_max_value)
                    {
                        local_max_value = self.data()[i];
                        local_max_idx = i;
                    }
                }

#pragma omp critical
                {
                    if (local_max_value > max_value)
                    {
                        max_value = local_max_value;
                        max_index = local_max_idx;
                    }
                }
            }

            return max_index;
        }

        [[nodiscard]] auto argmin() const
        {
            const Derived &self = static_cast<const Derived &>(*this);

            if (self.size() == 0)
                throw std::runtime_error("Cannot compute argmin of empty tensor");

            size_t min_index = 0;
            typename Derived::scalar_type min_value = self.data()[0];

#pragma omp parallel
            {
                size_t local_min_idx = 0;
                typename Derived::scalar_type local_min_value = min_value;

#pragma omp for nowait
                for (size_t i = 0; i < self.size(); ++i)
                {
                    if (self.data()[i] < local_min_value)
                    {
                        local_min_value = self.data()[i];
                        local_min_idx = i;
                    }
                }

#pragma omp critical
                {
                    if (local_min_value > min_value)
                    {
                        min_value = local_min_value;
                        min_index = local_min_idx;
                    }
                }
            }

            return min_index;
        }

        Derived &normalize()
        {
            Derived &self = static_cast<Derived &>(*this);

            if (self.size() == 0)
                return self;

            auto m = mean();
            auto s = stddev();

            if (s == 0)
                return self;

#pragma omp parallel for simd
            for (size_t i = 0; i < self.size(); ++i)
                self.data()[i] = (self.data()[i] - m) / s;

            return self;
        }
    };

    // ==================================================================
    // 3. Dimension Reduction Methods (DANGEROUS for static tensors)
    //    Operations that dynamically change the shape of a tensor (sum(dim)...)
    // ==================================================================
    template <typename Derived>
    class DimReductionMethods
    {
    public:
        [[nodiscard]] auto sum(int dim) const
        {
            const Derived &self = static_cast<const Derived &>(*this);

            if (dim < 0)
            {
                dim = self.ndim() + dim;
            }

            if (dim >= static_cast<int>(self.ndim()))
            {
                throw std::runtime_error("dim out of range");
            }

            std::vector<size_t> result_shape;
            for (size_t i = 0; i < self.shape().size(); ++i)
            {
                if (i != static_cast<size_t>(dim))
                {
                    result_shape.push_back(self.shape()[i]);
                }
            }

            if (result_shape.empty())
            {
                result_shape.push_back(1);
            }

            Derived result = Derived::zeros(result_shape);

            std::vector<size_t> indices(self.shape().size(), 0);
            do
            {
                std::vector<size_t> result_indices;
                for (size_t i = 0; i < indices.size(); ++i)
                {
                    if (i != static_cast<size_t>(dim))
                        result_indices.push_back(indices[i]);
                }

                if (result_indices.empty())
                    result_indices.push_back(0);

                result(result_indices) += self(indices);

            } while (increment_indices_for_shape(indices, self.shape()));

            return result;
        }

        [[nodiscard]] auto mean(int dim) const
        {
            const Derived &self = static_cast<const Derived &>(*this);

            auto sum_result = self.sum(dim);
            auto count = static_cast<typename Derived::scalar_type>(self.shape()[dim]);
            return sum_result / count;
        }

        [[nodiscard]] auto max(int dim) const
        {
            const Derived &self = static_cast<const Derived &>(*this);

            if (dim < 0)
            {
                dim = self.ndim() + dim;
            }

            std::vector<size_t> result_shape;
            for (size_t i = 0; i < self.shape().size(); ++i)
            {
                if (i != static_cast<size_t>(dim))
                {
                    result_shape.push_back(self.shape()[i]);
                }
            }

            if (result_shape.empty())
            {
                result_shape.push_back(1);
            }

            Derived result(result_shape);
            result.fill_(std::numeric_limits<typename Derived::scalar_type>::lowest());

            std::vector<size_t> indices(self.shape().size(), 0);
            do
            {
                std::vector<size_t> result_indices;
                for (size_t i = 0; i < indices.size(); ++i)
                {
                    if (i != static_cast<size_t>(dim))
                        result_indices.push_back(indices[i]);
                }

                if (result_indices.empty())
                    result_indices.push_back(0);

                result(result_indices) = std::max(result(result_indices), self(indices));

            } while (increment_indices_for_shape(indices, self.shape()));

            return result;
        }

        [[nodiscard]] auto min(int dim) const
        {
            const Derived &self = static_cast<const Derived &>(*this);

            if (dim < 0)
            {
                dim = self.ndim() + dim;
            }

            std::vector<size_t> result_shape;
            for (size_t i = 0; i < self.shape().size(); ++i)
            {
                if (i != static_cast<size_t>(dim))
                {
                    result_shape.push_back(self.shape()[i]);
                }
            }

            if (result_shape.empty())
            {
                result_shape.push_back(1);
            }

            Derived result(result_shape);
            result.fill_(std::numeric_limits<typename Derived::scalar_type>::lowest());

            std::vector<size_t> indices(self.shape().size(), 0);
            do
            {
                std::vector<size_t> result_indices;

                for (size_t i = 0; i < indices.size(); ++i)
                {
                    if (i != static_cast<size_t>(dim))
                    {
                        result_indices.push_back(indices[i]);
                    }
                }

                if (result_indices.empty())
                {
                    result_indices.push_back(0);
                }

                result(result_indices) = std::min(result(result_indices), self(indices));

            } while (increment_indices_for_shape(indices, self.shape()));

            return result;
        }

        [[nodiscard]] auto argmax(int dim) const
        {
            const Derived &self = static_cast<const Derived &>(*this);

            if (dim < 0)
                dim += self.ndim();

            std::vector<size_t> result_shape;
            for (size_t i = 0; i < self.ndim(); ++i)
            {
                if (i != static_cast<size_t>(dim))
                    result_shape.push_back(self.shape()[i]);
            }

            Derived result = Derived::zeros(result_shape);

            size_t fixed_dim_size = 1;
            for (size_t i = 0; i < self.ndim(); ++i)
            {
                if (i != static_cast<size_t>(dim))
                    fixed_dim_size *= self.shape()[i];
            }

            std::vector<size_t> fixed_indices(self.ndim(), 0);

#pragma omp parallel for
            for (size_t linear_idx = 0; linear_idx < fixed_dim_size; ++linear_idx)
            {
                std::vector<size_t> fixed_multi_idx(result_shape.size());
                size_t remainder = linear_idx;
                for (int i = (int)result_shape.size() - 1; i >= 0; --i)
                {
                    fixed_multi_idx[i] = remainder % result_shape[i];
                    remainder /= result_shape[i];
                }

                size_t dim_len = self.shape()[dim];
                std::vector<size_t> full_idx(self.ndim(), 0);
                for (size_t i = 0, j = 0; i < self.ndim(); ++i)
                {
                    if (i == static_cast<size_t>(dim))
                    {
                        full_idx[i] = 0;
                    }
                    else
                    {
                        full_idx[i] = fixed_multi_idx[j++];
                    }
                }

                size_t max_index = 0;
                auto max_val = self(full_idx);
                for (size_t k = 1; k < dim_len; ++k)
                {
                    full_idx[dim] = k;
                    auto val = self(full_idx);
                    if (val > max_val)
                    {
                        max_val = val;
                        max_index = k;
                    }
                }

                result(fixed_multi_idx) = max_index;
            }

            return result;
        }

        [[nodiscard]] auto argmin(int dim) const
        {
            const Derived &self = static_cast<const Derived &>(*this);

            if (dim < 0)
                dim += self.ndim();

            std::vector<size_t> result_shape;
            for (size_t i = 0; i < self.ndim(); ++i)
            {
                if (i != static_cast<size_t>(dim))
                    result_shape.push_back(self.shape()[i]);
            }

            Derived result = Derived::zeros(result_shape);

            size_t fixed_dim_size = 1;
            for (size_t i = 0; i < self.ndim(); ++i)
            {
                if (i != static_cast<size_t>(dim))
                    fixed_dim_size *= self.shape()[i];
            }

            std::vector<size_t> fixed_indices(self.ndim(), 0);

#pragma omp parallel for
            for (size_t linear_idx = 0; linear_idx < fixed_dim_size; ++linear_idx)
            {
                std::vector<size_t> fixed_multi_idx(result_shape.size());
                size_t remainder = linear_idx;
                for (int i = (int)result_shape.size() - 1; i >= 0; --i)
                {
                    fixed_multi_idx[i] = remainder % result_shape[i];
                    remainder /= result_shape[i];
                }

                size_t dim_len = self.shape()[dim];
                std::vector<size_t> full_idx(self.ndim(), 0);
                for (size_t i = 0, j = 0; i < self.ndim(); ++i)
                {
                    if (i == static_cast<size_t>(dim))
                    {
                        full_idx[i] = 0;
                    }
                    else
                    {
                        full_idx[i] = fixed_multi_idx[j++];
                    }
                }

                size_t min_index = 0;
                auto min_val = self(full_idx);
                for (size_t k = 1; k < dim_len; ++k)
                {
                    full_idx[dim] = k;
                    auto val = self(full_idx);
                    if (val < min_val)
                    {
                        min_val = val;
                        min_index = k;
                    }
                }

                result(fixed_multi_idx) = min_index;
            }

            return result;
        }
    };

    // ==================================================================
    // 4. Unified Interface for dynamic tensor
    // ==================================================================
    template <typename Derived>
    class TensorMethods : public ElementWiseMethods<Derived>,
                          public GlobalReductionMethods<Derived>,
                          public DimReductionMethods<Derived>
    {
    public:
        using GlobalReductionMethods<Derived>::mean;
        using DimReductionMethods<Derived>::mean;

        using GlobalReductionMethods<Derived>::sum;
        using DimReductionMethods<Derived>::sum;

        using GlobalReductionMethods<Derived>::max;
        using DimReductionMethods<Derived>::max;

        using GlobalReductionMethods<Derived>::min;
        using DimReductionMethods<Derived>::min;

        using GlobalReductionMethods<Derived>::argmax;
        using DimReductionMethods<Derived>::argmax;

        using GlobalReductionMethods<Derived>::argmin;
        using DimReductionMethods<Derived>::argmin;
    };

} // namespace arcana::tensor