#pragma once

#include <algorithm>
#include <cmath>
#include <omp.h>

namespace arcana::tensor
{
    template <typename Derived>
    class TensorMethods
    {
    public:
        // ============ REDUCTION OPERATIONS ============

        auto mean() const
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

        auto sum() const
        {
            const Derived &self = static_cast<const Derived &>(*this);
            typename Derived::scalar_type sum = 0;

#pragma omp parallel for reduction(+ : sum)
            for (size_t i = 0; i < self.size(); ++i)
                sum += self.data()[i];

            return sum;
        }

        auto stddev() const
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

        auto max() const
        {
            const Derived &self = static_cast<const Derived &>(*this);
            if (self.size() == 0)
                throw std::runtime_error("Cannot compute max of empty tensor");
            return *std::max_element(self.begin(), self.end());
        }

        auto min() const
        {
            const Derived &self = static_cast<const Derived &>(*this);
            if (self.size() == 0)
                throw std::runtime_error("Cannot compute min of empty tensor");
            return *std::min_element(self.begin(), self.end());
        }

        // ============ ELEMENT-WISE OPERATIONS (immutable) ============

        Derived exp() const
        {
            const Derived &self = static_cast<const Derived &>(*this);
            Derived result = Derived::empty_like(self);

#pragma omp parallel for simd
            for (size_t i = 0; i < self.size(); ++i)
                result.data()[i] = std::exp(self.data()[i]);

            return result;
        }

        Derived log() const
        {
            const Derived &self = static_cast<const Derived &>(*this);
            Derived result = Derived::empty_like(self);

#pragma omp parallel for simd
            for (size_t i = 0; i < self.size(); ++i)
                result.data()[i] = std::log(self.data()[i]);

            return result;
        }

        template <typename T>
        Derived pow(T exponent) const
        {
            const Derived &self = static_cast<const Derived &>(*this);
            Derived result = Derived::empty_like(self);

#pragma omp parallel for simd
            for (size_t i = 0; i < self.size(); ++i)
                result.data()[i] = std::pow(self.data()[i], exponent);

            return result;
        }

        Derived sqrt() const
        {
            const Derived &self = static_cast<const Derived &>(*this);
            Derived result = Derived::empty_like(self);

#pragma omp parallel for simd
            for (size_t i = 0; i < self.size(); ++i)
                result.data()[i] = std::sqrt(self.data()[i]);

            return result;
        }

        // ============ IN-PLACE OPERATIONS ============

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

        // ============ ACTIVATION FUNCTIONS ============

        Derived relu() const
        {
            const Derived &self = static_cast<const Derived &>(*this);
            Derived result = Derived::empty_like(self);

#pragma omp parallel for simd
            for (size_t i = 0; i < self.size(); ++i)
                result.data()[i] = std::max(typename Derived::scalar_type(0), self.data()[i]);

            return result;
        }

        Derived sigmoid() const
        {
            const Derived &self = static_cast<const Derived &>(*this);
            Derived result = Derived::empty_like(self);

#pragma omp parallel for simd
            for (size_t i = 0; i < self.size(); ++i)
                result.data()[i] = typename Derived::scalar_type(1) /
                                   (typename Derived::scalar_type(1) + std::exp(-self.data()[i]));

            return result;
        }

        Derived tanh() const
        {
            const Derived &self = static_cast<const Derived &>(*this);
            Derived result = Derived::empty_like(self);

#pragma omp parallel for simd
            for (size_t i = 0; i < self.size(); ++i)
                result.data()[i] = std::tanh(self.data()[i]);

            return result;
        }

        Derived &relu_()
        {
            Derived &self = static_cast<Derived &>(*this);

#pragma omp parallel for simd
            for (size_t i = 0; i < self.size(); ++i)
                self.data()[i] = std::max(typename Derived::scalar_type(0), self.data()[i]);

            return self;
        }

        // ============ NORMALIZE ============

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
}