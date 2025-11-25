#pragma once
#include "arcana/tensor/tensor_operations.hpp"

namespace arcana::tensor
{
    template <typename Derived, typename T>
    class ArithmeticMixin
    {
    public:
        Derived &derived() { return static_cast<Derived &>(*this); }
        const Derived &derived() const { return static_cast<const Derived &>(*this); }

        // ============ IN-PLACE TENSOR ARITHMETIC ============

        Derived &operator+=(const Derived &other)
        {
            if (derived().size() != other.size())
                throw std::runtime_error("Shape mismatch");
            AddOp<T> op(derived().data(), other.data(), derived().data(), derived().size());
            op.execute();
            return derived();
        }

        Derived &operator-=(const Derived &other)
        {
            if (derived().size() != other.size())
                throw std::runtime_error("Shape mismatch");
            SubOp<T> op(derived().data(), other.data(), derived().data(), derived().size());
            op.execute();
            return derived();
        }

        Derived &operator*=(const Derived &other)
        {
            if (derived().size() != other.size())
                throw std::runtime_error("Shape mismatch");
            MulOp<T> op(derived().data(), other.data(), derived().data(), derived().size());
            op.execute();
            return derived();
        }

        Derived &operator/=(const Derived &other)
        {
            if (derived().size() != other.size())
                throw std::runtime_error("Shape mismatch");
            DivOp<T> op(derived().data(), other.data(), derived().data(), derived().size());
            op.execute();
            return derived();
        }

        // ============ BINARY TENSOR ARITHMETIC (Creates new tensor) ============

        Derived operator+(const Derived &other) const
        {
            Derived result = derived(); // Copy constructor
            result += other;            // Uses SIMD in-place
            return result;
        }

        Derived operator-(const Derived &other) const
        {
            Derived result = derived();
            result -= other;
            return result;
        }

        Derived operator*(const Derived &other) const
        {
            Derived result = derived();
            result *= other;
            return result;
        }

        Derived operator/(const Derived &other) const
        {
            Derived result = derived();
            result /= other;
            return result;
        }

        // ============ IN-PLACE SCALAR ARITHMETIC ============

        Derived &operator+=(T scalar)
        {
            AddScalarOp<T> op(derived().data(), scalar, derived().size());
            op.execute();
            return derived();
        }

        Derived &operator-=(T scalar)
        {
            SubScalarOp<T> op(derived().data(), scalar, derived().size());
            op.execute();
            return derived();
        }

        Derived &operator*=(T scalar)
        {
            MulScalarOp<T> op(derived().data(), scalar, derived().size());
            op.execute();
            return derived();
        }

        Derived &operator/=(T scalar)
        {
            DivScalarOp<T> op(derived().data(), scalar, derived().size());
            op.execute();
            return derived();
        }

        // ============ SCALAR ARITHMETIC (New Tensor) ============

        Derived operator+(T scalar) const
        {
            Derived result = derived();
            result += scalar;
            return result;
        }

        Derived operator-(T scalar) const
        {
            Derived result = derived();
            result -= scalar;
            return result;
        }

        Derived operator*(T scalar) const
        {
            Derived result = derived();
            result *= scalar;
            return result;
        }

        Derived operator/(T scalar) const
        {
            Derived result = derived();
            result /= scalar;
            return result;
        }
    };
}