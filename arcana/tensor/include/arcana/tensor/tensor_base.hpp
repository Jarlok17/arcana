#pragma once

#include <concepts>
#include <cstddef>

#include "arcana/expression/expression.hpp"

namespace arcana::tensor
{

    template <typename T>
    concept Numeric = std::integral<T> || std::floating_point<T>;

    template <typename Derived>
    class TensorBase : public arcana::expression::ExpressionBase
    {
    public:
        Derived &derived() { return *static_cast<Derived *>(this); }

        const Derived &derived() const { return *static_cast<const Derived *>(this); }

        Derived eval_impl() const { return derived(); }

        template <typename OtherDerived>
        auto operator+(const TensorBase<OtherDerived> &other) const
        {
            return BinaryExpression<Derived, OtherDerived, expression::AddOp>(derived(), other.derived());
        }

        template <typename OtherDerived>
        auto operator-(const TensorBase<OtherDerived> &other) const
        {
            return BinaryExpression<Derived, OtherDerived, expression::SubOp>(derived(), other.derived());
        }

        template <typename OtherDerived>
        auto operator*(const TensorBase<OtherDerived> &other) const
        {
            return BinaryExpression<Derived, OtherDerived, expression::MulOp>(derived(), other.derived());
        }

        template <typename OtherDerived>
        auto operator/(const TensorBase<OtherDerived> &other) const
        {
            return BinaryExpression<Derived, OtherDerived, expression::DivOp>(derived(), other.derived());
        }

        auto rows() const { return derived().rows(); }
        auto cols() const { return derived().cols(); }
        auto size() const { return derived().size(); }

        auto &operator()(size_t i, size_t j) { return derived()(i, j); }

        const auto &operator()(size_t i, size_t j) const { return derived()(i, j); }

    protected:
        TensorBase() = default;
        ~TensorBase() = default;
    };
} // namespace arcana::tensor
