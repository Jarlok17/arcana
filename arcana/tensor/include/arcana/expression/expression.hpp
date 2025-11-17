#pragma once

namespace arcana::expression {

template <typename Op, typename L, typename R>
concept HasApply = requires(const L &l, const R &r) {
    { Op::apply(l, r) };
};

struct AddOp
{
        template <typename L, typename R> static auto apply(const L &lhs, const R &rhs) { return lhs + rhs; }
};

struct SubOp
{
        template <typename L, typename R> static auto apply(const L &lhs, const R &rhs) { return lhs - rhs; }
};

struct MulOp
{
        template <typename L, typename R> static auto apply(const L &lhs, const R &rhs) { return lhs * rhs; }
};

struct DivOp
{
        template <typename L, typename R> static auto apply(const L &lhs, const R &rhs) { return lhs / rhs; }
};

class ExpressionBase
{
    protected:
        ExpressionBase() = default;

    public:
        inline auto eval(this const auto &self) { return self.eval_impl(); }

        template <typename RHS> inline auto operator+(this const auto &self, const RHS &rhs)
        {
            return BinaryExpression<decltype(self), RHS, AddOp>(self, rhs);
        }

        template <typename RHS> inline auto operator-(this const auto &self, const RHS &rhs)
        {
            return BinaryExpression<decltype(self), RHS, SubOp>(self, rhs);
        }

        template <typename RHS> inline auto operator*(this const auto &self, const RHS &rhs)
        {
            return BinaryExpression<decltype(self), RHS, MulOp>(self, rhs);
        }

        template <typename RHS> inline auto operator/(this const auto &self, const RHS &rhs)
        {
            return BinaryExpression<decltype(self), RHS, DivOp>(self, rhs);
        }
};

template <typename LHS, typename RHS, typename Op> class BinaryExpression : public ExpressionBase
{
    private:
        const LHS &lhs;
        const RHS &rhs;

    public:
        BinaryExpression(const LHS &l, const RHS &r) : lhs(l), rhs(r) {}

        inline auto eval_impl() const
        {
            auto lhs_eval = eval_helper(lhs);
            auto rhs_eval = eval_helper(rhs);
            return Op::apply(lhs_eval, rhs_eval);
        }

    private:
        template <typename T> inline auto eval_helper(const T &val) const
        {
            if constexpr (requires { val.eval_impl(); }) {
                return val.eval_impl();
            } else {
                return val;
            }
        }
};

} // namespace arcana::expression
