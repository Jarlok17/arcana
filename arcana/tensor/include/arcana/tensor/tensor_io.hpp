#pragma once

#include <format>
#include <iomanip>
#include <iostream>
#include <print>
#include <sstream>

#include "arcana/tensor/tensor.hpp"

namespace arcana::tensor
{
    // ============ HELPER FUNCTIONS ============

    template <Numeric T>
    void print_tensor_recursive(std::ostream &os, const Tensor<T> &tensor, std::vector<size_t> &indices, size_t depth);

    template <Numeric T>
    std::string format_tensor(const Tensor<T> &tensor);
} // namespace arcana::tensor

// ============ std::formatter SPECIALIZATIONS ============

// Formatter for Tensor
template <arcana::tensor::Numeric T>
struct std::formatter<arcana::tensor::Tensor<T>>
{
    constexpr auto parse(std::format_parse_context &ctx) { return ctx.begin(); }

    auto format(const arcana::tensor::Tensor<T> &tensor, std::format_context &ctx) const
    {
        return std::format_to(ctx.out(), "{}", arcana::tensor::format_tensor(tensor));
    }
};

// ============ IMPLEMENTATIONS ============

namespace arcana::tensor
{
    // Format Tensor to string
    template <Numeric T>
    std::string format_tensor(const Tensor<T> &tensor)
    {
        std::ostringstream os;

        os << "Tensor(shape=[";

        const auto &shape = tensor.shape();
        for (size_t i = 0; i < shape.size(); ++i)
        {
            os << shape[i];
            if (i < shape.size() - 1)
                os << ", ";
        }
        os << "])\n";

        if (tensor.size() <= 1000)
        {
            std::vector<size_t> indices(shape.size(), 0);
            print_tensor_recursive(os, tensor, indices, 0);
        }
        else
        {
            os << "... " << tensor.size() << " elements (too large to display) ...";
        }

        return os.str();
    }

    // Recursive printer for N-D tensors
    template <Numeric T>
    void print_tensor_recursive(std::ostream &os, const Tensor<T> &tensor, std::vector<size_t> &indices, size_t depth)
    {
        const auto &shape = tensor.shape();

        if (depth == shape.size() - 1)
        {
            os << "[";
            for (size_t i = 0; i < shape[depth]; ++i)
            {
                indices[depth] = i;
                os << std::format("{:8.4f}", static_cast<double>(tensor(indices)));
                if (i < shape[depth] - 1)
                    os << ", ";
            }
            os << "]";
            return;
        }

        os << "[";

        size_t max_show = std::min(shape[depth], size_t(5));

        for (size_t i = 0; i < max_show; ++i)
        {
            if (i > 0)
            {
                os << "\n";
                for (size_t j = 0; j <= depth; ++j)
                    os << " ";
            }

            indices[depth] = i;
            print_tensor_recursive(os, tensor, indices, depth + 1);

            if (i < shape[depth] - 1)
                os << ",";
        }

        if (shape[depth] > max_show)
        {
            os << "\n";
            for (size_t j = 0; j <= depth; ++j)
                os << " ";
            os << "... (" << (shape[depth] - max_show) << " more)";
        }

        os << "]";
    }

    // operator<< for Tensor
    template <Numeric T>
    std::ostream &operator<<(std::ostream &os, const Tensor<T> &tensor)
    {
        os << format_tensor(tensor);
        return os;
    }

    // ============ HELPER FUNCTIONS ============

    template <Numeric T>
    void print_shape(const Tensor<T> &tensor, const std::string &name = "")
    {
        if (!name.empty())
            std::print("{}: ", name);

        std::print("shape=[");
        const auto &shape = tensor.shape();
        for (size_t i = 0; i < shape.size(); ++i)
        {
            std::print("{}", shape[i]);
            if (i < shape.size() - 1)
                std::print(", ");
        }
        std::print("]\n");
    }

} // namespace arcana::tensor
