#pragma once

#include <algorithm>
#include "arcana/pcg_random.hpp"
#include "arcana/pcg_extras.hpp"
#include <random>

namespace arcana::tensor::gen
{
    // ============ FACTORY METHODS ============

    template <typename TensorType, typename T = typename TensorType::scalar_type>
    void fill_uniform(TensorType &t, T min = 0, T max = 1)
    {
        pcg_extras::seed_seq_from<std::random_device> seed_source;
        pcg32 gen(seed_source);

        std::uniform_real_distribution<T> dis(min, max);
        for (auto &x : t)
            x = dis(gen);
    }

    template <typename TensorType, typename T = typename TensorType::scalar_type>
    void fill_randn(TensorType &t, T mean = 0, T stddev = 1)
    {
        pcg_extras::seed_seq_from<std::random_device> seed_source;
        pcg32 gen(seed_source);

        std::normal_distribution<T> dis(mean, stddev);
        for (auto &x : t)
            x = dis(gen);
    }

    template <typename TensorType, typename T = typename TensorType::scalar_type>
    void fill_(TensorType &t, T scalar)
    {
        std::fill(t.begin(), t.end(), scalar);
    }
}