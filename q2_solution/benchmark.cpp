#include "benchmark/benchmark.h"
#include "TickAggregator.h"
#include <string>
#include <random>

// Helper to generate a large, realistic CSV dataset.
static std::string generate_benchmark_data(int num_ticks) {
    std::string data;
    data.reserve(num_ticks * 40); // Pre-allocate memory
    long long timestamp = 1672531200000000000;
    double price = 100.0;

    std::mt19937 gen(123); // Fixed seed for reproducibility
    std::uniform_real_distribution<> price_dist(-0.1, 0.1);
    std::uniform_int_distribution<> vol_dist(1, 100);
    std::uniform_int_distribution<> time_dist(1000000, 500000000); // 1ms to 500ms

    for (int i = 0; i < num_ticks; ++i) {
        timestamp += time_dist(gen);
        price += price_dist(gen);
        data += std::to_string(timestamp) + "," + std::to_string(price) + "," + std::to_string(vol_dist(gen)) + "\n";
    }
    return data;
}

static void BM_TickAggregation(benchmark::State& state) {
    TickAggregator aggregator(13);
    std::string data = generate_benchmark_data(state.range(0));

    for (auto _ : state) {
        auto candles = aggregator.process_csv(data);
        benchmark::ClobberMemory(); // Prevent optimizations from removing the call
    }
    state.SetBytesProcessed(state.iterations() * data.size());
}

BENCHMARK(BM_TickAggregation)->Range(1000, 1000000)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN(); 
