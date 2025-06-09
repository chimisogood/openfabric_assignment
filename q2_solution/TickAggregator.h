 
#ifndef TICK_AGGREGATOR_H
#define TICK_AGGREGATOR_H

#include <iostream>
#include <vector>
#include <string_view>
#include <chrono>
#include <charconv>
#include <system_error>
#include <stdexcept>

// Represents a single financial tick.
struct Tick {
    std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds> timestamp;
    double price;
    int volume;
};

// Represents an OHLCV candle bar.
struct Candle {
    std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds> open_time;
    double open;
    double high;
    double low;
    double close;
    int volume;

    // Equality operator for easy testing.
    bool operator==(const Candle& other) const {
        return open_time == other.open_time &&
               open == other.open &&
               high == other.high &&
               low == other.low &&
               close == other.close &&
               volume == other.volume;
    }
};

// Overload for printing Candle objects, useful for debugging.
std::ostream& operator<<(std::ostream& os, const Candle& candle) {
    auto to_string_time = [](const auto& time_point) {
        auto tt = std::chrono::system_clock::to_time_t(
            std::chrono::time_point_cast<std::chrono::seconds>(time_point));
        std::tm tm = *std::localtime(&tt);
        char buffer[32];
        std::strftime(buffer, 32, "%Y-%m-%d %H:%M:%S", &tm);
        return std::string(buffer);
    };

    os << "Candle("
       << "OpenTime: " << to_string_time(candle.open_time)
       << ", O: " << candle.open
       << ", H: " << candle.high
       << ", L: " << candle.low
       << ", C: " << candle.close
       << ", V: " << candle.volume << ")";
    return os;
}

// High-performance CSV parser and candle aggregator.
class TickAggregator {
public:
    TickAggregator(int bar_duration_seconds)
        : bar_duration_(std::chrono::seconds(bar_duration_seconds)) {}

    // Processes a CSV file from a string view (e.g., from a memory-mapped file).
    std::vector<Candle> process_csv(std::string_view csv_data) {
        std::vector<Candle> candles;
        Candle current_candle{};
        bool is_first_tick = true;

        std::string_view::size_type pos = 0;
        while (pos < csv_data.size()) {
            auto next_pos = csv_data.find('\n', pos);
            if (next_pos == std::string_view::npos) {
                next_pos = csv_data.size();
            }

            std::string_view line = csv_data.substr(pos, next_pos - pos);
            pos = next_pos + 1;

            if (line.empty()) continue;

            Tick tick = parse_line(line);

            if (is_first_tick) {
                initialize_first_candle(current_candle, tick);
                is_first_tick = false;
            } else if (tick.timestamp >= current_candle.open_time + bar_duration_) {
                candles.push_back(current_candle);
                initialize_next_candle(current_candle, tick);
            } else {
                update_candle(current_candle, tick);
            }
        }

        if (!is_first_tick) {
            candles.push_back(current_candle);
        }

        return candles;
    }

private:
    std::chrono::seconds bar_duration_;

    // Parses a single line of CSV.
    // Expected format: timestamp (Unix nano),price,volume
    Tick parse_line(std::string_view line) {
        Tick tick;
        auto comma1 = line.find(',');
        auto comma2 = line.find(',', comma1 + 1);

        // Timestamp
        long long timestamp_ns;
        std::from_chars(line.data(), line.data() + comma1, timestamp_ns);
        tick.timestamp = std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds>(
            std::chrono::nanoseconds(timestamp_ns));

        // Price
        std::from_chars(line.data() + comma1 + 1, line.data() + comma2, tick.price);

        // Volume
        std::from_chars(line.data() + comma2 + 1, line.data() + line.size(), tick.volume);

        return tick;
    }

    void initialize_first_candle(Candle& candle, const Tick& tick) {
        auto timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(tick.timestamp.time_since_epoch());
        auto bar_duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(bar_duration_);
        auto bucket = timestamp_ns.count() / bar_duration_ns.count();
        
        candle.open_time = std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds>(
            std::chrono::nanoseconds(bucket * bar_duration_ns.count()));

        candle.open = tick.price;
        candle.high = tick.price;
        candle.low = tick.price;
        candle.close = tick.price;
        candle.volume = tick.volume;
    }

    void initialize_next_candle(Candle& candle, const Tick& tick) {
        initialize_first_candle(candle, tick);
    }

    void update_candle(Candle& candle, const Tick& tick) {
        if (tick.price > candle.high) candle.high = tick.price;
        if (tick.price < candle.low) candle.low = tick.price;
        candle.close = tick.price;
        candle.volume += tick.volume;
    }
};

#endif // TICK_AGGREGATOR_H