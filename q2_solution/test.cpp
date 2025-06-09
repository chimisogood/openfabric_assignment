#include "gtest/gtest.h"
#include "TickAggregator.h"
#include <string>

TEST(TickAggregatorTest, BasicAggregation) {
    TickAggregator aggregator(13); // 13-second bars
    std::string csv_data =
        "1672531200000000000,100.0,10\n" // 2023-01-01 00:00:00
        "1672531201000000000,102.5,5\n"  // 2023-01-01 00:00:01
        "1672531212000000000,99.0,8\n"   // 2023-01-01 00:00:12
        "1672531213000000000,105.0,12\n" // 2023-01-01 00:00:13 (New Bar)
        "1672531215000000000,106.0,20\n"; // 2023-01-01 00:00:15

    auto candles = aggregator.process_csv(csv_data);

    ASSERT_EQ(candles.size(), 2);

    Candle expected_candle1;
    expected_candle1.open_time = std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds>(
        std::chrono::nanoseconds(1672531200000000000 / (13 * 1000000000LL) * (13 * 1000000000LL)));
    expected_candle1.open = 100.0;
    expected_candle1.high = 102.5;
    expected_candle1.low = 99.0;
    expected_candle1.close = 99.0;
    expected_candle1.volume = 23;

    Candle expected_candle2;
    expected_candle2.open_time = std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds>(
        std::chrono::nanoseconds(1672531213000000000 / (13 * 1000000000LL) * (13 * 1000000000LL)));
    expected_candle2.open = 105.0;
    expected_candle2.high = 106.0;
    expected_candle2.low = 105.0;
    expected_candle2.close = 106.0;
    expected_candle2.volume = 32;


    EXPECT_EQ(candles[0], expected_candle1);
    EXPECT_EQ(candles[1], expected_candle2);
}

TEST(TickAggregatorTest, EmptyInput) {
    TickAggregator aggregator(13);
    auto candles = aggregator.process_csv("");
    EXPECT_TRUE(candles.empty());
}

TEST(TickAggregatorTest, SingleTick) {
    TickAggregator aggregator(13);
    std::string csv_data = "1672531200000000000,100.0,10\n";
    auto candles = aggregator.process_csv(csv_data);
    ASSERT_EQ(candles.size(), 1);
    EXPECT_EQ(candles[0].open, 100.0);
    EXPECT_EQ(candles[0].volume, 10);
} 
