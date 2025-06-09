Performance Optimizations and Design Choices

    Memory-Mapped I/O (Conceptual): The process_csv function takes a std::string_view. This is by design. In a real-world, high-performance scenario, you would memory-map your CSV file into memory and then construct a std::string_view that points to this memory. This avoids copying the file data into a std::string.
    std::string_view and std::from_chars: std::string_view allows processing substrings of the data without incurring memory allocation overhead. std::from_chars is the fastest standard way to convert strings to numbers as it is locale-independent and avoids the overhead of iostreams.
    Minimal State: The aggregator only keeps one Candle object in memory at a time during processing. This results in extremely low memory usage, regardless of the size of the input CSV.
    Efficient Time Arithmetic: Using std::chrono with nanosecond precision ensures accuracy. The candle bucketing logic (timestamp_ns.count() / bar_duration_ns.count()) is an efficient integer division operation to determine which time bucket a tick belongs to.
    Cache-Friendly Data Structures: The Tick and Candle structs are simple POD-like (Plain Old Data) structures, which are cache-friendly. The std::vector<Candle> stores the final results contiguously.


How to Compile and Run
You will need to have Google Test and Google Benchmark libraries installed.

1. Directory Structure:

tick_project/
|-- TickAggregator.h
|-- tests.cpp
|-- benchmarks.cpp
|-- CMakeLists.txt

