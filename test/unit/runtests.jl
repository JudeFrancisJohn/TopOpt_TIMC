"""
runtests.jl

Master test runner for all unit tests.

This file discovers and runs all test_*.jl files in the unit test directory.

Usage:
    julia --project=. test/unit/runtests.jl
"""

using Test
using Dates

println("="^80)
println("TOPOPT UNIT TEST SUITE")
println("="^80)
println("Started: $(Dates.now())")
println()

# Discover all test files
test_dir = @__DIR__
test_files = filter(f -> startswith(f, "test_") && endswith(f, ".jl"), readdir(test_dir))

println("Found $(length(test_files)) test file(s):")
for f in test_files
    println("  - $f")
end
println()

# Run each test file
total_passed = 0
total_failed = 0

for test_file in test_files
    println("\n" * "="^80)
    println("Running: $test_file")
    println("="^80)
    
    try
        include(test_file)
        
        # Assume each test file defines run_*_tests() function
        test_name = replace(test_file, "test_" => "", ".jl" => "")
        runner_fn = Symbol("run_$(test_name)_tests")
        
        if isdefined(Main, runner_fn)
            passed, failed = getfield(Main, runner_fn)()
            total_passed += passed
            total_failed += failed
        else
            println("⚠️  Warning: No runner function $runner_fn found")
        end
        
    catch e
        println("❌ ERROR running $test_file:")
        showerror(stdout, e, catch_backtrace())
        println()
        total_failed += 1
    end
end

# Final summary
println("\n" * "="^80)
println("FINAL TEST SUMMARY")
println("="^80)
println("✅ Total Passed: $total_passed")
println("❌ Total Failed: $total_failed")
println("📊 Total Tests:  $(total_passed + total_failed)")
println("⏱️  Completed:   $(Dates.now())")
println("="^80)

# Exit with appropriate code
exit(total_failed > 0 ? 1 : 0)
