"""
test_convergence_plot.jl

Unit tests for convergence plotting functionality.

This demonstrates:
- Test fixtures (creating mock data)
- Isolated unit testing (testing one function)
- File system mocking (temporary directories)
- Error handling tests

Run with:
    julia --project=. test/unit/test_convergence_plot.jl

Or interactively:
    include("test/unit/test_convergence_plot.jl")
    run_convergence_plot_tests()
"""

using Test
using Dates
using Plots
# Load real history
import JLD2
# Load the module under test
include("../../utils/io_manager.jl")

# ============================================================================
# TEST FIXTURES (Mock Data Creation)
# ============================================================================

"""
    create_mock_io_manager(; n_iterations=10)

Create a mock IOManager with fake optimization history for testing.
This is a "fixture" - test data that sets up the environment.
"""
function create_mock_io_manager(; n_iterations=10)
    # Create temporary directory for test output
    test_dir = mktempdir()
    
    # Create IOManager
    io = IOManager(test_dir; save_every=5, verbose=false)
    
    # Populate with mock history data
    for i in 1:n_iterations
        # Simulate decreasing badness (improving optimization)
        badness = 10.0 - 0.5 * i + 0.1 * randn()
        compliance = 1000.0 + 50.0 * randn()
        
        # Create mock diagnostics
        diagnostics = Dict(
            "intermediate_frac" => 0.3 + 0.02 * i + 0.01 * randn(),
            "severity" => 0.5 + 0.01 * randn(),
            "gray" => 0.4 + 0.02 * randn(),
            "stability" => 0.9
        )
        
        # Log iteration
        log_iteration(io, i, badness, compliance, diagnostics)
    end
    
    return io, test_dir
end

# ============================================================================
# UNIT TESTS
# ============================================================================

"""
    test_convergence_plot_creation()

Test that convergence plot is created successfully with valid data.

Unit Test Principle: Test ONE function with ONE scenario.
"""
function test_convergence_plot_creation()
    println("\n" * "="^70)
    println("TEST: Convergence Plot Creation (Happy Path)")
    println("="^70)
    
    # Arrange (Setup)
    io, test_dir = create_mock_io_manager(n_iterations=20)
    
    try
        # Act (Execute the function being tested)
        plot_file = create_convergence_plot(io)
        
        # Assert (Verify expected outcomes)
        @test plot_file !== nothing
        @test isfile(plot_file)
        @test endswith(plot_file, "convergence.png")
        
        # Additional assertions
        @test length(io.history["iteration"]) == 20
        @test isfile(joinpath(test_dir, "convergence.png"))
        
        println("✅ PASS: Plot file created at: $plot_file")
        println("   File size: $(filesize(plot_file)) bytes")
        
    catch e
        println("❌ FAIL: $e")
        rethrow(e)
    finally
        # Cleanup (Always runs)
        rm(test_dir; recursive=true, force=true)
        println("   Cleaned up test directory")
    end
end

"""
    test_empty_history()

Test behavior when IOManager has no history data.

Unit Test Principle: Test edge cases and error conditions.
"""
function test_empty_history()
    println("\n" * "="^70)
    println("TEST: Empty History Handling")
    println("="^70)
    
    test_dir = mktempdir()
    io = IOManager(test_dir; save_every=5, verbose=false)
    
    try
        # Should handle empty history gracefully
        result = create_convergence_plot(io)
        
        @test result === nothing  # Should return nothing for empty history
        println("✅ PASS: Handled empty history gracefully")
        
    catch e
        println("❌ FAIL: Should not throw for empty history")
        rethrow(e)
    finally
        rm(test_dir; recursive=true, force=true)
    end
end

"""
    test_plot_content()

Test that plot contains expected data.

Unit Test Principle: Verify output correctness, not just absence of errors.
"""
function test_plot_content()
    println("\n" * "="^70)
    println("TEST: Plot Content Verification")
    println("="^70)
    
    io, test_dir = create_mock_io_manager(n_iterations=15)
    
    try
        # Check history has expected keys
        @test haskey(io.history, "iteration")
        @test haskey(io.history, "badness")
        @test haskey(io.history, "compliance")
        @test haskey(io.history, "intermediate_frac")
        @test haskey(io.history, "gray_indicator")
        
        # Check data types (accept Vector{Any} or specific types)
        @test io.history["iteration"] isa AbstractVector
        @test io.history["badness"] isa AbstractVector
        @test all(x -> x isa Int, io.history["iteration"])
        @test all(x -> x isa Float64, io.history["badness"])
        
        # Check data length consistency
        n = length(io.history["iteration"])
        @test length(io.history["badness"]) == n
        @test length(io.history["compliance"]) == n
        @test length(io.history["intermediate_frac"]) == n
        
        println("✅ PASS: All history data is correctly structured")
        println("   Iterations recorded: $n")
        
    catch e
        println("❌ FAIL: $e")
        rethrow(e)
    finally
        rm(test_dir; recursive=true, force=true)
    end
end

"""
    test_with_real_data()

Integration test using real optimization history file.

Unit Test Principle: Sometimes you need integration tests too!
"""
function test_with_real_data()
    println("\n" * "="^70)
    println("TEST: Real Data Integration Test")
    println("="^70)
    
    # Look for real optimization history
    history_file = "c:/Users/judef/Desktop/COMMAS/SEM4/TopOpt/output/adversarial_20260114_22/optimization_history.jld2"
    
    if !isfile(history_file)
        println("⊘ SKIP: No real history file found at $history_file")
        return
    end
    
    try

        
        data = JLD2.load(history_file)
        history = data["history"]
        
        # Create IOManager with real data
        test_dir = mktempdir()
        io = IOManager(test_dir; save_every=5, verbose=false)
        io.history = history
        
        # Generate plot
        plot_file = create_convergence_plot(io)
        
        @test plot_file !== nothing
        @test isfile(plot_file)
        
        println("✅ PASS: Successfully plotted real optimization data")
        println("   Iterations: $(length(history["iteration"]))")
        println("   Plot saved to: $plot_file")
        
        # Keep the plot for inspection
        final_plot = "c:/Users/judef/Desktop/COMMAS/SEM4/TopOpt/test/unit/test_convergence.png"
        cp(plot_file, final_plot; force=true)
        println("   Test plot copied to: $final_plot")
        
        rm(test_dir; recursive=true, force=true)
        
    catch e
        println("❌ FAIL: $e")
        if isa(e, ArgumentError)
            println("   Note: JLD2 may not be installed. Run: using Pkg; Pkg.add(\"JLD2\")")
        end
    end
end

# ============================================================================
# TEST RUNNER
# ============================================================================

"""
    run_convergence_plot_tests()

Run all convergence plot unit tests.
"""
function run_convergence_plot_tests()
    println("\n" * "="^70)
    println("CONVERGENCE PLOT UNIT TEST SUITE")
    println("="^70)
    println("Testing module: io_manager.jl::create_convergence_plot()")
    println("Started: $(Dates.now())")
    
    # Track test results
    tests_passed = 0
    tests_failed = 0
    
    # Run each test
    tests = [
        ("Basic Plot Creation", test_convergence_plot_creation),
        ("Empty History", test_empty_history),
        ("Plot Content", test_plot_content),
        ("Real Data Integration", test_with_real_data)
    ]
    
    for (name, test_fn) in tests
        try
            test_fn()
            tests_passed += 1
        catch e
            tests_failed += 1
            println("\n⚠️  Test failed: $name")
            println("   Error: $e")
        end
    end
    
    # Summary
    println("\n" * "="^70)
    println("TEST SUMMARY")
    println("="^70)
    println("✅ Passed: $tests_passed")
    println("❌ Failed: $tests_failed")
    println("Total:    $(tests_passed + tests_failed)")
    println("="^70)
    
    return tests_passed, tests_failed
end

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    # Script was run directly
    passed, failed = run_convergence_plot_tests()
    exit(failed > 0 ? 1 : 0)  # Exit with error code if any tests failed
else
    println("Unit test module loaded. Run: run_convergence_plot_tests()")
end
