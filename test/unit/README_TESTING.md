# Unit Testing Guide

## Overview

This directory contains **unit tests** for the TopOpt project. Unit tests focus on testing individual functions or components in isolation.

## Testing Principles

### 1. **Unit Tests vs Integration Tests**

```
Unit Test:        Tests ONE function with controlled inputs
Integration Test: Tests how multiple components work together
End-to-End Test:  Tests the entire system from start to finish
```

**Current Structure:**
- `test/unit/` - Unit tests (isolated component testing)
- `test/` - Integration/E2E tests (full workflow testing)

### 2. **The AAA Pattern**

Every test should follow:

```julia
function test_something()
    # Arrange - Set up test data and conditions
    input = create_mock_data()
    
    # Act - Execute the function being tested
    result = function_to_test(input)
    
    # Assert - Verify expected outcomes
    @test result == expected_value
end
```

### 3. **Test Fixtures**

Fixtures are reusable test data:

```julia
function create_mock_io_manager(; n_iterations=10)
    # Creates standardized test data
    # Used by multiple tests
    # Ensures consistency
end
```

### 4. **Test Isolation**

Each test should:
- ✅ Run independently (not depend on other tests)
- ✅ Clean up after itself (remove temp files)
- ✅ Not modify global state
- ✅ Be repeatable (same result every time)

### 5. **What to Test**

**DO Test:**
- ✅ Core business logic
- ✅ Edge cases (empty input, null, negative numbers)
- ✅ Error conditions (what happens when things go wrong)
- ✅ Boundary values (min/max)

**DON'T Test:**
- ❌ Third-party library internals
- ❌ Language features
- ❌ Trivial getters/setters

## Running Tests

### Run All Unit Tests
```bash
julia --project=. test/unit/test_convergence_plot.jl
```

### Run Interactively (for debugging)
```julia
julia> include("test/unit/test_convergence_plot.jl")
julia> run_convergence_plot_tests()
```

### Run Specific Test
```julia
julia> include("test/unit/test_convergence_plot.jl")
julia> test_convergence_plot_creation()  # Just one test
```

## Writing New Tests

### Template

```julia
"""
test_my_feature.jl

Tests for my_feature functionality.
"""

using Test
include("../../utils/my_module.jl")

# Fixture
function create_mock_data()
    # Create test data
    return mock_data
end

# Unit test
function test_basic_case()
    # Arrange
    input = create_mock_data()
    
    # Act
    result = my_function(input)
    
    # Assert
    @test result == expected_value
end

# Test runner
function run_my_tests()
    @testset "My Feature Tests" begin
        test_basic_case()
        # Add more tests...
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_my_tests()
end
```

## Example: Testing Convergence Plot

### What We Test

1. **Happy Path** - Does it work with good data?
2. **Edge Cases** - What about empty data?
3. **Output Verification** - Is the file actually created?
4. **Content Validation** - Does it contain expected data?

### Test Structure

```julia
function test_convergence_plot_creation()
    # Arrange: Create mock IOManager with fake history
    io, test_dir = create_mock_io_manager(n_iterations=20)
    
    # Act: Generate plot
    plot_file = create_convergence_plot(io)
    
    # Assert: Verify output
    @test plot_file !== nothing
    @test isfile(plot_file)
    @test endswith(plot_file, "convergence.png")
    
    # Cleanup
    rm(test_dir; recursive=true, force=true)
end
```

## Best Practices

### ✅ DO

```julia
# Clear, descriptive test names
function test_handles_negative_coefficients()
    ...
end

# Test one thing per test
function test_file_creation()
    @test isfile(output)
end

function test_file_content()
    @test read(output) == expected_content
end

# Use fixtures for complex setup
data = create_mock_optimization_history()
```

### ❌ DON'T

```julia
# Vague names
function test1()
    ...
end

# Testing multiple things
function test_everything()
    @test result.a == 1
    @test result.b == 2
    @test result.c == 3
    # Too much in one test!
end

# Brittle tests (depend on external state)
function test_uses_global()
    global_var = 10  # Bad - modifies global state
    ...
end
```

## Test Coverage Goals

Aim for:
- **70-80%** code coverage (lines of code tested)
- **100%** critical path coverage (important functions)
- **All edge cases** covered

## Continuous Integration

When setting up CI/CD:

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: julia-actions/setup-julia@v1
      - name: Run tests
        run: julia --project=. -e 'using Pkg; Pkg.test()'
```

## Resources

- [Julia Testing Docs](https://docs.julialang.org/en/v1/stdlib/Test/)
- [Test-Driven Development (TDD)](https://en.wikipedia.org/wiki/Test-driven_development)
- [Martin Fowler on Testing](https://martinfowler.com/tags/testing.html)

## Next Steps

1. **Read** this guide
2. **Run** existing tests: `julia test/unit/test_convergence_plot.jl`
3. **Modify** a test to see what happens
4. **Write** your own test for a new feature
5. **Refactor** with confidence (tests catch regressions)
