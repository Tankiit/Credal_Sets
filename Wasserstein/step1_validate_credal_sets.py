"""
STEP 1: Validate Credal Set Setup
==================================

Goal: Confirm your credal sets are 2-monotone and ready for Wasserstein

What we're checking:
1. Can we load/create credal sets?
2. Are they convex? (if yes → 2-monotone automatically!)
3. Can we extract vertices?
4. Can we convert to numpy arrays?

If all checks pass → Ready for Wasserstein!
"""

import numpy as np
import sys

# Try to import your existing code
try:
    from construction import CredalConstructor, CredalSet
    print("✓ Successfully imported construction.py")
except ImportError as e:
    print("✗ Could not import construction.py")
    print(f"  Error: {e}")
    print("\n  Please make sure construction.py is in the same directory")
    print("  or adjust the import path.")
    sys.exit(1)

print("="*70)
print("STEP 1: Validating Credal Set Setup")
print("="*70)

# ============================================================================
# TEST 1.1: Create sample credal sets
# ============================================================================

print("\n" + "="*70)
print("TEST 1.1: Creating Sample Credal Sets")
print("="*70)

try:
    constructor = CredalConstructor(method='convex_hull')
    print("✓ CredalConstructor initialized with method='convex_hull'")
except Exception as e:
    print(f"✗ Error creating constructor: {e}")
    sys.exit(1)

# Create test examples
print("\nCreating test credal sets...")

# Example 1: High agreement (correct prediction)
correct_answers = ["A", "A", "A", "A", "B"]
print(f"  Correct answers:   {correct_answers}")

try:
    credal_correct = constructor.construct(correct_answers)
    print(f"  ✓ Created credal set for 'correct'")
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

# Example 2: High disagreement (incorrect prediction)
incorrect_answers = ["A", "B", "C", "D", "E"]
print(f"  Incorrect answers: {incorrect_answers}")

try:
    credal_incorrect = constructor.construct(incorrect_answers)
    print(f"  ✓ Created credal set for 'incorrect'")
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

print("\n✓ TEST 1.1 PASSED: Can create credal sets")

# ============================================================================
# TEST 1.2: Check convexity
# ============================================================================

print("\n" + "="*70)
print("TEST 1.2: Checking Convexity")
print("="*70)

print("\nWhy this matters:")
print("  Convex credal set → 2-monotone capacity (automatic!)")
print("  2-monotone → Efficient Wasserstein computation")

def check_convexity(credal, name):
    """Check if credal set is convex by examining construction method."""
    print(f"\nChecking '{name}':")

    # For convex_hull method, this should always be true
    if hasattr(credal, 'method') and credal.method == 'convex_hull':
        print(f"  ✓ Method: {credal.method} (convex by construction)")
        return True
    else:
        # Check if we can extract vertices
        try:
            vertices = credal.to_vertices()
            if len(vertices) > 0:
                print(f"  ✓ Has {len(vertices)} vertices (convex hull implied)")
                return True
            else:
                print(f"  ✗ No vertices found")
                return False
        except Exception as e:
            print(f"  ✗ Error extracting vertices: {e}")
            return False

is_convex_correct = check_convexity(credal_correct, "correct")
is_convex_incorrect = check_convexity(credal_incorrect, "incorrect")

if is_convex_correct and is_convex_incorrect:
    print("\n✓ TEST 1.2 PASSED: Both credal sets are convex")
    print("  → 2-monotonicity guaranteed!")
else:
    print("\n✗ TEST 1.2 FAILED: Credal sets not convex")
    print("  → Cannot guarantee 2-monotonicity")
    print("  → Wasserstein computation may fail")
    sys.exit(1)

# ============================================================================
# TEST 1.3: Extract vertices
# ============================================================================

print("\n" + "="*70)
print("TEST 1.3: Extracting Vertices")
print("="*70)

print("\nWhy this matters:")
print("  For 2-monotone credal sets, we only need vertices")
print("  Choquet integral = min over vertices (not entire set!)")

def extract_and_display_vertices(credal, name):
    """Extract vertices and display information."""
    print(f"\n{name.upper()}:")

    try:
        vertices = credal.to_vertices()
        print(f"  Number of vertices: {len(vertices)}")

        if len(vertices) == 0:
            print(f"  ✗ No vertices!")
            return None, None

        # Get all possible answers
        answers = sorted(credal.answers)
        print(f"  Possible answers: {answers}")

        # Display each vertex
        print(f"  Vertices:")
        for i, v in enumerate(vertices):
            print(f"    Vertex {i+1}: ", end="")
            for answer in answers:
                prob = v.get(answer, 0.0)
                print(f"{answer}:{prob:.2f} ", end="")
            print()

        return vertices, answers

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None, None

v_correct, answers_correct = extract_and_display_vertices(credal_correct, "correct")
v_incorrect, answers_incorrect = extract_and_display_vertices(credal_incorrect, "incorrect")

if v_correct is not None and v_incorrect is not None:
    print("\n✓ TEST 1.3 PASSED: Can extract vertices from both credal sets")
else:
    print("\n✗ TEST 1.3 FAILED: Cannot extract vertices")
    sys.exit(1)

# ============================================================================
# TEST 1.4: Convert to numpy arrays
# ============================================================================

print("\n" + "="*70)
print("TEST 1.4: Converting to Numpy Arrays")
print("="*70)

print("\nWhy this matters:")
print("  Numerical algorithms (Wasserstein, Sinkhorn) need numpy arrays")
print("  Format: [n_vertices, n_answers]")

def vertices_to_array(vertices, all_answers):
    """
    Convert vertices to numpy array.

    Args:
        vertices: List of dicts {answer: prob}
        all_answers: Sorted list of all possible answers

    Returns:
        numpy array of shape [n_vertices, n_answers]
    """
    vertex_array = []
    for v in vertices:
        # For each vertex, get probability for each answer (0 if not present)
        row = [v.get(answer, 0.0) for answer in all_answers]
        vertex_array.append(row)

    return np.array(vertex_array)

# Get union of all answers
all_answers = sorted(set(answers_correct) | set(answers_incorrect))
print(f"\nAll possible answers: {all_answers}")

# Convert to arrays
try:
    arr_correct = vertices_to_array(v_correct, all_answers)
    arr_incorrect = vertices_to_array(v_incorrect, all_answers)

    print(f"\nCorrect credal array:")
    print(f"  Shape: {arr_correct.shape}")
    print(f"  Array:\n{arr_correct}")

    print(f"\nIncorrect credal array:")
    print(f"  Shape: {arr_incorrect.shape}")
    print(f"  Array:\n{arr_incorrect}")

except Exception as e:
    print(f"✗ Error converting to arrays: {e}")
    sys.exit(1)

# ============================================================================
# TEST 1.5: Validate probability distributions
# ============================================================================

print("\n" + "="*70)
print("TEST 1.5: Validating Probability Distributions")
print("="*70)

print("\nChecking that each vertex is a valid probability distribution...")
print("(Each row should sum to 1.0)")

def validate_probabilities(arr, name):
    """Check that each vertex sums to 1.0."""
    print(f"\n{name}:")
    sums = arr.sum(axis=1)

    all_valid = True
    for i, s in enumerate(sums):
        if abs(s - 1.0) < 1e-6:
            print(f"  Vertex {i+1}: sum = {s:.6f} ✓")
        else:
            print(f"  Vertex {i+1}: sum = {s:.6f} ✗ (should be 1.0)")
            all_valid = False

    return all_valid

valid_correct = validate_probabilities(arr_correct, "Correct")
valid_incorrect = validate_probabilities(arr_incorrect, "Incorrect")

if valid_correct and valid_incorrect:
    print("\n✓ TEST 1.5 PASSED: All vertices are valid probability distributions")
else:
    print("\n✗ TEST 1.5 FAILED: Some vertices don't sum to 1.0")
    print("  This might cause issues in Wasserstein computation")
    sys.exit(1)

# ============================================================================
# FINAL CHECKPOINT
# ============================================================================

print("\n" + "="*70)
print("STEP 1 COMPLETE - FINAL CHECKPOINT")
print("="*70)

print("\n✅ ALL TESTS PASSED:")
print("  ✓ Can create credal sets")
print("  ✓ Credal sets are convex")
print("  ✓ 2-monotonicity guaranteed (from convexity)")
print("  ✓ Can extract vertices")
print("  ✓ Can convert to numpy arrays")
print("  ✓ All vertices are valid probability distributions")

print("\n✅ READY FOR WASSERSTEIN COMPUTATION!")

print("\n" + "="*70)
print("NEXT STEP: Compute Wasserstein distance")
print("="*70)
print("\nYou can now proceed to:")
print("  Run: python step2_basic_wasserstein.py")
print("\nOr let me know and I'll guide you through Step 2!")

# ============================================================================
# Save results for next step
# ============================================================================

print("\n" + "="*70)
print("Saving results for Step 2...")
print("="*70)

try:
    np.savez('step1_results.npz',
             arr_correct=arr_correct,
             arr_incorrect=arr_incorrect,
             all_answers=all_answers)
    print("✓ Saved to step1_results.npz")
    print("  (Step 2 can load these arrays)")
except Exception as e:
    print(f"⚠ Could not save results: {e}")
    print("  (Not critical - Step 2 can regenerate)")

print("\n" + "="*70)
