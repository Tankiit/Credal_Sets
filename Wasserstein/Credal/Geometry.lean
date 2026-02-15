/-
  Credal/Geometry.lean
  Identity: Mean squared deviation equals the trace of the diagonal covariance.
-/

import Mathlib
import Mathlib.Data.Real.Basic
import Mathlib.LinearAlgebra.Matrix.Trace

noncomputable section -- Must be above the definitions using ℝ

open scoped BigOperators
open scoped Matrix

namespace Credal

-- This line is crucial! It defines H and K for everything following it.
variable {K H : ℕ}

/-- Use `Vec K` for vectors in ℝ^K. -/
abbrev Vec (K : ℕ) := (Fin K → ℝ)

/-- Squared Euclidean norm on `Vec K`. -/
def norm2Sq (v : Vec K) : ℝ :=
  ∑ k : Fin K, (v k) ^ 2

/-- Pointwise mean of `H` vectors `p : Fin H → Vec K`. -/
def meanVec (p : Fin H → Vec K) : Vec K :=
  fun k => (1 / (H : ℝ)) * ∑ h : Fin H, p h k

/-- Per-coordinate variance: σ²_k = (1/H) ∑_h (p_hk - μ_k)^2. -/
def varCoord (p : Fin H → Vec K) (k : Fin K) : ℝ :=
  (1 / (H : ℝ)) * ∑ h : Fin H, (p h k - meanVec p k) ^ 2

/-- Diagonal covariance matrix Σ = diag(σ²_1,...,σ²_K). -/
def SigmaDiag (p : Fin H → Vec K) : Matrix (Fin K) (Fin K) ℝ :=
  Matrix.diagonal (fun k => varCoord p k)

/-- Trace of Σ for a diagonal matrix is sum of diagonal. -/
lemma trace_SigmaDiag (p : Fin H → Vec K) :
    Matrix.trace (SigmaDiag p) = ∑ k : Fin K, varCoord p k := by
  unfold SigmaDiag
  rw [Matrix.trace_diagonal]

theorem mean_sq_dev_eq_trace (p : Fin H → Vec K) :
    (1 / (H : ℝ)) * ∑ h : Fin H, norm2Sq (fun k => p h k - meanVec p k)
      = Matrix.trace (SigmaDiag p) := by
  classical
  -- Use the trace lemma to rewrite the RHS as a sum of variances
  rw [trace_SigmaDiag]

  -- Unfold everything so both sides become (1/H) times a double sum
  unfold norm2Sq varCoord

  -- Goal is now:
  -- (1/H) * ∑ h, ∑ k, (p h k - meanVec p k)^2
  --   = ∑ k, (1/H) * ∑ h, (p h k - meanVec p k)^2

  -- Pull (1/H) out of the RHS sum over k
  rw [← Finset.mul_sum]

  -- Now both sides have the same outer factor (1/H); reduce to a double-sum swap
  congr 1

  -- Swap the order of summation
  simpa using (Finset.sum_comm :
    (∑ h : Fin H, ∑ k : Fin K, (p h k - meanVec p k) ^ 2)
      =
    (∑ k : Fin K, ∑ h : Fin H, (p h k - meanVec p k) ^ 2))

end Credal
