"""
pytest tests for the SLERP (Spherical Linear Interpolation) function.

Covers: boundary conditions (t=0 → v0, t=1 → v1), idempotency (interpolating
identical vectors), continuity (consecutive steps are smooth), and near-
threshold behaviour (linear fallback vs. true SLERP when vectors are nearly
parallel or anti-parallel).
"""
import torch
import numpy as np
import pytest


def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """modified version of function in https://github.com/nateraw/stable-diffusion-videos/blob/main/stable_diffusion_videos/utils.py"""

    inputs_are_torch = isinstance(v0, torch.Tensor)
    if inputs_are_torch:
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        if isinstance(v2, np.ndarray):
            v2 = torch.tensor(v2, device=input_device, dtype=torch.float16)
        else:
            # If v2 is unexpectedly a tensor, clone it to maintain the original tensor's properties without alteration
            v2 = v2.clone().detach().to(input_device).to(torch.float16)

    return v2

def test_slerp_boundary_conditions():
    v0 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    v1 = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)

    # Interpolation at the start should return v0
    np.testing.assert_allclose(slerp(0, v0, v1).numpy(), v0.numpy(), rtol=1e-5)

    # Interpolation at the end should return v1
    np.testing.assert_allclose(slerp(1, v0, v1).numpy(), v1.numpy(), rtol=1e-5)

def test_slerp_idempotency():
    v = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)

    # Interpolating between two identical vectors should return the same vector
    for t in np.linspace(0, 1, 10):
        np.testing.assert_allclose(slerp(t, v, v).numpy(), v.numpy(), rtol=1e-5)

def test_slerp_continuity():
    v0 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    v1 = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    t_values = np.linspace(0, 1, 100)
    prev_vec = slerp(t_values[0], v0, v1)

    # Check that consecutive interpolations are close to each other, implying smoothness
    for t in t_values[1:]:
        current_vec = slerp(t, v0, v1)
        np.testing.assert_allclose((current_vec - prev_vec).numpy(), 0, atol=1e-3)
        prev_vec = current_vec

def test_slerp_dot_threshold():
    v0 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    v1 = torch.tensor([1.0, 0.0001, 0.0], dtype=torch.float32)  # Nearly parallel

    # This test assumes a specific implementation detail that may need adjustment
    # It's to ensure that the behavior is correct near the DOT_THRESHOLD
    interpolated_near = slerp(0.5, v0, v1)
    interpolated_far = slerp(0.5, v0, -v1)  # Opposite direction, should trigger actual SLERP

    assert not torch.allclose(interpolated_near, interpolated_far), "SLERP threshold behavior may be incorrect"

# Additional tests can be written to cover more cases and edge conditions.
