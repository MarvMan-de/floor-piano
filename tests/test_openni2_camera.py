"""Tests for the OpenNI2 Astra Pro backend + camera-backend factory.

The real depth path can only be exercised with the camera attached (see
docs/ASTRA_PRO_PI5_SETUP.md). What *is* testable without hardware: the factory
picks the right backend from the env var, and the backend fails loudly and
correctly when used out of order or without the `openni` bindings installed.
"""
import pytest

import depth_camera as dc


def test_factory_defaults_to_orbbec(monkeypatch):
    monkeypatch.delenv("FLOOR_PIANO_CAMERA", raising=False)
    assert isinstance(dc.make_depth_camera(), dc.DepthCamera)


@pytest.mark.parametrize("value", ["openni2", "OpenNI2", " openni ", "astra", "astrapro"])
def test_factory_selects_openni2(monkeypatch, value):
    monkeypatch.setenv("FLOOR_PIANO_CAMERA", value)
    assert isinstance(dc.make_depth_camera(), dc.OpenNI2DepthCamera)


def test_factory_unknown_falls_back_to_orbbec(monkeypatch):
    monkeypatch.setenv("FLOOR_PIANO_CAMERA", "potato")
    assert isinstance(dc.make_depth_camera(), dc.DepthCamera)


def test_read_before_start_raises():
    cam = dc.OpenNI2DepthCamera()
    with pytest.raises(dc.DepthCameraError):
        cam.read_depth()


def test_start_without_bindings_raises_clean_error():
    # `openni` is not installed in the dev/test env, so start() must surface a
    # DepthCameraError with an actionable hint — never a raw ImportError.
    if _openni_installed():
        pytest.skip("openni bindings are installed; the no-bindings path can't be tested here")
    cam = dc.OpenNI2DepthCamera()
    with pytest.raises(dc.DepthCameraError) as exc:
        cam.start()
    assert "openni" in str(exc.value).lower()


def test_stop_is_safe_before_start():
    # stop() during cleanup must never raise even if nothing was opened.
    dc.OpenNI2DepthCamera().stop()


def _openni_installed():
    try:
        import openni  # noqa: F401
        return True
    except Exception:
        return False
