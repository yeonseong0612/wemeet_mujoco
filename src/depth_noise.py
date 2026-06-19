"""[Noise-1] Intel RealSense D435 기준 depth/RGB 노이즈 모델 (CLAUDE.md §5).

모든 노이즈는 파이프라인 외부에서 주입한다 — GT depth 와 noisy depth 를 독립적으로
보관할 수 있게, 입력을 변형하지 않고 항상 새 배열을 반환한다.

구성:
    5-1 gaussian_depth_noise : 거리 비례 가우시안 (σ ≈ 0.001 + 0.0005·d)
    5-2 apply_depth_holes    : dark/reflective 표면 무효값 + 경계 edge noise (seg_mask 기반)
    5-3 quantize_depth       : 1 mm 양자화
    5-4 apply_rgb_noise      : RGB 가우시안 + 밝기 지터 (선택)
    NOISE_PRESETS            : none / low / medium / high (ablation)
    apply_preset             : preset 이름으로 위 요소를 조합 주입
"""
import numpy as np
import cv2


# ---------------------------------------------------------------------------
# 5-1. Gaussian Depth Noise (거리 비례)
# ---------------------------------------------------------------------------
def gaussian_depth_noise(depth: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    depth: (H, W) float32, metric meter
    scale: 노이즈 강도 배율 (1.0 = D435 기준)
    """
    sigma = (0.001 + 0.0005 * depth) * scale
    noise = np.random.normal(0, sigma, depth.shape).astype(np.float32)
    return np.clip(depth + noise, 0, None)


# ---------------------------------------------------------------------------
# 5-2. Depth Hole (반사 / 검정 플라스틱)
# ---------------------------------------------------------------------------
def apply_depth_holes(depth: np.ndarray,
                      seg_mask: np.ndarray,
                      hole_rate: float = 0.20,
                      edge_thickness: int = 3) -> np.ndarray:
    """
    seg_mask: 충전구 영역 binary mask (H, W) bool
    hole_rate: 충전구 영역 내 무효값 비율 (D435 dark surface 기준 0.15~0.30)
    edge_thickness: 물체 경계 depth mixing 두께 (px)
    """
    depth_out = depth.copy()
    # 1) 충전구 영역 내 랜덤 hole
    port_pixels = np.where(seg_mask)
    n_port = len(port_pixels[0])
    if n_port > 0 and hole_rate > 0:
        n_holes = int(n_port * hole_rate)
        if n_holes > 0:
            idx = np.random.choice(n_port, n_holes, replace=False)
            depth_out[port_pixels[0][idx], port_pixels[1][idx]] = 0.0
    # 2) 경계 edge noise
    if edge_thickness > 0:
        edge = cv2.dilate(seg_mask.astype(np.uint8),
                          np.ones((edge_thickness, edge_thickness), np.uint8)) \
             - seg_mask.astype(np.uint8)
        edge_sel = edge.astype(bool)
        if edge_sel.any():
            depth_out[edge_sel] *= np.random.uniform(0.5, 1.0, depth_out[edge_sel].shape)
    return depth_out


# ---------------------------------------------------------------------------
# 5-3. Quantization
# ---------------------------------------------------------------------------
def quantize_depth(depth: np.ndarray, resolution: float = 0.001) -> np.ndarray:
    return (np.round(depth / resolution) * resolution).astype(np.float32)


# ---------------------------------------------------------------------------
# 5-4. RGB Noise (선택)
# ---------------------------------------------------------------------------
def apply_rgb_noise(rgb: np.ndarray,
                    gaussian_std: float = 5.0,
                    brightness_jitter: float = 0.1) -> np.ndarray:
    noisy = rgb.astype(np.float32)
    noisy += np.random.normal(0, gaussian_std, noisy.shape)
    noisy *= (1.0 + np.random.uniform(-brightness_jitter, brightness_jitter))
    return np.clip(noisy, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 노이즈 레벨 프리셋 (ablation용)
# ---------------------------------------------------------------------------
NOISE_PRESETS = {
    "none": {
        "depth_gaussian_scale": 0.0,
        "hole_rate":            0.00,
        "quantize":             False,
        "rgb_gaussian_std":     0.0,
    },
    "low": {
        "depth_gaussian_scale": 0.5,   # D435 절반
        "hole_rate":            0.05,
        "quantize":             True,
        "rgb_gaussian_std":     3.0,
    },
    "medium": {                         # D435 실측 기준
        "depth_gaussian_scale": 1.0,
        "hole_rate":            0.20,
        "quantize":             True,
        "rgb_gaussian_std":     5.0,
    },
    "high": {
        "depth_gaussian_scale": 2.0,   # D435 열화/원거리
        "hole_rate":            0.35,
        "quantize":             True,
        "rgb_gaussian_std":     10.0,
    },
}


def apply_preset(depth: np.ndarray,
                 seg_mask: np.ndarray,
                 preset: str = "medium",
                 rgb: np.ndarray = None):
    """preset 이름으로 depth(+선택 rgb)에 노이즈 요소를 순서대로 주입.

    순서: gaussian → holes(seg_mask) → quantize. rgb 가 주어지면 rgb noise 도 적용.
    입력은 변형하지 않고 새 배열을 반환한다.

    Returns: noisy_depth  또는  (noisy_depth, noisy_rgb)  (rgb 제공 시)
    """
    if preset not in NOISE_PRESETS:
        raise KeyError(f"unknown preset '{preset}' (choices: {list(NOISE_PRESETS)})")
    cfg = NOISE_PRESETS[preset]

    out = depth.astype(np.float32, copy=True)
    if cfg["depth_gaussian_scale"] > 0:
        out = gaussian_depth_noise(out, scale=cfg["depth_gaussian_scale"])
    if cfg["hole_rate"] > 0:
        out = apply_depth_holes(out, seg_mask, hole_rate=cfg["hole_rate"])
    if cfg["quantize"]:
        out = quantize_depth(out)

    if rgb is None:
        return out
    noisy_rgb = rgb
    if cfg["rgb_gaussian_std"] > 0:
        noisy_rgb = apply_rgb_noise(rgb, gaussian_std=cfg["rgb_gaussian_std"])
    return out, noisy_rgb
