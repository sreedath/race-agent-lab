"""Export a trained SB3 PPO policy to a policy.json the race arena can run.

Dumps mlp_extractor.policy_net + action_net as row-major weight matrices.
Deterministic action = clip(mean, -1, 1) (SB3 Box PPO uses the clipped
Gaussian mean; there is NO tanh on the output). Embeds test vectors so the
browser can self-verify at load, plus your agent's display name.

From the notebook:  export_from_model(model, "SpeedyMcSpeedFace")
From a shell:       python -m racing.train.export_policy \
                        --model runs/ppo_race --name "SpeedyMcSpeedFace"
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parent.parent.parent
NUM_TEST_VECTORS = 5


def extract_layers(model: PPO):
    """Ordered [(W, b, activation), ...] for the deterministic actor path."""
    layers = []
    policy_net = model.policy.mlp_extractor.policy_net
    for module in policy_net:
        if isinstance(module, torch.nn.Linear):
            layers.append(
                {
                    "W": module.weight.detach().numpy(),
                    "b": module.bias.detach().numpy(),
                    "activation": None,
                }
            )
        elif isinstance(module, torch.nn.Tanh):
            layers[-1]["activation"] = "tanh"
        else:
            raise ValueError(f"unsupported module in policy_net: {module}")
    action_net = model.policy.action_net
    layers.append(
        {
            "W": action_net.weight.detach().numpy(),
            "b": action_net.bias.detach().numpy(),
            "activation": None,
        }
    )
    return layers


def forward_reference(layers, obs: np.ndarray) -> np.ndarray:
    """NumPy forward pass; must equal both SB3 predict and the JS mlp."""
    h = obs
    for layer in layers:
        h = layer["W"] @ h + layer["b"]
        if layer["activation"] == "tanh":
            h = np.tanh(h)
    return np.clip(h, -1.0, 1.0)


def export_from_model(model: PPO, name: str, out_path=None) -> Path:
    """Export an in-memory PPO model (e.g. right after training) to JSON."""
    if not name or not str(name).strip():
        raise ValueError("give your agent a name (shown in the race)")
    name = str(name).strip()[:24]
    layers = extract_layers(model)
    obs_dim = layers[0]["W"].shape[1]

    # Cross-check against SB3's own deterministic predict on random obs.
    rng = np.random.default_rng(123)
    test_vectors = []
    for _ in range(NUM_TEST_VECTORS):
        obs = rng.uniform(-1.0, 1.0, size=obs_dim).astype(np.float64)
        ours = forward_reference(layers, obs)
        sb3, _ = model.predict(obs.astype(np.float32), deterministic=True)
        assert np.max(np.abs(ours - sb3)) < 1e-5, "export mismatch vs SB3 predict"
        test_vectors.append({"obs": obs.tolist(), "action": ours.tolist()})

    out = {
        "name": name,
        "obs_dim": obs_dim,
        "act_dim": layers[-1]["W"].shape[0],
        "speed_scale": 1.0,
        "layers": [
            {
                "W": layer["W"].astype(np.float64).tolist(),  # row-major [out][in]
                "b": layer["b"].astype(np.float64).tolist(),
                "activation": layer["activation"],
            }
            for layer in layers
        ],
        "test_vectors": test_vectors,
    }
    safe_stem = "".join(c if c.isalnum() or c in "-_ " else "_" for c in name)
    safe_stem = safe_stem.strip().replace(" ", "_") or "agent"
    path = Path(out_path) if out_path else ROOT / f"{safe_stem}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f)
        f.write("\n")
    n_params = sum(layer["W"].size + layer["b"].size for layer in layers)
    print(f"wrote {path} ({n_params} params, {path.stat().st_size // 1024} KiB)")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=str(ROOT / "runs" / "ppo_race"))
    parser.add_argument("--name", type=str, required=True, help="agent display name")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    model = PPO.load(args.model, device="cpu")
    export_from_model(model, args.name, args.out)


if __name__ == "__main__":
    main()
