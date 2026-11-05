"""(Re)generate the experiment grid in configs/exp/. Edit DATASETS / HOLDOUT here,
not the generated files.

    python scripts/make_grid.py
"""
from pathlib import Path

EXP = Path(__file__).resolve().parents[1] / "configs" / "exp"
DATASETS = ["taco", "zerowaste", "uavvaste", "mju", "bepli"]
HOLDOUT = ["Glass bottle", "Pop tab", "Plastic straw", "Crisp packet", "Paper bag", "Foam cup", "Egg carton"]


def write(name, header, body):
    (EXP / f"{name}.yaml").write_text(f"# {header}\n{body.strip()}\n")


def main():
    EXP.mkdir(parents=True, exist_ok=True)
    for d in DATASETS:
        write(f"e1_indomain_{d}", f"E1 in-domain: train and test on {d} only (conventional reference).",
              f"_base_: ../base.yaml\nrun_name: e1_indomain_{d}\ntrain_datasets: [{d}]\ntest_datasets: [{d}]\n"
              f"train: {{transplant_p: 0.0}}")
    write("e2_multi_domain", "E2 multi-domain: one model on every dataset, metrics per dataset.",
          "_base_: ../base.yaml\nrun_name: e2_multi_domain")
    for d in DATASETS:
        others = ", ".join(x for x in DATASETS if x != d)
        write(f"e3_lodo_{d}", f"E3 leave-one-dataset-out: {d} is never seen, not even for model selection.",
              f"_base_: ../base.yaml\nrun_name: e3_lodo_{d}\ntrain_datasets: [{others}]\n"
              f"val_datasets: [{others}]\ntest_datasets: [{d}]")
    hold = ", ".join(f'"{h}"' for h in HOLDOUT)
    write("e4_heldout_classes",
          "E4 class independence: these TACO classes never reach training (whole images dropped from train AND "
          "val);\n# they are named and retrieved at test time from their atlas entry alone. Each is composable "
          "from\n# descriptors other classes carry (Glass bottle = glass [jars, shards] x bottle-shaped "
          "[plastic bottles]).",
          f"_base_: ../base.yaml\nrun_name: e4_heldout_classes\nholdout: [{hold}]")
    ablations = {
        "no_transplant": ("without cross-dataset object transplant", "train: {transplant_p: 0.0}"),
        "no_adversary": ("without the environment adversary", "model: {adversary: false}\ntrain: {loss: {adv: 0.0}}"),
        "no_mixstyle": ("without MixStyle", "model: {mixstyle_p: 0.0}"),
        "no_debias": ("no transplant, no adversary, no MixStyle",
                      "model: {adversary: false, mixstyle_p: 0.0}\ntrain: {transplant_p: 0.0, loss: {adv: 0.0}}"),
        "no_dense": ("without dense pixel-level descriptor supervision", "train: {loss: {dense: 0.0}}"),
        "lora8": ("LoRA r=8 on qkv/proj instead of adapters", "model: {adapter_dim: 0, lora_rank: 8}"),
        "adapters_both": ("adapters after attention AND MLP (Houlsby)", "model: {adapter_where: both}"),
        "probe": ("closed-vocabulary linear probe instead of the open descriptor head (reference; naming n/a)",
                  "model: {closed_vocab_probe: true}"),
        "lente_rgb": ("LENTE recipe only: waste mask, no descriptor head, no debiasing",
                      "model: {adversary: false, mixstyle_p: 0.0}\n"
                      "train: {transplant_p: 0.0, loss: {desc: 0.0, dense: 0.0, adv: 0.0}}"),
        "geometry": ("depth-gated early guidance (run scripts/precompute_depth.py and set depth_dir first)",
                     "model: {geometry: true}"),
    }
    for k, (h, b) in ablations.items():
        write(f"e5_ablate_{k}", f"E5 ablation: {h}.", f"_base_: e4_heldout_classes.yaml\nrun_name: e5_ablate_{k}\n{b}")
    write("e6_edge_640", "E6 edge model: same recipe at 640 px, the deployment resolution on Jetson Orin Nano.",
          "_base_: e4_heldout_classes.yaml\nrun_name: e6_edge_640\nmodel: {image_size: 640}\n"
          "train: {batch_size: 12, val_batch_size: 12}")
    print("\n".join(sorted(p.name for p in EXP.glob("*.yaml"))))


if __name__ == "__main__":
    main()
