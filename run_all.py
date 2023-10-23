import subprocess


def run(model, attention, oversample):
    att = "-a" if attention else ""
    ov = "-o" if oversample else ""
    subprocess.run(["make", "train_eval",
                   f"MODEL={model}", f"ATTENTION={att}", f"OVERSAMPLE={ov}"])


def run_lod(model, attention, oversample=False):
    att = "-a" if attention else ""
    ov = "-o" if oversample else ""
    subprocess.run(["make", "train_eval_lod",
                   f"MODEL={model}", f"ATTENTION={att}", f"OVERSAMPLE={ov}"])


for attention in [True, False]:
    for model in ['alexnet', 'densenet', 'resnet']:
        run(model, attention)


for attention in [True, False]:
    for model in ['alexnet', 'densenet', 'resnet']:
        run_lod(model, attention)
