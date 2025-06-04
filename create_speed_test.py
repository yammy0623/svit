for r in [10] + list(range(100, 1001, 100)):
    config_path = f"configs/diff_r_test_speed/tome-vit-adapter-t-3x_{r}.py"
    model_path = "pretrained/vit-adapter-t-3x.pth"
    print(f"{config_path}, {model_path}")
