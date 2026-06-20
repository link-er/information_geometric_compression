import torch

def compute_cluster(train_repr: torch.Tensor, y_predict_train: torch.Tensor):
    """
    Vectorized clustering regularization.

    Args:
        train_repr: (N, D) tensor of features
        y_predict_train: (N,) tensor of predicted class labels

    Returns:
        all_train_cdnvs: scalar clustering score
        means: (C, D) tensor of per-class means
        variances: (C,) tensor of per-class variances
    """
    device = train_repr.device
    num_classes = y_predict_train.max().item() + 1

    # --- Compute per-class counts ---
    counts = torch.bincount(y_predict_train, minlength=num_classes).float().to(device)

    # --- Compute per-class means ---
    sums = torch.zeros(num_classes, train_repr.size(1), device=device)
    sums.index_add_(0, y_predict_train, train_repr)
    means = sums / counts.unsqueeze(1).clamp(min=1)

    # --- Compute per-class variances (mean squared distance to class mean) ---
    diffs = train_repr - means[y_predict_train]
    sq_norms = (diffs ** 2).sum(dim=1)
    var_sums = torch.zeros(num_classes, device=device)
    var_sums.index_add_(0, y_predict_train, sq_norms)
    variances = var_sums / counts.clamp(min=1)

    # --- Pairwise distances between class means ---
    diff = means[:, None, :] - means[None, :, :]   # (C, C, D)
    dist2 = diff.pow(2).sum(dim=2)                 # (C, C)

    # --- Avoid divide-by-zero and self-pairs ---
    mask = (dist2 < 1e-12)  # all zero-distance pairs
    dist2 = dist2.masked_fill(mask, float('inf'))

    # --- Compute clustering metric for all pairs ---
    var1 = variances[:, None]  # (C, 1)
    var2 = variances[None, :]  # (1, C)
    temp_results = (var1 + var2) / (2 * dist2)     # (C, C)

    # Mean clustering score (exclude inf)
    all_train_cdnvs = temp_results[~mask].mean()

    return all_train_cdnvs, means, variances

# ---------------- TESTS ---------------- #

def compute_cluster_slow(train_repr, y_predict_train):
    class_dict = {key.item(): [] for key in torch.unique(y_predict_train)}
    for sub_train_repr, sub_y_predict_train in zip(train_repr, y_predict_train):
        class_dict[sub_y_predict_train.item()].append(sub_train_repr)

    mean_var_batch_dict = {}
    for key, value in class_dict.items():
        temp_mean = torch.mean(torch.stack(value), dim=0)
        temp_var = torch.mean(torch.stack([torch.linalg.norm(f - temp_mean) ** 2 for f in value]))
        mean_var_batch_dict[key] = [temp_mean, temp_var]

    all_train_cdnvs = []
    for c1 in class_dict.keys():
        for c2 in class_dict.keys():
            if c2 == c1:
                continue
            mu1, var1 = mean_var_batch_dict[c1]
            mu2, var2 = mean_var_batch_dict[c2]
            temp_result = (var1 + var2) / (2 * torch.linalg.norm(mu1 - mu2) ** 2)
            all_train_cdnvs.append(temp_result)
    all_train_cdnvs = torch.mean(torch.stack(all_train_cdnvs))
    return all_train_cdnvs

def test_small_case():
    torch.manual_seed(0)
    x = torch.randn(10, 5)  # 10 samples, 5-dim
    y = torch.randint(0, 3, (10,))  # 3 classes
    fast_val, _, _ = compute_cluster(x, y)
    slow_val = compute_cluster_slow(x, y)
    print("Small case")
    print("Fast:", fast_val.item())
    print("Slow:", slow_val.item())
    assert torch.allclose(fast_val, slow_val, atol=1e-5), "Mismatch with not vectorized in small case"

def test_random_large():
    torch.manual_seed(42)
    x = torch.randn(500, 20)  # 500 samples, 20-dim
    y = torch.randint(0, 10, (500,))  # 10 classes
    fast_val, _, _ = compute_cluster(x, y)
    slow_val = compute_cluster_slow(x, y)
    print("\nLarge case")
    print("Fast:", fast_val.item())
    print("Slow:", slow_val.item())
    assert torch.allclose(fast_val, slow_val, atol=1e-5), "Mismatch with not vectorized in large case"

def test_numeric_simple():
    """
    Two classes, 1D points.
    Class 0: [0, 0]
    Class 1: [2, 2]
    """
    x = torch.tensor([[0.0], [0.0], [2.0], [2.0]])
    y = torch.tensor([0, 0, 1, 1])

    cdnv, means, vars = compute_cluster(x, y)

    # Means
    expected_means = torch.tensor([[0.0], [2.0]])
    assert torch.allclose(means, expected_means), f"Means mismatch: {means} vs {expected_means}"

    # Variances: each class has identical points -> var = 0
    expected_vars = torch.tensor([0.0, 0.0])
    assert torch.allclose(vars, expected_vars), f"Vars mismatch: {vars} vs {expected_vars}"

    # CDNVS: (0+0)/(2 * (0-2)^2) = 0
    assert abs(cdnv.item() - 0.0) < 1e-8, f"CDNVS mismatch: {cdnv.item()} vs 0.0"

def test_numeric_variance():
    """
    Two classes, 1D points.
    Class 0: [0, 1]
    Class 1: [2, 3]
    """
    x = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    y = torch.tensor([0, 0, 1, 1])

    cdnv, means, vars = compute_cluster(x, y)

    # Means
    # Class 0 mean = (0+1)/2 = 0.5
    # Class 1 mean = (2+3)/2 = 2.5
    expected_means = torch.tensor([[0.5], [2.5]])
    assert torch.allclose(means, expected_means), f"Means mismatch: {means} vs {expected_means}"

    # Variances
    # Class 0: ((0-0.5)^2 + (1-0.5)^2)/2 = (0.25 + 0.25)/2 = 0.25
    # Class 1: ((2-2.5)^2 + (3-2.5)^2)/2 = (0.25 + 0.25)/2 = 0.25
    expected_vars = torch.tensor([0.25, 0.25])
    assert torch.allclose(vars, expected_vars), f"Vars mismatch: {vars} vs {expected_vars}"

    # CDNVS
    # dist^2 = (0.5 - 2.5)^2 = 4
    # (0.25 + 0.25)/(2 * 4) = 0.5/8 = 0.0625
    expected_cdnv = 0.0625
    assert abs(cdnv.item() - expected_cdnv) < 1e-8, f"CDNVS mismatch: {cdnv.item()} vs {expected_cdnv}"

if __name__ == "__main__":
    test_small_case()
    test_random_large()
    test_numeric_simple()
    test_numeric_variance()
    print("\nAll tests passed")
