def micro_f1_clustering(y_true, y_pred):
    """
    Micro-F1 for clustering = accuracy after optimal label alignment.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Label lists must have the same length.")

    M, true_labels, pred_labels = confusion_matrix(y_true, y_pred)

    n = max(len(true_labels), len(pred_labels))

    # Pad to square matrix
    square = [[0] * n for _ in range(n)]
    for i in range(len(M)):
        for j in range(len(M[0])):
            square[i][j] = M[i][j]

    assignment = hungarian_maximize(square)

    correct = sum(square[i][assignment[i]] for i in range(n))
    total = len(y_true)

    accuracy = correct / total
    return accuracy
