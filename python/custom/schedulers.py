def step_scheduler(epoch):
    """Decreases learning rate by step
    """
    if epoch < 20:
        return 0.1
    elif epoch < 40:
        return 0.02
    elif epoch < 60:
        return 0.005
    else:
        return 0.001
