class AdaptiveLOSAT:
    """Adaptive thresholding"""

    def __init__(self, alpha: float = 0.8, beta: float = 0.2, init_threshold: float = 0.5):
        self.alpha = alpha
        self.beta = beta
        self.threshold = init_threshold

    def update(self, score: float, motion: float) -> tuple[float, bool]:
        # T_t = alpha * T_prev + (1-alpha) * score + beta * M_t
        self.threshold = self.alpha * self.threshold + (1.0 - self.alpha) * score + self.beta * motion

        raw_alert = score > self.threshold
        return self.threshold, raw_alert
