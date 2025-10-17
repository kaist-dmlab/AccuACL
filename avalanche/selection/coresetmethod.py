class CoresetMethod(object):
    def __init__(self, dst_train, fraction=0.5, **kwargs):
        if fraction <= 0.0 or fraction > 1.0:
            raise ValueError("Illegal Coreset Size.")
        self.dst_train = dst_train
        # self.num_classes = len(dst_train.classes)
        self.fraction = fraction
        self.index = []

        self.n_train = len(dst_train)
        self.coreset_size = round(self.n_train * fraction)

    def select(self, **kwargs):
        return