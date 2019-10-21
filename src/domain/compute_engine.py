class CuPyEngine:
    """
    Wrapper to manage the computing engine in the case of CuPy
    """

    def __init__(self, engine, seed):
        self.engine = engine
        self.engine.random.seed(seed)

    def send_to_device(self, data):
        return self.engine.asarray(data)


class NumPyEngine:
    """
    Wrapper to manage the computing engine in the case of NumPy
    """

    def __init__(self, engine, seed):
        self.engine = engine
        self.engine.random.seed(seed)

    def send_to_device(self, data):
        return data
