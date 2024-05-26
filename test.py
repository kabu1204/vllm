class Test:
    def __init__(self):
        self.a = 10
        self.b = False

def func(config: Test):
    import copy
    config = copy.copy(config)
    config.a = 20
    config.b = True
    return config

if __name__ == "__main__":
    t1 = Test()
    print(t1.a, t1.b)
    t2 = func(t1)
    print(t1.a, t1.b)
    print(t2.a, t2.b)