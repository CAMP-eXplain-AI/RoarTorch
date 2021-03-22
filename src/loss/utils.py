import importlib


def create_loss(kwargs: dict):
    loss_name = kwargs['name']
    kwargs = kwargs.copy()
    kwargs.pop('name')

    p, m = loss_name.rsplit('.', 1)  # p is module(filename), m is Class Name
    module_name = p
    module_obj = importlib.import_module(module_name)
    loss = getattr(module_obj, m)

    return loss(**kwargs)
