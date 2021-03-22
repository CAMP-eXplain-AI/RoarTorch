import importlib


def create_optimizer(weights, kwargs: dict):
    optimizer_name = kwargs['name']
    kwargs = kwargs.copy()
    kwargs.pop('name')

    module_name, attribute = optimizer_name.rsplit('.', 1)
    module_obj = importlib.import_module(module_name)
    optimizer = getattr(module_obj, attribute)

    return optimizer(weights, **kwargs)


def create_scheduler(optimizer, scheduler_args):
    if scheduler_args:
        scheduler_name = scheduler_args['name']
        kwargs = scheduler_args.copy()
        kwargs.pop('name')

        module_name, attribute = scheduler_name.rsplit('.', 1)  # p is module(filename), m is Class Name
        module_obj = importlib.import_module(module_name)
        scheduler = getattr(module_obj, attribute)

        return scheduler(optimizer, **kwargs)

    return None
