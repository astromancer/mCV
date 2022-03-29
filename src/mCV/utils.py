# std
import inspect

# third-party
from astropy import units as u

# local
from recipes.decorators import Decorator


NULL = object()


def get_value(x):
    """Get numeric value from quantity or return object itself."""
    return getattr(x, 'value', x)


def has_unit(val):
    """Check if object has a physically meaningful unit."""
    return not no_unit(val)


def no_unit(val):
    return getattr(val, 'unit', None) in (None, u.dimensionless_unscaled)


def get_unit_string(val, default='', style='latex'):
    if hasattr(val, 'unit'):
        unit = val.unit
    elif isinstance(val, u.UnitBase):
        unit = val
    else:
        return default

    return unit.to_string(style).strip('$')


class default_units(Decorator):
    """
    Decorator for applying default units to function input parameters.
    """

    def __init__(self, default_units=(), **kws):
        self.sig = None     # placeholder
        self.default_units = dict(default_units, **kws)
        for unit in self.default_units.values():
            assert isinstance(unit, u.UnitBase)

    def __call__(self, func):
        self.sig = inspect.signature(func)
        return super().__call__(func)

    def apply(self, *args, **kws):
        # positional params
        if args:
            new_args = (val * unit if no_unit(val) and unit else val
                        for val, unit in zip(args, self.default_units.values()))
            return (next, tuple)[len(args) > 1](new_args)

        # keyword params
        return {name: (val * unit  # NOTE: >> does not copy underlying data
                       if no_unit(val) and
                       ((unit := self.default_units.get(name)) is not None)
                       else
                       val)
                for name, val in kws.items()}

    def __wrapper__(self, func, *args, **kws):
        ba = self.sig.bind(*args, **kws).arguments
        return func(
            *(() if (_self := ba.pop('self', NULL)) is NULL else (_self,)),
            **self.apply(**ba)
        )


def _check_optional_units(namespace, allowed_physical_types):
    for name, kinds in allowed_physical_types.items():
        if name not in namespace:
            continue

        obj = namespace[name]
        if isinstance(kinds, (str, u.PhysicalType)):
            kinds = (kinds, )

        if isinstance(obj, u.Quantity) and (obj.unit.physical_type not in kinds):
            raise u.UnitTypeError(f'Parameter {name} should have physical '
                                  f'type(s) {kinds}, not '
                                  f'{obj.unit.physical_type}.')
        # else:
        #     logger.info
