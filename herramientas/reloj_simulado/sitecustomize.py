"""Reloj simulado para reproducir eventos pasados como si ocurrieran ahora.

Solo actúa si existe la variable CLIMAAR_RELOJ_SIMULADO (fecha ISO en UTC).
Python carga este archivo solo si su carpeta está en PYTHONPATH; la usa
únicamente herramientas/simular_evento_real.py. No se usa en producción.
"""
import datetime as _dt
import os
import time as _time

_valor = os.environ.get("CLIMAAR_RELOJ_SIMULADO")

if _valor:
    _real = _dt.datetime
    _tiempo_real = _time.time
    _base = _real.fromisoformat(_valor.replace("Z", "+00:00"))
    if _base.tzinfo is None:
        _base = _base.replace(tzinfo=_dt.timezone.utc)
    _desfase = _base.timestamp() - _tiempo_real()

    def _ahora_ts():
        return _tiempo_real() + _desfase

    class _Meta(type):
        def __instancecheck__(cls, obj):
            return isinstance(obj, _real)

        def __subclasscheck__(cls, sub):
            return issubclass(sub, _real)

    class _RelojSimulado(_real, metaclass=_Meta):
        @classmethod
        def now(cls, tz=None):
            return _real.fromtimestamp(_ahora_ts(), tz)

        @classmethod
        def utcnow(cls):
            return _real.fromtimestamp(_ahora_ts(), _dt.timezone.utc).replace(tzinfo=None)

        @classmethod
        def today(cls):
            return _real.fromtimestamp(_ahora_ts())

    _dt.datetime = _RelojSimulado
    _time.time = _ahora_ts
