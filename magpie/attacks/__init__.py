"""Attacks on the attacker ladder (L0 natural → L3 weight), one interface.

Every attack subclasses :class:`~magpie.attacks.base.Attack` and is driven by
:class:`~magpie.attacks.base.AttackRunner`, which logs cost per attempt so the
survival analysis is automatic. Concrete attacks land per PLAN §6.
"""

from magpie.attacks.base import Attack, AttackRun, AttackRunner, Judge, Variant

__all__ = ["Attack", "AttackRun", "AttackRunner", "Judge", "Variant"]
