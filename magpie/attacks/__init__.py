"""Attacks on the attacker ladder (L0 natural -> L3 weight), one interface.

Every attack subclasses :class:`~magpie.attacks.base.Attack` and is driven by
:class:`~magpie.attacks.base.AttackRunner`, which logs cost per attempt so the
survival analysis is automatic. Concrete attacks land per PLAN §6.
"""

from magpie.attacks.base import Attack, AttackRun, AttackRunner, Judge, Variant
from magpie.attacks.l0_direct import L0Direct
from magpie.attacks.l1_bon_decoding import L1BonDecoding
from magpie.attacks.l1_bon_input import L1BonInput
from magpie.attacks.l1_templates import L1Templates

ATTACKS: dict[str, type[Attack]] = {
    cls.name: cls for cls in (L0Direct, L1Templates, L1BonInput, L1BonDecoding)
}

__all__ = [
    "Attack", "AttackRun", "AttackRunner", "Judge", "Variant",
    "L0Direct", "L1Templates", "L1BonInput", "L1BonDecoding", "ATTACKS",
]
