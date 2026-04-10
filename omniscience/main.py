"""OMNISCIENCE — Main programmatic entry point.

Usage:
    from omniscience.main import omniscience_attack
    report = omniscience_attack(public_key, plaintext, ciphertext, target, modulus=p)
"""

from __future__ import annotations

import logging
from typing import Any

from omniscience.core.config import OmniscienceConfig
from omniscience.core.types import AttackReport, CryptoInstance
from omniscience.dispatcher import Dispatcher


def omniscience_attack(
    public_key: bytes | list[int] | int,
    plaintext: bytes | list[int],
    ciphertext_known: bytes | list[int],
    ciphertext_target: bytes | list[int],
    modulus: int | None = None,
    config: OmniscienceConfig | None = None,
) -> AttackReport:
    """Run a full OMNISCIENCE attack.

    Args:
        public_key: The target's public key.
        plaintext: Known plaintext(s).
        ciphertext_known: Ciphertext(s) corresponding to the known plaintext.
        ciphertext_target: Ciphertext to decrypt.
        modulus: Modulus of the cipher (if known).
        config: Optional framework configuration.

    Returns:
        AttackReport with the best decryption result (if successful).
    """
    instance = CryptoInstance(
        public_key=public_key,
        plaintext=plaintext,
        ciphertext_known=ciphertext_known,
        ciphertext_target=ciphertext_target,
        modulus=modulus,
    )
    dispatcher = Dispatcher(config or OmniscienceConfig())
    return dispatcher.attack(instance)
