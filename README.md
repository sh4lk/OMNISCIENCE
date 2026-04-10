# OMNISCIENCE

**Framework de cryptanalyse asymetrique en boite noire** -- 14 moteurs de resolution, acceleration GPU, des chiffrements classiques jusqu'aux attaques post-quantiques sur les reseaux euclidiens.

OMNISCIENCE prend un texte chiffre inconnu et, en utilisant uniquement des paires clair/chiffre connues, identifie automatiquement le schema de chiffrement et le casse. Aucune connaissance prealable de l'algorithme n'est requise.

---

## Fonctionnalites

| Categorie | Solveurs |
|---|---|
| **Algebrique** | Elimination de Gauss sur F_p, bases de Groebner (SymPy + pont SageMath) |
| **Reseaux** | Reduction LLL/BKZ, sac a dos (CJLOSS), LWE (Kannan), HNP, NTRU, GGH, Coppersmith, SIS |
| **SMT/SAT** | Z3 bitvecteur, recuperation de S-Box, contraintes modulaires |
| **Neural** | Transformer + MLP PyTorch, approximation de fonction sur GPU |
| **Factorisation** | Division, Fermat, Pollard rho/p-1, Williams p+1, ECM, Wiener, Boneh-Durfee, Hastad |
| **Logarithme discret** | BSGS, Pohlig-Hellman, Pollard rho DLP, Index Calculus |
| **Courbe elliptique** | Lift p-adique de Smart, couplage MOV/Weil, courbe singuliere, EC Pohlig-Hellman |
| **AGCD** | Approximation diophantienne simultanee, reseau orthogonal, arbre de PGCD |
| **Force brute** | Noyau CUDA + CuPy vectorise modpow, attaque par collision (anniversaire) |
| **MITM** | Double chiffrement, MITM fonctionnel sqrt-split, composition affine |
| **Oracle** | Bleichenbacher (RSA PKCS#1), Vaudenay (padding CBC), oracle LSB adaptatif |
| **Classique** | Cesar, Affine, Vigenere, Beaufort, Autokey, Hill, substitution, rail-fence, XOR mono/multi |
| **Croisement** | Two-time pad, crib dragging, decomposition de composition, correlation de cles liees |

### Architecture

```
Entree (paires PT/CT + CT cible)
        |
        v
  +--------------------+
  |   Reconnaissance    |  Analyse statistique : entropie, correlation,
  |   (auto-detection)  |  Walsh-Hadamard, degre polynomial, structure
  +--------------------+  de reseau, heuristiques RSA/DLog/EC/AGCD
        |
        v
  +--------------------+
  |   Dispatcher        |  Priorise les solveurs selon la famille detectee
  |   (orchestrateur)   |  Execution parallele via Ray ou ThreadPool
  +--------------------+
        |
        v
  +--------------------+
  |   14 moteurs de     |  Chaque solveur tourne independamment avec timeout
  |   resolution        |  Le premier succes arrete tous les autres
  +--------------------+
        |
        v
  +--------------------+
  |   Rapport           |  Export JSON / HTML (theme sombre) / texte brut
  +--------------------+
```

### Saturation materielle

OMNISCIENCE est concu pour utiliser **100% des ressources disponibles** :

- **CPU** : Tous les coeurs via Ray (calcul distribue) ou ThreadPoolExecutor
- **GPU** : Noyaux CUDA (PyCUDA) + operations vectorisees CuPy + modeles neuraux PyTorch
- **RAM** : Limites memoire configurables par solveur
- Tableau de bord de monitoring en temps reel (jauges CPU/GPU/RAM)

---

## Installation

### De base

```bash
git clone https://github.com/sh4lk/omniscience.git
cd omniscience
pip install -e .
```

### Avec support GPU

```bash
pip install -e ".[gpu]"
```

### Avec calcul distribue (Ray)

```bash
pip install -e ".[ray]"
```

### Installation complete

```bash
pip install -e ".[all]"
```

### Pre-requis

- Python >= 3.10
- Voir [requirements.txt](requirements.txt) pour la liste complete des dependances
- Optionnel : GPU NVIDIA avec CUDA 12.x pour l'acceleration GPU
- Optionnel : [SageMath](https://www.sagemath.org) pour les calculs algebriques avances

---

## Utilisation

### Ligne de commande (CLI)

```bash
# Attaque complete sur un chiffrement
omniscience attack \
  --pub "3" \
  --pt "0,1,2,3,4,5,6,7,8,9" \
  --ct "7,10,13,16,19,22,25,28,31,34" \
  --target "223" \
  --mod 251 \
  --format int \
  --verbose

# Reconnaissance uniquement (pas de resolution)
omniscience recon \
  --pub "3" \
  --pt "0,1,2,3,4,5" \
  --ct "7,10,13,16,19,22" \
  --mod 251

# Informations systeme
omniscience info

# Exporter les resultats
omniscience attack ... --export-json rapport.json --export-html rapport.html
```

### Interface graphique (GUI)

```bash
omniscience-gui
```

Interface a theme sombre avec :
- Champs de saisie pour cle publique, texte clair, texte chiffre, cible
- Jauges CPU/GPU/RAM en temps reel
- Barres de confiance par solveur
- Console de logs en direct

### API Python

```python
from omniscience.core.types import CryptoInstance
from omniscience.core.config import OmniscienceConfig
from omniscience.dispatcher import Dispatcher

instance = CryptoInstance(
    public_key=[3],
    plaintext=list(range(50)),
    ciphertext_known=[(3 * p + 7) % 251 for p in range(50)],
    ciphertext_target=[223],
    modulus=251,
)

dispatcher = Dispatcher(OmniscienceConfig())
rapport = dispatcher.attack(instance)

if rapport.success():
    print(f"Dechiffre : {rapport.best_result.decrypted}")
    print(f"Solveur : {rapport.best_result.solver_name}")
    print(f"Confiance : {rapport.best_result.confidence:.1%}")
```

---

## Formats d'entree

| Format | Drapeau | Exemple |
|---|---|---|
| Entiers separes par des virgules | `--format int` | `0,1,2,3,4` |
| Hexadecimal | `--format hex` | `48656c6c6f` |
| Base64 | `--format base64` | `SGVsbG8=` |
| JSON | `--format json` | `[0, 1, 2, 3]` |
| Chemin de fichier | `--format file` | `chemin/vers/donnees.txt` |

---

## Structure du projet

```
omniscience/
  core/
    types.py          # CryptoInstance, SolverResult, AttackReport, AlgoFamily
    config.py         # OmniscienceConfig, HardwareConfig, SolverTimeouts
    report.py         # Export JSON/HTML/texte
  recon/
    statistical.py    # Auto-detection : entropie, correlation, heuristiques
  solvers/
    algebraic.py      # Elimination de Gauss, bases de Groebner
    lattice.py        # LLL, BKZ, sac a dos, LWE, HNP
    lattice_advanced.py  # NTRU, GGH, Coppersmith, SIS
    smt.py            # Z3 bitvecteur, S-Box, modulaire
    neural.py         # Transformer + MLP (PyTorch)
    factorization.py  # Attaques RSA (Pollard, ECM, Wiener, Boneh-Durfee)
    dlog.py           # Log discret (BSGS, Pohlig-Hellman, Index Calculus)
    elliptic_curve.py # Attaques EC (Smart, MOV, singuliere, BSGS)
    agcd.py           # Attaques PGCD approche
    bruteforce_gpu.py # Force brute CUDA + collision
    mitm.py           # Meet-in-the-Middle
    oracle.py         # Bleichenbacher, Vaudenay, oracle LSB
    classical.py      # Cesar, Vigenere, Hill, substitution, XOR
    cross_cipher.py   # Two-time pad, crib dragging, composition
    sage_bridge.py    # Pont subprocess SageMath
  dispatcher.py       # Orchestrateur avec routage par priorite
  hardware/
    resource_manager.py  # Monitoring CPU/GPU/RAM
  cli/
    app.py            # CLI Typer
  gui/
    app.py            # GUI CustomTkinter
tests/
  test_basic.py       # 40+ tests sur tous les modules
```

---

## Familles d'attaque supportees

OMNISCIENCE detecte automatiquement et route vers les solveurs appropries :

- `LINEAR` -- Affine, Cesar, XOR, Vigenere
- `POLYNOMIAL` -- Chiffrements quadratiques/cubiques sur corps finis
- `SUBSTITUTION` -- Monoalphabetique, base sur S-Box
- `LATTICE_BASED` -- LWE, sac a dos, problemes de reseaux
- `RSA_LIKE` -- Variantes RSA, exponentiation modulaire
- `EC_LIKE` -- Cryptosystemes sur courbes elliptiques
- `DLOG` -- Logarithme discret (DH, ElGamal)
- `AGCD` -- PGCD approche (schemas de chiffrement homomorphe)
- `NTRU_LIKE` -- NTRU, reseaux sur anneaux
- `LWE_BASED` -- Learning With Errors
- `HYBRID` -- Chiffrements multi-couches / composes
- `UNKNOWN` -- Les 14 moteurs sont testes par ordre de priorite

---

## Avertissement legal

Cet outil est destine exclusivement aux **tests de securite autorises**, aux **competitions CTF**, a la **recherche academique** et a des **fins educatives**.

**N'utilisez pas** cet outil pour attaquer des systemes que vous ne possedez pas ou pour lesquels vous n'avez pas d'autorisation ecrite explicite. L'acces non autorise a des systemes informatiques est illegal dans la plupart des juridictions.

Les auteurs ne sont pas responsables de toute utilisation abusive de ce logiciel.

---

## Licence

Ce projet est distribue sous la licence MIT. Voir [LICENSE](LICENSE) pour les details.

---

## Contribuer

Les contributions sont les bienvenues. Ouvrez une issue ou une pull request sur GitHub.

Domaines ou les contributions sont particulierement utiles :
- Nouveaux moteurs de resolution (ex: cryptanalyse differentielle, cryptanalyse lineaire)
- Optimisations de performance (noyaux CUDA, SIMD)
- Support additionnel de chiffrements classiques
- Amelioration de la couverture de tests
- Documentation et exemples
