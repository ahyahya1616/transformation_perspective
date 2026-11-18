# Transformation de Perspective — Devoir 2

Programme C++ / OpenCV qui **détecte automatiquement les 4 coins d'un document** dans une photo et applique une **transformation de perspective** pour obtenir une vue de dessus redressée.

---

## Fonctionnalités

| Étape | Description |
|---|---|
| 1 | Détection automatique des 4 coins (2 méthodes selon le fond) |
| 2 | Affichage du résultat de détection |
| 3 | Ajustement interactif des coins (glisser-déposer à la souris) |
| 4 | Application de la transformation de perspective (`warpPerspective`) |

---

## Méthodes de détection

### Fond simple (uniforme et contrasté)
**Méthode : Canny + `findContours` + `approxPolyDP`**

Le document génère un fort gradient sur ses bords par rapport au fond uni.  
`Canny` extrait les contours → `findContours` les énumère → `approxPolyDP` sélectionne le plus grand quadrilatère convexe.

### Fond texturé (complexe ou bruité)
**Méthode : Transformée de Hough probabiliste (`HoughLinesP`)**

Sur un fond texturé, `Canny` produit trop de bruit parasite.  
`HoughLinesP` détecte les **longues lignes droites** formant les bords du document, robuste aux textures. Les 4 bords (haut/bas/gauche/droite) sont identifiés et leurs intersections donnent les coins.

**Détection automatique** : la variance du Laplacien de l'image sert de critère — variance élevée → fond texturé.

---

## Compilation

```bash
mkdir build && cd build
cmake ..
make
```

## Utilisation

```bash
# Détection automatique du type de fond
./transform_perspective image.jpg

# Forcer le mode fond simple
./transform_perspective image.jpg simple

# Forcer le mode fond texturé
./transform_perspective image.jpg texture
```

### Contrôles interactifs

| Touche / Action | Effet |
|---|---|
| Glisser un coin vert | Déplacer ce coin |
| `Entrée` | Appliquer la transformation |
| `R` | Réinitialiser les coins à la détection automatique |
| `Q` | Quitter |

Le résultat est sauvegardé automatiquement sous `<nom_original>_rectified.jpg`.

---

## Dépendances

- CMake ≥ 3.10
- OpenCV ≥ 4.x (`opencv2/opencv.hpp`)
- Compilateur C++17

