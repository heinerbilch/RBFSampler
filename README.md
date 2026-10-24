# RBFToolkit

Random-Fourier-Features (RFF) für den Gauß-Kernel plus lineare SVM
(Hinge-Loss, Mini-Batch-SGD) auf Apple Silicon – implementiert mit [MLX](https://github.com/ml-explore/mlx-swift).

Der Ansatz ersetzt die O(n²)-Kernel-Matrix durch eine Zufalls-Feature-Abbildung
z(x) = sqrt(2/m) * cos(Wx + b) (Rahimi & Recht, NIPS 2007), sodass das Training
des nichtlinearen Klassifikators zu einem linearen Problem wird, das per SGD
auf der GPU läuft.

## Struktur

- Sources/RBFToolkit – Bibliothek: RBFSamplerMLX, LinearSGDSVM (fit + fitEpoch),
  Datensätze (synthetische Blobs, WDBC-CSV-Lader), Vorverarbeitung
  (Standardisierung, deterministischer Train/Test-Split), PCA per Potenzmethode,
  k-fold-CrossValidate, gamma-Tuning (Plateau-Logik), Exporte (VTK für ParaView,
  interaktives HTML via Plotly), RBFSampler-Visualisierungen (Wellen, Wellenfeld,
  Kernel-Näherung, Frequenzraum, Entscheidungslandschaft, 3D-Entscheidungsfläche,
  Epochen-Zeitsequenz, m-Fehlerkurve), WDBC-Entscheidungsfläche in der PCA-Ebene
- Sources/RBFToolkitDemo – Experimente als CLI (pipeline, visuals, cancer):
  Protokoll + Registry, jedes Experiment eine eigene Datei; visuals kennt
  Einzelschritte (waves field kernel omega boundary surface epochs components),
  cancer ebenfalls (data tune surface run)
- Tests/RBFToolkitTests – Korrektheit (Feature-Map gegen CPU-Nachrechnung),
  Kernel-Näherung (z(x)·z(x') ≈ e^-gamma·||x-x'||²), Reproduzierbarkeit
  (gleiche Seeds → identische Gewichte, fit == fitEpoch), Vorverarbeitung
  (Standardizer, Split, CSV-Lader), PCA (Potenzmethode)
- docs/ – Projekt-Aufsatz (essay.md), Logo (logo.svg) und
  Abkürzungswörterbuch (glossary.md)
- data/ – WDBC-Datensatz (wdbc.data, einmalig per curl geladen, siehe data/)

## Nutzung

swift run RBFToolkitDemo               # Liste der Experimente
swift run RBFToolkitDemo pipeline      # Pipeline (schreibt VTK/HTML auf den Desktop)
swift run RBFToolkitDemo visuals       # alle Visualisierungen
swift run RBFToolkitDemo visuals omega # nur ein Schritt: waves field kernel omega boundary surface epochs components
swift run RBFToolkitDemo cancer data   # WDBC laden: Zeilen, Klassen, Merkmals-Skalen
swift run RBFToolkitDemo cancer        # Split → Standardisierung → gamma-Tuning → Test vs. lineare Baseline → m-Sweep
swift run RBFToolkitDemo cancer surface  # PCA-Ebene des 30D-RFF-Modells als VTK für ParaView
swift test                             # Test-Suite
open Package.swift                     # Paket in Xcode öffnen

Voraussetzung: macOS 14+ auf Apple Silicon (MLX braucht Metal).

Der WDBC-Datensatz liegt nicht im Repo. Einmalig im Paket-Root:

```
mkdir -p data && curl -L -o data/wdbc.data https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data
```
