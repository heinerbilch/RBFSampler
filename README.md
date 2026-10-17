# RBFToolkit

Random-Fourier-Features (RFF) für den Gauß-Kernel plus lineare SVM
(Hinge-Loss, Mini-Batch-SGD) auf Apple Silicon – implementiert mit [MLX](https://github.com/ml-explore/mlx-swift).

Der Ansatz ersetzt die O(n²)-Kernel-Matrix durch eine Zufalls-Feature-Abbildung
z(x) = sqrt(2/m) * cos(Wx + b) (Rahimi & Recht, NIPS 2007), sodass das Training
des nichtlinearen Klassifikators zu einem linearen Problem wird, das per SGD
auf der GPU läuft.

## Struktur

- Sources/RBFToolkit – Bibliothek: RBFSamplerMLX, LinearSGDSVM,
  k-fold-CrossValidate, gamma-Tuning (Plateau-Logik), Exporte (VTK für ParaView,
  interaktives HTML via Plotly)
- Sources/RBFToolkitDemo – Kommandozeilen-Demo: synthetische Daten, Training,
  gamma-Tuning, lambda-Vergleich, VTK-Export
- Tests/RBFToolkitTests – Korrektheit (Feature-Map gegen CPU-Nachrechnung),
  Kernel-Näherung (z(x)·z(x') ≈ e^{-gamma||x-x'||²}), Reproduzierbarkeit
  (gleiche Seeds → identische Gewichte)

## Nutzung

swift run RBFToolkitDemo    # Demo (schreibt VTK/HTML auf den Desktop)
swift test                  # Test-Suite
open Package.swift          # Paket in Xcode öffnen

Voraussetzung: macOS 14+ auf Apple Silicon (MLX braucht Metal).
