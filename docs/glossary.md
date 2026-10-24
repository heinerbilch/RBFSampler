# Abkürzungen und Symbole

Kurzreferenz für alle Abkürzungen, die in diesem Projekt und in der SVM-Literatur
vorkommen – gruppiert nach Thema, mit der jeweiligen Bedeutung im Projekt.
Am Ende ein Abschnitt zu den griechischen Symbolen und Parametern, die streng
genommen keine Abkürzungen sind, aber ständig auftauchen.

## Projekt und Werkzeuge

| Abkürzung | Steht für | Bedeutung im Projekt |
| --- | --- | --- |
| RBF | Radial Basis Function | Radiale Basisfunktion. Der Gauß-Kernel ist eine RBF – davon leitet sich der Projektname RBFToolkit ab. |
| RFF | Random Fourier Features | Zufällige Fourier-Features (Rahimi & Recht, NIPS 2007). Herzstück des Projekts: der Gauß-Kernel wird durch eine feste, zufällige Feature-Map approximiert, danach ist alles linear. |
| SVM | Support Vector Machine | Klassifikator mit maximalem Abstand. Hier als lineare SVM mit Hinge-Loss, trainiert per Mini-Batch-SGD. |
| SGD | Stochastic Gradient Descent | Stochastischer Gradientenabstieg, hier als Mini-Batch-Variante: pro Schritt der Gradient über einen zufälligen Batch (deterministisch per SeededRNG). |
| MLX | MLX (Eigenname, keine offizielle Langform) | Apples Array-Framework für Apple Silicon, rechnet transparent auf CPU oder GPU (Metal). Einzige Abhängigkeit des Pakets, gepinnt via Package.resolved. |
| SPM | Swift Package Manager | Swifts Paketmanager: definiert das Projekt über Package.swift, Bibliothek/Demo/Tests als Targets. |
| CLI | Command Line Interface | Kommandozeile. Das Experiment-Tool wird per `swift run RBFToolkitDemo <experiment>` aufgerufen. |
| API | Application Programming Interface | Die öffentlichen Funktionen der Bibliothek (RBFSamplerMLX, LinearSGDSVM, Exporte usw.). |
| LIBSVM | LIBrary for Support Vector Machines | Die Referenzimplementierung für exakte Kernel-SVMs (Chang & Lin, SMO nach Fan et al. 2005). Dieses Projekt ist der approximative Gegenentwurf: RFF + lineares SGD statt QP auf der Kernel-Matrix. |

## Mathematik und Optimierung

| Abkürzung | Steht für | Bedeutung im Projekt |
| --- | --- | --- |
| SMO | Sequential Minimal Optimization | Löst das duale QP der exakten Kernel-SVM, indem es immer nur zwei Lagrange-Multiplikatoren gleichzeitig optimiert (jedes 2er-Teilproblem hat eine geschlossene Formel). Der exakte Weg, den dieses Projekt bewusst nicht geht. |
| QP | Quadratic Programming | Quadratische Optimierung: Zielfunktion quadratisch, Nebenbedingungen linear. Das SVM-Dualproblem ist ein QP. |
| KKT | Karush-Kuhn-Tucker | Optimalitätsbedingungen für Optimierung mit Nebenbedingungen. SMO stoppt, wenn keine KKT-Verletzung mehr übrig ist. |
| CV | Cross-Validation | Kreuzvalidierung. Hier 5-fach auf dem Trainings-Split, für das gamma-Tuning (Grid + Plateau-Logik). |
| PCA | Principal Component Analysis | Hauptkomponentenanalyse, hier von Hand per Potenzmethode implementiert (pcaTopTwo), um die 30 WDBC-Merkmale auf eine zeichnbare Ebene zu projizieren. |
| PC1, PC2 | Principal Component 1 / 2 | Erste und zweite Hauptkomponente – die Achsen der PCA-Ebene. |
| O(n²), O(n·m) | Landau-Notation | Aufwandsklassen: die exakte Kernel-SVM wird quadratisch in n, das RFF-Training skaliert mit n mal m. |
| ν-SVM | Nu-SVM | SVM-Variante, die über ν den Support-Vektor-Anteil und Margin-Verletzungen steuert (statt über C). |
| ϵ-SVR | Epsilon-Support-Vector-Regression | Regressions-SVM mit ϵ-Toleranzrohr. In LIBSVM enthalten, nicht Teil dieser Pipeline. |
| JMLR | Journal of Machine Learning Research | Fachjournal, in dem die SMO-Arbeit von Fan et al. (2005) erschien. |
| NIPS | Neural Information Processing Systems | Konferenz, auf der die RFF-Arbeit von Rahimi & Recht (2007) erschien (heute NeurIPS). |

## Daten, Datensätze, Dateiformate

| Abkürzung | Steht für | Bedeutung im Projekt |
| --- | --- | --- |
| WDBC | Wisconsin Diagnostic Breast Cancer | UCI-Datensatz: 569 Punkte, 30 Merkmale, Tumor-Klassifikation (M/B). Der erste echte Datensatz des Projekts. |
| UCI | University of California, Irvine | Betreiber des UCI Machine Learning Repository, woher wdbc.data stammt. |
| M / B | Malignant / Benign | Klassenlabels in WDBC: bösartig (+1) bzw. gutartig (−1). |
| CSV | Comma-Separated Values | Textformat, eine Zeile pro Datenpunkt – so kommt wdbc.data daher. |
| VTK | Visualization Toolkit | Dateiformat (legacy, ASCII) für ParaView: Entscheidungsoberflächen, Punkte-POLYDATA, Zeitsequenzen (.series). |
| ASCII | American Standard Code for Information Interchange | Klartext. „Legacy VTK ASCII“ heißt: lesbares Textformat statt Binärformat. |
| HTML | Hypertext Markup Language | Format der interaktiven Plotly-Diagramme. |
| SVG | Scalable Vector Graphics | Vektorgrafik-Format (Logo: docs/logo.svg). |

## Hardware und Drumherum

| Abkürzung | Steht für | Bedeutung im Projekt |
| --- | --- | --- |
| GPU | Graphics Processing Unit | MLX läuft darüber via Metal. Der erste transform-Aufruf kompiliert die Kernel (Sekunden), danach Warmstart – Benchmarks nur mit Warmstarts. |
| CPU | Central Processing Unit | Hauptprozessor; auch MLX kann darauf rechnen. |

## Symbole und griechische Buchstaben

| Zeichen | Name | Bedeutung |
| --- | --- | --- |
| γ | Gamma | Bandbreite des Gauß-Kernels, hier als Wellenlänge gelesen: klein = glatte globale Wellen, groß = Rauschen. Wird per CV getunt (bei WDBC gewählt: 0.01). |
| λ | Lambda | Regularisierungsstärke. Im Projekt eher eine Skalenfrage als eine Grenzformfrage (λ=1e-4 vs. 1e-2: gleiche Genauigkeit, ~20% kleinere f-Beträge). |
| α | Alpha | Lagrange-Multiplikator des SVM-Duals, einer pro Punkt; α > 0 heißt Support-Vektor. Bei RFF + SGD taucht er nicht auf – dort gibt es direkte Gewichte w. |
| C | (Parameter) | Kostenparameter der Soft-Margin-SVM: Strafe für Margin-Verletzungen. |
| ν | Ny | Parameter der ν-SVM: Steuergröße für Support-Vektor-Anteil und Verletzungen. |
| ϵ | Epsilon | Toleranzrohr bei der Regression (ϵ-SVR). |
| m | – | Anzahl der RFF-Features. Bei WDBC ab m=100 gesättigt (Standard: 500). |
| n | – | Anzahl der Datenpunkte (WDBC: 569, davon 399 Train / 170 Test). |
| f(x) | – | Entscheidungsfunktion, im RFF-Fall f(x) = wᵀ z(x). Die Marge ist \|f\| = 1; grenznahe Punkte zählt man per Threshold \|f\| ≤ 1.1. |
