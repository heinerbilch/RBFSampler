# data/

Hier wird der WDBC-Datensatz erwartet: `data/wdbc.data`
(569 Zeilen, kommagetrennt, keine Kopfzeile, keine fehlenden Werte).

Quelle: UCI Machine Learning Repository, Breast Cancer Wisconsin (Diagnostic).

Einmalig im Paket-Root herunterladen:

```
mkdir -p data && curl -L -o data/wdbc.data https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data
```

Die Datei wird bewusst nicht committet. Wer den Klon vollständig offline
reproduzierbar will, kann sie nach dem Download selbst committen; der
CSV-Lader validiert Spaltenzahl und Label-Domain und meldet kaputte
Downloads mit Zeilennummer.
