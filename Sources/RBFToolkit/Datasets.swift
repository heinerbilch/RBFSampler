//
//  Datasets.swift
//  RBFToolkit
//
//  Created by Heiner Strauss on 18.10.26.
//
import Foundation
import MLX
import MLXRandom

// MARK: - Synthetische Testdaten

/// Zwei Gauß-Blobs: Standardnormalverteilung, per Zufall gelabelt und um
/// ±shift verschoben – jedes Experiment startet mit denselben Daten.
/// Liefert (x: n×d, y: n×1 mit Labels in {-1, +1}).
public func makeBlobs(n: Int, d: Int, shift: Float = 0.8, seed: UInt64 = 1234) -> (x: MLXArray, y: MLXArray) {
    MLXRandom.seed(seed)
    let x = MLXRandom.normal([n, d])
    let y = MLXRandom.uniform(low: 0.0, high: 1.0, [n, 1]) .< 0.5
    let ySigned = y.asType(.float32) * 2 - 1
    let xShifted = x + ySigned * shift   // Blobs um ±shift verschoben
    return (xShifted, ySigned)
}

// MARK: - Echte Daten: Breast Cancer Wisconsin (WDBC)

/// Fehler beim Laden eines Datensatzes, mit klarer Meldung statt rotem
/// Foundation-Würfel – die curl-Zeile sagt exakt, was zu tun ist.
public enum DatasetError: Error, CustomStringConvertible {
    case fileNotFound(String)
    case badRow(row: Int, reason: String)

    public var description: String {
        switch self {
        case .fileNotFound(let path):
            return "wdbc.data nicht gefunden (" + path + ").\n" +
                "Einmalig im Paket-Root herunterladen (liegt danach offline im Repo):\n" +
                "  mkdir -p data && curl -L -o data/wdbc.data " +
                "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
        case .badRow(let row, let reason):
            return "Zeile " + String(row + 1) + " der CSV ist unbrauchbar: " + reason
        }
    }
}

/// Lädt Breast Cancer Wisconsin (Diagnostic): 569 Punkte, 30 reelle
/// Merkmale, Label M (maligne) → +1, B (benigne) → -1. Das UCI-Format
/// (wdbc.data) ist kommagetrennt ohne Kopfzeile, 32 Spalten: ID, Label,
/// 30 Werte. Der Lader ist bewusst handgeschnitzt: reine Zahlen, keine
/// Quotes, keine fehlenden Werte – Foundation allein reicht, keine
/// Dependency nötig (und der Parse-Code ist ohnehin der Lerninhalt).
/// Gegen kaputte Downloads wehrt er sich mit Zeilennummer und Grund.
public func loadWDBC(path: String) throws -> (x: MLXArray, y: MLXArray) {
    let text: String
    do {
        text = try String(contentsOfFile: path, encoding: .utf8)
    } catch {
        throw DatasetError.fileNotFound(path)
    }

    let d = 30
    var features: [Float] = []
    var labels: [Float] = []
    var n = 0

    for (i, rawLine) in text.split(separator: "\n").enumerated() {
        let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
        if line.isEmpty { continue }
        let cols = line.split(separator: ",", omittingEmptySubsequences: false)
        guard cols.count == d + 2 else {
            throw DatasetError.badRow(row: i,
                reason: "erwartet \(d + 2) Spalten (ID, Label, \(d) Merkmale), gefunden \(cols.count)")
        }
        guard cols[1] == "M" || cols[1] == "B" else {
            throw DatasetError.badRow(row: i, reason: "Label muss M oder B sein, gefunden \(cols[1])")
        }
        for c in cols[2...] {
            guard let v = Float(String(c)) else {
                throw DatasetError.badRow(row: i, reason: "Wert '\(c)' ist keine Zahl")
            }
            features.append(v)
        }
        labels.append(cols[1] == "M" ? 1.0 : -1.0)
        n += 1
    }
    guard n > 0 else {
        throw DatasetError.badRow(row: 0, reason: "Datei enthält keine Zeilen")
    }

    return (MLXArray(features).reshaped([n, d]), MLXArray(labels).reshaped([n, 1]))
}
