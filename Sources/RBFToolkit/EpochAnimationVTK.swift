//
//  EpochAnimationVTK.swift
//  RBFToolkit
//
//  Created by Heiner Strauss on 19.10.26.
//
import Foundation
import MLX

// MARK: - RBFSampler-Visualisierung: Trainingsepochen als Zeitsequenz (VTK)

/// Das Training als Film: dieselbe 3D-Entscheidungsflaeche wie im
/// surface-Schritt, aber einmal pro `snapshotEvery` Epochen in eine eigene
/// VTK-Datei geschrieben – plus eine ParaView-`.series`-Datei, die alle
/// Snapshots zu einer Zeitsequenz verdrahtet. Beim Oeffnen der .series
/// gibt ParaView eine Zeitleiste her: Play druecken und zusehen, wie die
/// Grenze aus dem flachen Zufallszustand (f = 0 ueberall) in die finale
/// Wellenform waechst.
///
/// Zwei Serien pro Aufruf, alles landet in einem Verzeichnis:
/// - Flaeche: STRUCTURED_GRID je Snapshot, z = f(x)·height, Skalar f.
/// - Trainingspunkte: POLYDATA je Snapshot auf Hoehe ihrer eigenen
///   Entscheidungswerte (Skalare label und decision) – die Punkte
///   wandern mit wachsender Marge auf ihre Huegel hinauf.
///
/// Zeitwert eines Snapshots = Anzahl abgeschlossener Epochen; Zeit 0 ist
/// das untrainierte Modell (w = b = 0, flache Karte). fitEpoch macht das
/// Training epochenweise und damit snapshot-faehig – bit-identisch zum
/// normalen fit.
///
/// ParaView-Rezept:
/// - `epoch_surface.vtk.series` oeffnen (nicht die Einzel-Dateien!) –
///   die Zeitleiste mit Play-Button erscheint automatisch; darueber
///   laesst sich die Animation auch als Video exportieren.
/// - Color by 'f', diverging Colormap: der Wertebereich gilt gemeinsam
///   ueber alle Zeitschritte, die Farben bleiben vergleichbar.
/// - Contour f=0 zeigt die Grenze, Threshold -1.1 bis 1.1 die
///   Support-Vektor-Schale – beides animiert.
/// - `epoch_points.vtk.series` dazuladen, Glyph (Sphere), Color by 'label'.
/// - height = 0 -> flache Karte, nur Farb- und Konturanimation.
public func writeVtkEpochAnimation(gamma: Float, shift: Float = 1.2,
                                   nTrain: Int = 4000, nComponents: Int = 500,
                                   epochs: Int = 20, snapshotEvery: Int = 2,
                                   batchSize: Int = 256,
                                   extent: Float = 4, resolution: Int = 100,
                                   height: Float = 0.5,
                                   toDirectory directory: String) {
    do {
        try FileManager.default.createDirectory(atPath: directory,
                                               withIntermediateDirectories: true)
    } catch {
        print("Fehler beim Anlegen von \(directory): \(error)")
        return
    }

    // Daten wie im surface-Schritt: 2D-Blobs -> Features -> lineare SVM,
    // diesmal aber Epoche fuer Epoche (fitEpoch), damit zwischen den
    // Epochen gesnapshottet werden kann.
    let (x, y) = makeBlobs(n: nTrain, d: 2, shift: shift)
    let n = x.shape[0]
    let sampler = RBFSamplerMLX(dimension: 2, nComponents: nComponents,
                                gamma: gamma, seed: 42)
    let z = sampler.transform(x)
    z.eval()
    var svm = LinearSGDSVM(featureCount: nComponents, lambda: 1e-4, learningRate: 0.05)

    // Gitter-Features einmal vorberechnen: die Feature-Map des Gitters ist
    // in jeder Epoche identisch, nur w und b aendern sich.
    let res = resolution
    var xs: [Float] = [], ys: [Float] = []
    for i in 0..<res { xs.append(-extent + 2 * extent * Float(i) / Float(res - 1)) }
    for i in 0..<res { ys.append(-extent + 2 * extent * Float(i) / Float(res - 1)) }
    var pts: [Float] = []
    pts.reserveCapacity(res * res * 2)
    for iy in 0..<res {
        for ix in 0..<res {
            pts.append(xs[ix]); pts.append(ys[iy])
        }
    }
    let xg = MLXArray(pts).reshaped([res * res, 2])
    let zg = sampler.transform(xg)
    zg.eval()

    let xArr = x.asArray(Float.self)
    let yArr = y.asArray(Float.self)

    // 12 Werte pro Zeile, wie in WriteVTK
    func vtkBlock(_ v: [Float]) -> [String] {
        var lines: [String] = []
        for start in stride(from: 0, to: v.count, by: 12) {
            let end = min(start + 12, v.count)
            lines.append(v[start..<end].map { String(format: "%.6g", $0) }
                .joined(separator: " "))
        }
        return lines
    }

    var surfaceFiles: [(name: String, time: Int)] = []
    var pointFiles: [(name: String, time: Int)] = []

    // Ein Snapshot = zwei Dateien (Flaeche + Punkte) zum gleichen Zeitwert.
    func writeSnapshot(_ time: Int) {
        let fG = svm.decisionFunction(zg)
        fG.eval()
        let fArr = fG.asArray(Float.self)
        let fPt = svm.decisionFunction(z)
        fPt.eval()
        let fpArr = fPt.asArray(Float.self)
        let pred = svm.predict(z)
        pred.eval()
        let acc = 100.0 * Float((pred * y .> 0).sum().item(Int.self)) / Float(n)
        print(String(format: "Epoche %3d: Genauigkeit %.1f%%, f der Punkte in [%.2f, %.2f]",
                     time, acc, fpArr.min() ?? 0, fpArr.max() ?? 0))

        // 1) Flaeche: z-Koordinate = f(x)·height
        let surfaceName = String(format: "epoch_%04d.vtk", time)
        var out: [String] = []
        out.append("# vtk DataFile Version 3.0")
        out.append("RFF Entscheidungsflaeche, gamma=" + String(format: "%.3g", gamma)
                   + ", Epoche " + String(time))
        out.append("ASCII")
        out.append("DATASET STRUCTURED_GRID")
        out.append(String(format: "DIMENSIONS %d %d 1", res, res))
        out.append(String(format: "POINTS %d float", res * res))
        for iy in 0..<res {
            for ix in 0..<res {
                let i = iy * res + ix
                out.append(String(format: "%.6g %.6g %.6g", xs[ix], ys[iy], fArr[i] * height))
            }
        }
        out.append(String(format: "POINT_DATA %d", res * res))
        out.append("SCALARS f float 1")
        out.append("LOOKUP_TABLE default")
        out.append(contentsOf: vtkBlock(fArr))

        // 2) Trainingspunkte auf Entscheidungshoehe ueber der Flaeche
        let pointsName = String(format: "epoch_points_%04d.vtk", time)
        var out2: [String] = []
        out2.append("# vtk DataFile Version 3.0")
        out2.append("RFF Trainingspunkte auf der Entscheidungsflaeche, gamma="
                    + String(format: "%.3g", gamma) + ", Epoche " + String(time))
        out2.append("ASCII")
        out2.append("DATASET POLYDATA")
        out2.append(String(format: "POINTS %d float", n))
        for i in 0..<n {
            out2.append(String(format: "%.6g %.6g %.6g", xArr[i * 2], xArr[i * 2 + 1], fpArr[i] * height))
        }
        out2.append(String(format: "VERTICES %d %d", n, 2 * n))
        for i in 0..<n { out2.append("1 " + String(i)) }
        out2.append(String(format: "POINT_DATA %d", n))
        out2.append("SCALARS label float 1")
        out2.append("LOOKUP_TABLE default")
        out2.append(contentsOf: vtkBlock(yArr))
        out2.append("SCALARS decision float 1")
        out2.append("LOOKUP_TABLE default")
        out2.append(contentsOf: vtkBlock(fpArr))

        do {
            try out.joined(separator: "\n").write(toFile: directory + "/" + surfaceName,
                                                  atomically: true, encoding: .utf8)
            try out2.joined(separator: "\n").write(toFile: directory + "/" + pointsName,
                                                   atomically: true, encoding: .utf8)
            surfaceFiles.append((surfaceName, time))
            pointFiles.append((pointsName, time))
        } catch {
            print("Fehler beim Schreiben: \(error)")
        }
    }

    // Zeit 0: untrainiertes Modell (flache Karte).
    writeSnapshot(0)
    if epochs > 0 {
        for e in 1...epochs {
            svm.fitEpoch(z, y, epoch: e - 1, batchSize: batchSize)
            if e % max(snapshotEvery, 1) == 0 || e == epochs {
                writeSnapshot(e)
            }
        }
    }

    // ParaView-File-Series: kleines JSON, das die Snapshots zur
    // Zeitsequenz macht (Zeitwert = Epochennummer).
    func writeSeries(_ files: [(name: String, time: Int)], _ seriesName: String) {
        var json = "{\n  \"file-series-version\": \"1.0\",\n  \"files\": [\n"
        json += files.map { "    { \"name\": \"\($0.name)\", \"time\": \($0.time) }" }
            .joined(separator: ",\n")
        json += "\n  ]\n}"
        do {
            try json.write(toFile: directory + "/" + seriesName,
                           atomically: true, encoding: .utf8)
        } catch {
            print("Fehler beim Schreiben der Series-Datei: \(error)")
        }
    }
    writeSeries(surfaceFiles, "epoch_surface.vtk.series")
    writeSeries(pointFiles, "epoch_points.vtk.series")

    print("Zeitsequenz geschrieben: \(directory) (\(surfaceFiles.count) Snapshots, je Flaeche + Punkte, Hoehe = f·\(height))")
    print("ParaView: epoch_surface.vtk.series oeffnen (nicht die Einzel-Dateien) – Zeitleiste/Play erscheint automatisch; epoch_points.vtk.series dazuladen, Glyph nach 'label' faerben.")
}
