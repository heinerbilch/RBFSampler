
//
//  DecisionSurfaceVTK.swift
//  RBFToolkit
//
//  Created by Heiner Strauss on 18.10.26.
//
import Foundation
import MLX

// MARK: - RBFSampler-Visualisierung: Entscheidungsflaeche 3D (VTK für ParaView)

/// Die Luxus-Version der Entscheidungslandschaft für ParaView: das Gitter
/// wird als 3D-Flaeche geschrieben, deren Hoehe die Entscheidungsfunktion
/// ist – f(x) direkt als Gelaende, nicht nur als Farbe. Zwei Dateien pro
/// Aufruf: die Flaeche (STRUCTURED_GRID, Skalar f) und die Trainingspunkte
/// als POLYDATA (Skalare label und decision), auf Hoehe ihrer eigenen
/// Entscheidungswerte ueber der Flaeche.
///
/// ParaView-Rezepte:
/// - Flaeche oeffnen, Color by 'f', diverging Colormap (z.B. Blau-Rot) –
///   Klasse +1 blau (Berg), Klasse -1 rot (Tal), Grenze am Farbwechsel.
/// - Contour-Filter auf f = 0 -> die Grenzkurve auf der Flaeche.
/// - Contour auf f = ±1 -> die Marge; Threshold -1.1 <= f <= 1.1 ->
///   die "Support-Vektor-Schale" zwischen den Margin-Hoehen.
/// - Punkte-Datei daneben laden, Glyph (Sphere), Color by 'label' ->
///   die Klassen liegen auf ihren Huegeln.
/// - height = 0 -> flache Karte, nur Farbe und Konturen.
public func writeVtkDecisionSurface(gamma: Float, shift: Float = 1.2,
                                    nTrain: Int = 4000, nComponents: Int = 500,
                                    epochs: Int = 20, extent: Float = 4,
                                    resolution: Int = 100, height: Float = 0.5,
                                    toSurface surfacePath: String = "decision_surface.vtk",
                                    toPoints pointsPath: String = "surface_points.vtk") {
    // Training wie in plotDecisionLandscape: 2D-Blobs -> Features -> lineare SVM
    let (x, y) = makeBlobs(n: nTrain, d: 2, shift: shift)
    let n = x.shape[0]
    let sampler = RBFSamplerMLX(dimension: 2, nComponents: nComponents,
                                gamma: gamma, seed: 42)
    let z = sampler.transform(x)
    z.eval()
    var svm = LinearSGDSVM(featureCount: nComponents, lambda: 1e-4, learningRate: 0.05)
    svm.fit(z, y, epochs: epochs, batchSize: 256)
    let pred = svm.predict(z)
    pred.eval()
    let acc = 100.0 * Float((pred * y .> 0).sum().item(Int.self)) / Float(n)
    let fPt = svm.decisionFunction(z)
    fPt.eval()
    let fMin = fPt.min().item(Float.self)
    let fMax = fPt.max().item(Float.self)
    print(String(format: "Oberflaeche gamma=%.3g: Genauigkeit %.1f%%, f der Punkte in [%.2f, %.2f]",
                 gamma, acc, fMin, fMax))

    // 1) Flaeche: Gitter, z-Koordinate = f(x)·height
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
    let f = svm.decisionFunction(sampler.transform(xg))
    f.eval()
    let fArr = f.asArray(Float.self)

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

    var out: [String] = []
    out.append("# vtk DataFile Version 3.0")
    out.append("RFF Entscheidungsflaeche, gamma=" + String(format: "%.3g", gamma))
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

    do {
        try out.joined(separator: "\n").write(toFile: surfacePath, atomically: true, encoding: .utf8)
        print("Flaeche geschrieben: \(surfacePath) (STRUCTURED_GRID \(res)x\(res), Hoehe = f·\(height))")
    } catch {
        print("Fehler beim Schreiben: \(error)")
    }

    // 2) Trainingspunkte auf Entscheidungshoehe ueber der Flaeche
    let xArr = x.asArray(Float.self)
    let yArr = y.asArray(Float.self)
    let fpArr = fPt.asArray(Float.self)

    var out2: [String] = []
    out2.append("# vtk DataFile Version 3.0")
    out2.append("RFF Trainingspunkte auf der Entscheidungsflaeche, gamma=" + String(format: "%.3g", gamma))
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
        try out2.joined(separator: "\n").write(toFile: pointsPath, atomically: true, encoding: .utf8)
        print("Punkte geschrieben: \(pointsPath) (\(n) Punkte, auf Hoehe ihrer Entscheidungswerte)")
        print("ParaView: Flaeche nach 'f' faerben (diverging), Contour f=0 = Grenze, Threshold -1.1 bis 1.1 = Support-Vektor-Schale; Punkte als Glyph nach 'label' faerben.")
    } catch {
        print("Fehler beim Schreiben: \(error)")
    }
}
