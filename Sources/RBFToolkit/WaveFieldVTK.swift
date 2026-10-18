//
//  WaveFieldVTK.swift
//  RBFToolkit
//
//  Created by Heiner Strauss on 18.10.26.
//
import Foundation
import MLX

// MARK: - RBFSampler-Visualisierung: 2D-Wellenfeld für ParaView

/// Bildet EINE Komponente eines 2D-Samplers über einem Gitter als
/// Kosinus-Ebenenwelle ab (VTK STRUCTURED_POINTS für ParaView).
/// Sichtbar: jede Komponente ist eine Welle mit zufälliger Richtung
/// (Streifenmuster), Phase und Wellenlänge. Die anderen Komponenten
/// sind dasselbe mit anderen Richtungen – parameter component variiieren.
public func writeVtkWaveField(gamma: Float, component: Int = 0,
                              to path: String = "wave.vtk",
                              extent: Float = 3, resolution: Int = 120,
                              nComponents: Int = 500) {
    let n = resolution * resolution
    var pts: [Float] = []
    pts.reserveCapacity(n * 2)
    // VTK-Reihenfolge: x läuft am schnellsten
    for iy in 0..<resolution {
        let yv = -extent + 2 * extent * Float(iy) / Float(resolution - 1)
        for ix in 0..<resolution {
            let xv = -extent + 2 * extent * Float(ix) / Float(resolution - 1)
            pts.append(xv); pts.append(yv)
        }
    }
    let x = MLXArray(pts).reshaped([n, 2])
    let sampler = RBFSamplerMLX(dimension: 2, nComponents: nComponents,
                                gamma: gamma, seed: 42)
    let z = sampler.transform(x)
    z.eval()
    let zArr = z.asArray(Float.self)

    let spacing = 2 * extent / Float(resolution - 1)
    var out: [String] = []
    out.append("# vtk DataFile Version 3.0")
    out.append("RBFSampler Wellenfeld, Komponente " + String(component))
    out.append("ASCII")
    out.append("DATASET STRUCTURED_POINTS")
    out.append(String(format: "DIMENSIONS %d %d 1", resolution, resolution))
    out.append(String(format: "ORIGIN %.6g %.6g 0", -extent, -extent))
    out.append(String(format: "SPACING %.6g %.6g 1", spacing, spacing))
    out.append(String(format: "POINT_DATA %d", n))
    out.append("SCALARS wave float 1")
    out.append("LOOKUP_TABLE default")
    for start in stride(from: 0, to: n, by: 12) {
        let end = min(start + 12, n)
        out.append((start..<end).map { String(format: "%.6g", zArr[$0 * nComponents + component]) }
            .joined(separator: " "))
    }
    do {
        try out.joined(separator: "\n").write(toFile: path, atomically: true, encoding: .utf8)
        print("Wellenfeld geschrieben: \(path) (Komponente \(component), in ParaView öffnen)")
    } catch {
        print("Fehler beim Schreiben: \(error)")
    }
}
