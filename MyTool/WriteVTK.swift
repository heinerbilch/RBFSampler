//
//  WriteVTK.swift
//  MyTool
//
//  Created by Heiner Strauss on 16.10.26.
//

import Foundation
import MLX

// MARK: - Export für VisIt / ParaView (Legacy-VTK, ASCII)

/// Schreibt die Datenpunkte als Legacy-VTK-Datei – das Standard-Einleseformat
/// von VisIt und ParaView. Koordinaten sind die drei gewählten Dimensionen;
/// Label und Entscheidungsfunktion werden als Punkt-Skalare mitgeschrieben,
/// sodass man im Tool nach Klasse färben, Thresholds setzen und die
/// Entscheidungsgrenze explorieren kann.
func writeVtk(x: MLXArray, label: MLXArray, decision: MLXArray? = nil,
              to path: String = "points.vtk", dims: [Int] = [0, 1, 2]) {
    precondition(dims.count == 3, "Genau 3 Koordinaten-Dimensionen angeben")
    let n = x.shape[0], d = x.shape[1]
    let xArr = x.asArray(Float.self)
    let yArr = label.asArray(Float.self)
    let sArr = decision?.asArray(Float.self)

    // VTK mag Zeilen moderater Länge: 12 Werte pro Zeile.
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
    out.append("RBFSampler Testdaten")
    out.append("ASCII")
    out.append("DATASET POLYDATA")
    out.append(String(format: "POINTS %d float", n))
    for i in 0..<n {
        out.append(dims.map { String(format: "%.6g", xArr[i * d + $0]) }
            .joined(separator: " "))
    }
    out.append(String(format: "VERTICES %d %d", n, 2 * n))
    for i in 0..<n { out.append("1 \(i)") }

    out.append(String(format: "POINT_DATA %d", n))
    out.append("SCALARS label float 1")
    out.append("LOOKUP_TABLE default")
    out.append(contentsOf: vtkBlock(yArr))
    if let s = sArr {
        out.append("SCALARS decision float 1")
        out.append("LOOKUP_TABLE default")
        out.append(contentsOf: vtkBlock(s))
    }

    do {
        try out.joined(separator: "\n").write(toFile: path, atomically: true, encoding: .utf8)
        print("VTK geschrieben: \(path) (\(n) Punkte, in VisIt/ParaView öffnen)")
    } catch {
        print("Fehler beim Schreiben: \(error)")
    }
}
