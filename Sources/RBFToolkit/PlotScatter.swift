//
//  PlotScatter.swift
//  RBFToolkit
//
//  Created by Heiner Strauss on 16.10.26.
//
import Foundation
import MLX

// MARK: - Plotting (interaktives HTML via Plotly, keine Swift-Pakete)

/// Schreibt ein interaktives 2D-Streidiagramm der ersten zwei Dimensionen
/// von x als eigenständige HTML-Datei (Plotly/WebGL): Zoom, Pan, Hover,
/// alle Punkte ohne Dünnung. plotly.js kommt per CDN.
public func plotScatter(x: MLXArray, label: MLXArray, to path: String = "scatter.html") {
    precondition(x.shape[1] >= 2, "x braucht mindestens 2 Dimensionen")
    let n = x.shape[0]
    let xArr = x.asArray(Float.self)      // row-major, n*d
    let yArr = label.asArray(Float.self)  // n Werte

    // Zwei Klassen als getrennte Traces (Legende, einzeln ausblendbar)
    var pX: [String] = [], pY: [String] = [], nX: [String] = [], nY: [String] = []
    for i in 0..<n {
        let a = String(format: "%.4g", xArr[i * 2 + 0])
        let b = String(format: "%.4g", xArr[i * 2 + 1])
        if yArr[i] > 0 { pX.append(a); pY.append(b) }
        else           { nX.append(a); nY.append(b) }
    }

    func trace(_ xs: [String], _ ys: [String], color: String, name: String) -> String {
        "  { type: 'scattergl', mode: 'markers', name: '" + name + "',\n" +
        "    x: [" + xs.joined(separator: ", ") + "], y: [" + ys.joined(separator: ", ") + "],\n" +
        "    marker: { size: 3, color: '" + color + "', opacity: 0.5 } }"
    }

    let html = """
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    </head>
    <body style="margin:0">
      <div id="plot" style="width:100vw;height:100vh"></div>
      <script>
    Plotly.newPlot('plot', [
    \(trace(pX, pY, color: "#1f77b4", name: "Klasse +1")),
    \(trace(nX, nY, color: "#d62728", name: "Klasse -1"))
    ], {
      margin: { l: 50, r: 20, t: 20, b: 50 },
      xaxis: { title: "x[0]" }, yaxis: { title: "x[1]" },
      legend: { orientation: "h" }
    }, { responsive: true });
      </script>
    </body>
    </html>
    """
    do {
        try html.write(toFile: path, atomically: true, encoding: .utf8)
        print("Plot geschrieben: \(path) (\(n) Punkte, interaktiv – im Browser öffnen)")
    } catch {
        print("Fehler beim Schreiben: \(error)")
    }
}
