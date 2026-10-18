
//
//  PlotOmegaCloud.swift
//  RBFToolkit
//
//  Created by Heiner Strauss on 18.10.26.
//
import Foundation
import MLX

// MARK: - RBFSampler-Visualisierung: Frequenzraum (Plotly-HTML)

/// Der Sampler hinter dem Sampler: die Spalten von W sind Wellenvektoren,
/// gezogen aus N(0, 2·gamma) pro Dimension. Für d=2 liegen sie als Punktwolke
/// in der Ebene – jeder Punkt ist eine Ebenewelle mit zufälliger Richtung.
/// Mit dem gleichen Seed sind es für alle gamma-Werte dieselben Richtungen,
/// nur um sqrt(gamma) gestreckt: gamma skaliert die Frequenzen, sonst nichts.
/// Der gestrichelte Kreis ist die 1-Sigma-Kontur pro Achse (Radius sqrt(2·gamma)).
/// Die Phasen b ~ U[0, 2π) kommen als zweite Ziehung dazu und stecken nicht im Bild.
public func plotOmegaCloud(gammas: [Float] = [0.05, 2.0], nComponents: Int = 500,
                           to path: String = "omega_cloud.html") {
    precondition(gammas.count >= 1, "Mindestens ein gamma-Wert")
    precondition(gammas.count <= 6, "Maximal 6 gamma-Werte pro Plot")

    let colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
    var traces: [String] = []
    var shapes: [String] = []

    for (g, gamma) in gammas.enumerated() {
        // Gleicher Seed für alle gamma: gleiche Richtungen, andere Skala.
        let sampler = RBFSamplerMLX(dimension: 2, nComponents: nComponents,
                                    gamma: gamma, seed: 42)
        let wArr = sampler.W.asArray(Float.self)   // 2 × m, row-major
        var xs: [String] = [], ys: [String] = []
        for k in 0..<nComponents {
            xs.append(String(format: "%.4g", wArr[k]))
            ys.append(String(format: "%.4g", wArr[nComponents + k]))
        }
        let sigma = pow(2 * gamma, 0.5)
        let c = colors[g % colors.count]
        traces.append("  { type: 'scatter', mode: 'markers', name: 'gamma=" + String(format: "%.2g", gamma) + "',\n" +
            "    x: [" + xs.joined(separator: ", ") + "], y: [" + ys.joined(separator: ", ") + "],\n" +
            "    marker: { size: 4, color: '" + c + "', opacity: 0.45 } }")
        shapes.append("  { type: 'circle', xref: 'x', yref: 'y',\n" +
            "    x0: " + String(format: "%.4g", -sigma) + ", x1: " + String(format: "%.4g", sigma) + ",\n" +
            "    y0: " + String(format: "%.4g", -sigma) + ", y1: " + String(format: "%.4g", sigma) + ",\n" +
            "    line: { color: '" + c + "', width: 1.5, dash: 'dot' } }")
    }

    let html = """
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8">
      <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    </head>
    <body style="margin:0">
      <div id="plot" style="width:100vw;height:100vh"></div>
      <script>
    Plotly.newPlot('plot', [
    \(traces.joined(separator: ",\n"))
    ], {
      margin: { l: 50, r: 20, t: 60, b: 50 },
      title: "Gezogene Wellenvektoren W (d=2, m=\(nComponents))",
      xaxis: { title: "w[0]", zeroline: true },
      yaxis: { title: "w[1]", zeroline: true, scaleanchor: "x", scaleratio: 1 },
      legend: { orientation: "h" },
      shapes: [
    \(shapes.joined(separator: ",\n"))
      ]
    }, { responsive: true });
      </script>
    </body>
    </html>
    """
    do {
        try html.write(toFile: path, atomically: true, encoding: .utf8)
        print("Frequenzraum-Plot geschrieben: \(path)")
    } catch {
        print("Fehler beim Schreiben: \(error)")
    }
}
