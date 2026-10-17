//
//  SamplerUtils.swift
//  MyTool
//
//  Created by Heiner Strauss on 16.10.26.
//
import Foundation
import MLX


/// Rechnet Z[0] auf der CPU nach und vergleicht mit der MLX-Ausgabe.
func verifyFeatureMap(sampler: RBFSamplerMLX, x: MLXArray, z: MLXArray, d: Int) {
    let m = sampler.nComponents
    let scale = Float(sqrt(2.0 / Double(m)))
    let xRow = x.asArray(Float.self).prefix(d)   // Zeile 0
    let wArr = sampler.W.asArray(Float.self)
    let bArr = sampler.b.asArray(Float.self)
    let zArr = z.asArray(Float.self)
    
    var maxErr: Float = 0
    for k in 0..<m {
        var s: Float = 0
        for j in 0..<d { s += xRow[j] * wArr[j * m + k] }
        let ref = scale * cos(s + bArr[k])
        let gpu = zArr[k]
        maxErr = max(maxErr, abs(ref - gpu))
    }
    print(String(format: "Max. Abweichung Feature-Map (Zeile 0): %.2e  (sollte < 1e-5 sein)", maxErr))
}
