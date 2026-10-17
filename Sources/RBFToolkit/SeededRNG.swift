//
//  SeededRNG.swift
//  RBFToolkit
//
//  Created by Heiner Strauss on 16.10.26.
//

/// Deterministischer SplitMix64-RNG – reproduzierbare Folds, gleiche Technik
/// wie in der MPSGraph-Version.
public struct SeededRNG: RandomNumberGenerator {
    public var state: UInt64
    public init(seed: UInt64) { state = seed }
    public mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}
