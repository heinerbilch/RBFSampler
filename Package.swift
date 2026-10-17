// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "RBFToolkit",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "RBFToolkit", targets: ["RBFToolkit"]),
        .executable(name: "RBFToolkitDemo", targets: ["RBFToolkitDemo"])
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.32.0")
    ],
    targets: [
        .target(
            name: "RBFToolkit",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift")
            ]),
        .executableTarget(
            name: "RBFToolkitDemo",
            dependencies: ["RBFToolkit"]),
        .testTarget(
            name: "RBFToolkitTests",
            dependencies: [
                "RBFToolkit",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift")
            ])
    ]
)
