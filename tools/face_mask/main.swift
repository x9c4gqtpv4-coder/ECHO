import CoreGraphics
import CoreImage
import Foundation
import UniformTypeIdentifiers
import Vision

enum FaceMaskError: LocalizedError {
    case usage
    case unreadableImage(String)
    case cannotRenderMask
    case cannotWrite(String)

    var errorDescription: String? {
        switch self {
        case .usage:
            return "Usage: batch-color-face-mask <input> <output.png>"
        case .unreadableImage(let path):
            return "Cannot read image: \(path)"
        case .cannotRenderMask:
            return "Cannot render the Vision face mask"
        case .cannotWrite(let path):
            return "Cannot write mask: \(path)"
        }
    }
}

@main
struct FaceMaskTool {
    static func normalizedImage(at url: URL, context: CIContext) throws -> CGImage {
        guard let source = CIImage(contentsOf: url, options: [.applyOrientationProperty: true]) else {
            throw FaceMaskError.unreadableImage(url.path)
        }
        let normalized = source.transformed(
            by: CGAffineTransform(translationX: -source.extent.minX, y: -source.extent.minY)
        )
        guard let image = context.createCGImage(normalized, from: normalized.extent) else {
            throw FaceMaskError.unreadableImage(url.path)
        }
        return image
    }

    static func writePNG(_ image: CGImage, to url: URL) throws {
        guard let destination = CGImageDestinationCreateWithURL(
            url as CFURL, UTType.png.identifier as CFString, 1, nil
        ) else {
            throw FaceMaskError.cannotWrite(url.path)
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw FaceMaskError.cannotWrite(url.path)
        }
    }

    static func run() throws {
        let arguments = CommandLine.arguments
        guard arguments.count == 3 else { throw FaceMaskError.usage }
        let inputURL = URL(fileURLWithPath: arguments[1])
        let outputURL = URL(fileURLWithPath: arguments[2])
        let ciContext = CIContext(options: [.useSoftwareRenderer: false])
        let source = try normalizedImage(at: inputURL, context: ciContext)

        let request = VNDetectFaceRectanglesRequest()
        let handler = VNImageRequestHandler(cgImage: source, orientation: .up, options: [:])
        try handler.perform([request])
        let observations = request.results ?? []

        let colorSpace = CGColorSpaceCreateDeviceGray()
        guard let context = CGContext(
            data: nil,
            width: source.width,
            height: source.height,
            bitsPerComponent: 8,
            bytesPerRow: source.width,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) else {
            throw FaceMaskError.cannotRenderMask
        }
        context.setFillColor(gray: 0, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: source.width, height: source.height))
        context.setFillColor(gray: 1, alpha: 1)

        for observation in observations {
            let box = observation.boundingBox
            // CoreGraphics and Vision both use a lower-left origin here.
            let rect = CGRect(
                x: box.minX * CGFloat(source.width),
                y: box.minY * CGFloat(source.height),
                width: box.width * CGFloat(source.width),
                height: box.height * CGFloat(source.height)
            )
            let oval = rect.insetBy(dx: rect.width * 0.10, dy: rect.height * 0.03)
            context.fillEllipse(in: oval)
        }
        guard let rendered = context.makeImage() else {
            throw FaceMaskError.cannotRenderMask
        }
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true
        )
        try writePNG(rendered, to: outputURL)
        print("{\"backend\":\"vision-face\",\"faces\":\(observations.count),\"width\":\(source.width),\"height\":\(source.height)}")
    }

    static func main() {
        do {
            try run()
        } catch {
            FileHandle.standardError.write(Data("face-mask: \(error.localizedDescription)\n".utf8))
            Foundation.exit(1)
        }
    }
}
