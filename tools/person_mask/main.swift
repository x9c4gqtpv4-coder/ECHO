import CoreImage
import CoreVideo
import Foundation
import UniformTypeIdentifiers
import Vision

enum PersonMaskError: LocalizedError {
    case usage
    case unreadableImage(String)
    case noPerson
    case cannotRenderMask
    case cannotWrite(String)
    case unknownQuality(String)

    var errorDescription: String? {
        switch self {
        case .usage:
            return "Usage: batch-color-person-mask <input> <output.png> [accurate|balanced|fast]"
        case .unreadableImage(let path):
            return "Cannot read image: \(path)"
        case .noPerson:
            return "Vision did not detect a person"
        case .cannotRenderMask:
            return "Cannot render the Vision person mask"
        case .cannotWrite(let path):
            return "Cannot write mask: \(path)"
        case .unknownQuality(let value):
            return "Unknown quality: \(value)"
        }
    }
}

@main
struct PersonMaskTool {
    static func qualityLevel(_ name: String) throws -> VNGeneratePersonSegmentationRequest.QualityLevel {
        switch name.lowercased() {
        case "accurate":
            return .accurate
        case "balanced":
            return .balanced
        case "fast":
            return .fast
        default:
            throw PersonMaskError.unknownQuality(name)
        }
    }

    static func normalizedImage(at url: URL, context: CIContext) throws -> CGImage {
        guard let source = CIImage(
            contentsOf: url,
            options: [.applyOrientationProperty: true]
        ) else {
            throw PersonMaskError.unreadableImage(url.path)
        }
        let normalized = source.transformed(
            by: CGAffineTransform(
                translationX: -source.extent.minX,
                y: -source.extent.minY
            )
        )
        guard let image = context.createCGImage(normalized, from: normalized.extent) else {
            throw PersonMaskError.unreadableImage(url.path)
        }
        return image
    }

    static func writePNG(_ image: CGImage, to url: URL) throws {
        guard let destination = CGImageDestinationCreateWithURL(
            url as CFURL,
            UTType.png.identifier as CFString,
            1,
            nil
        ) else {
            throw PersonMaskError.cannotWrite(url.path)
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw PersonMaskError.cannotWrite(url.path)
        }
    }

    static func run() throws {
        let arguments = CommandLine.arguments
        guard arguments.count == 3 || arguments.count == 4 else {
            throw PersonMaskError.usage
        }

        let inputURL = URL(fileURLWithPath: arguments[1])
        let outputURL = URL(fileURLWithPath: arguments[2])
        let qualityName = arguments.count == 4 ? arguments[3] : "accurate"
        let quality = try qualityLevel(qualityName)

        let context = CIContext(options: [.useSoftwareRenderer: false])
        let source = try normalizedImage(at: inputURL, context: context)

        let request = VNGeneratePersonSegmentationRequest()
        request.qualityLevel = quality
        request.outputPixelFormat = kCVPixelFormatType_OneComponent8
        let handler = VNImageRequestHandler(cgImage: source, orientation: .up, options: [:])
        try handler.perform([request])

        guard let observation = request.results?.first else {
            throw PersonMaskError.noPerson
        }

        let mask = CIImage(cvPixelBuffer: observation.pixelBuffer)
        let scaleX = CGFloat(source.width) / mask.extent.width
        let scaleY = CGFloat(source.height) / mask.extent.height
        let resized = mask.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
        let outputRect = CGRect(x: 0, y: 0, width: source.width, height: source.height)
        guard let rendered = context.createCGImage(
            resized,
            from: outputRect,
            format: .L8,
            colorSpace: CGColorSpaceCreateDeviceGray()
        ) else {
            throw PersonMaskError.cannotRenderMask
        }

        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try writePNG(rendered, to: outputURL)
        print(
            "{\"backend\":\"vision\",\"quality\":\"\(qualityName)\","
            + "\"width\":\(source.width),\"height\":\(source.height)}"
        )
    }

    static func main() {
        do {
            try run()
        } catch {
            FileHandle.standardError.write(
                Data("person-mask: \(error.localizedDescription)\n".utf8)
            )
            Foundation.exit(1)
        }
    }
}
