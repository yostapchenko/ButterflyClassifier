import SwiftUI
import CoreML
import Vision
import UIKit

struct ContentView: View {
    @State private var image: UIImage?
    @State private var classificationLabel: String = "Nie wybrano zdjęcia"
    @State private var isShowingImagePicker = false
    @State private var identifiedButterfly: String? = nil
    @State private var isShowingLogin = true
    @State private var history: [String] = []

    // Lista etykiet klas (nazwy motyli)
    let labels = [
        "SOUTHERN DOGFACE", "ADONIS", "BROWN SIPROETA", "MONARCH", "GREEN CELLED CATTLEHEART",
        "CAIRNS BIRDWING", "EASTERN DAPPLE WHITE", "RED POSTMAN", "MANGROVE SKIPPER", "BLACK HAIRSTREAK",
        "CABBAGE WHITE", "RED ADMIRAL", "PAINTED LADY", "PAPER KITE", "SOOTYWING",
        "PINE WHITE", "PEACOCK", "CHECQUERED SKIPPER", "JULIA", "COMMON WOOD-NYMPH",
        "BLUE MORPHO", "CLOUDED SULPHUR", "STRAITED QUEEN", "ORANGE OAKLEAF", "PURPLISH COPPER",
        "ATALA", "IPHICLUS SISTER", "DANAID EGGFLY", "LARGE MARBLE", "PIPEVINE SWALLOW",
        "BLUE SPOTTED CROW", "RED CRACKER", "QUESTION MARK", "CRIMSON PATCH", "BANDED PEACOCK",
        "SCARCE SWALLOW", "COPPER TAIL", "GREAT JAY", "INDRA SWALLOW", "VICEROY",
        "MALACHITE", "APPOLLO", "TWO BARRED FLASHER", "MOURNING CLOAK", "TROPICAL LEAFWING",
        "POPINJAY", "ORANGE TIP", "GOLD BANDED", "BECKERS WHITE", "RED SPOTTED PURPLE",
        "MILBERTS TORTOISESHELL", "SILVER SPOT SKIPPER", "AMERICAN SNOOT", "AN 88", "ULYSES",
        "COMMON BANDED AWL", "CRECENT", "METALMARK", "SLEEPY ORANGE", "PURPLE HAIRSTREAK",
        "ELBOWED PIERROT", "GREAT EGGFLY", "ORCHARD SWALLOW", "ZEBRA LONG WING", "WOOD SATYR",
        "MESTRA", "EASTERN PINE ELFIN", "EASTERN COMA", "YELLOW SWALLOW TAIL", "CLEOPATRA",
        "GREY HAIRSTREAK", "BANDED ORANGE HELICONIAN", "AFRICAN GIANT SWALLOWTAIL", "CHESTNUT", "CLODIUS PARNASSIAN"
    ]

    var body: some View {
        NavigationView {
            VStack {
                if isShowingLogin {
                    LoginView(isShowingLogin: $isShowingLogin)
                } else {
                    // Wyświetl wybrane zdjęcie (jeśli istnieje)
                    if let uiImage = image {
                        Image(uiImage: uiImage)
                            .resizable()
                            .scaledToFit()
                            .frame(height: 300)
                    } else {
                        Rectangle()
                            .fill(Color.gray.opacity(0.2))
                            .overlay(Text("Brak zdjęcia").foregroundColor(.gray))
                            .frame(height: 300)
                    }

                    // Wynik klasyfikacji
                    Text(classificationLabel)
                        .font(.title2)
                        .padding()

                    // Przycisk wyszukiwania informacji o motylu
                    if let butterfly = identifiedButterfly {
                        Button(action: {
                            let searchQuery = (butterfly + " butterfly").addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? ""
                            if let url = URL(string: "https://www.google.com/search?q=\(searchQuery)") {
                                UIApplication.shared.open(url)
                            }
                        }) {
                            Text("Więcej informacji o \(butterfly)")
                                .foregroundColor(.blue)
                        }
                        .padding()
                    }

                    Spacer()

                    // Historia predykcji
                    VStack(alignment: .leading) {
                        Text(" Historia predykcji:")
                            .font(.headline)
                        List(history, id: \.self) { item in
                            Text(item)
                        }
                    }

                    // Przycisk wybierania zdjęcia z galerii
                    Button("Wybierz zdjęcie") {
                        isShowingImagePicker = true
                    }
                    .padding()
                }
            }
            .navigationTitle("Butterfly Classifier")
        }
        .sheet(isPresented: $isShowingImagePicker) {
            ImagePicker(image: $image)
        }
        .onChange(of: image) { newImage in
            if let img = newImage {
                classifyImage(img)
            }
        }
    }

    /// Klasyfikacja obrazu przy pomocy Vision i modelu Core ML
    func classifyImage(_ uiImage: UIImage) {
        guard let ciImage = CIImage(image: uiImage) else {
            classificationLabel = "Nie udało się skonwertować zdjęcia na CIImage."
            return
        }

        let resizedImage = uiImage.resize(to: CGSize(width: 224, height: 224))

        guard let scaledCIImage = CIImage(image: resizedImage) else {
            classificationLabel = "Nie udało się przeskalować obrazu."
            return
        }

        do {
            // Wczytaj model
            let config = MLModelConfiguration()
            let model = try ButterflyClassifier(configuration: config)

            // Stwórz model Vision
            let visionModel = try VNCoreMLModel(for: model.model)

            // Utwórz zapytanie Vision
            let request = VNCoreMLRequest(model: visionModel) { request, error in
                if let results = request.results as? [VNCoreMLFeatureValueObservation],
                   let multiArray = results.first?.featureValue.multiArrayValue {
                    // Przetwórz MultiArray
                    let probabilities = (0..<multiArray.count).compactMap { index in
                        return multiArray[index].floatValue
                    }
                    let total = probabilities.reduce(0, +) // Suma wszystkich wartości
                    let normalizedProbabilities = probabilities.map { $0 / total } // Normalizacja

                    if let maxIndex = normalizedProbabilities.indices.max(by: { normalizedProbabilities[$0] < normalizedProbabilities[$1] }) {
                        if maxIndex < labels.count {
                            let label = labels[maxIndex]

                            DispatchQueue.main.async {
                                self.classificationLabel = "\(label)"
                                self.identifiedButterfly = label
                                self.history.insert(label, at: 0)
                                print("Rozpoznano: \(label)")
                            }
                        } else {
                            DispatchQueue.main.async {
                                self.classificationLabel = "Błąd: Index out of range."
                                self.identifiedButterfly = nil
                            }
                        }
                    } else {
                        DispatchQueue.main.async {
                            self.classificationLabel = "Brak wyników."
                            self.identifiedButterfly = nil
                        }
                    }
                } else {
                    DispatchQueue.main.async {
                        self.classificationLabel = "Błąd: Nie udało się zinterpretować wyników."
                        self.identifiedButterfly = nil
                    }
                }
            }

            let handler = VNImageRequestHandler(ciImage: scaledCIImage, orientation: .up)
            try handler.perform([request])

        } catch {
            classificationLabel = "Błąd klasyfikacji: \(error.localizedDescription)"
            identifiedButterfly = nil
        }
    }
}

// Widok logowania
struct LoginView: View {
    @Binding var isShowingLogin: Bool
    @State private var username: String = ""
    @State private var password: String = ""

    var body: some View {
        VStack {
            Text("Zaloguj się")
                .font(.largeTitle)
                .padding()

            TextField("Nazwa użytkownika", text: $username)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .padding()

            SecureField("Hasło", text: $password)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .padding()

            Button("Zaloguj się") {
                if !username.isEmpty && !password.isEmpty {
                    isShowingLogin = false
                }
            }
            .padding()
        }
        .padding()
    }
}

// ImagePicker do wybierania zdjęć
struct ImagePicker: UIViewControllerRepresentable {
    @Environment(\.presentationMode) var presentationMode
    @Binding var image: UIImage?

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) { }

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    class Coordinator: NSObject, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
        let parent: ImagePicker

        init(_ parent: ImagePicker) {
            self.parent = parent
        }

        func imagePickerController(_ picker: UIImagePickerController,
                                   didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
            if let uiImage = info[.originalImage] as? UIImage {
                parent.image = uiImage
            }
            parent.presentationMode.wrappedValue.dismiss()
        }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            parent.presentationMode.wrappedValue.dismiss()
        }
    }
}

// Rozszerzenie UIImage do zmiany rozmiaru obrazu
extension UIImage {
    func resize(to size: CGSize) -> UIImage {
        UIGraphicsBeginImageContextWithOptions(size, false, 1.0)
        self.draw(in: CGRect(origin: .zero, size: size))
        let resizedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return resizedImage ?? self
    }
}
